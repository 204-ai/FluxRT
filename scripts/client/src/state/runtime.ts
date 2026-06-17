// Long-lived imperative objects (rail, vision workers) live OUTSIDE React —
// module singletons, created once, immune to StrictMode double-mounts.
// Stores drive them; components only read store state and call actions.

import { Rail } from '../pipeline/rail'
import { VideoFileSource } from '../pipeline/videoFileSource'
import { VisionClient } from '../pipeline/visionClient'
import type { VisionResult } from '../vision/types'

type LogFn = (msg: string) => void
let logFn: LogFn = () => {}
export function setRuntimeLogger(fn: LogFn): void {
  logFn = fn
}
export function rtLog(msg: string): void {
  logFn(msg)
}

export const rail = new Rail({ onLog: (m) => rtLog(m) })

/** File-backed <video> input layer for the SEEDED video layer — outlives rail
 *  restarts. Added (overlay) video layers get their own pooled source below, so
 *  multiple video clips can present at once (one element presents one clip). */
export const videoSource = new VideoFileSource()

// Per-clip <video> pool for ADDED video layers (beyond the seeded one). Keyed by
// clip id so each overlay clip drives its own element, transport and listeners.
const videoPool = new Map<string, VideoFileSource>()

/** Get (creating on first use) the pooled video source for a clip. */
export function acquireVideoSource(clipId: string): VideoFileSource {
  let s = videoPool.get(clipId)
  if (!s) {
    s = new VideoFileSource()
    videoPool.set(clipId, s)
  }
  return s
}

/** Unload + drop a clip's pooled video source (on layer removal). */
export function releaseVideoSource(clipId: string): void {
  const s = videoPool.get(clipId)
  if (s) {
    s.unload()
    videoPool.delete(clipId)
  }
}

// Camera device-stream pool. One getUserMedia per device (keyed '' = default);
// each camera clip gets a CLONE of that device's track, so the same webcam can
// back several camera clips and different devices open independently. The base
// device stream is stopped once its last clone is released.
const deviceStreams = new Map<string, MediaStream>()
const deviceRefs = new Map<string, Set<string>>()
const cameraClones = new Map<string, MediaStreamTrack>()
const cloneDevice = new Map<string, string>() // clipId → deviceKey

function camConstraints(deviceId: string): MediaStreamConstraints {
  return {
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30 },
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
    },
  }
}

/** Acquire a camera MediaStream for a clip. The FIRST clip on a device gets the
 *  raw device stream (a MediaStreamTrackProcessor needs the real capture track —
 *  cloned tracks can yield no frames); additional clips on the same device get a
 *  clone so the webcam can back several clips. Re-acquiring for the same clip
 *  releases its prior handle first. */
export async function acquireCamera(clipId: string, deviceId: string): Promise<MediaStream> {
  const key = deviceId || ''
  let base = deviceStreams.get(key)
  if (!base || base.getVideoTracks()[0]?.readyState !== 'live') {
    base = await navigator.mediaDevices.getUserMedia(camConstraints(deviceId))
    deviceStreams.set(key, base)
    deviceRefs.set(key, deviceRefs.get(key) ?? new Set())
  }
  releaseCamera(clipId)
  const refs = deviceRefs.get(key)!
  const track = base.getVideoTracks()[0]
  if (!track) throw new Error('camera has no video track')
  cloneDevice.set(clipId, key)
  refs.add(clipId)
  if (refs.size === 1) {
    // Sole consumer → hand over the raw capture stream (MSTP-friendly).
    return base
  }
  const clone = track.clone()
  cameraClones.set(clipId, clone)
  return new MediaStream([clone])
}

/** Stop a clip's camera clone; stop the device's base stream when its last clone
 *  is released (clears the browser "camera in use" indicator). */
export function releaseCamera(clipId: string): void {
  const clone = cameraClones.get(clipId)
  if (clone) {
    try {
      clone.stop()
    } catch {
      /* already stopped */
    }
    cameraClones.delete(clipId)
  }
  const key = cloneDevice.get(clipId)
  if (key === undefined) return
  cloneDevice.delete(clipId)
  const refs = deviceRefs.get(key)
  if (!refs) return
  refs.delete(clipId)
  if (refs.size === 0) {
    deviceStreams.get(key)?.getTracks().forEach((t) => {
      try {
        t.stop()
      } catch {
        /* already stopped */
      }
    })
    deviceStreams.delete(key)
    deviceRefs.delete(key)
  }
}

export type VisionConsumer = 'marker' | 'sense'

/**
 * Shares one vision worker between the hand-marker (pose only) and the Sense
 * panel (face+pose) on the input rail. Reinitializes the worker when the
 * required capability set grows/shrinks; forwards pose results into the rail
 * bus so the marker effect can read them at render time.
 */
class InputVisionManager {
  private client: VisionClient | null = null
  private caps = { face: false, pose: false }
  private consumers = new Map<VisionConsumer, { face: boolean; pose: boolean }>()
  private listeners = new Set<(r: VisionResult) => void>()
  private statusListeners = new Set<(msg: string) => void>()
  private starting: Promise<void> | null = null

  subscribe(l: (r: VisionResult) => void): () => void {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }

  onStatus(l: (msg: string) => void): () => void {
    this.statusListeners.add(l)
    return () => this.statusListeners.delete(l)
  }

  async acquire(consumer: VisionConsumer, caps: { face: boolean; pose: boolean }): Promise<void> {
    this.consumers.set(consumer, caps)
    await this.reconcile()
  }

  async release(consumer: VisionConsumer): Promise<void> {
    this.consumers.delete(consumer)
    await this.reconcile()
  }

  private wanted(): { face: boolean; pose: boolean } | null {
    if (this.consumers.size === 0) return null
    let face = false
    let pose = false
    for (const c of this.consumers.values()) {
      face = face || c.face
      pose = pose || c.pose
    }
    return { face, pose }
  }

  private async reconcile(): Promise<void> {
    if (this.starting) await this.starting.catch(() => {})
    const want = this.wanted()
    if (!want) {
      this.teardown()
      return
    }
    // Keep the existing worker if it already provides everything wanted — only
    // rebuild to ADD a capability, never to drop one. Tearing down on a shrink
    // would kill the worker a still-active consumer depends on (e.g. releasing
    // 'sense' must not stall the 'marker' that still needs pose).
    if (this.client && (!want.face || this.caps.face) && (!want.pose || this.caps.pose)) return
    this.teardown()
    const client = new VisionClient()
    this.client = client
    this.caps = want
    client.onStatus((m) => this.statusListeners.forEach((l) => l(m)))
    client.subscribe((r) => {
      if (r.poseLandmarks.length) rail.busPush('pose', { landmarks: r.poseLandmarks[0] })
      this.listeners.forEach((l) => l(r))
    })
    this.starting = client.init(want)
    try {
      await this.starting
    } catch (e) {
      rtLog('vision init failed: ' + (e instanceof Error ? e.message : e))
      this.teardown()
      throw e
    } finally {
      this.starting = null
    }
    // ~15 Hz sampling: smooth enough for marker + sense, far below video rate.
    // setTap sets the worker's tap CADENCE (and is the canvas backend's frame
    // path); the streams backend's frames flow through the direct port below.
    rail.setTap(66, (bitmap, tsMs) => {
      if (this.client) this.client.push(bitmap, tsMs)
      else bitmap.close()
    })
    // Streams backend: hand the pipeline worker a direct frame-port to this
    // vision worker so composite frames skip the main thread. Re-minted per rail
    // (re)start. The canvas backend ignores it and uses the tap callback above.
    rail.setVisionPortFactory(() => (this.client ? this.client.mintFramePort() : null))
  }

  private teardown(): void {
    rail.setTap(0, null)
    rail.setVisionPortFactory(null)
    this.client?.dispose()
    this.client = null
    this.caps = { face: false, pose: false }
  }
}

export const inputVision = new InputVisionManager()

/**
 * Sense on the remote AI output: its own vision worker fed by sampling the
 * remote <video> element at ~10 Hz on the main thread.
 */
class OutputVisionManager {
  private client: VisionClient | null = null
  private timer = 0
  private sampling = false
  private video: HTMLVideoElement | null = null
  private listeners = new Set<(r: VisionResult) => void>()
  private statusListeners = new Set<(msg: string) => void>()

  subscribe(l: (r: VisionResult) => void): () => void {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }

  onStatus(l: (msg: string) => void): () => void {
    this.statusListeners.add(l)
    return () => this.statusListeners.delete(l)
  }

  setVideo(el: HTMLVideoElement | null): void {
    this.video = el
  }

  async start(): Promise<void> {
    if (this.client) return
    const client = new VisionClient()
    this.client = client
    client.onStatus((m) => this.statusListeners.forEach((l) => l(m)))
    client.subscribe((r) => this.listeners.forEach((l) => l(r)))
    try {
      await client.init({ face: true, pose: true })
    } catch (e) {
      // Init failed — clear the slot and drop the worker so a later start()
      // retries instead of early-returning on a permanently-dead client.
      this.client = null
      client.dispose()
      throw e
    }
    this.timer = window.setInterval(() => void this.sample(), 100)
  }

  private async sample(): Promise<void> {
    const v = this.video
    if (this.sampling || !v || !this.client?.isReady || v.readyState < 2 || v.videoWidth === 0) return
    this.sampling = true
    try {
      const bitmap = await createImageBitmap(v)
      // stop() may have run during the await — don't push to a disposed client.
      if (this.client) this.client.push(bitmap, performance.now())
      else bitmap.close()
    } catch {
      /* video not ready — skip */
    } finally {
      this.sampling = false
    }
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer)
    this.timer = 0
    this.client?.dispose()
    this.client = null
  }
}

export const outputVision = new OutputVisionManager()
