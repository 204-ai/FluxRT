// Long-lived imperative objects (rail, vision workers) live OUTSIDE React —
// module singletons, created once, immune to StrictMode double-mounts.
// Stores drive them; components only read store state and call actions.

import { Rail } from '../pipeline/rail'
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
    if (this.client && this.caps.face === want.face && this.caps.pose === want.pose) return
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
    rail.setTap(66, (bitmap, tsMs) => {
      if (this.client) this.client.push(bitmap, tsMs)
      else bitmap.close()
    })
  }

  private teardown(): void {
    rail.setTap(0, null)
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
    await client.init({ face: true, pose: true })
    this.timer = window.setInterval(() => void this.sample(), 100)
  }

  private async sample(): Promise<void> {
    const v = this.video
    if (!v || !this.client?.isReady || v.readyState < 2 || v.videoWidth === 0) return
    try {
      const bitmap = await createImageBitmap(v)
      this.client.push(bitmap, performance.now())
    } catch {
      /* video not ready — skip */
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
