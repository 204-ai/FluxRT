// Rail — the one video pipeline: sources (camera and/or video file) → effect
// chain → output stream (WebRTC + preview). Owns getUserMedia and the
// backend; exposes a stable API the UI/stores drive. Framework-agnostic
// (no React).

import { detectBackend } from './backends/detect'
import { StreamsBackend } from './backends/streamsBackend'
import { CanvasBackend } from './backends/canvasBackend'
import type { CompositeOptions, RailBackend, RailBackendKind, TapCallback } from './core/types'
import type { DrawLayerConfig } from './effects/drawLayer'
import type { MarkerConfig } from './effects/marker'

export interface RailEvents {
  onLog?: (msg: string) => void
}

export interface RailSources {
  deviceId: string | null
  camera: boolean
  videoEl: HTMLVideoElement | null
  /** Force the output/canvas aspect ratio (width/height); the source is
   *  cover-cropped into it. Used to match the server's generation aspect. */
  targetAspect?: number | null
}

/** Longest output edge when the video file is the sole source. */
const MAX_VIDEO_EDGE = 1280

function videoDims(el: HTMLVideoElement): { width: number; height: number } {
  let w = el.videoWidth || 1280
  let h = el.videoHeight || 720
  const long = Math.max(w, h)
  if (long > MAX_VIDEO_EDGE) {
    const scale = MAX_VIDEO_EDGE / long
    w *= scale
    h *= scale
  }
  // Even dims keep downstream encoders happy.
  return { width: Math.round(w / 2) * 2, height: Math.round(h / 2) * 2 }
}

/** Canvas dims forced to a target aspect (the server's output aspect), keeping
 *  the source height; the source is cover-cropped into this by the compositor. */
function aspectDims(srcHeight: number, aspect: number): { width: number; height: number } {
  let h = srcHeight || 720
  let w = h * aspect
  const long = Math.max(w, h)
  if (long > MAX_VIDEO_EDGE) {
    const scale = MAX_VIDEO_EDGE / long
    w *= scale
    h *= scale
  }
  return { width: Math.round(w / 2) * 2, height: Math.round(h / 2) * 2 }
}

export class Rail {
  private backend: RailBackend | null = null
  private rawStream: MediaStream | null = null
  private mirrored = false
  private markerConfig: Partial<MarkerConfig> = {}
  private drawConfig: Partial<DrawLayerConfig> = {}
  private composite: CompositeOptions = { order: 'camera-over', opacity: 0.5, blend: 'normal' }
  private tap: { intervalMs: number; cb: TapCallback } | null = null
  private onLog: (m: string) => void
  private onOutputTrack: (track: MediaStreamTrack | null) => void = () => {}

  constructor(events: RailEvents = {}) {
    this.onLog = events.onLog ?? (() => {})
  }

  get active(): boolean {
    return this.backend !== null
  }

  get backendKind(): RailBackendKind | null {
    return this.backend?.kind ?? null
  }

  get previewEl(): HTMLCanvasElement | HTMLVideoElement | null {
    return this.backend?.previewEl ?? null
  }

  get outputStream(): MediaStream | null {
    return this.backend?.outputStream ?? null
  }

  async start(sources: RailSources): Promise<{ label: string }> {
    if (this.backend) this.stop()
    if (!sources.camera && !sources.videoEl) throw new Error('no input source')

    let label = 'video file'
    let width: number
    let height: number
    if (sources.camera) {
      this.rawStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 },
          ...(sources.deviceId ? { deviceId: { exact: sources.deviceId } } : {}),
        },
      })
      const [track] = this.rawStream.getVideoTracks()
      const s = track.getSettings()
      width = s.width || 1280
      height = s.height || 720
      label = track.label || 'camera'
      if (sources.videoEl) label += ' + video file'
    } else {
      ;({ width, height } = videoDims(sources.videoEl!))
    }

    // Force the output aspect to match the server's generation aspect; the
    // source is cover-cropped into it (no stretch/letterbox).
    if (sources.targetAspect && sources.targetAspect > 0) {
      ;({ width, height } = aspectDims(height, sources.targetAspect))
    }

    const kind = detectBackend()
    this.backend = kind === 'streams' ? new StreamsBackend(this.onLog) : new CanvasBackend()
    this.onLog(`pipeline backend: ${kind} (${width}x${height}) [${label}]`)

    await this.backend.start(
      {
        width,
        height,
        fps: 30,
        mirrored: this.mirrored,
        composite: { ...this.composite },
        effects: [
          { name: 'marker', config: this.markerConfig },
          { name: 'drawLayer', config: this.drawConfig },
        ],
      },
      { cameraStream: this.rawStream, videoEl: sources.videoEl },
    )
    if (this.tap) this.backend.setTap(this.tap.intervalMs, this.tap.cb)
    // A (re)start builds a brand-new output track; notify so the session can
    // replaceTrack on its live sender (a restart otherwise strands the peer
    // connection on the old, ended track — frozen remote output).
    const [outTrack] = this.backend.outputStream.getVideoTracks()
    this.onOutputTrack(outTrack ?? null)
    return { label }
  }

  /** Notified with the new output video track on every (re)start. The session
   *  uses it to replaceTrack on its RTCRtpSender without renegotiation. */
  setOutputTrackHandler(fn: (track: MediaStreamTrack | null) => void): void {
    this.onOutputTrack = fn
  }

  /** Hot-swap the video-file source in place (no restart): re-feed the worker
   *  the element's new track while keeping the output stream/track alive. */
  swapVideoSource(videoEl: HTMLVideoElement): void {
    this.backend?.swapVideo(videoEl)
  }

  /** Hot-remove the video-file overlay in place (no restart): drop the overlay
   *  layer while the camera keeps feeding and the output stream stays alive. */
  clearVideoSource(): void {
    this.backend?.clearVideo()
  }

  /** Hot-swap the camera device in place (no restart): acquire the new device's
   *  stream, feed it to the backend, then stop the old one. Keeps the output
   *  track / WebRTC sender alive. */
  async swapCameraDevice(deviceId: string | null): Promise<void> {
    if (!this.backend) return
    const newStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    })
    this.backend.swapCamera(newStream)
    // Stop the previous camera track now that the new one is feeding.
    this.rawStream?.getTracks().forEach((t) => {
      try {
        t.stop()
      } catch {
        /* already stopped */
      }
    })
    this.rawStream = newStream
  }

  stop(): void {
    this.backend?.stop()
    this.backend = null
    this.rawStream?.getTracks().forEach((t) => {
      try {
        t.stop()
      } catch {
        /* already stopped */
      }
    })
    this.rawStream = null
  }

  setMirror(on: boolean): void {
    this.mirrored = on
    this.backend?.setMirror(on)
  }

  setComposite(patch: Partial<CompositeOptions>): void {
    Object.assign(this.composite, patch)
    this.backend?.setComposite(patch)
  }

  configureMarker(patch: Partial<MarkerConfig>): void {
    Object.assign(this.markerConfig, patch)
    this.backend?.configureEffect('marker', patch)
  }

  configureDraw(patch: Partial<DrawLayerConfig>): void {
    Object.assign(this.drawConfig, patch)
    this.backend?.configureEffect('drawLayer', patch)
  }

  /** Stroke coords are normalized [0..1] relative to the preview element. */
  beginStroke(x: number, y: number): void {
    this.backend?.effectMessage('drawLayer', { type: 'begin', x, y })
  }
  moveStroke(x: number, y: number): void {
    this.backend?.effectMessage('drawLayer', { type: 'move', x, y })
  }
  endStroke(): void {
    this.backend?.effectMessage('drawLayer', { type: 'end' })
  }
  clearDrawing(): void {
    this.backend?.effectMessage('drawLayer', { type: 'clear' })
  }

  /** Push analyzer results to worker-side effects (e.g. pose for the marker). */
  busPush(key: string, value: unknown): void {
    this.backend?.busPush(key, value)
  }

  /** Sampled COMPOSITE frames (already mirrored when the camera mirror is on)
   *  for the vision analyzer — NOT pre-mirror base frames. Consumers must treat
   *  landmarks as being in the same (mirrored) space the preview displays, so
   *  they render 1:1 with no extra flip (see marker.ts / OverlayCanvas). */
  setTap(intervalMs: number, cb: TapCallback | null): void {
    this.tap = cb ? { intervalMs, cb } : null
    this.backend?.setTap(intervalMs, cb)
  }

  snapshot(type?: string): Promise<Blob> {
    if (!this.backend) return Promise.reject(new Error('pipeline inactive'))
    return this.backend.snapshot(type)
  }
}
