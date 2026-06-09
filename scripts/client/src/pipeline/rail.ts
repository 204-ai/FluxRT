// Rail — the one video pipeline: camera source → effect chain → output
// stream (WebRTC + preview). Owns getUserMedia and the backend; exposes a
// stable API the UI/stores drive. Framework-agnostic (no React).

import { detectBackend } from './backends/detect'
import { StreamsBackend } from './backends/streamsBackend'
import { CanvasBackend } from './backends/canvasBackend'
import type { RailBackend, RailBackendKind, TapCallback } from './core/types'
import type { DrawLayerConfig } from './effects/drawLayer'
import type { MarkerConfig } from './effects/marker'

export interface RailEvents {
  onLog?: (msg: string) => void
}

export class Rail {
  private backend: RailBackend | null = null
  private rawStream: MediaStream | null = null
  private mirrored = false
  private markerConfig: Partial<MarkerConfig> = {}
  private drawConfig: Partial<DrawLayerConfig> = {}
  private tap: { intervalMs: number; cb: TapCallback } | null = null
  private onLog: (m: string) => void

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

  async start(deviceId: string | null): Promise<{ label: string }> {
    if (this.backend) this.stop()
    this.rawStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    })
    const [track] = this.rawStream.getVideoTracks()
    const s = track.getSettings()
    const width = s.width || 1280
    const height = s.height || 720

    const kind = detectBackend()
    this.backend = kind === 'streams' ? new StreamsBackend(this.onLog) : new CanvasBackend()
    this.onLog(`pipeline backend: ${kind} (${width}x${height})`)

    await this.backend.start(
      {
        deviceId,
        width,
        height,
        fps: 30,
        mirrored: this.mirrored,
        effects: [
          { name: 'marker', config: this.markerConfig },
          { name: 'drawLayer', config: this.drawConfig },
        ],
      },
      this.rawStream,
    )
    if (this.tap) this.backend.setTap(this.tap.intervalMs, this.tap.cb)
    return { label: track.label || 'camera' }
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

  /** Sampled source frames (pre-mirror) for the vision analyzer. */
  setTap(intervalMs: number, cb: TapCallback | null): void {
    this.tap = cb ? { intervalMs, cb } : null
    this.backend?.setTap(intervalMs, cb)
  }

  snapshot(type?: string): Promise<Blob> {
    if (!this.backend) return Promise.reject(new Error('pipeline inactive'))
    return this.backend.snapshot(type)
  }
}
