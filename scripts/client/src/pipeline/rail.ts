// Rail — the one video pipeline: sources (camera and/or video file) → effect
// chain → output stream (WebRTC + preview). Owns getUserMedia and the
// backend; exposes a stable API the UI/stores drive. Framework-agnostic
// (no React).

import { detectBackend } from './backends/detect'
import { StreamsBackend } from './backends/streamsBackend'
import { CanvasBackend } from './backends/canvasBackend'
import type {
  BaseSource,
  ClipKind,
  Composite,
  CompositeOp,
  LayerId,
  LayerSourceInput,
  RailBackend,
  RailBackendKind,
  TapCallback,
} from './core/types'
import {
  applyCompositeOp,
  CAMERA_LAYER,
  defaultComposite,
  FEEDBACK_LAYER,
  hasVideoTrack,
  VIDEO_LAYER,
} from './core/types'
import type { DrawLayerConfig, StrokeMessage } from './effects/drawLayer'
import type { MarkerConfig } from './effects/marker'

export interface RailEvents {
  onLog?: (msg: string) => void
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

export class Rail {
  private backend: RailBackend | null = null
  private markerConfig: Partial<MarkerConfig> = {}
  private drawConfig: Partial<DrawLayerConfig> = {}
  private composite: Composite = defaultComposite()
  // The remote output stream fed back in as a layer. Remembered so a pipeline
  // restart (camera/video toggle) re-attaches it to the new backend.
  private feedbackStream: MediaStream | null = null
  // Draw ops recorded so a pipeline restart (which rebuilds the draw layer from
  // scratch) can replay the strokes instead of wiping the user's drawing. Each
  // stroke is preceded by a 'cfg' marker capturing its color/size at draw time.
  private drawHistory: Array<StrokeMessage | { type: 'cfg'; patch: Partial<DrawLayerConfig> }> = []
  // True only between begin/end so hover-driven pointermove (no button held)
  // is ignored — otherwise it would flood drawHistory and the worker with no-ops.
  private inStroke = false
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

  /** Start the pipeline on an already-acquired BASE source (the store owns
   *  getUserMedia/getDisplayMedia/file). Output dims come from `outDims` when
   *  given (the server's output resolution → preview matches output), else from
   *  the base source; every other clip hot-attaches afterward via setLayerSource. */
  async start(base: BaseSource, outDims?: { width: number; height: number } | null): Promise<{ label: string }> {
    if (this.backend) this.stop()

    let label: string
    let width: number
    let height: number
    if (base.stream) {
      const [track] = base.stream.getVideoTracks()
      if (!track) throw new Error('base source has no video track')
      const s = track.getSettings()
      width = s.width || 1280
      height = s.height || 720
      label = track.label || base.kind
    } else if (base.videoEl) {
      ;({ width, height } = videoDims(base.videoEl))
      label = 'video file'
    } else {
      throw new Error('no base source')
    }

    // Pin the output canvas to the server's generation resolution when known —
    // the input composite (and the frames sent upstream) then match the output's
    // aspect AND resolution exactly; sources are cover-cropped into it (no stretch
    // / letterbox). Dims are fixed at start and never derive from a single layer's
    // liveness, so activating/deactivating a clip can't change them.
    if (outDims && outDims.width > 0 && outDims.height > 0) {
      width = outDims.width
      height = outDims.height
    }

    const kind = detectBackend()
    this.backend = kind === 'streams' ? new StreamsBackend(this.onLog) : new CanvasBackend()
    this.onLog(`pipeline backend: ${kind} (${width}x${height}) [${label}]`)

    await this.backend.start(
      {
        width,
        height,
        fps: 30,
        composite: this.composite,
        effects: [
          { name: 'marker', config: this.markerConfig },
          { name: 'drawLayer', config: this.drawConfig },
        ],
      },
      { base },
    )
    if (this.tap) this.backend.setTap(this.tap.intervalMs, this.tap.cb)
    // Replay any persisted drawing onto the freshly-built draw layer so a
    // source-set restart doesn't wipe the user's strokes.
    if (this.drawHistory.length) {
      for (const m of this.drawHistory) {
        if (m.type === 'cfg') this.backend.configureEffect('drawLayer', m.patch)
        else this.backend.effectMessage('drawLayer', m)
      }
      // Replay left the layer's config at the last stroke's — restore the live one.
      this.backend.configureEffect('drawLayer', this.drawConfig)
    }
    // Re-attach the feedback layer (remote output) onto the freshly-built backend
    // so a source-set restart keeps the output→input loop alive across it. Guard:
    // a stale/ended remote track (track.clone / MSTP can throw) must NOT abort the
    // start and tear down the whole input pipeline — just drop the feedback layer.
    if (this.feedbackStream) {
      try {
        this.backend.setLayerSource(FEEDBACK_LAYER, 'feedback', this.feedbackStream)
      } catch (e) {
        this.onLog('feedback re-attach failed: ' + (e instanceof Error ? e.message : e))
        this.feedbackStream = null
      }
    }
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

  /** Hot add/swap/remove one layer's live source in place (no restart): keeps
   *  the output stream/track alive. Generalizes the old swap-video/swap-camera/
   *  clear-video/set-feedback paths. */
  setLayerSource(id: LayerId, kind: ClipKind, source: LayerSourceInput): void {
    this.backend?.setLayerSource(id, kind, source)
  }

  /** Hot-swap the video-file source in place (no restart): re-feed the worker
   *  the element's new track while keeping the output stream/track alive. */
  swapVideoSource(videoEl: HTMLVideoElement): void {
    this.backend?.setLayerSource(VIDEO_LAYER, 'video', videoEl)
  }

  /** Hot-remove the video-file overlay in place (no restart): drop the overlay
   *  layer while the camera keeps feeding and the output stream stays alive. */
  clearVideoSource(): void {
    this.backend?.setLayerSource(VIDEO_LAYER, 'video', null)
  }

  stop(): void {
    this.backend?.stop()
    this.backend = null
    // Camera/screen streams are owned by the store now (it acquires + releases).
  }

  /** Per-layer selfie flip. In the migration era this targets the camera layer
   *  (selfie view) — once clips own their mirror (P1+) the store drives it via
   *  a composite patch directly. */
  setMirror(on: boolean): void {
    this.setComposite({ op: 'patch', layers: [{ id: CAMERA_LAYER, mirror: on }] })
  }

  /** Apply a structural/mix op to the composite (patch/add/remove/reorder),
   *  remembered so a pipeline restart re-establishes it. */
  setComposite(op: CompositeOp): void {
    applyCompositeOp(this.composite, op)
    this.backend?.setComposite(op)
  }

  /** Replace the whole composite (the reconciler's "match this stack"). */
  setCompositeAll(composite: Composite): void {
    this.setComposite({ op: 'replace', layers: composite })
  }

  /** Set or clear the feedback layer — the remote output stream fed back into
   *  the input compositor. Remembered so a pipeline restart re-attaches it. */
  setFeedback(stream: MediaStream | null): void {
    this.feedbackStream = hasVideoTrack(stream) ? stream : null
    this.backend?.setLayerSource(FEEDBACK_LAYER, 'feedback', this.feedbackStream)
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
    this.inStroke = true
    // Snapshot the active config so a replay reproduces this stroke's color/size.
    this.drawHistory.push({ type: 'cfg', patch: { ...this.drawConfig } }, { type: 'begin', x, y })
    this.backend?.effectMessage('drawLayer', { type: 'begin', x, y })
  }
  moveStroke(x: number, y: number): void {
    if (!this.inStroke) return // hover with no active stroke — nothing to draw
    this.drawHistory.push({ type: 'move', x, y })
    this.backend?.effectMessage('drawLayer', { type: 'move', x, y })
  }
  endStroke(): void {
    if (!this.inStroke) return
    this.inStroke = false
    this.drawHistory.push({ type: 'end' })
    this.backend?.effectMessage('drawLayer', { type: 'end' })
  }
  clearDrawing(): void {
    this.inStroke = false
    this.drawHistory = []
    this.backend?.effectMessage('drawLayer', { type: 'clear' })
  }

  /** Push analyzer results to worker-side effects (e.g. pose for the marker). */
  busPush(key: string, value: unknown): void {
    this.backend?.busPush(key, value)
  }

  /** Sampled COMPOSITE frames (already mirrored when a layer's mirror is on)
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
