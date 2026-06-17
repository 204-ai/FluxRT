// Canvas fallback backend (Safari/Firefox): hidden <video> elements (one per
// stream-backed layer) + a video-file element + rVFC/rAF loop + main-thread
// compositing + canvas.captureStream() — the legacy InputProcessor approach
// behind the same RailBackend API. The same effect implementations run here as
// in the pipeline worker. Video-file layers are drawn straight from their
// element (Safari lacks HTMLMediaElement.captureStream).

import { Compositor, type FrameMap } from '../core/compositor'
import type {
  ClipKind,
  CompositeOp,
  LayerId,
  LayerSourceInput,
  RailBackend,
  RailStartOptions,
  SourceSet,
  TapCallback,
} from '../core/types'
import { clipMeta } from '../core/clipKinds'

/** One layer's draw source: a <video> element. `owned` elements were created
 *  here (camera / feedback, fed a srcObject) and are torn down on remove; an
 *  un-owned element (a video file) belongs to its caller — we only hold a ref. */
interface ElementSource {
  el: HTMLVideoElement
  owned: boolean
  kind: ClipKind
}

export class CanvasBackend implements RailBackend {
  readonly kind = 'canvas' as const
  readonly previewEl: HTMLCanvasElement
  outputStream: MediaStream = new MediaStream()

  private sources = new Map<LayerId, ElementSource>()
  private baseLayerId: LayerId = ''
  private compositor: Compositor | null = null
  private rafId = 0
  private rvfcId = 0
  private rvfcEl: HTMLVideoElement | null = null
  private running = false
  private composeErrored = false
  private tapCb: TapCallback | null = null
  private tapIntervalMs = 0
  private lastTapMs = 0
  private tapBusy = false

  constructor() {
    this.previewEl = document.createElement('canvas')
  }

  async start(opts: RailStartOptions, { base }: SourceSet): Promise<void> {
    if (base.stream) {
      const video = document.createElement('video')
      video.srcObject = base.stream
      video.muted = true
      video.playsInline = true
      await video.play()
      this.sources.set(base.layerId, { el: video, owned: true, kind: base.kind })
    } else if (base.videoEl) {
      // Direct reference — playback state belongs to the element's owner.
      this.sources.set(base.layerId, { el: base.videoEl, owned: false, kind: base.kind })
    } else {
      throw new Error('no base source')
    }
    this.baseLayerId = base.layerId

    this.previewEl.width = opts.width
    this.previewEl.height = opts.height
    const ctx = this.previewEl.getContext('2d')
    if (!ctx) throw new Error('2d context unavailable')
    this.compositor = new Compositor(ctx, opts.width, opts.height)
    this.compositor.setComposite(opts.composite)
    this.compositor.setEffects(opts.effects)

    this.outputStream = this.previewEl.captureStream(opts.fps)
    this.running = true

    const onFrame = () => {
      if (!this.running || !this.compositor) return
      const tsMs = performance.now()
      try {
        const frames: FrameMap = {}
        for (const [id, s] of this.sources) frames[id] = s.el
        this.compositor.drawComposite(frames, tsMs)
        this.maybeTap(tsMs)
      } catch (err) {
        // One bad frame (e.g. a source element transiently in an invalid state)
        // must never kill the loop — that would freeze the output/captureStream
        // for good. Keep scheduling; warn once so a persistent fault is visible.
        if (!this.composeErrored) {
          this.composeErrored = true
          console.warn('canvas composite frame failed (loop kept alive):', err)
        }
      }
      this.schedule(onFrame)
    }
    this.schedule(onFrame)
  }

  /** rVFC doesn't fire on a paused video — with a file layer, drive by rAF so
   *  seek-while-paused and live composite tweaks still redraw. A live
   *  camera/feedback stream needs no rAF: it's redrawn at the base camera's rVFC
   *  cadence (matching the worker), and rAF would freeze the output while the
   *  tab is backgrounded. */
  private schedule(onFrame: () => void): void {
    if (!this.running) return
    this.rvfcEl = null
    const base = this.sources.get(this.baseLayerId)
    const hasFileVideo = [...this.sources.values()].some((s) => s.kind === 'video')
    if (
      !hasFileVideo &&
      base &&
      base.kind !== 'video' &&
      'requestVideoFrameCallback' in HTMLVideoElement.prototype
    ) {
      this.rvfcEl = base.el
      this.rvfcId = base.el.requestVideoFrameCallback(() => onFrame())
    } else {
      this.rafId = requestAnimationFrame(onFrame)
    }
  }

  private maybeTap(tsMs: number): void {
    if (!this.tapCb || this.tapBusy || tsMs - this.lastTapMs < this.tapIntervalMs) return
    // Sample the COMPOSITE canvas so sensing reflects all layers.
    if (this.previewEl.width === 0) return
    this.lastTapMs = tsMs
    this.tapBusy = true
    createImageBitmap(this.previewEl)
      .then((bitmap) => {
        this.tapBusy = false
        if (this.tapCb) this.tapCb(bitmap, tsMs)
        else bitmap.close()
      })
      .catch(() => {
        this.tapBusy = false
      })
  }

  stop(): void {
    this.running = false
    if (this.rafId) cancelAnimationFrame(this.rafId)
    if (this.rvfcId && this.rvfcEl) {
      try {
        this.rvfcEl.cancelVideoFrameCallback(this.rvfcId)
      } catch {
        /* ignore */
      }
    }
    this.rvfcEl = null
    this.outputStream.getTracks().forEach((t) => t.stop())
    for (const s of this.sources.values()) if (s.owned) s.el.srcObject = null
    this.sources.clear()
    this.compositor?.disposeEffects()
    this.compositor = null
    this.tapCb = null
  }

  setComposite(op: CompositeOp): void {
    this.compositor?.setComposite(op)
  }

  setLayerSource(id: LayerId, kind: ClipKind, source: LayerSourceInput): void {
    if (!source) {
      const s = this.sources.get(id)
      if (s?.owned) s.el.srcObject = null
      this.sources.delete(id)
      return
    }
    if (clipMeta(kind).sourceForm === 'element') {
      // Video file: the same element is redrawn every frame, so a new src is
      // picked up automatically — just keep the reference current.
      this.sources.set(id, { el: source as HTMLVideoElement, owned: false, kind })
      return
    }
    // Stream-backed (camera / feedback): re-point an owned hidden <video>, or
    // create one. Multiple elements may share one MediaStream.
    const stream = source as MediaStream
    const existing = this.sources.get(id)
    const el = existing?.owned ? existing.el : document.createElement('video')
    el.muted = true
    el.playsInline = true
    el.srcObject = stream
    void el.play().catch(() => {})
    this.sources.set(id, { el, owned: true, kind })
  }

  configureEffect(name: string, patch: Record<string, unknown>): void {
    this.compositor?.configureEffect(name, patch)
  }

  effectMessage(name: string, data: unknown): void {
    this.compositor?.effectMessage(name, data)
  }

  busPush(key: string, value: unknown): void {
    this.compositor?.bus.set(key, value)
  }

  setTap(intervalMs: number, cb: TapCallback | null): void {
    this.tapIntervalMs = intervalMs
    this.tapCb = cb
  }

  async snapshot(type = 'image/png'): Promise<Blob> {
    return new Promise((res, rej) =>
      this.previewEl.toBlob((b) => (b ? res(b) : rej(new Error('snapshot failed'))), type),
    )
  }
}
