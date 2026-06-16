// Canvas fallback backend (Safari/Firefox): hidden <video> + rVFC/rAF loop +
// main-thread compositing + canvas.captureStream() — the legacy
// InputProcessor approach behind the same RailBackend API. The same effect
// implementations run here as in the pipeline worker. The video-file layer is
// drawn straight from its element (Safari lacks HTMLMediaElement.captureStream).

import { Compositor } from '../core/compositor'
import type { CompositePatch, RailBackend, RailStartOptions, SourceSet, TapCallback } from '../core/types'
import { hasVideoTrack } from '../core/types'

export class CanvasBackend implements RailBackend {
  readonly kind = 'canvas' as const
  readonly previewEl: HTMLCanvasElement
  outputStream: MediaStream = new MediaStream()

  private hiddenVideo: HTMLVideoElement | null = null
  private fileVideo: HTMLVideoElement | null = null
  // Hidden <video> playing the remote output stream, drawn as the feedback layer.
  private feedbackVideo: HTMLVideoElement | null = null
  private compositor: Compositor | null = null
  private rafId = 0
  private rvfcId = 0
  private running = false
  private composeErrored = false
  private tapCb: TapCallback | null = null
  private tapIntervalMs = 0
  private lastTapMs = 0
  private tapBusy = false

  constructor() {
    this.previewEl = document.createElement('canvas')
  }

  async start(opts: RailStartOptions, sources: SourceSet): Promise<void> {
    if (sources.cameraStream) {
      const video = document.createElement('video')
      video.srcObject = sources.cameraStream
      video.muted = true
      video.playsInline = true
      await video.play()
      this.hiddenVideo = video
    }
    // Direct reference — playback state belongs to the element's owner.
    this.fileVideo = sources.videoEl

    this.previewEl.width = opts.width
    this.previewEl.height = opts.height
    const ctx = this.previewEl.getContext('2d')
    if (!ctx) throw new Error('2d context unavailable')
    this.compositor = new Compositor(ctx, opts.width, opts.height)
    this.compositor.mirrored = opts.mirrored
    this.compositor.setComposite(opts.composite)
    this.compositor.setEffects(opts.effects)

    this.outputStream = this.previewEl.captureStream(opts.fps)
    this.running = true

    const onFrame = () => {
      if (!this.running || !this.compositor) return
      const tsMs = performance.now()
      try {
        this.compositor.drawComposite(this.hiddenVideo, this.fileVideo, this.feedbackVideo, tsMs)
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
      schedule()
    }
    const schedule = () => {
      if (!this.running) return
      // rVFC doesn't fire on a paused video — with a file layer, drive by rAF so
      // seek-while-paused and live composite tweaks still redraw. A live feedback
      // stream needs no rAF: it's redrawn at the camera's rVFC cadence (matching
      // the worker, where the camera base drives the loop), and rAF would freeze
      // the output while the tab is backgrounded.
      if (
        !this.fileVideo &&
        this.hiddenVideo &&
        'requestVideoFrameCallback' in HTMLVideoElement.prototype
      ) {
        this.rvfcId = this.hiddenVideo.requestVideoFrameCallback(() => onFrame())
      } else {
        this.rafId = requestAnimationFrame(onFrame)
      }
    }
    schedule()
  }

  private maybeTap(tsMs: number): void {
    if (!this.tapCb || this.tapBusy || tsMs - this.lastTapMs < this.tapIntervalMs) return
    // Sample the COMPOSITE canvas so sensing reflects camera + video.
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
    if (this.rvfcId && this.hiddenVideo) {
      try {
        this.hiddenVideo.cancelVideoFrameCallback(this.rvfcId)
      } catch {
        /* ignore */
      }
    }
    this.outputStream.getTracks().forEach((t) => t.stop())
    if (this.hiddenVideo) {
      this.hiddenVideo.srcObject = null
      this.hiddenVideo = null
    }
    if (this.feedbackVideo) {
      this.feedbackVideo.srcObject = null
      this.feedbackVideo = null
    }
    this.fileVideo = null
    this.compositor?.disposeEffects()
    this.compositor = null
    this.tapCb = null
  }

  setMirror(on: boolean): void {
    if (this.compositor) this.compositor.mirrored = on
  }

  setComposite(patch: CompositePatch): void {
    this.compositor?.setComposite(patch)
  }

  setFeedback(stream: MediaStream | null): void {
    if (!hasVideoTrack(stream)) {
      // Hot-remove: drop the feedback <video>; the next onFrame composites
      // without it. The output canvas / captureStream track is untouched.
      if (this.feedbackVideo) {
        this.feedbackVideo.srcObject = null
        this.feedbackVideo = null
      }
      return
    }
    // Hot-add (or re-point): a hidden <video> on the remote stream, drawn every
    // frame as the bottom layer. Multiple elements may share one MediaStream.
    if (!this.feedbackVideo) {
      const v = document.createElement('video')
      v.muted = true
      v.playsInline = true
      this.feedbackVideo = v
    }
    this.feedbackVideo.srcObject = stream
    void this.feedbackVideo.play().catch(() => {})
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

  swapVideo(videoEl: HTMLVideoElement): void {
    // The same element is redrawn every frame, so a new src is picked up
    // automatically — just keep the reference current.
    this.fileVideo = videoEl
  }

  swapCamera(cameraStream: MediaStream): void {
    // Re-point the hidden <video> at the new camera stream; the rAF loop draws
    // it every frame, so the output canvas/captureStream keeps flowing.
    if (this.hiddenVideo) {
      this.hiddenVideo.srcObject = cameraStream
      void this.hiddenVideo.play().catch(() => {})
    }
  }

  clearVideo(): void {
    // Drop the overlay reference; the next onFrame composites camera-only and
    // schedule() hands the driver back to rVFC-on-camera. The output canvas /
    // captureStream track is untouched.
    this.fileVideo = null
  }

  async snapshot(type = 'image/png'): Promise<Blob> {
    return new Promise((res, rej) =>
      this.previewEl.toBlob((b) => (b ? res(b) : rej(new Error('snapshot failed'))), type),
    )
  }
}
