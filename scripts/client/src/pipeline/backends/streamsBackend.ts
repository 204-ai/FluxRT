// WebCodecs/insertable-streams backend. Compositing runs in pipeline.worker;
// this side owns the worker, the MSTP/MSTG pairs (camera + optional video
// file layer), and a <video> preview element fed by the generator's output
// stream. The video-file layer is captured via HTMLMediaElement.captureStream
// — Chrome-only, but so is this backend.

import type {
  CompositeOptions,
  EffectInit,
  RailBackend,
  RailStartOptions,
  SourceSet,
  TapCallback,
} from '../core/types'

export class StreamsBackend implements RailBackend {
  readonly kind = 'streams' as const
  readonly previewEl: HTMLVideoElement
  outputStream: MediaStream = new MediaStream()

  private worker: Worker | null = null
  private generator: MediaStreamTrackGenerator<VideoFrame> | null = null
  private capturedStream: MediaStream | null = null
  private tapCb: TapCallback | null = null
  private onLog: (m: string) => void

  constructor(onLog: (m: string) => void = () => {}) {
    this.onLog = onLog
    this.previewEl = document.createElement('video')
    this.previewEl.muted = true
    this.previewEl.playsInline = true
    this.previewEl.autoplay = true
  }

  async start(opts: RailStartOptions, sources: SourceSet): Promise<void> {
    let cameraReadable: ReadableStream<VideoFrame> | null = null
    if (sources.cameraStream) {
      const [track] = sources.cameraStream.getVideoTracks()
      cameraReadable = new MediaStreamTrackProcessor({ track }).readable
    }

    let videoReadable: ReadableStream<VideoFrame> | null = null
    if (sources.videoEl) {
      // Don't touch the element itself — playback state belongs to its owner.
      this.capturedStream = sources.videoEl.captureStream()
      const [track] = this.capturedStream.getVideoTracks()
      videoReadable = new MediaStreamTrackProcessor({ track }).readable
    }

    this.generator = new MediaStreamTrackGenerator<VideoFrame>({ kind: 'video' })
    this.outputStream = new MediaStream([this.generator])
    this.previewEl.srcObject = this.outputStream
    void this.previewEl.play().catch(() => {})

    this.worker = new Worker(new URL('../workers/pipeline.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.worker.onmessage = (e) => {
      const m = e.data
      if (m.type === 'tap-frame') {
        if (this.tapCb) this.tapCb(m.bitmap as ImageBitmap, m.tsMs as number)
        else (m.bitmap as ImageBitmap).close()
      } else if (m.type === 'leak') {
        this.onLog(`pipeline worker: ${m.open} VideoFrames open (possible leak)`)
      } else if (m.type === 'error') {
        this.onLog('pipeline worker error: ' + m.message)
      }
    }
    const transfer: Transferable[] = [this.generator.writable as unknown as Transferable]
    if (cameraReadable) transfer.push(cameraReadable)
    if (videoReadable) transfer.push(videoReadable)
    this.worker.postMessage(
      {
        type: 'init',
        camera: cameraReadable,
        video: videoReadable,
        writable: this.generator.writable,
        width: opts.width,
        height: opts.height,
        mirrored: opts.mirrored,
        composite: opts.composite,
        effects: opts.effects satisfies EffectInit[],
      },
      transfer,
    )
  }

  stop(): void {
    this.worker?.postMessage({ type: 'stop' })
    // Give the worker a beat to flush/close, then terminate.
    const w = this.worker
    if (w) setTimeout(() => w.terminate(), 250)
    this.worker = null
    this.generator?.stop()
    this.generator = null
    this.capturedStream?.getTracks().forEach((t) => {
      try {
        t.stop()
      } catch {
        /* already stopped */
      }
    })
    this.capturedStream = null
    this.previewEl.srcObject = null
    this.tapCb = null
  }

  setMirror(on: boolean): void {
    this.worker?.postMessage({ type: 'mirror', on })
  }

  setComposite(patch: Partial<CompositeOptions>): void {
    this.worker?.postMessage({ type: 'composite', patch })
  }

  configureEffect(name: string, patch: Record<string, unknown>): void {
    this.worker?.postMessage({ type: 'effect-config', name, patch })
  }

  effectMessage(name: string, data: unknown): void {
    this.worker?.postMessage({ type: 'effect-msg', name, data })
  }

  busPush(key: string, value: unknown): void {
    this.worker?.postMessage({ type: 'bus', key, value })
  }

  setTap(intervalMs: number, cb: TapCallback | null): void {
    this.tapCb = cb
    this.worker?.postMessage({ type: 'tap', intervalMs: cb ? intervalMs : 0 })
  }

  swapVideo(videoEl: HTMLVideoElement): void {
    if (!this.worker) return
    // The element's previous captured track ended when its src changed; capture
    // its NEW video track and hand a fresh readable to the worker. The output
    // generator/track is left untouched, so the WebRTC stream keeps flowing.
    const stream = videoEl.captureStream()
    this.capturedStream = stream
    const [track] = stream.getVideoTracks()
    if (!track) {
      this.onLog('swapVideo: no video track after re-capture')
      return
    }
    const readable = new MediaStreamTrackProcessor({ track }).readable
    this.worker.postMessage({ type: 'swap-video', video: readable }, [
      readable as unknown as Transferable,
    ])
  }

  async snapshot(type = 'image/png'): Promise<Blob> {
    const c = document.createElement('canvas')
    c.width = this.previewEl.videoWidth
    c.height = this.previewEl.videoHeight
    c.getContext('2d')!.drawImage(this.previewEl, 0, 0)
    return new Promise((res, rej) =>
      c.toBlob((b) => (b ? res(b) : rej(new Error('snapshot failed'))), type),
    )
  }
}
