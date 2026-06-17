// WebCodecs/insertable-streams backend. Compositing runs in pipeline.worker;
// this side owns the worker, the MSTP/MSTG pairs (one per layer source), and a
// <video> preview element fed by the generator's output stream. Video-file
// layers are captured via HTMLMediaElement.captureStream — Chrome-only, but so
// is this backend.

import type {
  ClipKind,
  CompositeOp,
  EffectInit,
  LayerId,
  LayerSourceInput,
  RailBackend,
  RailStartOptions,
  SourceSet,
  TapCallback,
} from '../core/types'
import { clipMeta } from '../core/clipKinds'

/** Backend-owned resources for one layer's live source, stopped on replace /
 *  remove / shutdown. The camera stream is owned by the rail (not here). */
interface OwnedSource {
  /** captureStream() result for a video-file layer. */
  capturedStream?: MediaStream
  /** Our clone of a shared MediaStream track (e.g. the remote output → feedback;
   *  one MediaStreamTrackProcessor per track, so we never process the shared
   *  track directly). */
  clonedTrack?: MediaStreamTrack
}

export class StreamsBackend implements RailBackend {
  readonly kind = 'streams' as const
  readonly previewEl: HTMLVideoElement
  outputStream: MediaStream = new MediaStream()

  private worker: Worker | null = null
  private generator: MediaStreamTrackGenerator<VideoFrame> | null = null
  private owned = new Map<LayerId, OwnedSource>()
  private tapCb: TapCallback | null = null
  private onLog: (m: string) => void

  constructor(onLog: (m: string) => void = () => {}) {
    this.onLog = onLog
    this.previewEl = document.createElement('video')
    this.previewEl.muted = true
    this.previewEl.playsInline = true
    this.previewEl.autoplay = true
  }

  async start(opts: RailStartOptions, { base }: SourceSet): Promise<void> {
    const transfer: Transferable[] = []

    // Build the base layer's readable from its acquired source.
    let readable: ReadableStream<VideoFrame>
    if (base.stream) {
      const [track] = base.stream.getVideoTracks()
      if (!track) throw new Error('base source has no video track')
      readable = new MediaStreamTrackProcessor({ track }).readable
    } else if (base.videoEl) {
      // Don't touch the element itself — playback state belongs to its owner.
      const captured = base.videoEl.captureStream()
      this.owned.set(base.layerId, { capturedStream: captured })
      const [track] = captured.getVideoTracks()
      readable = new MediaStreamTrackProcessor({ track }).readable
    } else {
      throw new Error('no base source')
    }
    transfer.push(readable as unknown as Transferable)
    const layers: { id: LayerId; readable: ReadableStream<VideoFrame> | null }[] = [
      { id: base.layerId, readable },
    ]
    const baseLayerId = base.layerId

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
    transfer.push(this.generator.writable as unknown as Transferable)
    this.worker.postMessage(
      {
        type: 'init',
        layers,
        baseLayerId,
        writable: this.generator.writable,
        width: opts.width,
        height: opts.height,
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
    for (const owned of this.owned.values()) this.freeOwned(owned)
    this.owned.clear()
    this.previewEl.srcObject = null
    this.tapCb = null
  }

  private freeOwned(owned: OwnedSource): void {
    owned.capturedStream?.getTracks().forEach((t) => {
      try {
        t.stop()
      } catch {
        /* already stopped */
      }
    })
    if (owned.clonedTrack) {
      try {
        owned.clonedTrack.stop()
      } catch {
        /* already stopped */
      }
    }
  }

  setComposite(op: CompositeOp): void {
    this.worker?.postMessage({ type: 'composite', op })
  }

  setLayerSource(id: LayerId, kind: ClipKind, source: LayerSourceInput): void {
    if (!this.worker) return
    if (!source) {
      this.worker.postMessage({ type: 'layer-source', id, readable: null })
      const owned = this.owned.get(id)
      if (owned) {
        this.freeOwned(owned)
        this.owned.delete(id)
      }
      return
    }
    switch (clipMeta(kind).sourceForm) {
      case 'mediastream':
        this.bindMediaStream(id, source as MediaStream)
        break
      case 'mediastream-clone':
        this.bindClonedStream(id, source as MediaStream)
        break
      case 'element':
        this.bindElement(id, source as HTMLVideoElement)
        break
    }
  }

  /** Camera: process the track directly (getUserMedia tracks are live
   *  immediately, no first-frame wait). The MediaStream is owned by the rail. */
  private bindMediaStream(id: LayerId, stream: MediaStream): void {
    const [track] = stream.getVideoTracks()
    if (!track) {
      this.onLog('setLayerSource: no video track for ' + id)
      return
    }
    const readable = new MediaStreamTrackProcessor({ track }).readable
    this.worker!.postMessage({ type: 'layer-source', id, readable }, [
      readable as unknown as Transferable,
    ])
  }

  /** Feedback: clone the shared remote track (one processor per track; the
   *  remote track also feeds the output <video>). We own and stop the clone. */
  private bindClonedStream(id: LayerId, stream: MediaStream): void {
    const src = stream.getVideoTracks()[0]
    if (!src) {
      this.onLog('setLayerSource: no video track to clone for ' + id)
      return
    }
    const prev = this.owned.get(id)
    if (prev?.clonedTrack) {
      try {
        prev.clonedTrack.stop()
      } catch {
        /* already stopped */
      }
    }
    const clone = src.clone()
    this.owned.set(id, { ...prev, clonedTrack: clone })
    const readable = new MediaStreamTrackProcessor({ track: clone }).readable
    this.worker!.postMessage({ type: 'layer-source', id, readable }, [
      readable as unknown as Transferable,
    ])
  }

  /** Video file: wait for the element's first presented frame so the
   *  re-captured stream has a LIVE track before handing it to the worker
   *  (re-capturing immediately would grab the old, just-ended track and freeze
   *  the layer). A paused / autoplay-blocked element never fires rVFC, so also
   *  fall back on a timer (the `sent` guard makes whichever fires first win). */
  private bindElement(id: LayerId, videoEl: HTMLVideoElement): void {
    let sent = false
    const send = () => {
      if (sent) return
      sent = true
      const w = this.worker
      if (!w) return
      // captureStream() returns the SAME MediaStream per element; on a src
      // change the old track ends and a new one is added once the element
      // presents frames. Do NOT stop the previous capture here — it IS this same
      // stream, so stopping it would kill the just-added LIVE track. The old
      // track ends on its own; the stream is freed on remove / stop.
      const stream = videoEl.captureStream()
      this.owned.set(id, { capturedStream: stream })
      const tracks = stream.getVideoTracks()
      const track = tracks.find((t) => t.readyState === 'live') ?? tracks[0]
      if (!track) {
        this.onLog('setLayerSource: no live video track after re-capture for ' + id)
        return
      }
      const readable = new MediaStreamTrackProcessor({ track }).readable
      w.postMessage({ type: 'layer-source', id, readable }, [readable as unknown as Transferable])
    }
    if ('requestVideoFrameCallback' in videoEl) {
      videoEl.requestVideoFrameCallback(() => send())
      setTimeout(send, 600)
    } else {
      send()
    }
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

  async snapshot(type = 'image/png'): Promise<Blob> {
    // The output <video> has 0x0 dimensions until it presents its first frame —
    // drawing it then yields a blank blob. Fail loudly so callers (e.g. Comfy
    // snap → edit) surface "not ready" instead of uploading an empty image.
    const w = this.previewEl.videoWidth
    const h = this.previewEl.videoHeight
    if (!w || !h) throw new Error('output not ready yet — start the stream first')
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    c.getContext('2d')!.drawImage(this.previewEl, 0, 0)
    return new Promise((res, rej) =>
      c.toBlob((b) => (b ? res(b) : rej(new Error('snapshot failed'))), type),
    )
  }
}
