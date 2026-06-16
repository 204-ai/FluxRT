// Pipeline worker — WebCodecs streams backend compositing loop.
// Receives up to two MSTP readables (camera + video file layers, transferred)
// plus the MSTG writable, composites each pass onto an OffscreenCanvas through
// the effect chain, writes the result out. Keeps running when the tab is
// backgrounded (no rAF throttle).
//
// VideoFrame ownership: each layer has an eager-read valve holding at most one
// pending frame, closing superseded ones. The base frame (camera when present,
// else video) drives the loop and is closed right after drawing (and after the
// analyzer-tap bitmap is created). The overlay's newest frame is RETAINED
// across passes — a paused/slower video must keep compositing under a 30fps
// camera — and closed only on supersede or shutdown. Unclosed frames silently
// stall the camera — the open-frame counter guards against regressions.

import { Compositor } from '../core/compositor'
import type { CompositeOptions, CompositePatch, EffectInit } from '../core/types'

type InMsg =
  | {
      type: 'init'
      camera: ReadableStream<VideoFrame> | null
      video: ReadableStream<VideoFrame> | null
      writable: WritableStream<VideoFrame>
      width: number
      height: number
      mirrored: boolean
      composite: CompositeOptions
      effects: EffectInit[]
    }
  | { type: 'mirror'; on: boolean }
  | { type: 'composite'; patch: CompositePatch }
  | { type: 'effect-config'; name: string; patch: Record<string, unknown> }
  | { type: 'effect-msg'; name: string; data: unknown }
  | { type: 'bus'; key: string; value: unknown }
  | { type: 'tap'; intervalMs: number }
  | { type: 'stop' }
  | { type: 'swap-video'; video: ReadableStream<VideoFrame> }
  | { type: 'swap-camera'; video: ReadableStream<VideoFrame> }
  | { type: 'clear-video' }
  | { type: 'set-feedback'; video: ReadableStream<VideoFrame> }
  | { type: 'clear-feedback' }

let compositor: Compositor | null = null
let running = false
let tapIntervalMs = 0
let lastTapMs = 0
let openFrames = 0
let wake: (() => void) | null = null
let requestVideoSwap: ((readable: ReadableStream<VideoFrame>) => void) | null = null
let requestCameraSwap: ((readable: ReadableStream<VideoFrame>) => void) | null = null
let requestOverlayClear: (() => void) | null = null
let requestFeedbackSet: ((readable: ReadableStream<VideoFrame>) => void) | null = null
let requestFeedbackClear: (() => void) | null = null

function post(msg: Record<string, unknown>, transfer: Transferable[] = []) {
  ;(self as unknown as Worker).postMessage(msg, transfer)
}

/** Eager-read valve: always consume, keep only the newest frame. */
interface Valve {
  reader: ReadableStreamDefaultReader<VideoFrame>
  pending: VideoFrame | null
  take(): VideoFrame | null
}
function startValve(readable: ReadableStream<VideoFrame>): Valve {
  const valve = {
    reader: readable.getReader(),
    pending: null as VideoFrame | null,
    take(): VideoFrame | null {
      const f = valve.pending
      valve.pending = null
      return f
    },
  }
  void (async () => {
    try {
      for (;;) {
        const { value, done } = await valve.reader.read()
        if (done || !running) {
          value?.close()
          break
        }
        openFrames++
        if (valve.pending) {
          valve.pending.close()
          openFrames--
        }
        valve.pending = value
        wake?.()
      }
    } catch {
      /* reader cancelled on stop */
    }
  })()
  return valve
}

async function run(msg: Extract<InMsg, { type: 'init' }>) {
  const { camera, video, writable, width, height } = msg
  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    post({ type: 'error', message: '2d context unavailable in worker' })
    return
  }
  compositor = new Compositor(ctx, width, height)
  compositor.mirrored = msg.mirrored
  compositor.setComposite(msg.composite)
  compositor.setEffects(msg.effects)

  const writer = writable.getWriter()
  running = true

  // Base layer drives the loop cadence and the vision tap.
  const baseIsCamera = camera !== null
  let base = startValve(baseIsCamera ? camera! : video!)
  let overlay = camera && video ? startValve(video) : null
  let retainedOverlay: VideoFrame | null = null
  // The feedback layer (remote output stream) arrives after init via
  // set-feedback — it's always a standalone layer, never the base/overlay.
  // Retained across passes like the overlay so it keeps compositing between its
  // own (possibly slower) frames.
  // Typed-null init (not a bare `null` literal) so control-flow analysis keeps
  // the `Valve | null` union across the closures below — matching `overlay`.
  let feedback = null as Valve | null
  let retainedFeedback: VideoFrame | null = null
  // Steady-state holds one extra open frame per retained overlay/feedback;
  // recomputed when a layer is added/removed at runtime. Declared before the
  // swap/clear closures that call it (no use-before-init).
  let leakThreshold = 4
  const recomputeLeak = () => {
    leakThreshold = 4 + (overlay ? 1 : 0) + (feedback ? 1 : 0)
  }
  recomputeLeak()

  // Hot-swap the video-file input without restarting: cancel the old video
  // valve and start a fresh one on the re-captured readable. The video is the
  // `base` valve when there's no camera, otherwise the `overlay`.
  requestVideoSwap = (readable) => {
    if (!running) {
      void readable.cancel().catch(() => {})
      return
    }
    const videoIsBase = !baseIsCamera
    const oldValve = videoIsBase ? base : overlay
    if (oldValve) {
      try {
        void oldValve.reader.cancel()
      } catch {
        /* already done */
      }
      if (oldValve.pending) {
        oldValve.pending.close()
        openFrames--
        oldValve.pending = null
      }
    }
    const fresh = startValve(readable)
    if (videoIsBase) base = fresh
    else overlay = fresh
    // A camera-only pipeline gaining its first video overlay now retains one
    // extra frame in steady state — widen the leak guard to match init.
    recomputeLeak()
  }

  // Hot-swap the camera: it's always the `base` valve when a camera is present.
  requestCameraSwap = (readable) => {
    if (!running || !baseIsCamera) {
      void readable.cancel().catch(() => {})
      return
    }
    try {
      void base.reader.cancel()
    } catch {
      /* already done */
    }
    if (base.pending) {
      base.pending.close()
      openFrames--
      base.pending = null
    }
    base = startValve(readable)
  }

  // Hot-remove the overlay: drop the video layer while the camera keeps
  // feeding. Cancels the overlay valve, frees any held frames, and narrows the
  // leak guard back to camera-only. No-op when there is no overlay.
  requestOverlayClear = () => {
    if (overlay) {
      try {
        void overlay.reader.cancel()
      } catch {
        /* already done */
      }
      if (overlay.pending) {
        overlay.pending.close()
        openFrames--
        overlay.pending = null
      }
      overlay = null
    }
    if (retainedOverlay) {
      retainedOverlay.close()
      openFrames--
      retainedOverlay = null
    }
    recomputeLeak()
  }

  // Hot-add / re-point the feedback layer (remote output stream). Mirrors
  // requestVideoSwap's valve handling but is always a standalone overlay layer.
  requestFeedbackSet = (readable) => {
    if (!running) {
      void readable.cancel().catch(() => {})
      return
    }
    if (feedback) {
      try {
        void feedback.reader.cancel()
      } catch {
        /* already done */
      }
      if (feedback.pending) {
        feedback.pending.close()
        openFrames--
        feedback.pending = null
      }
    }
    if (retainedFeedback) {
      retainedFeedback.close()
      openFrames--
      retainedFeedback = null
    }
    feedback = startValve(readable)
    recomputeLeak()
  }

  // Hot-remove the feedback layer (mirror of requestOverlayClear).
  requestFeedbackClear = () => {
    if (feedback) {
      try {
        void feedback.reader.cancel()
      } catch {
        /* already done */
      }
      if (feedback.pending) {
        feedback.pending.close()
        openFrames--
        feedback.pending = null
      }
      feedback = null
    }
    if (retainedFeedback) {
      retainedFeedback.close()
      openFrames--
      retainedFeedback = null
    }
    recomputeLeak()
  }

  try {
    while (running) {
      if (!base.pending) {
        await new Promise<void>((res) => {
          wake = res
          if (base.pending || !running) res() // re-check after registering: avoids lost wakeup
        })
        wake = null
        continue
      }
      const frame: VideoFrame = base.take()!

      if (overlay) {
        const fresh = overlay.take()
        if (fresh) {
          if (retainedOverlay) {
            retainedOverlay.close()
            openFrames--
          }
          retainedOverlay = fresh
        }
      }

      if (feedback) {
        const fresh = feedback.take()
        if (fresh) {
          if (retainedFeedback) {
            retainedFeedback.close()
            openFrames--
          }
          retainedFeedback = fresh
        }
      }

      const tsMs = frame.timestamp / 1000
      const cameraFrame = baseIsCamera ? frame : null
      const videoFrame = baseIsCamera ? retainedOverlay : frame
      compositor.drawComposite(cameraFrame, videoFrame, retainedFeedback, tsMs)

      // Analyzer tap: sample the full COMPOSITE (all layers, incl. feedback) at
      // cadence, so sensing reflects what's actually composited — not just the base.
      if (tapIntervalMs > 0 && tsMs - lastTapMs >= tapIntervalMs) {
        lastTapMs = tsMs
        try {
          const bitmap = await createImageBitmap(canvas)
          post({ type: 'tap-frame', bitmap, tsMs }, [bitmap])
        } catch {
          /* frame raced close — skip */
        }
      }

      const out = new VideoFrame(canvas, { timestamp: frame.timestamp })
      frame.close()
      openFrames--
      try {
        await writer.write(out)
      } catch {
        out.close()
        break
      }
      if (openFrames > leakThreshold) post({ type: 'leak', open: openFrames })
    }
  } finally {
    requestVideoSwap = null
    requestCameraSwap = null
    requestOverlayClear = null
    requestFeedbackSet = null
    requestFeedbackClear = null
    for (const v of [base, overlay, feedback]) {
      if (!v) continue
      try {
        v.reader.cancel()
      } catch {
        /* already done */
      }
      ;(v.pending as VideoFrame | null)?.close()
      v.pending = null
    }
    retainedOverlay?.close()
    retainedFeedback?.close()
    try {
      await writer.close()
    } catch {
      /* already closed */
    }
    compositor?.disposeEffects()
    post({ type: 'stopped' })
  }
}

self.onmessage = (e: MessageEvent<InMsg>) => {
  const m = e.data
  switch (m.type) {
    case 'init':
      void run(m)
      break
    case 'mirror':
      if (compositor) compositor.mirrored = m.on
      break
    case 'composite':
      compositor?.setComposite(m.patch)
      break
    case 'effect-config':
      compositor?.configureEffect(m.name, m.patch)
      break
    case 'effect-msg':
      compositor?.effectMessage(m.name, m.data)
      break
    case 'bus':
      compositor?.bus.set(m.key, m.value)
      break
    case 'tap':
      tapIntervalMs = m.intervalMs
      break
    case 'stop':
      running = false
      wake?.()
      break
    case 'swap-video':
      requestVideoSwap?.(m.video)
      break
    case 'swap-camera':
      requestCameraSwap?.(m.video)
      break
    case 'clear-video':
      requestOverlayClear?.()
      break
    case 'set-feedback':
      requestFeedbackSet?.(m.video)
      break
    case 'clear-feedback':
      requestFeedbackClear?.()
      break
  }
}
