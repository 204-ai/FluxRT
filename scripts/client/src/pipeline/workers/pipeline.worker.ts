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
import type { CompositeOptions, EffectInit } from '../core/types'

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
  | { type: 'composite'; patch: Partial<CompositeOptions> }
  | { type: 'effect-config'; name: string; patch: Record<string, unknown> }
  | { type: 'effect-msg'; name: string; data: unknown }
  | { type: 'bus'; key: string; value: unknown }
  | { type: 'tap'; intervalMs: number }
  | { type: 'stop' }

let compositor: Compositor | null = null
let running = false
let tapIntervalMs = 0
let lastTapMs = 0
let openFrames = 0
let wake: (() => void) | null = null

function post(msg: Record<string, unknown>, transfer: Transferable[] = []) {
  ;(self as unknown as Worker).postMessage(msg, transfer)
}

/** Eager-read valve: always consume, keep only the newest frame. */
function startValve(readable: ReadableStream<VideoFrame>) {
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
  const base = startValve(baseIsCamera ? camera! : video!)
  const overlay = camera && video ? startValve(video) : null
  // Steady-state holds one extra open frame for the retained overlay.
  const leakThreshold = 4 + (overlay ? 1 : 0)
  let retainedOverlay: VideoFrame | null = null

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

      const tsMs = frame.timestamp / 1000
      const cameraFrame = baseIsCamera ? frame : null
      const videoFrame = baseIsCamera ? retainedOverlay : frame
      compositor.drawComposite(cameraFrame, videoFrame, tsMs)

      // Analyzer tap: sample the BASE source frame (pre-composite) at cadence.
      if (tapIntervalMs > 0 && tsMs - lastTapMs >= tapIntervalMs) {
        lastTapMs = tsMs
        try {
          const bitmap = await createImageBitmap(frame)
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
    for (const v of [base, overlay]) {
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
  }
}
