// Pipeline worker — WebCodecs streams backend compositing loop.
// Receives one MSTP readable per layer that has a live source at start
// (transferred) plus the MSTG writable, composites each pass onto an
// OffscreenCanvas through the effect chain, writes the result out. Keeps
// running when the tab is backgrounded (no rAF throttle). Layers can be hot
// added/swapped/removed at runtime (layer-source) without restarting.
//
// VideoFrame ownership: each layer has an eager-read valve holding at most one
// pending frame, closing superseded ones. The BASE layer (the cadence/dims/tap
// source — camera when present, else video) drives the loop and its frame is
// closed right after drawing (and after the analyzer-tap bitmap is created).
// Every NON-BASE layer's newest frame is RETAINED across passes — a paused /
// slower video or feedback must keep compositing under a 30fps base — and is
// closed only on supersede, layer removal or shutdown. Unclosed frames silently
// stall the base — the open-frame counter guards against regressions.

import { Compositor, type FrameMap } from '../core/compositor'
import type { Composite, CompositeOp, EffectInit, LayerId } from '../core/types'

interface LayerInit {
  id: LayerId
  readable: ReadableStream<VideoFrame> | null
}

type InMsg =
  | {
      type: 'init'
      layers: LayerInit[]
      baseLayerId: LayerId
      writable: WritableStream<VideoFrame>
      width: number
      height: number
      composite: Composite
      effects: EffectInit[]
    }
  | { type: 'composite'; op: CompositeOp }
  | { type: 'effect-config'; name: string; patch: Record<string, unknown> }
  | { type: 'effect-msg'; name: string; data: unknown }
  | { type: 'bus'; key: string; value: unknown }
  | { type: 'tap'; intervalMs: number }
  | { type: 'stop' }
  // Hot add / swap (readable set) or remove (readable null) one layer's source.
  | { type: 'layer-source'; id: LayerId; readable: ReadableStream<VideoFrame> | null }
  // Re-designate the cadence/dims/tap base layer (e.g. base clip deactivated).
  | { type: 'base'; id: LayerId }

let compositor: Compositor | null = null
let running = false
let tapIntervalMs = 0
let lastTapMs = 0
let openFrames = 0
let wake: (() => void) | null = null
let setLayerSource: ((id: LayerId, readable: ReadableStream<VideoFrame> | null) => void) | null = null
let baseSetter: ((id: LayerId) => void) | null = null

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

/** One layer's live frame plumbing: its valve plus the newest frame retained
 *  across passes (non-base layers only — the base's frame is closed each pass). */
interface Slot {
  valve: Valve
  retained: VideoFrame | null
}

async function run(msg: Extract<InMsg, { type: 'init' }>) {
  const { layers, writable, width, height } = msg
  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    post({ type: 'error', message: '2d context unavailable in worker' })
    return
  }
  compositor = new Compositor(ctx, width, height)
  compositor.setComposite(msg.composite)
  compositor.setEffects(msg.effects)

  const writer = writable.getWriter()
  running = true

  // One slot per layer that has a live source at start; more hot-attach later.
  const slots = new Map<LayerId, Slot>()
  for (const l of layers) {
    if (l.readable) slots.set(l.id, { valve: startValve(l.readable), retained: null })
  }
  let baseLayerId = msg.baseLayerId

  // Steady-state holds one extra open frame per retained (non-base) slot;
  // recomputed when a layer is added/removed at runtime.
  let leakThreshold = 4
  const recomputeLeak = () => {
    const nonBase = slots.size - (slots.has(baseLayerId) ? 1 : 0)
    leakThreshold = 4 + Math.max(0, nonBase)
  }
  recomputeLeak()

  // Re-designate the cadence/dims/tap base layer (e.g. the base clip was
  // deactivated and another base-capable layer must drive the loop). The loop
  // reads `baseLayerId` fresh each pass, so this takes effect next frame.
  baseSetter = (id) => {
    baseLayerId = id
    recomputeLeak()
    wake?.()
  }

  const freeSlot = (slot: Slot) => {
    try {
      void slot.valve.reader.cancel()
    } catch {
      /* already done */
    }
    if (slot.valve.pending) {
      slot.valve.pending.close()
      openFrames--
      slot.valve.pending = null
    }
    if (slot.retained) {
      slot.retained.close()
      openFrames--
      slot.retained = null
    }
  }

  // Hot add / swap (readable set) or remove (readable null) one layer's source
  // without restarting. Swapping keeps the slot's retained frame on screen until
  // the fresh valve supplies a new one (no one-frame gap on the common
  // video-overlay swap); removal frees everything and narrows the leak guard.
  setLayerSource = (id, readable) => {
    if (!running) {
      void readable?.cancel().catch(() => {})
      return
    }
    const existing = slots.get(id)
    if (!readable) {
      if (existing) {
        freeSlot(existing)
        slots.delete(id)
        recomputeLeak()
      }
      return
    }
    if (existing) {
      // Swap in place: cancel the old reader + drop its pending frame, but keep
      // `retained` so the layer stays on screen until the new valve produces.
      try {
        void existing.valve.reader.cancel()
      } catch {
        /* already done */
      }
      if (existing.valve.pending) {
        existing.valve.pending.close()
        openFrames--
        existing.valve.pending = null
      }
      existing.valve = startValve(readable)
    } else {
      slots.set(id, { valve: startValve(readable), retained: null })
    }
    recomputeLeak()
  }

  // ~30fps ticker for base-less compositions (effect / feedback / image only).
  const TICK_MS = 33
  let lastOutTs = 0

  try {
    while (running) {
      const baseSlot = slots.get(baseLayerId)
      let baseFrame: VideoFrame | null
      if (baseSlot) {
        // A base source is present → block on its frames (it sets the cadence).
        if (!baseSlot.valve.pending) {
          await new Promise<void>((res) => {
            wake = res
            const b = slots.get(baseLayerId)
            if ((b && b.valve.pending) || !running) res() // re-check after registering: avoids lost wakeup
          })
          wake = null
          continue
        }
        baseFrame = baseSlot.valve.take()
      } else {
        // No base → the ticker drives: composite retained frames + effects every
        // TICK_MS (wakes early if a frame arrives). A frame-source becoming the
        // base again switches back to base-driven cadence next iteration.
        await new Promise<void>((res) => {
          let settled = false
          const done = () => {
            if (settled) return
            settled = true
            clearTimeout(timer)
            wake = null
            res()
          }
          const timer = setTimeout(done, TICK_MS)
          wake = done
          if (!running) done()
        })
        if (!running) break
        baseFrame = null
      }

      // Refresh the retained frame of every non-base slot from its valve.
      for (const [id, slot] of slots) {
        if (id === baseLayerId) continue
        const fresh = slot.valve.take()
        if (fresh) {
          if (slot.retained) {
            slot.retained.close()
            openFrames--
          }
          slot.retained = fresh
        }
      }

      // Monotonic output timestamp — the base frame's when present, else advance
      // by the tick so the encoder sees an increasing timeline in ticker mode.
      const tsOut = baseFrame ? Math.max(baseFrame.timestamp, lastOutTs + 1) : lastOutTs + TICK_MS * 1000
      lastOutTs = tsOut
      const tsMs = tsOut / 1000
      const frames: FrameMap = {}
      for (const [id, slot] of slots) {
        frames[id] = id === baseLayerId ? baseFrame : slot.retained
      }
      compositor.drawComposite(frames, tsMs)

      // Analyzer tap: sample the full COMPOSITE (all layers) at cadence, so
      // sensing reflects what's actually composited — not just the base.
      if (tapIntervalMs > 0 && tsMs - lastTapMs >= tapIntervalMs) {
        lastTapMs = tsMs
        try {
          const bitmap = await createImageBitmap(canvas)
          post({ type: 'tap-frame', bitmap, tsMs }, [bitmap])
        } catch {
          /* frame raced close — skip */
        }
      }

      const out = new VideoFrame(canvas, { timestamp: tsOut })
      if (baseFrame) {
        baseFrame.close()
        openFrames--
      }
      try {
        await writer.write(out)
      } catch {
        out.close()
        break
      }
      if (openFrames > leakThreshold) post({ type: 'leak', open: openFrames })
    }
  } finally {
    setLayerSource = null
    baseSetter = null
    for (const slot of slots.values()) {
      try {
        slot.valve.reader.cancel()
      } catch {
        /* already done */
      }
      slot.valve.pending?.close()
      slot.valve.pending = null
      slot.retained?.close()
      slot.retained = null
    }
    slots.clear()
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
    case 'composite':
      compositor?.setComposite(m.op)
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
    case 'layer-source':
      setLayerSource?.(m.id, m.readable)
      break
    case 'base':
      baseSetter?.(m.id)
      break
  }
}
