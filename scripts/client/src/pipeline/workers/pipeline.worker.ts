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
// stall the base — the open-frame counter guards against regressions. NOTE: it
// counts SOURCE-STREAM frames only (the valves), not the per-pass canvas
// snapshots (tap/output VideoFrame, depth clone) — those are owned + closed/
// transferred at their own call sites.

import { Compositor, type FrameMap } from '../core/compositor'
import { WebGpuCompositor } from '../core/webgpuCompositor'
import { getGpuContext, setGpuLostHandler } from '../core/gpu'
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
      profile?: boolean
      webgpu?: boolean
      depth?: boolean
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
  // Direct worker→worker frame channel to the vision worker (no main bounce).
  | { type: 'vision-port'; port: MessagePort | null }

let compositor: Compositor | WebGpuCompositor | null = null
let running = false
let tapIntervalMs = 0
let lastTapMs = 0
let openFrames = 0
let wake: (() => void) | null = null
// Direct frame channel to the vision worker. When set, the composite tap hands
// off a VideoFrame straight to it (no createImageBitmap await, no main bounce).
let visionPort: MessagePort | null = null
// Step-0 profiling: rolling per-frame composite/tap timing, summarized to onLog.
let profile = false
// Dedicated depth worker + the last depth input size forwarded to it (the depth
// effect layer's `size` config — the demo's "Image size" knob).
let depthWorker: Worker | null = null
let lastDepthSize = 0

/** The depth effect layer's configured input size from the live composite, 0 if none. */
function depthSizeFromComposite(): number {
  if (!compositor) return 0
  for (const l of compositor.composite) {
    if (l.effectName === 'depth') return Number((l.effectConfig as Record<string, unknown> | undefined)?.size) || 0
  }
  return 0
}

/** Forward an image-size change to the depth worker (re-allocates its tensor). */
function syncDepthSize(): void {
  if (!depthWorker) return
  const s = depthSizeFromComposite()
  if (s > 0 && s !== lastDepthSize) {
    lastDepthSize = s
    depthWorker.postMessage({ type: 'config', size: s })
  }
}
// Init gating: in the WebGPU path run() awaits the GPU device BEFORE its message
// handlers (setLayerSource/baseSetter) and the compositor exist. Messages that
// arrive during that await are buffered here and flushed in order once run() is
// ready — otherwise a non-base layer-source binding (posted right after start)
// hits a null setLayerSource and is dropped, leaving that layer uncomposited.
let initialized = false
let stopRequested = false
const preInit: Exclude<InMsg, { type: 'init' }>[] = []
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
  // Pick the compositor backend. WebGPU is opt-in (msg.webgpu) and probed; on any
  // failure we fall back to the 2D canvas. A canvas can hold only one context
  // type, so a failed WebGPU attempt (which may already have taken a 'webgpu'
  // context) gets a fresh canvas before the 2D path.
  let canvas = new OffscreenCanvas(width, height)
  let comp: Compositor | WebGpuCompositor | null = null
  if (msg.webgpu) {
    const gpu = await getGpuContext()
    if (gpu) {
      try {
        comp = new WebGpuCompositor(gpu, canvas, width, height)
      } catch (e) {
        console.error('[pipeline] webgpu compositor init failed, using 2d:', e)
        post({
          type: 'error',
          message: 'webgpu compositor init failed, using 2d: ' + (e instanceof Error ? e.message : String(e)),
        })
        canvas = new OffscreenCanvas(width, height)
        comp = null
      }
    }
  }
  if (!comp) {
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      post({ type: 'error', message: '2d context unavailable in worker' })
      return
    }
    comp = new Compositor(ctx, width, height)
  }
  compositor = comp
  const compositorMode = comp instanceof WebGpuCompositor ? 'webgpu' : '2d'
  console.info('[pipeline] compositor:', compositorMode, `${width}x${height}`)
  post({ type: 'info', text: `compositor: ${compositorMode}` })
  compositor.setComposite(msg.composite)
  compositor.setEffects(msg.effects)

  const webgpuComp = comp instanceof WebGpuCompositor ? comp : null

  // Depth Anything V2 runs in a DEDICATED worker (off this compositing thread),
  // continuously — so its inference never janks the 30fps composite. We stream
  // it composite frames and upload the depth maps it returns. Only on WebGPU.
  if (msg.depth && webgpuComp) {
    depthWorker = new Worker(new URL('./depth.worker.ts', import.meta.url), { type: 'module' })
    let depthReady = false
    depthWorker.onmessage = (de) => {
      const dm = de.data
      if (dm.type === 'depth') webgpuComp.setDepthData(dm.data as Uint8Array, dm.w as number, dm.h as number)
      else if (dm.type === 'ready') {
        depthReady = true
        post({ type: 'info', text: 'depth: ready' })
      } else if (dm.type === 'error') {
        post({ type: 'error', message: 'depth: ' + dm.message })
        // Init failure is terminal — terminate so we stop cloning+transferring a
        // frame to a dead session every composite frame.
        if (!depthReady) {
          depthWorker?.terminate()
          depthWorker = null
        }
      }
    }
    lastDepthSize = depthSizeFromComposite()
    depthWorker.postMessage({ type: 'init', size: lastDepthSize || undefined })
  }

  // Compositor (not depth) device loss freezes output forever otherwise — halt
  // the loop and signal so the host can restart the pipeline.
  if (webgpuComp) {
    setGpuLostHandler((info) => {
      post({ type: 'error', message: 'webgpu device lost — restart pipeline: ' + (info?.message ?? '') })
      running = false
      wake?.()
    })
  }

  const writer = writable.getWriter()
  running = !stopRequested // a stop during async init must not start the loop

  // Step-0 profiling accumulators (only touched when `profile`).
  profile = !!msg.profile
  let profWallStart = performance.now()
  let profFrames = 0
  let profCompMs = 0
  let profTapMs = 0
  let profTaps = 0

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

  // Init complete: flush any messages buffered during async setup, in order, so
  // layer-source bindings / composite ops that raced the GPU await are applied.
  initialized = true
  for (const m of preInit) handle(m)
  preInit.length = 0

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
      const comp0 = profile ? performance.now() : 0
      compositor.drawComposite(frames, tsMs)
      if (profile) profCompMs += performance.now() - comp0

      // Analyzer tap: sample the full COMPOSITE (all layers) at cadence, so
      // sensing reflects what's actually composited — not just the base.
      if (tapIntervalMs > 0 && tsMs - lastTapMs >= tapIntervalMs) {
        lastTapMs = tsMs
        const tap0 = profile ? performance.now() : 0
        let tapFrame: VideoFrame | null = null
        let tapBitmap: ImageBitmap | null = null
        try {
          if (visionPort) {
            // Direct worker→worker handoff: a SYNCHRONOUS canvas snapshot (no
            // createImageBitmap await stalling the loop) transferred straight to
            // the vision worker — no main-thread bounce.
            tapFrame = new VideoFrame(canvas, { timestamp: tsOut })
            visionPort.postMessage({ type: 'detect', frame: tapFrame, tsMs }, [tapFrame])
            tapFrame = null // ownership transferred to the vision worker
          } else {
            // Fallback when no vision port is wired (e.g. vision inactive): the
            // legacy bitmap relay via the main thread.
            tapBitmap = await createImageBitmap(canvas)
            post({ type: 'tap-frame', bitmap: tapBitmap, tsMs }, [tapBitmap])
            tapBitmap = null // transferred
          }
        } catch {
          // post failed before transfer (e.g. stale port) — close so it can't leak.
          tapFrame?.close()
          tapBitmap?.close()
        }
        if (profile) {
          profTapMs += performance.now() - tap0
          profTaps++
        }
      }

      const out = new VideoFrame(canvas, { timestamp: tsOut })

      // Feed depth the SOURCE (base) frame — NOT the composite output. The output
      // includes the depth layer itself (in Replace mode it IS the depth map), so
      // feeding it back would loop (depth-of-depth) and diverge after a few frames.
      // The realtime demo likewise runs depth on the raw video. clone() is a cheap
      // refcount; the depth worker drop-and-replaces and closes it.
      if (depthWorker && baseFrame) {
        const dframe = baseFrame.clone()
        // Match the base layer's selfie mirror so the depth aligns with the display.
        const baseMirror = !!compositor?.composite.find((l) => l.id === baseLayerId)?.mirror
        depthWorker.postMessage({ type: 'frame', frame: dframe, tsMs, mirror: baseMirror }, [dframe])
      }

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

      // Step-0 profiling: summarize composite/tap timing every ~2s to onLog.
      if (profile) {
        profFrames++
        const wall = performance.now() - profWallStart
        if (wall >= 2000) {
          const fps = (profFrames * 1000) / wall
          const comp = profCompMs / profFrames
          const tap = profTaps ? profTapMs / profTaps : 0
          post({
            type: 'perf',
            text: `composite ${comp.toFixed(2)}ms/f · tap ${tap.toFixed(2)}ms (${profTaps}/${profFrames}f) · ${fps.toFixed(1)}fps`,
          })
          profWallStart = performance.now()
          profFrames = 0
          profCompMs = 0
          profTapMs = 0
          profTaps = 0
        }
      }
    }
  } finally {
    setLayerSource = null
    baseSetter = null
    visionPort?.close()
    visionPort = null
    if (depthWorker) {
      depthWorker.postMessage({ type: 'close' }) // dispose the ort session, then terminate
      const dw = depthWorker
      setTimeout(() => dw.terminate(), 250)
      depthWorker = null
    }
    setGpuLostHandler(null)
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

/** Dispatch a non-init message against run()'s live state. Called directly once
 *  initialized, or replayed from `preInit` when init finishes. */
function handle(m: Exclude<InMsg, { type: 'init' }>): void {
  switch (m.type) {
    case 'composite':
      compositor?.setComposite(m.op)
      syncDepthSize() // forward a changed depth "image size" to the depth worker
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
      stopRequested = true
      wake?.()
      break
    case 'layer-source':
      setLayerSource?.(m.id, m.readable)
      break
    case 'base':
      baseSetter?.(m.id)
      break
    case 'vision-port':
      visionPort?.close()
      visionPort = m.port
      break
  }
}

self.onmessage = (e: MessageEvent<InMsg>) => {
  const m = e.data
  if (m.type === 'init') {
    void run(m)
    return
  }
  // 'stop' must act immediately, even mid-init, so an early abort is honored.
  if (m.type === 'stop') {
    running = false
    stopRequested = true
    wake?.()
    return
  }
  // Buffer everything that needs run()'s setup until init finishes (see preInit).
  if (!initialized) {
    preInit.push(m)
    return
  }
  handle(m)
}
