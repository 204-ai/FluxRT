// Pipeline worker — WebCodecs streams backend compositing loop.
// Receives the MSTP readable + MSTG writable (transferred), composites each
// VideoFrame onto an OffscreenCanvas through the effect chain, writes the
// result out. Keeps running when the tab is backgrounded (no rAF throttle).
//
// VideoFrame ownership: the eager-read valve holds at most one pending frame
// and closes superseded ones; the input frame is closed right after drawing
// (and after the analyzer-tap bitmap is created). Unclosed frames silently
// stall the camera — the open-frame counter guards against regressions.

import { Compositor } from '../core/compositor'
import type { EffectInit } from '../core/types'

type InMsg =
  | {
      type: 'init'
      readable: ReadableStream<VideoFrame>
      writable: WritableStream<VideoFrame>
      width: number
      height: number
      mirrored: boolean
      effects: EffectInit[]
    }
  | { type: 'mirror'; on: boolean }
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

async function run(msg: Extract<InMsg, { type: 'init' }>) {
  const { readable, writable, width, height } = msg
  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    post({ type: 'error', message: '2d context unavailable in worker' })
    return
  }
  compositor = new Compositor(ctx, width, height)
  compositor.mirrored = msg.mirrored
  compositor.setEffects(msg.effects)

  const reader = readable.getReader()
  const writer = writable.getWriter()
  running = true

  // Eager-read valve: always consume, keep only the newest frame.
  let pending: VideoFrame | null = null
  void (async () => {
    try {
      for (;;) {
        const { value, done } = await reader.read()
        if (done || !running) {
          value?.close()
          break
        }
        openFrames++
        if (pending) {
          pending.close()
          openFrames--
        }
        pending = value
        wake?.()
      }
    } catch {
      /* reader cancelled on stop */
    }
  })()

  try {
    while (running) {
      if (!pending) {
        await new Promise<void>((res) => {
          wake = res
          if (pending || !running) res() // re-check after registering: avoids lost wakeup
        })
        wake = null
        continue
      }
      // explicit type: TS can't track narrowing across the reader closure
      const frame: VideoFrame = pending
      pending = null

      const tsMs = frame.timestamp / 1000
      compositor.drawFrame(frame, tsMs)

      // Analyzer tap: sample the SOURCE frame (pre-composite) at cadence.
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
      if (openFrames > 4) post({ type: 'leak', open: openFrames })
    }
  } finally {
    try {
      reader.cancel()
    } catch {
      /* already done */
    }
    try {
      await writer.close()
    } catch {
      /* already closed */
    }
    ;(pending as VideoFrame | null)?.close()
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
