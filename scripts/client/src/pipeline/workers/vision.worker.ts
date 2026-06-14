// Vision worker — hosts the MediaPipe engine off the main thread so 15-40ms
// inference never stalls compositing or the UI. Depth-1 mailbox: a frame
// arriving while detection runs replaces any waiting frame (drop-and-replace,
// never queue) so results are always fresh.

import { VisionEngine, type EngineConfig } from '../../vision/engine'

type InMsg =
  | { type: 'init'; config: EngineConfig }
  | { type: 'detect'; bitmap: ImageBitmap; tsMs: number }
  | { type: 'close' }

const engine = new VisionEngine()
let ready = false
let busy = false
let waiting: { bitmap: ImageBitmap; tsMs: number } | null = null

function post(msg: Record<string, unknown>) {
  ;(self as unknown as Worker).postMessage(msg)
}

async function process(bitmap: ImageBitmap, tsMs: number): Promise<void> {
  busy = true
  try {
    const result = engine.detect(bitmap, tsMs)
    if (result) post({ type: 'result', result })
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) })
  } finally {
    bitmap.close()
    busy = false
    if (waiting) {
      const next = waiting
      waiting = null
      void process(next.bitmap, next.tsMs)
    }
  }
}

self.onmessage = async (e: MessageEvent<InMsg>) => {
  const m = e.data
  if (m.type === 'init') {
    try {
      await engine.init(m.config, (msg) => post({ type: 'status', message: msg }))
      ready = true
      post({ type: 'ready' })
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) })
    }
  } else if (m.type === 'detect') {
    if (!ready) {
      m.bitmap.close()
      return
    }
    if (busy) {
      waiting?.bitmap.close()
      waiting = m
      return
    }
    void process(m.bitmap, m.tsMs)
  } else if (m.type === 'close') {
    waiting?.bitmap.close()
    waiting = null
    engine.close()
    self.close()
  }
}
