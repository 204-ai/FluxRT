// Vision worker — hosts the MediaPipe engine off the main thread so 15-40ms
// inference never stalls compositing or the UI. Depth-1 mailbox: a frame
// arriving while detection runs replaces any waiting frame (drop-and-replace,
// never queue) so results are always fresh.
//
// Two frame ingresses, one mailbox:
//   • self.onmessage 'detect' — main-thread sources (canvas backend, output
//     <video> sampler) push ImageBitmaps here.
//   • a transferred MessagePort ('frame-port') — the STREAMS backend's pipeline
//     worker posts composite VideoFrames straight here, worker→worker, with no
//     main-thread bounce. Results still flow back to the main thread (VisionClient).

import { VisionEngine, type EngineConfig } from '../../vision/engine'

/** Both ImageBitmap and VideoFrame expose close(); the engine takes either. */
type Frame = ImageBitmap | VideoFrame

type InMsg =
  | { type: 'init'; config: EngineConfig }
  | { type: 'detect'; frame: Frame; tsMs: number }
  | { type: 'frame-port'; port: MessagePort }
  | { type: 'close' }

const engine = new VisionEngine()
let ready = false
let busy = false
let waiting: { frame: Frame; tsMs: number } | null = null
let framePort: MessagePort | null = null

function post(msg: Record<string, unknown>) {
  ;(self as unknown as Worker).postMessage(msg)
}

async function process(frame: Frame, tsMs: number): Promise<void> {
  busy = true
  try {
    const result = engine.detect(frame, tsMs)
    if (result) post({ type: 'result', result })
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) })
  } finally {
    frame.close()
    busy = false
    if (waiting) {
      const next = waiting
      waiting = null
      void process(next.frame, next.tsMs)
    }
  }
}

/** Drop-and-replace: never queue. A frame arriving mid-inference supersedes any
 *  already-waiting frame so detection always runs on the freshest sample. */
function enqueue(frame: Frame, tsMs: number): void {
  if (!ready) {
    frame.close()
    return
  }
  if (busy) {
    waiting?.frame.close()
    waiting = { frame, tsMs }
    return
  }
  void process(frame, tsMs)
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
  } else if (m.type === 'frame-port') {
    // The pipeline worker's direct frame channel — replace any previous port.
    framePort?.close()
    framePort = m.port
    framePort.onmessage = (ev: MessageEvent<{ type: 'detect'; frame: Frame; tsMs: number }>) => {
      const d = ev.data
      if (d.type === 'detect') enqueue(d.frame, d.tsMs)
    }
  } else if (m.type === 'detect') {
    enqueue(m.frame, m.tsMs)
  } else if (m.type === 'close') {
    waiting?.frame.close()
    waiting = null
    framePort?.close()
    framePort = null
    engine.close()
    self.close()
  }
}
