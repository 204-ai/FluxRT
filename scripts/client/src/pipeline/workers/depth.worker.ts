// Dedicated depth worker — hosts the ort-web DepthSession OFF the compositing
// thread and runs it CONTINUOUSLY (drop-and-replace), mirroring the smooth
// transformers.js webgpu-realtime-depth demo: depth-only execution, as fast as
// the GPU allows, nothing else competing for the thread. The pipeline worker
// (which owns the compositor) spawns this as a nested worker, streams it
// composite VideoFrames, and gets normalized depth maps back to upload as a
// GPU texture. Keeping inference here means it never janks the 30fps composite.

import { DepthSession } from '../core/depthSession'

type InMsg =
  | { type: 'init'; size?: number }
  | { type: 'frame'; frame: VideoFrame; tsMs: number }
  | { type: 'config'; size: number }
  | { type: 'close' }

let session: DepthSession | null = null
let busy = false
let pending: { frame: VideoFrame; tsMs: number } | null = null
// Throughput logging: compare our depth fps to the realtime demo's ~10fps.
let fpsCount = 0
let fpsMs = 0
let fpsLast = performance.now()

function post(msg: Record<string, unknown>, transfer: Transferable[] = []) {
  ;(self as unknown as Worker).postMessage(msg, transfer)
}

/** Run one inference; the freshly-allocated depth map is transferred back. */
async function process(frame: VideoFrame, tsMs: number): Promise<void> {
  busy = true
  try {
    const t0 = performance.now()
    const r = session ? await session.run(frame) : null
    if (r) post({ type: 'depth', data: r.data, w: r.w, h: r.h, tsMs }, [r.data.buffer])
    // Log inference cost + achieved depth fps every ~2s (vs the demo's ~10fps).
    fpsCount++
    fpsMs += performance.now() - t0
    const now = performance.now()
    if (now - fpsLast >= 2000) {
      console.info(`[depth] ${(fpsMs / fpsCount).toFixed(1)}ms/infer · ${((fpsCount * 1000) / (now - fpsLast)).toFixed(1)} fps`)
      fpsCount = 0
      fpsMs = 0
      fpsLast = now
    }
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) })
  } finally {
    frame.close()
    busy = false
    // Drop-and-replace: only the newest queued frame is ever processed next.
    if (pending) {
      const next = pending
      pending = null
      void process(next.frame, next.tsMs)
    }
  }
}

self.onmessage = async (e: MessageEvent<InMsg>) => {
  const m = e.data
  if (m.type === 'init') {
    session = await DepthSession.create(m.size)
    post(session ? { type: 'ready' } : { type: 'error', message: 'depth session failed to load' })
  } else if (m.type === 'config') {
    session?.setSize(m.size)
  } else if (m.type === 'frame') {
    if (!session) {
      m.frame.close()
      return
    }
    if (busy) {
      pending?.frame.close()
      pending = { frame: m.frame, tsMs: m.tsMs }
      return
    }
    void process(m.frame, m.tsMs)
  } else if (m.type === 'close') {
    pending?.frame.close()
    pending = null
    session?.dispose()
    session = null
    self.close()
  }
}
