// Dedicated depth worker — hosts the ort-web DepthSession OFF the compositing
// thread and runs it CONTINUOUSLY (drop-and-replace), mirroring the smooth
// transformers.js webgpu-realtime-depth demo: depth-only execution, as fast as
// the GPU allows, nothing else competing for the thread. The pipeline worker
// (which owns the compositor) spawns this as a nested worker, streams it
// composite VideoFrames, and gets normalized depth maps back to upload as a
// GPU texture. Keeping inference here means it never janks the 30fps composite.

import { DepthSession } from '../core/depthSession'

type InMsg =
  | { type: 'init' }
  | { type: 'frame'; frame: VideoFrame; tsMs: number }
  | { type: 'close' }

let session: DepthSession | null = null
let busy = false
let pending: { frame: VideoFrame; tsMs: number } | null = null

function post(msg: Record<string, unknown>, transfer: Transferable[] = []) {
  ;(self as unknown as Worker).postMessage(msg, transfer)
}

/** Run one inference; the freshly-allocated depth map is transferred back. */
async function process(frame: VideoFrame, tsMs: number): Promise<void> {
  busy = true
  try {
    const r = session ? await session.run(frame) : null
    if (r) post({ type: 'depth', data: r.data, w: r.w, h: r.h, tsMs }, [r.data.buffer])
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
    session = await DepthSession.create()
    post(session ? { type: 'ready' } : { type: 'error', message: 'depth session failed to load' })
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
