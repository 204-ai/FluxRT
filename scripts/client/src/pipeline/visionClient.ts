// Main-thread wrapper around the vision worker. One instance per frame
// source (input rail / remote output video). Frames go in as ImageBitmaps;
// typed results come back via subscription.

import type { EngineConfig } from '../vision/engine'
import { DEFAULT_ENGINE_PATHS } from '../vision/engine'
import type { VisionResult } from '../vision/types'

export type VisionListener = (result: VisionResult) => void

export class VisionClient {
  private worker: Worker | null = null
  private listeners = new Set<VisionListener>()
  private statusListeners = new Set<(msg: string) => void>()
  private ready = false
  private disposed = false

  async init(opts: { face: boolean; pose: boolean }): Promise<void> {
    const config: EngineConfig = { ...DEFAULT_ENGINE_PATHS, ...opts }
    this.worker = new Worker(new URL('./workers/vision.worker.ts', import.meta.url), {
      type: 'module',
    })
    await new Promise<void>((resolve, reject) => {
      this.worker!.onmessage = (e) => {
        const m = e.data
        if (m.type === 'ready') {
          this.ready = true
          resolve()
        } else if (m.type === 'status') {
          this.statusListeners.forEach((l) => l(m.message))
        } else if (m.type === 'result') {
          this.listeners.forEach((l) => l(m.result as VisionResult))
        } else if (m.type === 'error') {
          if (!this.ready) reject(new Error(m.message))
          else this.statusListeners.forEach((l) => l('vision error: ' + m.message))
        }
      }
      this.worker!.postMessage({ type: 'init', config })
    })
  }

  get isReady(): boolean {
    return this.ready
  }

  /** Takes ownership of the bitmap (transferred to the worker). Main-thread
   *  sources (canvas backend, output <video> sampler) feed frames this way. The
   *  streams backend bypasses this and posts straight to the worker via the port
   *  from mintFramePort(). */
  push(bitmap: ImageBitmap, tsMs: number): void {
    if (!this.worker || !this.ready || this.disposed) {
      bitmap.close()
      return
    }
    this.worker.postMessage({ type: 'detect', frame: bitmap, tsMs }, [bitmap])
  }

  /** Open a fresh worker→worker frame channel and return the producer end to
   *  hand to the pipeline worker. The vision worker replaces any prior port, so
   *  minting a new one per rail (re)start is safe — the dead worker's port is
   *  simply dropped. Returns null if the worker isn't up. */
  mintFramePort(): MessagePort | null {
    if (!this.worker || this.disposed) return null
    const ch = new MessageChannel()
    this.worker.postMessage({ type: 'frame-port', port: ch.port2 }, [ch.port2])
    return ch.port1
  }

  subscribe(l: VisionListener): () => void {
    this.listeners.add(l)
    return () => this.listeners.delete(l)
  }

  onStatus(l: (msg: string) => void): () => void {
    this.statusListeners.add(l)
    return () => this.statusListeners.delete(l)
  }

  dispose(): void {
    this.disposed = true
    this.worker?.postMessage({ type: 'close' })
    const w = this.worker
    if (w) setTimeout(() => w.terminate(), 250)
    this.worker = null
    this.listeners.clear()
    this.statusListeners.clear()
    this.ready = false
  }
}
