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

  /** Takes ownership of the bitmap (transferred to the worker). */
  push(bitmap: ImageBitmap, tsMs: number): void {
    if (!this.worker || !this.ready || this.disposed) {
      bitmap.close()
      return
    }
    this.worker.postMessage({ type: 'detect', bitmap, tsMs }, [bitmap])
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
