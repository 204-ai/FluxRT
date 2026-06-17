// Shared GPU context for the pipeline worker. ONE GPUDevice is requested lazily
// and reused by the WebGPU compositor and (later) the ort-web depth session:
// WebGPU resources are NOT transferable across workers, so everything that
// touches this device must run in the worker that owns it.
//
// getGpuContext() returns null whenever WebGPU can't be initialized (no
// navigator.gpu, no adapter, or a failed device request) so every caller can
// cleanly fall back to the 2D-canvas compositor. Device loss drops the cache so
// the next call rebuilds; register setGpuLostHandler to tear down dependent
// pipelines/textures and reseed.

export interface GpuContext {
  adapter: GPUAdapter
  device: GPUDevice
  /** Preferred format for a GPUCanvasContext on this platform. */
  canvasFormat: GPUTextureFormat
}

let current: GpuContext | null = null
let pending: Promise<GpuContext | null> | null = null
let onLost: ((info: GPUDeviceLostInfo) => void) | null = null

/** Notified when the shared device is lost; the cache is already cleared by the
 *  time this fires, so a fresh getGpuContext() rebuilds from scratch. */
export function setGpuLostHandler(fn: ((info: GPUDeviceLostInfo) => void) | null): void {
  onLost = fn
}

/** Cheap availability probe — does not force device creation. */
export function hasWebGpu(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && !!navigator.gpu
}

async function initGpu(): Promise<GpuContext | null> {
  try {
    if (!hasWebGpu()) return null
    const gpu = navigator.gpu
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!adapter) return null
    const device = await adapter.requestDevice()
    const canvasFormat = gpu.getPreferredCanvasFormat()
    // On device loss, drop the cache and notify the owner. `device.lost` resolves
    // (never rejects) on loss, including on explicit destroy().
    void device.lost.then((info) => {
      if (current?.device === device) current = null
      pending = null
      onLost?.(info)
    })
    return { adapter, device, canvasFormat }
  } catch {
    return null
  }
}

/** Lazily acquire the shared GPU context. Cached on success; concurrent callers
 *  share one in-flight request; a failure clears the latch so a later call can
 *  retry. */
export async function getGpuContext(): Promise<GpuContext | null> {
  if (current) return current
  if (!pending) {
    pending = initGpu().then((ctx) => {
      current = ctx
      if (!ctx) pending = null // allow a retry after a transient failure
      return ctx
    })
  }
  return pending
}

/** Destroy the shared device and clear the cache (worker shutdown). */
export function disposeGpu(): void {
  current?.device.destroy()
  current = null
  pending = null
}
