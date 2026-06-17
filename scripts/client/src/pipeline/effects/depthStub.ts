// Depth effect layer — a 2D-canvas NO-OP. The real depth render lives in the
// WebGPU compositor (it samples a GPU depth texture produced by the ort-web
// DepthSession). This stub exists only so createEffect('depth') is valid on the
// 2D-canvas fallback backend (depth has no CPU implementation), and so the
// effect config (strength/mode/near/far) round-trips through the registry.

import type { CanvasEffect } from '../core/types'

export interface DepthConfig {
  /** 0..1 effect strength (mix between the scene below and the depth result). */
  strength: number
  /** fog = depth-faded; replace = show the depth map; mask = keep near, cut far. */
  mode: 'fog' | 'replace' | 'mask'
  near: number
  far: number
  /** Model input size (multiple of 14). Bigger = sharper depth + slower. */
  size: number
}

export function createDepthStub(config?: Record<string, unknown>): CanvasEffect<DepthConfig> {
  const cfg: DepthConfig = { strength: 1, mode: 'replace', near: 0, far: 1, size: 518, ...(config as Partial<DepthConfig>) }
  return {
    name: 'depth',
    config: cfg,
    configure(patch) {
      Object.assign(cfg, patch)
    },
    // No-op on the 2D backend — depth requires WebGPU (ort-web + a GPU texture).
    render() {},
  }
}

/** Map the string blend `mode` to the numeric code the WGSL depth_fs expects. */
export function depthModeNum(mode: unknown): number {
  return mode === 'replace' ? 1 : mode === 'mask' ? 2 : 0 // default fog
}
