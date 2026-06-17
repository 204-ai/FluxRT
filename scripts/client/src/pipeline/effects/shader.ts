// Shader-style effect layer: re-draws the composite-so-far through a CSS canvas
// filter (hue-rotate / invert / saturate / blur / …), transforming everything
// composited below it in the stack. Runs in the worker (OffscreenCanvas) and on
// the main thread alike — DOM-free.

import type { BusReader, CanvasEffect, Ctx2D, FrameInfo } from '../core/types'

export interface ShaderConfig {
  /** A CSS filter string, e.g. 'hue-rotate(90deg) saturate(1.4)'. 'none' = off. */
  filter: string
}

export function createShaderEffect(config?: Record<string, unknown>): CanvasEffect<ShaderConfig> {
  const cfg: ShaderConfig = { filter: 'hue-rotate(90deg)', ...(config as Partial<ShaderConfig>) }
  let tmp: OffscreenCanvas | null = null
  let tmpCtx: OffscreenCanvasRenderingContext2D | null = null

  return {
    name: 'shader',
    config: cfg,
    configure(patch) {
      Object.assign(cfg, patch)
    },
    render(ctx: Ctx2D, info: FrameInfo, _bus: BusReader) {
      if (!cfg.filter || cfg.filter === 'none') return
      const W = info.width
      const H = info.height
      if (!tmp || tmp.width !== W || tmp.height !== H) {
        tmp = new OffscreenCanvas(W, H)
        tmpCtx = tmp.getContext('2d')
      }
      if (!tmpCtx) return
      // Snapshot what's been composited below, then paint the FILTERED copy back
      // ON TOP at the compositor's globalAlpha (= the layer opacity). Source-over
      // of filtered@opacity over the untouched original yields
      // mix(original, filtered, opacity) — opacity is the effect STRENGTH, 0=off,
      // 1=full — matching the WebGPU compositor. (Clearing first instead would
      // drop the original and make opacity fade to transparent/black.)
      tmpCtx.clearRect(0, 0, W, H)
      tmpCtx.drawImage(ctx.canvas as CanvasImageSource, 0, 0)
      ctx.save()
      ctx.filter = cfg.filter
      ctx.drawImage(tmp as CanvasImageSource, 0, 0)
      ctx.restore()
      ctx.filter = 'none'
    },
    dispose() {
      tmp = null
      tmpCtx = null
    },
  }
}
