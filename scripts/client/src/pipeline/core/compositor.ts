// Per-frame compositing shared by both backends: camera + video layers
// (order/opacity/blend) + ordered effect layers. Runs in the pipeline worker
// for the streams backend, on the main thread for the canvas fallback.

import { AnalyzerBus } from './bus'
import type { BlendMode, CanvasEffect, CompositeOptions, Ctx2D, EffectInit } from './types'
import { createEffect } from '../effects/registry'

type Layer = CanvasImageSource | VideoFrame

const BLEND_OP: Record<BlendMode, GlobalCompositeOperation> = {
  normal: 'source-over',
  screen: 'screen',
  multiply: 'multiply',
  difference: 'difference',
}

function dimsOf(src: Layer): { w: number; h: number } {
  if (typeof VideoFrame !== 'undefined' && src instanceof VideoFrame) {
    return { w: src.displayWidth, h: src.displayHeight }
  }
  // HTMLVideoElement is undefined in the worker (video arrives as VideoFrame there).
  if (typeof HTMLVideoElement !== 'undefined' && src instanceof HTMLVideoElement) {
    return { w: src.videoWidth, h: src.videoHeight }
  }
  return { w: 0, h: 0 }
}

export class Compositor {
  readonly bus = new AnalyzerBus()
  private effects: CanvasEffect[] = []
  mirrored = false
  composite: CompositeOptions = { order: 'camera-over', opacity: 0.5, blend: 'normal' }

  constructor(
    private ctx: Ctx2D,
    private width: number,
    private height: number,
  ) {}

  setEffects(inits: EffectInit[]): void {
    this.disposeEffects()
    this.effects = inits.map((e) => createEffect(e.name, e.config))
    for (const e of this.effects) e.init?.(this.width, this.height)
  }

  setComposite(patch: Partial<CompositeOptions>): void {
    Object.assign(this.composite, patch)
  }

  configureEffect(name: string, patch: Record<string, unknown>): void {
    this.effects.find((e) => e.name === name)?.configure(patch)
  }

  effectMessage(name: string, data: unknown): void {
    this.effects.find((e) => e.name === name)?.message?.(data)
  }

  /** Mirror applies to the camera layer only (selfie view). Cover-fit
   *  (center-crop) so a camera whose aspect differs from the output canvas is
   *  cropped, never stretched. */
  private drawCamera(src: Layer, alpha: number, blend: GlobalCompositeOperation): void {
    const { ctx, width: W, height: H } = this
    const { w, h } = dimsOf(src)
    let dx = 0
    let dy = 0
    let dw = W
    let dh = H
    if (w > 0 && h > 0) {
      const scale = Math.max(W / w, H / h)
      dw = w * scale
      dh = h * scale
      dx = (W - dw) / 2
      dy = (H - dh) / 2
    }
    ctx.save()
    ctx.globalAlpha = alpha
    ctx.globalCompositeOperation = blend
    if (this.mirrored) {
      ctx.scale(-1, 1)
      // In flipped space, an image spanning screen [dx, dx+dw] is drawn at -(dx+dw).
      ctx.drawImage(src as CanvasImageSource, -(dx + dw), dy, dw, dh)
    } else {
      ctx.drawImage(src as CanvasImageSource, dx, dy, dw, dh)
    }
    ctx.restore()
  }

  /** Video layer: cover-fit (center-crop) — letterbox bars would feed black
   *  into the model and break screen/multiply blends. */
  private drawVideo(src: Layer, alpha: number, blend: GlobalCompositeOperation): void {
    const { ctx, width: W, height: H } = this
    const { w, h } = dimsOf(src)
    let dx = 0
    let dy = 0
    let dw = W
    let dh = H
    if (w > 0 && h > 0) {
      const scale = Math.max(W / w, H / h)
      dw = w * scale
      dh = h * scale
      dx = (W - dw) / 2
      dy = (H - dh) / 2
    }
    ctx.save()
    ctx.globalAlpha = alpha
    ctx.globalCompositeOperation = blend
    ctx.drawImage(src as CanvasImageSource, dx, dy, dw, dh)
    ctx.restore()
  }

  drawComposite(camera: Layer | null, video: Layer | null, tsMs: number): void {
    const { ctx, width: W, height: H } = this
    if (camera && video) {
      const { order, opacity, blend } = this.composite
      const op = BLEND_OP[blend] ?? 'source-over'
      if (order === 'camera-over') {
        this.drawVideo(video, 1, 'source-over')
        this.drawCamera(camera, opacity, op)
      } else {
        this.drawCamera(camera, 1, 'source-over')
        this.drawVideo(video, opacity, op)
      }
    } else if (camera) {
      this.drawCamera(camera, 1, 'source-over')
    } else if (video) {
      this.drawVideo(video, 1, 'source-over')
    } else {
      ctx.clearRect(0, 0, W, H)
    }
    const info = { width: W, height: H, tsMs }
    for (const e of this.effects) e.render(ctx, info, this.bus)
  }

  disposeEffects(): void {
    for (const e of this.effects) e.dispose?.()
    this.effects = []
  }
}
