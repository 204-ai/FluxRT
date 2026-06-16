// Per-frame compositing shared by both backends: camera + video layers
// (order/opacity/blend) + ordered effect layers. Runs in the pipeline worker
// for the streams backend, on the main thread for the canvas fallback.

import { AnalyzerBus } from './bus'
import type {
  BlendMode,
  CanvasEffect,
  CompositeOptions,
  CompositePatch,
  Ctx2D,
  EffectInit,
  LayerOptions,
} from './types'
import { createEffect } from '../effects/registry'

type Layer = CanvasImageSource | VideoFrame

const BLEND_OP: Record<BlendMode, GlobalCompositeOperation> = {
  normal: 'source-over',
  screen: 'screen',
  multiply: 'multiply',
  difference: 'difference',
}

/** Cover-fit (center-crop) destination rect: scale the source up to fill the
 *  W×H canvas, centered — never stretched, never letterboxed. */
function coverRect(
  W: number,
  H: number,
  w: number,
  h: number,
): { dx: number; dy: number; dw: number; dh: number } {
  if (w <= 0 || h <= 0) return { dx: 0, dy: 0, dw: W, dh: H }
  const scale = Math.max(W / w, H / h)
  const dw = w * scale
  const dh = h * scale
  return { dx: (W - dw) / 2, dy: (H - dh) / 2, dw, dh }
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
  composite: CompositeOptions = {
    camera: { opacity: 1, blend: 'normal' },
    video: { opacity: 1, blend: 'normal' },
    feedback: { opacity: 1, blend: 'normal' },
  }

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

  setComposite(patch: CompositePatch): void {
    // Per-layer merge — a flat Object.assign would drop the untouched field
    // (e.g. a blend-only patch must keep the layer's opacity).
    for (const id of ['camera', 'video', 'feedback'] as const) {
      if (patch[id]) Object.assign(this.composite[id], patch[id])
    }
  }

  configureEffect(name: string, patch: Record<string, unknown>): void {
    this.effects.find((e) => e.name === name)?.configure(patch)
  }

  effectMessage(name: string, data: unknown): void {
    this.effects.find((e) => e.name === name)?.message?.(data)
  }

  /** Draw one layer with its own opacity + blend. Cover-fit (center-crop) so a
   *  source whose aspect differs from the output canvas is cropped, never
   *  stretched (letterbox bars would feed black into the model and break
   *  screen/multiply). `mirror` flips horizontally (selfie view, camera only). */
  private drawLayer(src: Layer, opts: LayerOptions, mirror: boolean): void {
    if (opts.opacity <= 0) return
    const { ctx, width: W, height: H } = this
    const { w, h } = dimsOf(src)
    const { dx, dy, dw, dh } = coverRect(W, H, w, h)
    ctx.save()
    ctx.globalAlpha = opts.opacity
    ctx.globalCompositeOperation = BLEND_OP[opts.blend] ?? 'source-over'
    if (mirror) {
      ctx.scale(-1, 1)
      // In flipped space, an image spanning screen [dx, dx+dw] is drawn at -(dx+dw).
      ctx.drawImage(src as CanvasImageSource, -(dx + dw), dy, dw, dh)
    } else {
      ctx.drawImage(src as CanvasImageSource, dx, dy, dw, dh)
    }
    ctx.restore()
  }

  /** Composite the layer stack back-to-front — feedback (bottom) → video →
   *  camera (top) — each with its own opacity + blend, over an opaque base so
   *  the output frame is never semi-transparent (transparency encodes as black
   *  upstream and breaks screen/multiply). Any layer may be null/absent. */
  drawComposite(
    camera: Layer | null,
    video: Layer | null,
    feedback: Layer | null,
    tsMs: number,
  ): void {
    const { ctx, width: W, height: H } = this
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, W, H)
    if (feedback) this.drawLayer(feedback, this.composite.feedback, false)
    if (video) this.drawLayer(video, this.composite.video, false)
    if (camera) this.drawLayer(camera, this.composite.camera, this.mirrored)
    const info = { width: W, height: H, tsMs }
    for (const e of this.effects) e.render(ctx, info, this.bus)
  }

  disposeEffects(): void {
    for (const e of this.effects) e.dispose?.()
    this.effects = []
  }
}
