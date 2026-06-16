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
import { defaultComposite, layerDrawRects, mergeComposite } from './types'
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
  composite: CompositeOptions = defaultComposite()

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
    mergeComposite(this.composite, patch)
  }

  configureEffect(name: string, patch: Record<string, unknown>): void {
    this.effects.find((e) => e.name === name)?.configure(patch)
  }

  effectMessage(name: string, data: unknown): void {
    this.effects.find((e) => e.name === name)?.message?.(data)
  }

  /** Draw one layer with its own opacity + blend. Geometry comes from the
   *  layer's transform (OBS-style move/resize/crop); absent, it defaults to
   *  cover-fit (center-crop) so a source whose aspect differs from the output
   *  canvas is cropped, never stretched (letterbox bars would feed black into
   *  the model and break screen/multiply). `mirror` flips the content
   *  horizontally about its own box center (selfie view, camera only) — for the
   *  full-canvas default box this is the legacy full-frame flip. */
  private drawLayer(src: Layer, opts: LayerOptions, mirror: boolean): void {
    if (opts.opacity <= 0) return
    const { ctx, width: W, height: H } = this
    const { w, h } = dimsOf(src)
    const r = layerDrawRects(W, H, w, h, opts.transform)
    if (!r) return
    ctx.save()
    ctx.globalAlpha = opts.opacity
    ctx.globalCompositeOperation = BLEND_OP[opts.blend] ?? 'source-over'
    if (mirror) {
      const cx = r.dx + r.dw / 2
      ctx.translate(cx, 0)
      ctx.scale(-1, 1)
      ctx.translate(-cx, 0)
    }
    ctx.drawImage(src as CanvasImageSource, r.sx, r.sy, r.sw, r.sh, r.dx, r.dy, r.dw, r.dh)
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
