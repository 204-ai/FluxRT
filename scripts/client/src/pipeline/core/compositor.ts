// Per-frame compositing shared by both backends: a dynamic ordered stack of
// layers (each with its own source frame, opacity, blend, transform, mirror) +
// ordered effect layers. Runs in the pipeline worker for the streams backend,
// on the main thread for the canvas fallback.

import { AnalyzerBus } from './bus'
import type {
  BlendMode,
  CanvasEffect,
  Composite,
  CompositeOp,
  Ctx2D,
  EffectInit,
  LayerId,
  LayerOptions,
} from './types'
import { applyCompositeOp, defaultComposite, layerDrawRects } from './types'
import { createEffect } from '../effects/registry'

type FrameSource = CanvasImageSource | VideoFrame
/** Live frame per layer id for the current pass (absent / null = layer not
 *  drawn this frame). */
export type FrameMap = Record<LayerId, FrameSource | null | undefined>

const BLEND_OP: Record<BlendMode, GlobalCompositeOperation> = {
  normal: 'source-over',
  screen: 'screen',
  multiply: 'multiply',
  difference: 'difference',
}

function dimsOf(src: FrameSource): { w: number; h: number } {
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
  // Legacy GLOBAL effect chain (marker/drawLayer), rendered after all layers.
  // Kept until the UI moves draw/marker to effect clips.
  private effects: CanvasEffect[] = []
  // Effect LAYERS, addressed by layer id — interleaved at their stack position.
  private effectLayers = new Map<LayerId, CanvasEffect>()
  composite: Composite = defaultComposite()

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

  /** Replace the whole composite (init) or apply a structural/mix op. */
  setComposite(value: Composite | CompositeOp): void {
    if (Array.isArray(value)) {
      // Init / full replace — copy so callers keep their own array.
      this.composite = value.map((l) => ({ ...l }))
    } else {
      applyCompositeOp(this.composite, value)
    }
    this.reconcileEffectLayers()
  }

  /** Create/configure/dispose effect-layer CanvasEffect instances to match the
   *  composite. Each effect layer owns its own instance (keyed by layer id), so
   *  two draw layers are independent. */
  private reconcileEffectLayers(): void {
    const wanted = new Set<LayerId>()
    for (const l of this.composite) {
      if (!l.effectName) continue
      wanted.add(l.id)
      let fx = this.effectLayers.get(l.id)
      if (!fx) {
        fx = createEffect(l.effectName, l.effectConfig)
        fx.init?.(this.width, this.height)
        this.effectLayers.set(l.id, fx)
      } else if (l.effectConfig) {
        fx.configure(l.effectConfig)
      }
    }
    for (const [id, fx] of this.effectLayers) {
      if (!wanted.has(id)) {
        fx.dispose?.()
        this.effectLayers.delete(id)
      }
    }
  }

  /** Forward an out-of-band message (e.g. a draw stroke) to one effect LAYER. */
  effectLayerMessage(layerId: LayerId, data: unknown): void {
    this.effectLayers.get(layerId)?.message?.(data)
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
   *  horizontally about its own box center (selfie view) — for the full-canvas
   *  default box this is the legacy full-frame flip. */
  private drawLayer(src: FrameSource, opts: LayerOptions, mirror: boolean): void {
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

  /** Composite the layer stack back-to-front (the composite is ordered front →
   *  back, so we iterate in reverse) — each layer with its own source frame,
   *  opacity, blend, transform and mirror — over an opaque base so the output
   *  frame is never semi-transparent (transparency encodes as black upstream
   *  and breaks screen/multiply). A layer with no frame this pass is skipped. */
  drawComposite(frames: FrameMap, tsMs: number): void {
    const { ctx, width: W, height: H } = this
    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, W, H)
    const info = { width: W, height: H, tsMs }
    // Back-to-front: source layers draw their frame; effect layers run their
    // CanvasEffect at this position, transforming everything composited below.
    for (let i = this.composite.length - 1; i >= 0; i--) {
      const layer = this.composite[i]
      if (layer.effectName) {
        const fx = this.effectLayers.get(layer.id)
        if (fx && layer.opacity > 0) {
          ctx.save()
          ctx.globalAlpha = layer.opacity
          ctx.globalCompositeOperation = 'source-over'
          fx.render(ctx, info, this.bus)
          ctx.restore()
        }
        continue
      }
      const src = frames[layer.id]
      if (src) this.drawLayer(src, layer, layer.mirror)
    }
    // Legacy global effects run last (compat until the UI uses effect clips).
    for (const e of this.effects) e.render(ctx, info, this.bus)
  }

  disposeEffects(): void {
    for (const e of this.effects) e.dispose?.()
    this.effects = []
    for (const fx of this.effectLayers.values()) fx.dispose?.()
    this.effectLayers.clear()
  }
}
