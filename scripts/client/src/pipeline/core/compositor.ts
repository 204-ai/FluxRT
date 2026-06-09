// Per-frame compositing shared by both backends: base frame (optionally
// mirrored) + ordered effect layers. Runs in the pipeline worker for the
// streams backend, on the main thread for the canvas fallback.

import { AnalyzerBus } from './bus'
import type { CanvasEffect, Ctx2D, EffectInit } from './types'
import { createEffect } from '../effects/registry'

export class Compositor {
  readonly bus = new AnalyzerBus()
  private effects: CanvasEffect[] = []
  mirrored = false

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

  configureEffect(name: string, patch: Record<string, unknown>): void {
    this.effects.find((e) => e.name === name)?.configure(patch)
  }

  effectMessage(name: string, data: unknown): void {
    this.effects.find((e) => e.name === name)?.message?.(data)
  }

  drawFrame(source: CanvasImageSource | VideoFrame, tsMs: number): void {
    const { ctx, width: W, height: H } = this
    if (this.mirrored) {
      ctx.save()
      ctx.scale(-1, 1)
      ctx.drawImage(source as CanvasImageSource, -W, 0, W, H)
      ctx.restore()
    } else {
      ctx.drawImage(source as CanvasImageSource, 0, 0, W, H)
    }
    const info = { width: W, height: H, tsMs, mirrored: this.mirrored }
    for (const e of this.effects) e.render(ctx, info, this.bus)
  }

  disposeEffects(): void {
    for (const e of this.effects) e.dispose?.()
    this.effects = []
  }
}
