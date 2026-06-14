// Effect factory registry — imported by BOTH the main thread (canvas
// backend) and the pipeline worker (streams backend), so effects are
// constructed wherever the compositing loop runs and configured by name.

import type { CanvasEffect } from '../core/types'
import { createMarkerEffect } from './marker'
import { createDrawLayerEffect } from './drawLayer'

const FACTORIES: Record<string, (config?: Record<string, unknown>) => CanvasEffect<any>> = {
  marker: (c) => createMarkerEffect(c),
  drawLayer: (c) => createDrawLayerEffect(c),
}

export function createEffect(name: string, config?: Record<string, unknown>): CanvasEffect<any> {
  const f = FACTORIES[name]
  if (!f) throw new Error('unknown effect: ' + name)
  return f(config)
}
