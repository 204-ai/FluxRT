// Persistent freehand drawing layer, composited topmost on every frame.
// Strokes arrive as normalized
// [0..1] coordinates via message() so the same code runs on the main thread
// or inside the pipeline worker.

import type { CanvasEffect, Ctx2D, FrameInfo } from '../core/types'

export interface DrawLayerConfig {
  color: string
  size: number
  erase: boolean
}

export type StrokeMessage =
  | { type: 'begin'; x: number; y: number }
  | { type: 'move'; x: number; y: number }
  | { type: 'end' }
  | { type: 'clear' }

export function createDrawLayerEffect(initial?: Partial<DrawLayerConfig>): CanvasEffect<DrawLayerConfig> {
  let layer: OffscreenCanvas | null = null
  let lctx: OffscreenCanvasRenderingContext2D | null = null
  let last: { x: number; y: number } | null = null
  let W = 0
  let H = 0

  return {
    name: 'drawLayer',
    config: { color: '#ffffff', size: 6, erase: false, ...initial },
    configure(patch) {
      Object.assign(this.config, patch)
    },
    init(width: number, height: number) {
      W = width
      H = height
      layer = new OffscreenCanvas(W, H)
      lctx = layer.getContext('2d')
      last = null
    },
    message(data: unknown) {
      const m = data as StrokeMessage
      if (!lctx || !layer) return
      const o = this.config
      const compose = o.erase ? 'destination-out' : 'source-over'
      if (m.type === 'begin') {
        const p = { x: m.x * W, y: m.y * H }
        last = p
        lctx.globalCompositeOperation = compose
        lctx.beginPath()
        lctx.fillStyle = o.color
        lctx.arc(p.x, p.y, o.size / 2, 0, Math.PI * 2)
        lctx.fill()
      } else if (m.type === 'move') {
        if (!last) return
        const p = { x: m.x * W, y: m.y * H }
        lctx.globalCompositeOperation = compose
        lctx.strokeStyle = o.color
        lctx.lineWidth = o.size
        lctx.lineCap = 'round'
        lctx.lineJoin = 'round'
        lctx.beginPath()
        lctx.moveTo(last.x, last.y)
        lctx.lineTo(p.x, p.y)
        lctx.stroke()
        last = p
      } else if (m.type === 'end') {
        last = null
      } else if (m.type === 'clear') {
        lctx.clearRect(0, 0, W, H)
      }
    },
    render(ctx: Ctx2D, info: FrameInfo) {
      if (layer) ctx.drawImage(layer, 0, 0, info.width, info.height)
    },
    dispose() {
      layer = null
      lctx = null
    },
  }
}
