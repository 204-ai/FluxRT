// Hand/body marker effect — colored circle (+ optional fading trail) on a
// chosen pose landmark. Port of input_processor.js step 2, except landmarks
// now come from the shared vision analyzer via the bus (key 'pose') instead
// of a bespoke PoseLandmarker. Detection runs on the COMPOSITE frame, which
// already reflects the camera mirror — so the landmarks are in the same space
// we render into. We use them 1:1 with no extra x-flip or left/right id swap,
// and the marker follows the mirror toggle automatically (same reasoning as
// the sense overlay, which also draws 1:1 — see OverlayCanvas).

import type { BusReader, CanvasEffect, Ctx2D, FrameInfo } from '../core/types'

export interface MarkerConfig {
  enabled: boolean
  landmark: number // BlazePose index: 15=L wrist, 16=R wrist, 19/20 index, 0 nose, 11/12 shoulders
  color: string
  size: number
  trail: boolean
  trailLen: number
}

export interface PoseBusValue {
  landmarks: Array<{ x: number; y: number; visibility?: number }>
}

function hexToRgb(hex: string): string {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || '').trim())
  if (!m) return '255, 60, 60'
  const n = parseInt(m[1], 16)
  return `${(n >> 16) & 0xff}, ${(n >> 8) & 0xff}, ${n & 0xff}`
}

export function createMarkerEffect(initial?: Partial<MarkerConfig>): CanvasEffect<MarkerConfig> {
  const trail: Array<{ x: number; y: number }> = []
  return {
    name: 'marker',
    config: {
      enabled: false,
      landmark: 15,
      color: '#ff3c3c',
      size: 32,
      trail: false,
      trailLen: 20,
      ...initial,
    },
    configure(patch) {
      if (
        (patch.landmark !== undefined && patch.landmark !== this.config.landmark) ||
        patch.trail === false ||
        patch.enabled === false
      ) {
        trail.length = 0
      }
      Object.assign(this.config, patch)
    },
    render(ctx: Ctx2D, info: FrameInfo, bus: BusReader) {
      const o = this.config
      if (!o.enabled) {
        trail.length = 0
        return
      }
      const pose = bus.get<PoseBusValue>('pose')
      let cx: number | null = null
      let cy: number | null = null
      const lm = pose?.landmarks?.[o.landmark]
      if (lm && (lm.visibility === undefined || lm.visibility > 0.5)) {
        cx = lm.x * info.width
        cy = lm.y * info.height
      }

      const rgb = hexToRgb(o.color)
      const baseR = o.size

      if (cx !== null && cy !== null && o.trail) {
        const last = trail[trail.length - 1]
        if (!last || last.x !== cx || last.y !== cy) trail.push({ x: cx, y: cy })
        while (trail.length > o.trailLen) trail.shift()
      } else if (!o.trail) {
        trail.length = 0
      }

      if (o.trail && trail.length > 1) {
        for (let i = 0; i < trail.length; i++) {
          const p = trail[i]
          const t = (i + 1) / trail.length // newest = 1
          const r = baseR * (0.35 + 0.65 * t)
          ctx.beginPath()
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(${rgb}, ${0.15 + 0.55 * t})`
          ctx.fill()
        }
      }

      if (cx !== null && cy !== null) {
        ctx.beginPath()
        ctx.arc(cx, cy, baseR, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${rgb}, 0.9)`
        ctx.fill()
        ctx.lineWidth = Math.max(2, baseR * 0.1)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)'
        ctx.stroke()
      }
    },
  }
}
