import { describe, expect, it } from 'vitest'
import {
  coverRect,
  identityTransform,
  layerDestRect,
  layerDrawRects,
  type LayerTransform,
} from './types'

describe('coverRect', () => {
  it('center-crops a wide source to fill (overflow left/right)', () => {
    // 200×100 into 100×100: scale 1, drawn at x=-50 spanning 200 (clipped by canvas).
    expect(coverRect(100, 100, 200, 100)).toEqual({ dx: -50, dy: 0, dw: 200, dh: 100 })
  })
  it('falls back to the full canvas for a degenerate source', () => {
    expect(coverRect(100, 80, 0, 0)).toEqual({ dx: 0, dy: 0, dw: 100, dh: 80 })
  })
})

describe('layerDrawRects — no transform (legacy cover-fit)', () => {
  it('draws the whole source into the cover rect', () => {
    expect(layerDrawRects(100, 100, 200, 100)).toEqual({
      sx: 0,
      sy: 0,
      sw: 200,
      sh: 100,
      dx: -50,
      dy: 0,
      dw: 200,
      dh: 100,
    })
  })
  it('returns null for a zero-sized source or canvas', () => {
    expect(layerDrawRects(100, 100, 0, 100)).toBeNull()
    expect(layerDrawRects(0, 100, 200, 100)).toBeNull()
  })
})

describe('layerDrawRects — identity transform', () => {
  it('is visually equivalent to cover-fit (pre-clipped center crop into full canvas)', () => {
    // The wide source overflows under cover; identity pre-clips to the visible
    // center span [50..150] and draws it into the full canvas [0..100].
    const r = layerDrawRects(100, 100, 200, 100, identityTransform())!
    expect(r).toEqual({ sx: 50, sy: 0, sw: 100, sh: 100, dx: 0, dy: 0, dw: 100, dh: 100 })
  })
})

describe('layerDestRect', () => {
  it('insets the frame by the crop fractions', () => {
    const t: LayerTransform = {
      frame: { x: 0.1, y: 0.2, w: 0.5, h: 0.5 },
      crop: { left: 0.1, right: 0.1, top: 0, bottom: 0 },
    }
    const d = layerDestRect(t)
    expect(d.x).toBeCloseTo(0.15)
    expect(d.y).toBeCloseTo(0.2)
    expect(d.w).toBeCloseTo(0.4)
    expect(d.h).toBeCloseTo(0.5)
  })
})

describe('layerDrawRects — crop keeps content anchored', () => {
  it('cropping the sides trims the source and shrinks the dest to match (no shift)', () => {
    // Square source, square canvas: cover is 1:1. Cropping 25% off each side
    // shows the source middle [25..75] in the dest middle [25..75] — content
    // stays put, the box just clips.
    const t: LayerTransform = {
      frame: { x: 0, y: 0, w: 1, h: 1 },
      crop: { left: 0.25, right: 0.25, top: 0, bottom: 0 },
    }
    const r = layerDrawRects(100, 100, 100, 100, t)!
    expect(r.sx).toBeCloseTo(25)
    expect(r.sw).toBeCloseTo(50)
    expect(r.dx).toBeCloseTo(25)
    expect(r.dw).toBeCloseTo(50)
    expect(r.dy).toBeCloseTo(0)
    expect(r.dh).toBeCloseTo(100)
  })
  it('returns null when a crop collapses the layer to nothing', () => {
    const t: LayerTransform = {
      frame: { x: 0, y: 0, w: 1, h: 1 },
      crop: { left: 1, right: 0, top: 0, bottom: 0 },
    }
    expect(layerDrawRects(100, 100, 100, 100, t)).toBeNull()
  })
})

describe('layerDrawRects — frame move/resize', () => {
  it('places the full content into a shrunken, offset frame', () => {
    const t: LayerTransform = {
      frame: { x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
      crop: { left: 0, right: 0, top: 0, bottom: 0 },
    }
    // Square source/canvas: whole source drawn into the centered half-rect.
    const r = layerDrawRects(100, 100, 100, 100, t)!
    expect(r).toEqual({ sx: 0, sy: 0, sw: 100, sh: 100, dx: 25, dy: 25, dw: 50, dh: 50 })
  })
})
