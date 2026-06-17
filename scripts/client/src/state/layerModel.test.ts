import { describe, it, expect } from 'vitest'
import {
  activeCell,
  activeClip,
  findClip,
  layerById,
  layerKind,
  makeClip,
  makeLayer,
  seedLayers,
} from './layerModel'

describe('seedLayers', () => {
  it('seeds camera/video/feedback front → back, each with one active clip', () => {
    const layers = seedLayers()
    expect(layers.map((l) => l.id)).toEqual(['camera', 'video', 'feedback'])
    for (const l of layers) {
      expect(l.cells).toHaveLength(1)
      expect(l.activeCellId).toBe(l.cells[0].id)
      expect(activeClip(l)).not.toBeNull()
    }
    expect(layers.map((l) => layerKind(l))).toEqual(['camera', 'video', 'feedback'])
  })

  it('gives the camera clip a selfie-mirror default, others not', () => {
    const layers = seedLayers()
    expect(activeClip(layers[0])!.mirror).toBe(true) // camera
    expect(activeClip(layers[1])!.mirror).toBe(false) // video
    expect(activeClip(layers[2])!.mirror).toBe(false) // feedback
  })
})

describe('makeClip / makeLayer', () => {
  it('makeClip defaults mirror from the kind', () => {
    expect(makeClip('camera').mirror).toBe(true)
    expect(makeClip('video').mirror).toBe(false)
    expect(makeClip('screen').mirror).toBe(false)
  })
  it('makeLayer pre-seeds one active clip of the kind', () => {
    const l = makeLayer('video', 'beach.mp4')
    expect(l.name).toBe('beach.mp4')
    expect(l.cells).toHaveLength(1)
    expect(activeClip(l)?.kind).toBe('video')
    expect(activeCell(l)).toBe(l.cells[0])
  })
  it('mints unique ids', () => {
    const a = makeLayer('video', 'a')
    const b = makeLayer('video', 'b')
    expect(a.id).not.toBe(b.id)
    expect(activeClip(a)!.id).not.toBe(activeClip(b)!.id)
  })
})

describe('lookups', () => {
  it('layerById finds a seeded layer', () => {
    const layers = seedLayers()
    expect(layerById(layers, 'video')?.name).toBe('Video')
    expect(layerById(layers, 'nope')).toBeUndefined()
  })
  it('findClip locates a clip and its layer anywhere in the stack', () => {
    const layers = seedLayers()
    const camClip = activeClip(layers[0])!
    const hit = findClip(layers, camClip.id)
    expect(hit?.layer.id).toBe('camera')
    expect(hit?.clip).toBe(camClip)
    expect(findClip(layers, 'ghost')).toBeNull()
  })
})
