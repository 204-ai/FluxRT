import { describe, it, expect } from 'vitest'
import {
  applyCompositeOp,
  defaultComposite,
  identityTransform,
  type Composite,
} from './types'

const ids = (c: Composite) => c.map((l) => l.id)

describe('defaultComposite', () => {
  it('seeds the three legacy layers front → back', () => {
    expect(ids(defaultComposite())).toEqual(['camera', 'video', 'feedback'])
    expect(defaultComposite().every((l) => l.opacity === 1 && l.blend === 'normal')).toBe(true)
  })
  it('returns a fresh array each call', () => {
    expect(defaultComposite()).not.toBe(defaultComposite())
  })
})

describe('applyCompositeOp patch', () => {
  it('updates mix fields by id, leaving order and other layers intact', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'patch', layers: [{ id: 'video', opacity: 0.3, blend: 'screen' }] })
    expect(ids(c)).toEqual(['camera', 'video', 'feedback'])
    expect(c.find((l) => l.id === 'video')).toMatchObject({ opacity: 0.3, blend: 'screen' })
    expect(c.find((l) => l.id === 'camera')!.opacity).toBe(1)
  })

  it('sets mirror per layer', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'patch', layers: [{ id: 'camera', mirror: true }] })
    expect(c.find((l) => l.id === 'camera')!.mirror).toBe(true)
  })

  it('clears transform when patched with undefined (reset to cover-fit)', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'patch', layers: [{ id: 'camera', transform: identityTransform() }] })
    expect(c.find((l) => l.id === 'camera')!.transform).toBeDefined()
    applyCompositeOp(c, { op: 'patch', layers: [{ id: 'camera', transform: undefined }] })
    expect(c.find((l) => l.id === 'camera')!.transform).toBeUndefined()
  })

  it('ignores a patch for an unknown id (no crash)', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'patch', layers: [{ id: 'ghost', opacity: 0 }] })
    expect(ids(c)).toEqual(['camera', 'video', 'feedback'])
  })
})

describe('applyCompositeOp add / remove', () => {
  it('adds at the given index (0 = frontmost) and dedupes by id', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'add', layer: { id: 'l2', opacity: 1, blend: 'normal', mirror: false } })
    expect(ids(c)).toEqual(['l2', 'camera', 'video', 'feedback'])
    applyCompositeOp(c, {
      op: 'add',
      layer: { id: 'l3', opacity: 1, blend: 'normal', mirror: false },
      index: 2,
    })
    expect(ids(c)).toEqual(['l2', 'camera', 'l3', 'video', 'feedback'])
    // duplicate id is a no-op
    applyCompositeOp(c, { op: 'add', layer: { id: 'l2', opacity: 0, blend: 'normal', mirror: false } })
    expect(ids(c).filter((i) => i === 'l2')).toHaveLength(1)
  })

  it('removes by id', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'remove', id: 'video' })
    expect(ids(c)).toEqual(['camera', 'feedback'])
    applyCompositeOp(c, { op: 'remove', id: 'nope' })
    expect(ids(c)).toEqual(['camera', 'feedback'])
  })
})

describe('applyCompositeOp reorder', () => {
  it('reorders by the given order', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'reorder', order: ['feedback', 'camera', 'video'] })
    expect(ids(c)).toEqual(['feedback', 'camera', 'video'])
  })
  it('appends any layer not named in the order', () => {
    const c = defaultComposite()
    applyCompositeOp(c, { op: 'reorder', order: ['video'] })
    expect(ids(c)).toEqual(['video', 'camera', 'feedback'])
  })
})
