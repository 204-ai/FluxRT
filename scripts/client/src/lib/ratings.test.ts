import { describe, expect, it } from 'vitest'
import { applyVerdict, parseStored, serializeRatings, type RatingMap } from './ratings'

const META = { ts: 100, seed: '52', steps: '2' }

describe('applyVerdict', () => {
  it('adds a rating keyed by trimmed prompt', () => {
    const m = applyVerdict({}, '  a cat  ', 'love', META)
    expect(m).toEqual({ 'a cat': { prompt: 'a cat', verdict: 'love', ts: 100, seed: '52', steps: '2' } })
  })

  it('overwrites with the latest verdict for the same prompt', () => {
    let m = applyVerdict({}, 'a cat', 'like', { ts: 1 })
    m = applyVerdict(m, 'a cat', 'love', { ts: 2 })
    expect(Object.keys(m)).toHaveLength(1)
    expect(m['a cat'].verdict).toBe('love')
    expect(m['a cat'].ts).toBe(2)
  })

  it('toggles off when the same verdict is re-applied', () => {
    let m = applyVerdict({}, 'a cat', 'skip', META)
    m = applyVerdict(m, 'a cat', 'skip', META)
    expect(m).toEqual({})
  })

  it('ignores blank prompts and does not mutate the input', () => {
    const src: RatingMap = { 'a cat': { prompt: 'a cat', verdict: 'like', ts: 1 } }
    expect(applyVerdict(src, '   ', 'love', META)).toBe(src)
  })
})

describe('serializeRatings', () => {
  it('emits an array sorted newest-first', () => {
    const m: RatingMap = {
      old: { prompt: 'old', verdict: 'like', ts: 1 },
      new: { prompt: 'new', verdict: 'love', ts: 9 },
    }
    expect(JSON.parse(serializeRatings(m)).map((r: { prompt: string }) => r.prompt)).toEqual(['new', 'old'])
  })
})

describe('parseStored', () => {
  it('round-trips a serialized map', () => {
    const m = applyVerdict({}, 'a cat', 'love', META)
    expect(parseStored(serializeRatings(m))).toEqual(m)
  })

  it('accepts a bare object map shape', () => {
    const raw = JSON.stringify({ 'a cat': { prompt: 'a cat', verdict: 'skip', ts: 5 } })
    expect(parseStored(raw)['a cat'].verdict).toBe('skip')
  })

  it('drops rows with bad/unknown verdicts or missing prompts', () => {
    const raw = JSON.stringify([
      { prompt: 'ok', verdict: 'like', ts: 1 },
      { prompt: 'bad', verdict: 'meh', ts: 1 },
      { verdict: 'love', ts: 1 },
    ])
    expect(Object.keys(parseStored(raw))).toEqual(['ok'])
  })

  it('returns {} for empty or corrupt input', () => {
    expect(parseStored(null)).toEqual({})
    expect(parseStored('')).toEqual({})
    expect(parseStored('{not json')).toEqual({})
  })
})
