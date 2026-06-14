import { describe, expect, it } from 'vitest'
import { applyFeatureChange, ratingSum, removePhrase } from './features'

describe('removePhrase', () => {
  it('drops only the exact segment', () => {
    expect(removePhrase('a cat, pointy elf ears, 8k', 'pointy elf ears')).toBe('a cat, 8k')
  })
  it('keeps superstrings', () => {
    expect(removePhrase('very pointy elf ears indeed', 'pointy elf ears')).toBe(
      'very pointy elf ears indeed',
    )
  })
})

describe('applyFeatureChange', () => {
  it('appends a new phrase', () => {
    const r = applyFeatureChange('a portrait', {}, 'ears', 'pointy elf ears')
    expect(r.prompt).toBe('a portrait, pointy elf ears')
    expect(r.state.ears).toBe('pointy elf ears')
  })

  it('replaces the slot phrase, leaves the rest', () => {
    const s1 = applyFeatureChange('a portrait', {}, 'ears', 'pointy elf ears')
    const s2 = applyFeatureChange(s1.prompt, s1.state, 'eyes', 'giant googly eyes')
    const s3 = applyFeatureChange(s2.prompt, s2.state, 'ears', 'pig snout nose')
    expect(s3.prompt).toBe('a portrait, giant googly eyes, pig snout nose')
  })

  it('starts from an empty prompt', () => {
    const r = applyFeatureChange('', {}, 'style', 'in the style of vaporwave')
    expect(r.prompt).toBe('in the style of vaporwave')
  })

  it('clearing a slot removes its phrase', () => {
    const s1 = applyFeatureChange('base', {}, 'mouth', 'sharp vampire fangs')
    const s2 = applyFeatureChange(s1.prompt, s1.state, 'mouth', '')
    expect(s2.prompt).toBe('base')
    expect(s2.state.mouth).toBe('')
  })
})

describe('ratingSum sort', () => {
  it('sorts best first', () => {
    const list = [
      { prompt: 'a', style: 1, tracking: 1, stability: 1 },
      { prompt: 'b', style: 5, tracking: 4, stability: 5 },
      { prompt: 'c', style: 0, tracking: 0, stability: 0 },
    ]
    list.sort((a, b) => ratingSum(b) - ratingSum(a))
    expect(list.map((e) => e.prompt)).toEqual(['b', 'a', 'c'])
  })
})
