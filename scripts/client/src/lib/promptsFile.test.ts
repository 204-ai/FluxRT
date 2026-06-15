import { describe, expect, it } from 'vitest'
import { parsePromptsFile } from './promptsFile'

describe('parsePromptsFile', () => {
  it('parses one prompt per line, skipping blanks and comments', () => {
    const text = 'a cat\n\n# a comment\n  a dog  \n'
    expect(parsePromptsFile(text)).toEqual([
      { prompt: 'a cat', style: 0, tracking: 0, stability: 0 },
      { prompt: 'a dog', style: 0, tracking: 0, stability: 0 },
    ])
  })

  it('parses a JSON array of strings', () => {
    expect(parsePromptsFile('["a cat", "a dog"]')).toEqual([
      { prompt: 'a cat', style: 0, tracking: 0, stability: 0 },
      { prompt: 'a dog', style: 0, tracking: 0, stability: 0 },
    ])
  })

  it('parses a JSON array of objects with ratings', () => {
    const text = JSON.stringify([{ prompt: 'a cat', style: 2, tracking: 1, stability: 3 }])
    expect(parsePromptsFile(text)).toEqual([
      { prompt: 'a cat', style: 2, tracking: 1, stability: 3 },
    ])
  })

  it('parses the {prompts:[...]} wrapper exported by the server', () => {
    const text = JSON.stringify({ prompts: [{ prompt: 'a cat', style: 1, tracking: 0, stability: 0 }] })
    expect(parsePromptsFile(text)).toEqual([
      { prompt: 'a cat', style: 1, tracking: 0, stability: 0 },
    ])
  })

  it('defaults missing/invalid ratings to 0', () => {
    const text = JSON.stringify([{ prompt: 'a cat', style: 'high' }])
    expect(parsePromptsFile(text)).toEqual([
      { prompt: 'a cat', style: 0, tracking: 0, stability: 0 },
    ])
  })

  it('drops whitespace-only and entryless items', () => {
    const text = JSON.stringify(['  ', { style: 5 }, { prompt: '   ' }, 'real'])
    expect(parsePromptsFile(text)).toEqual([
      { prompt: 'real', style: 0, tracking: 0, stability: 0 },
    ])
  })

  it('returns [] for empty input', () => {
    expect(parsePromptsFile('')).toEqual([])
    expect(parsePromptsFile('   \n  ')).toEqual([])
  })

  it('falls back to line parsing when JSON is malformed', () => {
    // Starts like JSON but is not valid — treated as a single text line.
    expect(parsePromptsFile('[not really json')).toEqual([
      { prompt: '[not really json', style: 0, tracking: 0, stability: 0 },
    ])
  })
})
