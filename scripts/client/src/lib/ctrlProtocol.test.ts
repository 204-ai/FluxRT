import { describe, expect, it } from 'vitest'
import { decodeCtrl, encodeCtrl } from './ctrlProtocol'

describe('encodeCtrl', () => {
  it('encodes prompt/seed/steps', () => {
    expect(encodeCtrl({ kind: 'prompt', text: 'a cat, in the style of: noir' })).toBe(
      'prompt:a cat, in the style of: noir',
    )
    expect(encodeCtrl({ kind: 'seed', value: 52 })).toBe('seed:52')
    expect(encodeCtrl({ kind: 'steps', value: 4 })).toBe('steps:4')
  })
})

describe('decodeCtrl', () => {
  it('decodes ref messages', () => {
    expect(decodeCtrl('ref:set:7')).toEqual({ kind: 'refSet', version: 7 })
    expect(decodeCtrl('ref:clear:9')).toEqual({ kind: 'refClear', version: 9 })
    expect(decodeCtrl('ref:clear')).toEqual({ kind: 'refClear', version: 0 })
  })

  it('decodes input roles', () => {
    expect(decodeCtrl('input:you')).toEqual({ kind: 'inputRole', role: 'you' })
    expect(decodeCtrl('input:peer')).toEqual({ kind: 'inputRole', role: 'peer' })
    expect(decodeCtrl('input:server')).toEqual({ kind: 'inputRole', role: 'server' })
  })

  it('decodes lip toggle', () => {
    expect(decodeCtrl('lip:on')).toEqual({ kind: 'lip', on: true })
    expect(decodeCtrl('lip:off')).toEqual({ kind: 'lip', on: false })
  })

  it('preserves colons inside state:prompt payloads', () => {
    expect(decodeCtrl('state:prompt:portrait, style: kubrick, 8k')).toEqual({
      kind: 'statePrompt',
      text: 'portrait, style: kubrick, 8k',
    })
  })

  it('decodes state seed/steps and prompts:changed', () => {
    expect(decodeCtrl('state:seed:42')).toEqual({ kind: 'stateSeed', value: '42' })
    expect(decodeCtrl('state:steps:3')).toEqual({ kind: 'stateSteps', value: '3' })
    expect(decodeCtrl('prompts:changed')).toEqual({ kind: 'promptsChanged' })
  })

  it('falls through to unknown (acks, errors)', () => {
    expect(decodeCtrl('ack:prompt')).toEqual({ kind: 'unknown', raw: 'ack:prompt' })
    expect(decodeCtrl('err:seed')).toEqual({ kind: 'unknown', raw: 'err:seed' })
  })
})
