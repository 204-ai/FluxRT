import { describe, expect, it } from 'vitest'
import { composeFromAnalysis } from './senseCompose'
import type { HumanAnalysis } from '../vision/types'

function analysis(over: {
  smile?: number
  expression?: 'neutral' | 'happy' | 'surprised' | 'frowning' | 'squinting' | 'talking'
  browRaise?: number
  jawOpen?: number
  yaw?: number
  pitch?: number
  tilt?: number
  leaning?: 'left' | 'right' | 'centered'
  attention?: 'engaged' | 'looking away' | 'distracted'
  leftHand?: boolean
  rightHand?: boolean
  posture?: 'upright' | 'slouching' | 'unknown'
  activity?: 'still' | 'calm' | 'active' | 'very active'
  present?: boolean
  noFace?: boolean
}): HumanAnalysis {
  return {
    present: over.present ?? true,
    fps: 15,
    inferenceMs: 20,
    face: over.noFace
      ? null
      : {
          expression: over.expression ?? 'neutral',
          expressionScore: 0.5,
          smile: over.smile ?? 0.3,
          jawOpen: over.jawOpen ?? 0,
          browRaise: over.browRaise ?? 0,
          eyeBlinkLeft: 0,
          eyeBlinkRight: 0,
          blinking: false,
          headPose: { yaw: over.yaw ?? 0, pitch: over.pitch ?? 0, roll: 0 },
          attention: over.attention ?? 'engaged',
          topBlendshapes: [],
        },
    body: {
      leftHandRaised: over.leftHand ?? false,
      rightHandRaised: over.rightHand ?? false,
      leaning: over.leaning ?? 'centered',
      shoulderTilt: over.tilt ?? 0,
      movementEnergy: 0,
      activity: over.activity ?? 'calm',
      posture: over.posture ?? 'upright',
    },
  }
}

describe('composeFromAnalysis (natural theme)', () => {
  it('absent when nobody present or null', () => {
    expect(composeFromAnalysis(null).key).toBe('natural|absent')
    expect(composeFromAnalysis(analysis({ present: false })).key).toBe('natural|absent')
    expect(composeFromAnalysis(null).prompt).toContain('forest clearing')
  })

  it('valence picks medium', () => {
    expect(composeFromAnalysis(analysis({ smile: 0.8 })).prompt).toContain('impasto oils')
    expect(composeFromAnalysis(analysis({ expression: 'happy' })).prompt).toContain('impasto oils')
    expect(composeFromAnalysis(analysis({ smile: 0.3 })).prompt).toContain('watercolor')
    expect(composeFromAnalysis(analysis({ smile: 0.05 })).prompt).toContain('ink wash')
    expect(composeFromAnalysis(analysis({ expression: 'frowning', smile: 0.3 })).prompt).toContain('ink wash')
  })

  it('arousal picks weather from face or body', () => {
    expect(composeFromAnalysis(analysis({ browRaise: 0.7 })).prompt).toContain('storm wind')
    expect(composeFromAnalysis(analysis({ jawOpen: 0.4 })).prompt).toContain('gentle breeze')
    expect(composeFromAnalysis(analysis({ activity: 'very active' })).prompt).toContain('storm wind')
    expect(composeFromAnalysis(analysis({ activity: 'active' })).prompt).toContain('gentle breeze')
    expect(composeFromAnalysis(analysis({})).prompt).toContain('misty air')
  })

  it('gaze picks light', () => {
    expect(composeFromAnalysis(analysis({})).prompt).toContain('window light')
    expect(composeFromAnalysis(analysis({ attention: 'looking away' })).prompt).toContain('long amber evening shadows')
    expect(composeFromAnalysis(analysis({ pitch: 20 })).prompt).toContain('shaft of sunlight')
  })

  it('expression accents: bloom and burst', () => {
    expect(composeFromAnalysis(analysis({ smile: 0.7 })).prompt).toContain('wildflowers blooming')
    expect(composeFromAnalysis(analysis({ expression: 'surprised' })).prompt).toContain('burst of petals')
  })

  it('gesture slot priority: both hands > one hand > lean > slouch', () => {
    expect(composeFromAnalysis(analysis({ leftHand: true, rightHand: true })).prompt).toContain('white birds')
    expect(composeFromAnalysis(analysis({ rightHand: true })).prompt).toContain('butterflies')
    expect(composeFromAnalysis(analysis({ tilt: 20 })).prompt).toContain('swept diagonally')
    expect(composeFromAnalysis(analysis({ leaning: 'left' })).prompt).toContain('swept diagonally')
    expect(composeFromAnalysis(analysis({ posture: 'slouching' })).prompt).toContain('wilting flowers')
    expect(
      composeFromAnalysis(analysis({ leftHand: true, rightHand: true, tilt: 20, posture: 'slouching' })).prompt,
    ).toContain('white birds')
  })

  it('key changes only with slot changes', () => {
    const a = composeFromAnalysis(analysis({ smile: 0.3 }))
    const b = composeFromAnalysis(analysis({ smile: 0.35 }))
    const c = composeFromAnalysis(analysis({ smile: 0.8 }))
    expect(a.key).toBe(b.key)
    expect(a.key).not.toBe(c.key)
  })

  it('no face still composes from body', () => {
    const r = composeFromAnalysis(analysis({ noFace: true, rightHand: true }))
    expect(r.prompt).toContain('watercolor')
    expect(r.prompt).toContain('butterflies')
  })
})

describe('composeFromAnalysis (glitch theme)', () => {
  it('keeps the original tables', () => {
    expect(composeFromAnalysis(analysis({ smile: 0.8 }), 'glitch').prompt).toContain('golden mosaic')
    expect(composeFromAnalysis(analysis({ smile: 0.05 }), 'glitch').prompt).toContain('phosphor pixels')
    expect(composeFromAnalysis(analysis({ browRaise: 0.7 }), 'glitch').prompt).toContain('glitch tearing')
    expect(composeFromAnalysis(analysis({ attention: 'looking away' }), 'glitch').prompt).toContain('Van Gogh')
    expect(composeFromAnalysis(null, 'glitch').prompt).toContain('analog static')
  })

  it('themes produce distinct keys', () => {
    expect(composeFromAnalysis(analysis({}), 'glitch').key).not.toBe(
      composeFromAnalysis(analysis({}), 'natural').key,
    )
  })
})
