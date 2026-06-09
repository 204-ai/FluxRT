import { describe, expect, it } from 'vitest'
import { composeFromAnalysis } from './senseCompose'
import type { HumanAnalysis } from '../vision/types'

function analysis(over: {
  smile?: number
  browRaise?: number
  jawOpen?: number
  yaw?: number
  tilt?: number
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
          expression: 'neutral',
          expressionScore: 0.5,
          smile: over.smile ?? 0.3,
          jawOpen: over.jawOpen ?? 0,
          browRaise: over.browRaise ?? 0,
          eyeBlinkLeft: 0,
          eyeBlinkRight: 0,
          blinking: false,
          headPose: { yaw: over.yaw ?? 0, pitch: 0, roll: 0 },
          attention: 'engaged',
          topBlendshapes: [],
        },
    body: {
      leftHandRaised: false,
      rightHandRaised: false,
      leaning: 'centered',
      shoulderTilt: over.tilt ?? 0,
      movementEnergy: 0,
      activity: 'calm',
      posture: 'upright',
    },
  }
}

describe('composeFromAnalysis', () => {
  it('absent when nobody present or null', () => {
    expect(composeFromAnalysis(null).key).toBe('absent')
    expect(composeFromAnalysis(analysis({ present: false })).key).toBe('absent')
  })

  it('smile picks material', () => {
    expect(composeFromAnalysis(analysis({ smile: 0.8 })).key).toMatch(/^warm\|/)
    expect(composeFromAnalysis(analysis({ smile: 0.3 })).key).toMatch(/^neutral\|/)
    expect(composeFromAnalysis(analysis({ smile: 0.1 })).key).toMatch(/^cold\|/)
    expect(composeFromAnalysis(analysis({ noFace: true })).key).toMatch(/^neutral\|/)
  })

  it('arousal picks energy from max(brow, jaw)', () => {
    expect(composeFromAnalysis(analysis({ browRaise: 0.7 })).key).toContain('|high|')
    expect(composeFromAnalysis(analysis({ jawOpen: 0.4 })).key).toContain('|medium|')
    expect(composeFromAnalysis(analysis({})).key).toContain('|calm|')
  })

  it('yaw flips background, tilt adds accent', () => {
    expect(composeFromAnalysis(analysis({ yaw: 30 })).key).toContain('|away|')
    expect(composeFromAnalysis(analysis({ yaw: -30 })).key).toContain('|away|')
    expect(composeFromAnalysis(analysis({ tilt: 20 })).key).toMatch(/\|tilted$/)
    const r = composeFromAnalysis(analysis({ tilt: 20 }))
    expect(r.prompt).toContain('speed lines')
  })

  it('level accent contributes no phrase', () => {
    const r = composeFromAnalysis(analysis({}))
    expect(r.prompt.split(', ').every((p) => p.length > 0)).toBe(true)
    expect(r.prompt).not.toContain(',,')
  })
})
