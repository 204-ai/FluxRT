// Sense → FluxRT combinatorial prompt composer.
//
// Reads typed HumanAnalysis and composes a prompt from orthogonal slots —
// each slot owns ONE visual dimension, so any combination stays coherent.
// A prompt is (re)sent only when the slot-combination key changes.
//
// Emotion mapping follows the valence/arousal/attention circumplex:
//   valence   <- smile + expression      what the person is made of / palette
//   arousal   <- max(browRaise, jawOpen) how much the atmosphere moves
//               (+ body activity)
//   attention <- head yaw/pitch          where the light comes from
// Gesture mapping:
//   hands raised (one/both), lean/tilt, slouch, presence

import type { HumanAnalysis } from '../vision/types'

export type ComposeTheme = 'natural' | 'glitch'

export interface ComposeResult {
  /** Slot combination id — send only when this changes. */
  key: string
  prompt: string
}

interface Signals {
  valence: 'joyful' | 'serene' | 'somber'
  arousal: 'still' | 'breezy' | 'stormy'
  gaze: 'engaged' | 'away' | 'upward'
  accent: 'bloom' | 'burst' | 'none'
  gesture: 'bothHands' | 'oneHand' | 'leaning' | 'slouching' | 'none'
}

function extractSignals(a: HumanAnalysis): Signals {
  const f = a.face
  const b = a.body

  const smile = f?.smile ?? null
  const expr = f?.expression
  const valence: Signals['valence'] =
    expr === 'happy' || (smile !== null && smile > 0.5)
      ? 'joyful'
      : expr === 'frowning' || (smile !== null && smile < 0.12)
        ? 'somber'
        : 'serene'

  const faceArousal = Math.max(f?.browRaise ?? 0, f?.jawOpen ?? 0)
  const bodyActive = b?.activity === 'active' || b?.activity === 'very active'
  const arousal: Signals['arousal'] =
    faceArousal > 0.55 || b?.activity === 'very active'
      ? 'stormy'
      : faceArousal > 0.3 || bodyActive
        ? 'breezy'
        : 'still'

  const pitch = f?.headPose.pitch ?? 0
  const gaze: Signals['gaze'] =
    pitch > 15 ? 'upward' : f?.attention === 'looking away' ? 'away' : 'engaged'

  const accent: Signals['accent'] =
    expr === 'surprised' ? 'burst' : valence === 'joyful' && (smile ?? 0) > 0.65 ? 'bloom' : 'none'

  const tilted = b !== null && (b.leaning !== 'centered' || Math.abs(b.shoulderTilt) > 12)
  const gesture: Signals['gesture'] =
    b?.leftHandRaised && b?.rightHandRaised
      ? 'bothHands'
      : b?.leftHandRaised || b?.rightHandRaised
        ? 'oneHand'
        : tilted
          ? 'leaning'
          : b?.posture === 'slouching'
            ? 'slouching'
            : 'none'

  return { valence, arousal, gaze, accent, gesture }
}

interface ThemeTables {
  absent: string
  medium: Record<Signals['valence'], string>
  weather: Record<Signals['arousal'], string>
  light: Record<Signals['gaze'], string>
  accent: Record<Signals['accent'], string>
  gesture: Record<Signals['gesture'], string>
}

// ── natural theme: painterly media, weather, light, flora, fauna ──────────

const NATURAL: ThemeTables = {
  absent:
    'an empty forest clearing at dawn, soft fog drifting between the trees, no one there, quiet pale morning light',
  medium: {
    joyful:
      'person painted in warm impasto oils, golden sunlit tones, a rosy glow on the skin',
    serene:
      'person painted in soft watercolor washes, gentle earth tones bleeding into the paper',
    somber:
      'person painted in muted ink wash, rain-blue and grey tones, edges dissolving like wet paper',
  },
  weather: {
    still: 'still misty air, dust motes floating in the light',
    breezy: 'a gentle breeze stirring leaves and loose strands of hair',
    stormy:
      'storm wind sweeping through the scene, swirling leaves and petals, dramatic rolling clouds behind',
  },
  light: {
    engaged: 'soft golden window light falling on the face',
    away: 'gazing into the distance, long amber evening shadows stretching across',
    upward: 'a shaft of sunlight breaking through from above, dust glittering in the beam',
  },
  accent: {
    bloom: 'wildflowers blooming across the shoulders and woven into the hair',
    burst: 'a sudden burst of petals scattering around the head',
    none: '',
  },
  gesture: {
    bothHands: 'white birds taking flight around the raised arms',
    oneHand: 'butterflies gathering around the raised hand',
    leaning: 'the whole composition swept diagonally by wind, hair and fabric streaming with the lean',
    slouching: 'wilting flowers and heavy drooping branches framing the figure',
    none: '',
  },
}

// ── glitch theme: the original sense_compose.js tables ────────────────────

const GLITCH: ThemeTables = {
  absent:
    'an empty dark room dissolving into soft analog static noise, faint phosphor afterglow where a person used to stand',
  medium: {
    joyful:
      'person as an ancient golden mosaic, gold leaf tesserae tiles glowing warmly, fine grout lines following the contours',
    serene: 'person as seamless liquid chrome metal, studio reflections sliding across the surface',
    somber:
      'person rendered in glowing green phosphor pixels, 1-bit monochrome, scanlines sweeping across the body',
  },
  weather: {
    still: 'smooth calm surface',
    breezy: 'mild scanline shimmer and faint RGB fringing at the edges',
    stormy:
      'violent glitch tearing, RGB channels split and shifted, blocky compression artifacts bursting with movement',
  },
  light: {
    engaged: 'minimal deep black void background',
    away: 'the background melting into swirling Van Gogh starry night brushstrokes',
    upward: 'the background melting into swirling Van Gogh starry night brushstrokes',
  },
  accent: { bloom: '', burst: '', none: '' },
  gesture: {
    bothHands: 'twin pillars of light rising from the lifted hands',
    oneHand: 'sparks of corrupted pixels trailing from the raised hand',
    leaning: 'bold manga speed lines radiating diagonally with the lean of the body',
    slouching: '',
    none: '',
  },
}

const THEMES: Record<ComposeTheme, ThemeTables> = {
  natural: NATURAL,
  glitch: GLITCH,
}

export function composeFromAnalysis(
  a: HumanAnalysis | null,
  theme: ComposeTheme = 'natural',
): ComposeResult {
  const t = THEMES[theme]
  if (!a || !a.present) return { key: theme + '|absent', prompt: t.absent }

  const s = extractSignals(a)
  const parts = [
    t.medium[s.valence],
    t.weather[s.arousal],
    t.accent[s.accent],
    t.gesture[s.gesture],
    t.light[s.gaze],
  ].filter(Boolean)

  return {
    key: [theme, s.valence, s.arousal, s.gaze, s.accent, s.gesture].join('|'),
    prompt: parts.join(', '),
  }
}
