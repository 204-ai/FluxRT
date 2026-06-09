// Sense → FluxRT combinatorial prompt composer. Port of the console hack
// scripts/sense_compose.js, reading typed HumanAnalysis instead of scraping
// panel text. Each slot is orthogonal — any combination stays coherent:
//
//   material   <- smile               emotional warmth picks what you're made of
//   energy     <- browRaise + jawOpen arousal picks how violently it glitches
//   background <- head yaw            looking away makes the world melt
//   accent     <- shoulderTilt        pose adds kinetic speed lines
//   presence   <- present             nobody there = room of static

import type { HumanAnalysis } from '../vision/types'

const MATERIAL = {
  warm: 'person as an ancient golden mosaic, gold leaf tesserae tiles glowing warmly, fine grout lines following the contours',
  neutral: 'person as seamless liquid chrome metal, studio reflections sliding across the surface',
  cold: 'person rendered in glowing green phosphor pixels, 1-bit monochrome, scanlines sweeping across the body',
} as const

const ENERGY = {
  calm: 'smooth calm surface',
  medium: 'mild scanline shimmer and faint RGB fringing at the edges',
  high: 'violent glitch tearing, RGB channels split and shifted, blocky compression artifacts bursting with movement',
} as const

const BACKGROUND = {
  focused: 'minimal deep black void background',
  away: 'the background melting into swirling Van Gogh starry night brushstrokes',
} as const

const ACCENT = {
  level: '',
  tilted: 'bold manga speed lines radiating diagonally with the lean of the body',
} as const

const ABSENT =
  'an empty dark room dissolving into soft analog static noise, faint phosphor afterglow where a person used to stand'

export interface ComposeResult {
  /** Slot combination id — send only when this changes. */
  key: string
  prompt: string
}

export function composeFromAnalysis(a: HumanAnalysis | null): ComposeResult {
  if (!a || !a.present) return { key: 'absent', prompt: ABSENT }

  const smile = a.face?.smile ?? null
  const arousal = Math.max(a.face?.browRaise ?? 0, a.face?.jawOpen ?? 0)
  const yaw = a.face?.headPose.yaw ?? null
  const tilt = a.body?.shoulderTilt ?? null

  const m = smile === null ? 'neutral' : smile > 0.5 ? 'warm' : smile > 0.2 ? 'neutral' : 'cold'
  const e = arousal > 0.55 ? 'high' : arousal > 0.3 ? 'medium' : 'calm'
  const b = yaw !== null && Math.abs(yaw) > 25 ? 'away' : 'focused'
  const acc = tilt !== null && Math.abs(tilt) > 12 ? 'tilted' : 'level'

  const parts = [MATERIAL[m], ENERGY[e], ACCENT[acc], BACKGROUND[b]].filter(Boolean)
  return { key: [m, e, b, acc].join('|'), prompt: parts.join(', ') }
}
