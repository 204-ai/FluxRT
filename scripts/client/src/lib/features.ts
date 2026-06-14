// Facial-feature builder data + prompt composition logic
// (FEATURES / STYLES / applyFeatureChange).

export const FEATURE_ORDER = ['eyes', 'eyebrows', 'nose', 'ears', 'mouth'] as const
export type FeatureKey = (typeof FEATURE_ORDER)[number] | 'style'

export const FEATURES: Record<(typeof FEATURE_ORDER)[number], { emoji: string; label: string; opts: string[] }> = {
  eyes: {
    emoji: '👁️',
    label: 'eyes',
    opts: [
      'giant googly eyes',
      'glowing neon cyber eyes',
      'heterochromia, one blue eye and one green eye',
      'huge sparkling anime eyes',
      'wise wrinkled squinting eyes',
    ],
  },
  eyebrows: {
    emoji: '🤨',
    label: 'eyebrows',
    opts: [
      'huge bushy caterpillar eyebrows',
      'thin dramatically arched eyebrows',
      'thick connected unibrow',
      'glowing painted neon eyebrows',
      'completely shaved off eyebrows',
    ],
  },
  nose: {
    emoji: '👃',
    label: 'nose',
    opts: [
      'big round red clown nose',
      'long crooked witch nose',
      'tiny upturned button nose',
      'pig snout nose',
      'golden nose ring through a wide nose',
    ],
  },
  ears: {
    emoji: '👂',
    label: 'ears',
    opts: [
      'pointy elf ears',
      'enormous floppy elephant ears',
      'furry pointed wolf ears on top of the head',
      'cybernetic robot ears with antennae',
      'stretched gauged earlobes with big hoops',
    ],
  },
  mouth: {
    emoji: '👄',
    label: 'mouth',
    opts: [
      'wide gold-tooth grin',
      'huge toothy cartoon smile',
      'bushy walrus mustache over the mouth',
      'sharp vampire fangs',
      'glowing neon lips',
    ],
  },
}

export const STYLES = [
  'in the style of Studio Ghibli watercolor',
  'in the style of Stanley Kubrick, symmetrical cinematic composition',
  'in the style of cyberpunk neon noir',
  'in the style of Wes Anderson pastel symmetry',
  'in the style of Salvador Dalí surrealism',
  'in the style of 1980s synthwave',
  'in the style of Tim Burton gothic animation',
  'in the style of a Renaissance oil painting',
  'in the style of vaporwave',
  'in the style of claymation stop-motion',
]

export type FeatureState = Partial<Record<FeatureKey, string>>

/** Comma-split a prompt, drop any segment equal to `phrase`, rejoin. */
export function removePhrase(text: string, phrase: string): string {
  return text
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s && s !== phrase)
    .join(', ')
}

/**
 * Append the newly selected phrase to the prompt, replacing this slot's
 * previous phrase (if any). Leaves the rest of the prompt untouched.
 * Returns the new prompt text and feature state.
 */
export function applyFeatureChange(
  prompt: string,
  state: FeatureState,
  key: FeatureKey,
  newPhrase: string,
): { prompt: string; state: FeatureState } {
  let text = prompt.trim()
  const old = state[key] || ''
  if (old) text = removePhrase(text, old)
  if (newPhrase) text = text ? text + ', ' + newPhrase : newPhrase
  return { prompt: text, state: { ...state, [key]: newPhrase } }
}

export function randomFeaturePrompt(): { prompt: string; state: FeatureState } {
  const k = FEATURE_ORDER[Math.floor(Math.random() * FEATURE_ORDER.length)]
  const opts = FEATURES[k].opts
  const featPick = opts[Math.floor(Math.random() * opts.length)]
  const stylePick = STYLES[Math.floor(Math.random() * STYLES.length)]
  return {
    prompt: `person with ${featPick} ${stylePick}`,
    state: { [k]: featPick, style: stylePick },
  }
}

export function ratingSum(e: { style?: number; tracking?: number; stability?: number }): number {
  return (e.style || 0) + (e.tracking || 0) + (e.stability || 0)
}

export function ratingLabel(e: { style?: number; tracking?: number; stability?: number }): string {
  const f = (n?: number) => (n ? String(n) : '–')
  return `${f(e.style)}/${f(e.tracking)}/${f(e.stability)}`
}
