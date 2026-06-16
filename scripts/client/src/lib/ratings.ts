// Quick prompt triage: skip / like / love, keyed by prompt text. Pure data
// layer — no DOM, no localStorage — so it tests under the node vitest env.
// The store (state/ratingStore.ts) wraps these with localStorage IO + zustand.

export type Verdict = 'skip' | 'like' | 'love'

export interface PromptRating {
  prompt: string
  verdict: Verdict
  ts: number // epoch ms when last rated
  seed?: string
  steps?: string
}

/** One rating per unique prompt; the latest verdict wins. */
export type RatingMap = Record<string, PromptRating>

export const STORAGE_KEY = 'fluxrt.ratings.v1'

/**
 * Set (or toggle off) the verdict for `prompt`. Re-applying the same verdict
 * clears it, so a mis-tap is one click to undo. Returns a NEW map (callers
 * persist + set state from it); an empty/blank prompt is a no-op.
 */
export function applyVerdict(
  map: RatingMap,
  prompt: string,
  verdict: Verdict,
  meta: { ts: number; seed?: string; steps?: string },
): RatingMap {
  const key = prompt.trim()
  if (!key) return map
  const next = { ...map }
  if (next[key]?.verdict === verdict) {
    delete next[key]
  } else {
    next[key] = { prompt: key, verdict, ts: meta.ts, seed: meta.seed, steps: meta.steps }
  }
  return next
}

/** Serialize for export / storage: a flat array, newest rating first. */
export function serializeRatings(map: RatingMap): string {
  const rows = Object.values(map).sort((a, b) => b.ts - a.ts)
  return JSON.stringify(rows, null, 2)
}

/**
 * Tolerant parse of a stored blob back into a map. Accepts either the array
 * export shape or a bare object map; drops anything malformed instead of
 * throwing, so a corrupt key can never wedge the UI.
 */
export function parseStored(raw: string | null): RatingMap {
  if (!raw) return {}
  let data: unknown
  try {
    data = JSON.parse(raw)
  } catch {
    return {}
  }
  const rows = Array.isArray(data) ? data : data && typeof data === 'object' ? Object.values(data) : []
  const out: RatingMap = {}
  for (const r of rows) {
    if (!r || typeof r !== 'object') continue
    const { prompt, verdict, ts, seed, steps } = r as Record<string, unknown>
    if (typeof prompt !== 'string' || !prompt.trim()) continue
    if (verdict !== 'skip' && verdict !== 'like' && verdict !== 'love') continue
    out[prompt.trim()] = {
      prompt: prompt.trim(),
      verdict,
      ts: typeof ts === 'number' ? ts : 0,
      seed: typeof seed === 'string' ? seed : undefined,
      steps: typeof steps === 'string' ? steps : undefined,
    }
  }
  return out
}
