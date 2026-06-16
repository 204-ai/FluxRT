// Quick skip/like/love triage of the current prompt, persisted to localStorage
// (no server round-trip) and exportable as a JSON file. Distinct from the
// star scorecard in RatingOverlay, which saves to the server's /prompts list.

import { create } from 'zustand'
import { applyVerdict, parseStored, serializeRatings, STORAGE_KEY, type RatingMap, type Verdict } from '../lib/ratings'
import { usePromptStore } from './promptStore'
import { useSessionStore } from './sessionStore'

function load(): RatingMap {
  try {
    return parseStored(localStorage.getItem(STORAGE_KEY))
  } catch {
    return {} // localStorage can throw in private mode / when disabled
  }
}

function persist(map: RatingMap): void {
  try {
    localStorage.setItem(STORAGE_KEY, serializeRatings(map))
  } catch (e) {
    useSessionStore.getState().logLine('Rating save failed: ' + (e instanceof Error ? e.message : e))
  }
}

interface RatingState {
  ratings: RatingMap
  /** Rate the current prompt; re-clicking the same verdict clears it. */
  rate(verdict: Verdict): void
  /** Download all ratings as a JSON file. */
  exportJson(): void
  /** Wipe all stored ratings. */
  clearAll(): void
}

export const useRatingStore = create<RatingState>((set, get) => ({
  ratings: load(),

  rate(verdict) {
    const ps = usePromptStore.getState()
    const prompt = ps.prompt.trim()
    if (!prompt) return
    const next = applyVerdict(get().ratings, prompt, verdict, {
      ts: Date.now(),
      seed: ps.seed,
      steps: ps.steps,
    })
    persist(next)
    set({ ratings: next })
  },

  exportJson() {
    const map = get().ratings
    const count = Object.keys(map).length
    if (!count) {
      useSessionStore.getState().logLine('No ratings to export')
      return
    }
    const blob = new Blob([serializeRatings(map)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'prompt-ratings.json'
    // Firefox only fires .click() download for an anchor in the document.
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
    useSessionStore.getState().logLine(`Exported ${count} prompt rating${count === 1 ? '' : 's'}`)
  },

  clearAll() {
    persist({})
    set({ ratings: {} })
  },
}))
