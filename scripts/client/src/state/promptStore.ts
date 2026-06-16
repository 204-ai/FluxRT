// Prompt, seed/steps, feature builder, saved prompts + autoloop.

import { create } from 'zustand'
import {
  applyFeatureChange,
  randomFeaturePrompt,
  ratingLabel,
  ratingSum,
  type FeatureKey,
  type FeatureState,
} from '../lib/features'
import { deletePrompt, getSavedPrompts, savePrompt, type SavedPrompt } from '../lib/api'
import { parsePromptsFile } from '../lib/promptsFile'
import type { Verdict } from '../lib/ratings'
import { useSessionStore } from './sessionStore'
import { useRatingStore } from './ratingStore'

interface PromptState {
  prompt: string
  seed: string
  steps: string
  featureState: FeatureState
  styleSelection: string
  savedPrompts: SavedPrompt[]
  rateStyle: number
  rateTracking: number
  rateStability: number
  savedStatus: string
  loopRunning: boolean
  loopDelay: number
  morph: boolean
  morphFrames: number
  ratingFilter: Verdict | 'all'

  setPromptLocal(text: string): void
  sendPrompt(text: string): void
  setMorph(on: boolean): void
  setSeed(v: string): void
  setSteps(v: string): void
  applyFeature(key: FeatureKey, phrase: string): void
  randomize(): void
  resetFeatures(): void
  setRating(axis: 'style' | 'tracking' | 'stability', v: number): void
  loadSavedPrompts(): Promise<void>
  loadPromptsFromFile(file: File): Promise<void>
  applySaved(i: number): void
  shuffleSelect(): void
  setRatingFilter(f: Verdict | 'all'): void
  saveCurrent(): Promise<void>
  deleteCurrent(): Promise<void>
  setLoopDelay(secs: number): void
  toggleLoop(): void
}

let loopTimer = 0
let loopIdx = -1

/** Indices into savedPrompts that match the active rating filter (all when off).
 *  Reads the triage verdicts from ratingStore lazily — only at call time, so the
 *  import cycle with ratingStore never bites during module init. */
function filteredIndices(s: PromptState): number[] {
  if (s.ratingFilter === 'all') return s.savedPrompts.map((_, i) => i)
  const { ratings } = useRatingStore.getState()
  const out: number[] = []
  s.savedPrompts.forEach((e, i) => {
    if (ratings[e.prompt.trim()]?.verdict === s.ratingFilter) out.push(i)
  })
  return out
}

function loopTick(): void {
  const s = usePromptStore.getState()
  const pool = filteredIndices(s)
  if (!pool.length) return
  // Advance to the next pool entry after the current one. If the current pick
  // isn't in the pool (filter changed mid-loop), indexOf → -1 → start at pool[0].
  const pos = pool.indexOf(loopIdx)
  loopIdx = pool[(pos + 1) % pool.length]
  s.applySaved(loopIdx)
}

export const usePromptStore = create<PromptState>((set, get) => ({
  prompt: '',
  seed: '52',
  steps: '2',
  featureState: {},
  styleSelection: '',
  savedPrompts: [],
  rateStyle: 0,
  rateTracking: 0,
  rateStability: 0,
  savedStatus: '',
  loopRunning: false,
  loopDelay: 10,
  morph: false,
  morphFrames: 48,
  ratingFilter: 'all',

  setPromptLocal(text) {
    set({ prompt: text })
  },

  sendPrompt(text) {
    set({ prompt: text })
    const { morph, morphFrames } = get()
    if (morph) {
      // Slerp morphing on: smoothly travel to the new prompt instead of
      // hard-swapping. All prompt sources (editor, features, saved, autoloop)
      // funnel through here, so the toggle covers them all.
      useSessionStore
        .getState()
        .sendCtrl({ kind: 'promptTravel', text, frames: morphFrames, mode: 'slerp' })
    } else {
      useSessionStore.getState().sendCtrl({ kind: 'prompt', text })
    }
  },

  setMorph(on) {
    set({ morph: on })
  },

  setSeed(v) {
    set({ seed: v })
    // Only push a real number — an empty/invalid field shouldn't silently send a
    // coerced value that diverges from what's shown. The server validates range.
    const n = parseInt(v, 10)
    if (!Number.isNaN(n)) useSessionStore.getState().sendCtrl({ kind: 'seed', value: n })
  },

  setSteps(v) {
    set({ steps: v })
    const n = parseInt(v, 10)
    if (!Number.isNaN(n)) useSessionStore.getState().sendCtrl({ kind: 'steps', value: n })
  },

  applyFeature(key, phrase) {
    const { prompt, featureState } = get()
    const r = applyFeatureChange(prompt, featureState, key, phrase)
    set({
      featureState: r.state,
      styleSelection: key === 'style' ? phrase : get().styleSelection,
    })
    get().sendPrompt(r.prompt)
  },

  randomize() {
    const r = randomFeaturePrompt()
    set({ featureState: r.state, styleSelection: r.state.style || '' })
    get().sendPrompt(r.prompt)
  },

  resetFeatures() {
    set({ featureState: {}, styleSelection: '' })
    const text = useSessionStore.getState().serverDefaultPrompt || ''
    set({ prompt: text })
    // Reset is a hard snap back to the default, not a creative transition —
    // always instant, even when slerp morphing is enabled for normal changes.
    if (text) useSessionStore.getState().sendCtrl({ kind: 'prompt', text })
  },

  setRating(axis, v) {
    if (axis === 'style') set({ rateStyle: v })
    else if (axis === 'tracking') set({ rateTracking: v })
    else set({ rateStability: v })
  },

  async loadSavedPrompts() {
    try {
      const prompts = await getSavedPrompts()
      prompts.sort((a, b) => ratingSum(b) - ratingSum(a))
      set({ savedPrompts: prompts })
    } catch (e) {
      useSessionStore.getState().logLine('Saved prompts load error: ' + e)
    }
  },

  async loadPromptsFromFile(file) {
    let parsed: SavedPrompt[]
    try {
      parsed = parsePromptsFile(await file.text())
    } catch (e) {
      set({ savedStatus: 'file read error: ' + (e instanceof Error ? e.message : e) })
      return
    }
    if (!parsed.length) {
      set({ savedStatus: 'no prompts found in file' })
      return
    }
    // The file defines the current selection — replace the dropdown list with
    // exactly the file's prompts. Not merged with the server's saved prompts
    // and not persisted; a page reload (or save/delete, which refetch) restores
    // the server list. Re-base the autoloop so play/shuffle start from the head.
    loopIdx = -1
    set({ savedPrompts: parsed, savedStatus: `loaded ${parsed.length} from ${file.name}` })
    useSessionStore.getState().logLine(`Loaded ${parsed.length} prompts from ${file.name}`)
  },

  applySaved(i) {
    const e = get().savedPrompts[i]
    if (!e) return
    set({
      rateStyle: e.style || 0,
      rateTracking: e.tracking || 0,
      rateStability: e.stability || 0,
    })
    loopIdx = i // a manual pick re-bases the autoloop
    get().sendPrompt(e.prompt)
    useSessionStore.getState().logLine(`Saved prompt applied (${ratingLabel(e)})`)
  },

  shuffleSelect() {
    const s = get()
    const pool = filteredIndices(s)
    if (!pool.length) {
      set({ savedStatus: s.savedPrompts.length ? 'no prompts match the filter' : 'no saved prompts to shuffle' })
      return
    }
    let pick = pool[Math.floor(Math.random() * pool.length)]
    // Avoid repeating the current pick when there's more than one to choose from.
    if (pool.length > 1 && pick === loopIdx) {
      const pos = pool.indexOf(pick)
      pick = pool[(pos + 1) % pool.length]
    }
    get().applySaved(pick)
  },

  setRatingFilter(f) {
    set({ ratingFilter: f })
  },

  async saveCurrent() {
    const { prompt, rateStyle, rateTracking, rateStability } = get()
    const text = prompt.trim()
    if (!text) {
      set({ savedStatus: 'prompt box is empty' })
      return
    }
    try {
      await savePrompt({ prompt: text, style: rateStyle, tracking: rateTracking, stability: rateStability })
      set({ savedStatus: 'saved ♥' })
      void get().loadSavedPrompts()
    } catch (e) {
      set({ savedStatus: 'save failed: ' + (e instanceof Error ? e.message : e) })
    }
  },

  async deleteCurrent() {
    const text = get().prompt.trim()
    if (!text) {
      set({ savedStatus: 'prompt box is empty' })
      return
    }
    try {
      const ok = await deletePrompt(text)
      set({ savedStatus: ok ? 'deleted' : 'not in saved list' })
      if (ok) void get().loadSavedPrompts()
    } catch (e) {
      set({ savedStatus: 'delete error: ' + (e instanceof Error ? e.message : e) })
    }
  },

  setLoopDelay(secs) {
    const v = Math.max(2, secs || 10)
    set({ loopDelay: v })
    // Changing the delay while looping restarts the timer with the new period.
    if (loopTimer) {
      clearInterval(loopTimer)
      loopTimer = window.setInterval(loopTick, v * 1000)
      useSessionStore.getState().logLine(`Autoloop delay: ${v}s`)
    }
  },

  toggleLoop() {
    const s = get()
    if (loopTimer) {
      clearInterval(loopTimer)
      loopTimer = 0
      set({ loopRunning: false })
      useSessionStore.getState().logLine('Autoloop stopped')
      return
    }
    const pool = filteredIndices(s)
    if (!pool.length) {
      set({ savedStatus: s.savedPrompts.length ? 'no prompts match the filter' : 'no saved prompts to loop' })
      return
    }
    loopTimer = window.setInterval(loopTick, s.loopDelay * 1000)
    set({ loopRunning: true })
    useSessionStore
      .getState()
      .logLine(`Autoloop started: ${pool.length} prompts, every ${s.loopDelay}s`)
    loopTick() // apply the first one immediately
  },
}))
