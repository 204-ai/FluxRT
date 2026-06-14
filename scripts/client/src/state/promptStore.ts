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
import { useSessionStore } from './sessionStore'

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

  setPromptLocal(text: string): void
  sendPrompt(text: string): void
  setSeed(v: string): void
  setSteps(v: string): void
  applyFeature(key: FeatureKey, phrase: string): void
  randomize(): void
  resetFeatures(): void
  setRating(axis: 'style' | 'tracking' | 'stability', v: number): void
  loadSavedPrompts(): Promise<void>
  applySaved(i: number): void
  saveCurrent(): Promise<void>
  deleteCurrent(): Promise<void>
  setLoopDelay(secs: number): void
  toggleLoop(): void
}

let loopTimer = 0
let loopIdx = -1

function loopTick(): void {
  const s = usePromptStore.getState()
  if (!s.savedPrompts.length) return
  loopIdx = (loopIdx + 1) % s.savedPrompts.length
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
  loopDelay: 20,

  setPromptLocal(text) {
    set({ prompt: text })
  },

  sendPrompt(text) {
    set({ prompt: text })
    useSessionStore.getState().sendCtrl({ kind: 'prompt', text })
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
    const v = Math.max(2, secs || 20)
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
    if (!s.savedPrompts.length) {
      set({ savedStatus: 'no saved prompts to loop' })
      return
    }
    loopTimer = window.setInterval(loopTick, s.loopDelay * 1000)
    set({ loopRunning: true })
    useSessionStore
      .getState()
      .logLine(`Autoloop started: ${s.savedPrompts.length} prompts, every ${s.loopDelay}s`)
    loopTick() // apply the first one immediately
  },
}))
