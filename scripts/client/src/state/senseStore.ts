// Sense feature: human detection + feature/behavior analysis as a panel.
// Source 'input' shares the input rail's vision worker; 'output' runs its
// own worker sampling the remote video. The overlay canvas subscribes to raw
// results imperatively (full rate); React state updates at ~10 Hz.

import { create } from 'zustand'
import { inputVision, outputVision } from './runtime'
import { composeFromAnalysis, type ComposeTheme } from '../lib/senseCompose'
import { postPromptRest } from '../lib/api'
import { usePromptStore } from './promptStore'
import { useSessionStore } from './sessionStore'
import type { HumanAnalysis, VisionResult } from '../vision/types'

export type SenseSource = 'input' | 'output'
/** How the sense visualisation appears over the input preview. */
export type SenseOverlay = 'overlay' | 'only' | 'off'

interface SenseState {
  enabled: boolean
  source: SenseSource
  overlay: SenseOverlay
  status: string
  analysis: HumanAnalysis | null

  /** Drive the FLUX prompt from sensed features (sense_compose port). */
  composeEnabled: boolean
  composeTheme: ComposeTheme
  composeMinGapSecs: number
  composeKey: string
  composePrompt: string

  setEnabled(on: boolean): Promise<void>
  setSource(source: SenseSource): Promise<void>
  setOverlay(mode: SenseOverlay): void
  setComposeEnabled(on: boolean): void
  setComposeTheme(theme: ComposeTheme): void
  setComposeMinGap(secs: number): void
}

type ResultListener = (r: VisionResult) => void
const resultListeners = new Set<ResultListener>()
/** Overlay canvases subscribe here (imperative draw, no React re-render). */
export function onSenseResult(l: ResultListener): () => void {
  resultListeners.add(l)
  return () => resultListeners.delete(l)
}

let unsubResult: (() => void) | null = null
let unsubStatus: (() => void) | null = null
let lastUiUpdate = 0

// Compose throttle: send only when the slot combination changes AND the
// cooldown has elapsed (lastKey updates only on send, matching the original
// console script — a change during cooldown fires on the next tick after it).
let lastComposeKey = ''
let lastComposeSent = 0

function maybeCompose(analysis: HumanAnalysis): void {
  const s = useSenseStore.getState()
  if (!s.composeEnabled) return
  const { key, prompt } = composeFromAnalysis(analysis, s.composeTheme)
  const now = Date.now()
  if (key === lastComposeKey || now - lastComposeSent < s.composeMinGapSecs * 1000) {
    useSenseStore.setState({ composeKey: key })
    return
  }
  lastComposeKey = key
  lastComposeSent = now
  useSenseStore.setState({ composeKey: key, composePrompt: prompt })
  // Show in the prompt box + prefer the ctrl channel (syncs all clients);
  // REST fallback drives the pipeline even with no WebRTC session.
  usePromptStore.setState({ prompt })
  const session = useSessionStore.getState()
  const sent = session.sendCtrl({ kind: 'prompt', text: prompt })
  if (!sent) {
    postPromptRest(prompt).catch((e) => session.logLine('sense-compose REST error: ' + e))
  }
  session.logLine(`sense-compose: ${key}`)
}

function handleResult(r: VisionResult): void {
  resultListeners.forEach((l) => l(r))
  const now = performance.now()
  if (now - lastUiUpdate > 100) {
    lastUiUpdate = now
    useSenseStore.setState({ analysis: r.analysis })
    maybeCompose(r.analysis)
  }
}

async function attach(source: SenseSource): Promise<void> {
  detach()
  if (source === 'input') {
    unsubResult = inputVision.subscribe(handleResult)
    unsubStatus = inputVision.onStatus((m) => useSenseStore.setState({ status: m }))
    await inputVision.acquire('sense', { face: true, pose: true })
  } else {
    unsubResult = outputVision.subscribe(handleResult)
    unsubStatus = outputVision.onStatus((m) => useSenseStore.setState({ status: m }))
    await outputVision.start()
  }
}

function detach(): void {
  unsubResult?.()
  unsubStatus?.()
  unsubResult = null
  unsubStatus = null
}

export const useSenseStore = create<SenseState>((set, get) => ({
  enabled: false,
  source: 'input',
  overlay: 'overlay',
  status: '',
  analysis: null,

  composeEnabled: false,
  composeTheme: 'natural',
  composeMinGapSecs: 5,
  composeKey: '',
  composePrompt: '',

  setOverlay(mode) {
    set({ overlay: mode })
  },

  setComposeEnabled(on) {
    set({ composeEnabled: on })
    if (!on) {
      lastComposeKey = ''
      set({ composeKey: '', composePrompt: '' })
    }
  },

  setComposeTheme(theme) {
    set({ composeTheme: theme })
    lastComposeKey = '' // theme switch should re-send immediately on next tick
  },

  setComposeMinGap(secs) {
    set({ composeMinGapSecs: Math.max(1, secs || 5) })
  },

  async setEnabled(on) {
    set({ enabled: on })
    if (on) {
      try {
        await attach(get().source)
        set({ status: '' })
      } catch (e) {
        set({ enabled: false, status: 'sense init failed: ' + (e instanceof Error ? e.message : e) })
      }
    } else {
      detach()
      void inputVision.release('sense')
      outputVision.stop()
      set({ analysis: null, status: '' })
    }
  },

  async setSource(source) {
    const wasEnabled = get().enabled
    // Release the previous source before switching.
    if (wasEnabled) {
      detach()
      if (get().source === 'input') void inputVision.release('sense')
      else outputVision.stop()
    }
    // The two sources share the module-level compose dedup keys; reset them so
    // the first compose after a switch isn't suppressed as a stale duplicate.
    lastComposeKey = ''
    set({ source, analysis: null, composeKey: '' })
    if (wasEnabled) {
      try {
        await attach(source)
      } catch (e) {
        set({ enabled: false, status: 'sense init failed: ' + (e instanceof Error ? e.message : e) })
      }
    }
  },
}))
