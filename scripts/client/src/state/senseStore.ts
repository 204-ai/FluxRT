// Sense feature: human detection + feature/behavior analysis as a panel.
// Source 'input' shares the input rail's vision worker; 'output' runs its
// own worker sampling the remote video. The overlay canvas subscribes to raw
// results imperatively (full rate); React state updates at ~10 Hz.

import { create } from 'zustand'
import { inputVision, outputVision } from './runtime'
import type { HumanAnalysis, VisionResult } from '../vision/types'

export type SenseSource = 'input' | 'output'

interface SenseState {
  enabled: boolean
  source: SenseSource
  status: string
  analysis: HumanAnalysis | null

  setEnabled(on: boolean): Promise<void>
  setSource(source: SenseSource): Promise<void>
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

function handleResult(r: VisionResult): void {
  resultListeners.forEach((l) => l(r))
  const now = performance.now()
  if (now - lastUiUpdate > 100) {
    lastUiUpdate = now
    useSenseStore.setState({ analysis: r.analysis })
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
  status: '',
  analysis: null,

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
    set({ source, analysis: null })
    if (wasEnabled) {
      try {
        await attach(source)
      } catch (e) {
        set({ enabled: false, status: 'sense init failed: ' + (e instanceof Error ? e.message : e) })
      }
    }
  },
}))
