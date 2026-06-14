// Reference image + ComfyUI integration.

import { create } from 'zustand'
import {
  clearReference,
  comfyEdit,
  comfyPull,
  getComfyServers,
  referenceImageUrl,
  uploadReference,
  type ComfyServer,
} from '../lib/api'
import { rail } from './runtime'
import { useSessionStore } from './sessionStore'
import { usePromptStore } from './promptStore'

interface ReferenceState {
  enabled: boolean
  version: number
  previewUrl: string
  previewShown: boolean
  meta: string
  dropHint: string

  comfyServers: ComfyServer[]
  comfyServer: string
  comfyPrompt: string
  comfyStatus: string
  comfyBusy: boolean

  setDisabled(): void
  syncVersion(version: number): void
  remoteSet(version: number): void
  remoteClear(version: number): void
  upload(file: File | Blob): Promise<void>
  clear(): Promise<void>
  loadComfyServers(): Promise<void>
  setComfyServer(name: string): void
  setComfyPrompt(text: string): void
  doComfyPull(): Promise<void>
  doComfyEdit(): Promise<void>
}

// Revoke any previous blob: URL before replacing it — object URLs pin their
// blob in memory until revoked, so repeated uploads otherwise leak.
function swapPreviewUrl(prev: string, next: string): string {
  if (prev.startsWith('blob:')) URL.revokeObjectURL(prev)
  return next
}

const DEFAULT_DROP_HINT = 'Drop / click reference'

export const useReferenceStore = create<ReferenceState>((set, get) => ({
  enabled: true,
  version: 0,
  previewUrl: '',
  previewShown: false,
  meta: 'no reference',
  dropHint: DEFAULT_DROP_HINT,

  comfyServers: [],
  comfyServer: '',
  comfyPrompt: '',
  comfyStatus: '',
  comfyBusy: false,

  setDisabled() {
    set({
      enabled: false,
      meta: 'disabled in config',
      dropHint:
        'Reference disabled — start server with --config configs/config_with_reference.json',
    })
  },

  syncVersion(version) {
    set({
      version,
      previewUrl: swapPreviewUrl(get().previewUrl, referenceImageUrl()),
      previewShown: true,
      meta: version ? `reference v${version}` : 'reference active',
    })
  },

  remoteSet(version) {
    if (version <= get().version) return
    get().syncVersion(version)
    useSessionStore.getState().logLine(`Reference updated by another client (v${version})`)
  },

  remoteClear(version) {
    if (version <= get().version) return
    set({
      version,
      previewUrl: swapPreviewUrl(get().previewUrl, ''),
      previewShown: false,
      meta: 'no reference',
    })
    useSessionStore.getState().logLine(`Reference cleared by another client (v${version})`)
  },

  async upload(file) {
    const log = useSessionStore.getState().logLine
    if (!file.type.startsWith('image/')) {
      log('Not an image file')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      log('File too large (>10 MB)')
      return
    }
    set({ meta: 'uploading...' })
    try {
      const j = await uploadReference(file)
      log(`Reference set: ${j.size[0]}x${j.size[1]} (v${j.version})`)
      set({
        version: j.version || get().version,
        meta: `active ${j.size[0]}x${j.size[1]} (v${j.version})`,
        previewUrl: swapPreviewUrl(get().previewUrl, URL.createObjectURL(file)),
        previewShown: true,
      })
    } catch (e) {
      log('Reference upload failed: ' + (e instanceof Error ? e.message : e))
      set({ meta: 'upload failed' })
    }
  },

  async clear() {
    const log = useSessionStore.getState().logLine
    try {
      const j = await clearReference()
      set({
        version: j.version || get().version,
        previewUrl: swapPreviewUrl(get().previewUrl, ''),
        previewShown: false,
        meta: 'no reference',
      })
      log('Reference cleared' + (j.version ? ` (v${j.version})` : ''))
    } catch (e) {
      log('Clear error: ' + e)
    }
  },

  async loadComfyServers() {
    try {
      const servers = await getComfyServers()
      set({ comfyServers: servers, comfyServer: servers[0]?.name ?? '' })
    } catch (e) {
      useSessionStore.getState().logLine('Comfy server list error: ' + e)
    }
  },

  setComfyServer(name) {
    set({ comfyServer: name })
  },

  setComfyPrompt(text) {
    set({ comfyPrompt: text })
  },

  async doComfyPull() {
    const { comfyServer } = get()
    const log = useSessionStore.getState().logLine
    if (!comfyServer) return
    set({ comfyBusy: true, comfyStatus: `pulling from ${comfyServer}...` })
    try {
      const j = await comfyPull(comfyServer)
      set({ comfyStatus: `pulled ${j.filename} (v${j.version})` })
      get().syncVersion(j.version || get().version)
      log(`Comfy pulled: ${j.filename} from ${comfyServer} (v${j.version})`)
      // Drive the live pipeline to match the freshly pulled reference.
      usePromptStore.getState().sendPrompt('make this person look like the reference image')
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      set({ comfyStatus: 'error: ' + msg })
      log('Comfy pull failed: ' + msg)
    } finally {
      set({ comfyBusy: false })
    }
  },

  async doComfyEdit() {
    const { comfyServer, comfyPrompt } = get()
    const log = useSessionStore.getState().logLine
    if (!comfyServer) {
      set({ comfyStatus: 'pick a comfy server first' })
      return
    }
    if (!rail.active) {
      set({ comfyStatus: 'enable your camera (Input tab) first' })
      return
    }
    set({ comfyBusy: true, comfyStatus: `snapping → Qwen edit on ${comfyServer}...` })
    try {
      const blob = await rail.snapshot('image/png')
      // Dedicated Qwen-edit prompt; falls back to the live pipeline prompt.
      const prompt = comfyPrompt.trim() || usePromptStore.getState().prompt.trim()
      const j = await comfyEdit(comfyServer, prompt, blob)
      set({ comfyStatus: `qwen edit → reference (v${j.version})` })
      get().syncVersion(j.version || get().version)
      log(`Qwen edit done: ${j.filename} (v${j.version})`)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      set({ comfyStatus: 'edit error: ' + msg })
      log('Qwen edit error: ' + msg)
    } finally {
      set({ comfyBusy: false })
    }
  },
}))
