// Typed fetch wrappers for every FastAPI endpoint the client uses.
// Server contract: scripts/run_webrtc.py.
// Every URL goes through api() so a configured remote backend (serverBase) is
// honored — empty base = same origin (dev proxy / bundled client).

import { api } from './serverBase'

export interface Healthz {
  peers: number
  resolution?: { width: number; height: number }
  prompt?: string
  seed?: number
  steps?: number
  input_source?: 'peer' | 'server' | 'none'
  reference_enabled?: boolean
  reference_set?: boolean
  reference_version?: number
  lip_enabled?: boolean
  lip_active?: boolean
  fps_pipeline?: number
  fps_interpolated?: number
  proc_time_ms?: number
  vram_mb?: number
}

export interface SavedPrompt {
  prompt: string
  style: number
  tracking: number
  stability: number
}

export interface ComfyServer {
  name: string
  url: string
}

async function jsonOrDetail(r: Response): Promise<never> {
  const err = await r.json().catch(() => ({ detail: r.statusText }))
  throw new Error(err.detail || r.statusText)
}

export async function postOffer(desc: RTCSessionDescription): Promise<RTCSessionDescriptionInit> {
  const r = await fetch(api('/offer'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sdp: desc.sdp, type: desc.type }),
  })
  if (!r.ok) throw new Error('offer rejected')
  return r.json()
}

export async function getHealthz(): Promise<Healthz> {
  const r = await fetch(api('/healthz'), { cache: 'no-store' })
  if (!r.ok) throw new Error('healthz ' + r.status)
  return r.json()
}

export async function getSavedPrompts(): Promise<SavedPrompt[]> {
  const r = await fetch(api('/prompts'))
  const j = await r.json()
  return j.prompts || []
}

export async function savePrompt(entry: SavedPrompt): Promise<void> {
  const r = await fetch(api('/prompts'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entry),
  })
  if (!r.ok) await jsonOrDetail(r)
}

export async function deletePrompt(prompt: string): Promise<boolean> {
  const r = await fetch(api('/prompts?prompt=' + encodeURIComponent(prompt)), { method: 'DELETE' })
  return r.ok
}

export async function uploadReference(file: Blob): Promise<{ version: number; size: [number, number] }> {
  const r = await fetch(api('/reference'), {
    method: 'POST',
    headers: { 'Content-Type': file.type || 'application/octet-stream' },
    body: file,
  })
  if (!r.ok) await jsonOrDetail(r)
  return r.json()
}

export async function clearReference(): Promise<{ version?: number }> {
  const r = await fetch(api('/reference'), { method: 'DELETE' })
  if (!r.ok) return {}
  return r.json().catch(() => ({}))
}

export function referenceImageUrl(): string {
  return api('/reference?t=' + Date.now())
}

export async function getComfyServers(): Promise<ComfyServer[]> {
  const r = await fetch(api('/comfy/servers'))
  const j = await r.json()
  return j.servers || []
}

export async function comfyPull(server: string): Promise<{ version: number; filename: string }> {
  const r = await fetch(api('/comfy/pull?server=' + encodeURIComponent(server)), { method: 'POST' })
  if (!r.ok) await jsonOrDetail(r)
  return r.json()
}

export async function comfyEdit(
  server: string,
  prompt: string,
  png: Blob,
): Promise<{ version: number; filename: string }> {
  const r = await fetch(
    api('/comfy/edit?server=' + encodeURIComponent(server) + '&prompt=' + encodeURIComponent(prompt)),
    { method: 'POST', headers: { 'Content-Type': 'image/png' }, body: png },
  )
  if (!r.ok) await jsonOrDetail(r)
  return r.json()
}

/** REST prompt fallback — works without an open ctrl DataChannel. */
export async function postPromptRest(text: string): Promise<void> {
  const r = await fetch(api('/prompt'), {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: text,
  })
  if (!r.ok) await jsonOrDetail(r)
}

export async function setLipTransfer(on: boolean): Promise<{ lip_active: boolean }> {
  const r = await fetch(api('/lip-transfer?on=' + (on ? 'true' : 'false')), { method: 'POST' })
  if (!r.ok) await jsonOrDetail(r)
  return r.json()
}
