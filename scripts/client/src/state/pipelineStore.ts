// Input pipeline state — generic Resolume-style grid. The composition is an
// ordered list of LAYERS (rows); each layer holds CELLS (columns), one active
// at a time; a cell holds a CLIP of some kind (camera | video | feedback |
// screen | …). The store owns every clip's live source (camera clone pool,
// per-clip video <video>, screen capture, the remote feedback stream) and a
// single reconciler (`syncPipeline`) keeps the rail matching the desired state —
// starting on the first frame-producing source, hot-attaching the rest, and
// re-selecting the base (or stopping) as clips come and go, all without dropping
// the WebRTC output track. Draw + marker stay as global overlays for now.

import { create } from 'zustand'
import {
  acquireCamera,
  acquireVideoSource,
  inputVision,
  rail,
  releaseCamera,
  releaseVideoSource,
} from './runtime'
import { getHealthz } from '../lib/api'
import type { BaseSource, BlendMode, ClipId, ClipKind, Composite, LayerId, LayerTransform } from '../pipeline/core/types'
import { clipMeta, isEffectKind } from '../pipeline/core/clipKinds'

/** Default config for a fresh effect clip of a kind. */
function defaultEffectConfig(kind: ClipKind): Record<string, unknown> {
  if (kind === 'shader') return { filter: 'hue-rotate(90deg)' }
  return {}
}
import type { Clip, Layer } from './layerModel'
import {
  activeClip,
  findClip,
  layerById,
  makeClip,
  newEmptyLayer,
} from './layerModel'
import { loadComposition, saveComposition } from './persistence'

export type DrawMode = 'off' | 'brush' | 'eraser'

interface CameraDevice {
  deviceId: string
  label: string
}

/** Per-video-clip transport state, keyed by clip id. */
interface VideoClipState {
  name: string
  meta: string
  duration: number
  currentTime: number
  playing: boolean
  loop: boolean
  rate: number
}

interface PipelineState {
  // composition grid
  layers: Layer[]
  selectedClipId: ClipId | null
  active: boolean
  // Bumped on every rail (re)start — the preview element gets a fresh identity
  // when a new backend is built, so the host re-parents it on epoch change even
  // when `active` stays true (a hot restart that doesn't toggle active).
  previewEpoch: number

  devices: CameraDevice[]
  // per-video-clip transport, keyed by clip id
  videoState: Record<ClipId, VideoClipState>
  // a remote output stream is connected → feedback clips can go live
  feedbackAvailable: boolean

  drawMode: DrawMode
  drawColor: string
  drawSize: number

  // OBS-style framing — `layoutLayer` is the layer being framed (null = off).
  layoutLayer: LayerId | null
  cropMode: boolean

  markerEnabled: boolean
  markerLandmark: number
  markerColor: string
  markerSize: number
  markerTrail: boolean
  markerTrailLen: number
  poseStatus: string

  log: (msg: string) => void
  setLogger(fn: (msg: string) => void): void
  refreshCameras(): Promise<void>

  // --- grid structure ---
  addLayer(): void
  /** Tear down every clip + stop the pipeline; reset to one empty layer. */
  clearComposition(): void
  removeLayer(layerId: LayerId): Promise<void>
  moveLayer(layerId: LayerId, dir: -1 | 1): void
  addCell(layerId: LayerId): void
  removeClip(layerId: LayerId, cellId: string): Promise<void>
  activateCell(layerId: LayerId, cellId: string): Promise<void>
  /** Drag a clip from one cell to another (same layer = reorder/swap; across
   *  layers = move, swapping with whatever occupies the target). */
  moveClip(fromLayerId: LayerId, fromCellId: string, toLayerId: LayerId, toCellId: string): Promise<void>
  selectClip(clipId: ClipId | null): void

  // --- fill an empty cell with a source of a kind ---
  fillCellCamera(layerId: LayerId, cellId: string, deviceId: string): Promise<void>
  fillCellVideo(layerId: LayerId, cellId: string, file: File): Promise<void>
  fillCellScreen(layerId: LayerId, cellId: string): Promise<void>
  fillCellFeedback(layerId: LayerId, cellId: string): Promise<void>
  fillCellEffect(layerId: LayerId, cellId: string, kind: ClipKind): Promise<void>
  setEffectConfig(clipId: ClipId, patch: Record<string, unknown>): void

  // --- per-clip / per-layer controls ---
  setClipDevice(clipId: ClipId, deviceId: string): Promise<void>
  setClipMirror(clipId: ClipId, on: boolean): void
  setLayerOpacity(layerId: LayerId, opacity: number): void
  setLayerBlend(layerId: LayerId, blend: BlendMode): void

  toggleVideoPlay(clipId: ClipId): void
  seekVideo(clipId: ClipId, t: number): void
  setVideoLoop(clipId: ClipId, on: boolean): void
  setVideoRate(clipId: ClipId, r: number): void

  setLayoutLayer(id: LayerId | null): void
  setCropMode(on: boolean): void
  setLayerTransform(id: LayerId, transform: LayerTransform | null): void

  /** Wire the remote output stream in/out (feedback clips read it). */
  attachFeedback(stream: MediaStream | null): void

  setDrawMode(mode: DrawMode): void
  setDrawColor(c: string): void
  setDrawSize(n: number): void
  clearDrawing(): void

  setMarkerEnabled(on: boolean): Promise<void>
  setMarkerLandmark(n: number): void
  setMarkerColor(c: string): void
  setMarkerSize(n: number): void
  setMarkerTrail(on: boolean): void
  setMarkerTrailLen(n: number): void
}

function fmtTime(s: number): string {
  if (!isFinite(s)) return '–:––'
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

export { fmtTime }

// --- module-level runtime (live handles, never in renderable state) ---
const clipStreams = new Map<ClipId, MediaStream>() // camera / screen streams per clip
const videoDetachers = new Map<ClipId, () => void>() // per-video-clip <video> listeners
let feedbackStream: MediaStream | null = null
// The server's output resolution (from /healthz) — the input composite canvas is
// pinned to it so the preview matches the output's aspect AND resolution exactly.
let targetResolution: { width: number; height: number } | null = null

// Reconciler memo: the base layer driving cadence, and what source each layer is
// currently bound to (so we only re-bind on change — re-binding glitches).
let currentBaseLayerId: LayerId | null = null
const boundSource = new Map<LayerId, MediaStream | HTMLVideoElement>()

export const usePipelineStore = create<PipelineState>((set, get) => {
  /** Live source for a clip (or null if not acquired / loaded). */
  function liveSourceFor(clip: Clip): MediaStream | HTMLVideoElement | null {
    if (clip.kind === 'video') {
      const s = acquireVideoSource(clip.id)
      return s.loaded ? s.el : null
    }
    if (clip.kind === 'camera' || clip.kind === 'screen') return clipStreams.get(clip.id) ?? null
    if (clip.kind === 'feedback') return feedbackStream
    return null
  }

  /** Build the render composite (all layers, incl. effect layers) from the grid. */
  function compositeFrom(layers: Layer[]): Composite {
    return layers.map((l) => {
      const clip = activeClip(l)
      const render = {
        id: l.id,
        opacity: l.opacity,
        blend: l.blend,
        transform: clip?.transform ?? l.transform,
        mirror: clip?.mirror ?? false,
      }
      if (clip && isEffectKind(clip.kind)) {
        return { ...render, effectName: clipMeta(clip.kind).effectName, effectConfig: clip.effectConfig }
      }
      return render
    })
  }

  function baseSourceFor(layerId: LayerId): BaseSource | null {
    const layer = layerById(get().layers, layerId)
    const clip = layer ? activeClip(layer) : null
    if (!clip) return null
    const src = liveSourceFor(clip)
    if (!src) return null
    return {
      layerId,
      kind: clip.kind,
      stream: src instanceof MediaStream ? src : null,
      videoEl: src instanceof HTMLVideoElement ? src : null,
    }
  }

  // syncPipeline is async (it awaits rail.start) and fires from many triggers
  // (add/activate/remove a clip, the healthz re-pin, feedback). Serialize it so
  // concurrent invocations can't race rail.start/stop; coalesce overlapping calls
  // into one trailing re-run.
  let syncing = false
  let syncQueued = false
  async function syncPipeline(): Promise<void> {
    if (syncing) {
      syncQueued = true
      return
    }
    syncing = true
    try {
      do {
        syncQueued = false
        await syncPipelineImpl()
      } while (syncQueued)
    } finally {
      syncing = false
    }
  }

  /** The one place that drives the rail. Computes the desired source bindings
   *  from the grid and reconciles: (re)start on the first frame-producing source
   *  (base), hot-attach/detach the rest, replace the composite. Never drops the
   *  WebRTC output track unless the base layer itself changes. */
  async function syncPipelineImpl(): Promise<void> {
    const layers = get().layers
    const bindings: { layerId: LayerId; kind: string; source: MediaStream | HTMLVideoElement }[] = []
    for (const layer of layers) {
      const clip = activeClip(layer)
      if (!clip || isEffectKind(clip.kind)) continue
      const source = liveSourceFor(clip)
      if (source) bindings.push({ layerId: layer.id, kind: clip.kind, source })
    }

    if (!bindings.length) {
      // No frame source → nothing to drive cadence. (Effect/feedback-only ticker
      // comps arrive with image clips in G4.) Stop.
      if (rail.active) {
        rail.stop()
        boundSource.clear()
        currentBaseLayerId = null
        set({ active: false, drawMode: 'off', layoutLayer: null })
      }
      return
    }

    // Prefer a cadence-capable base (camera/video/screen); else any source.
    const base = bindings.find((b) => clipMeta(b.kind).canBeBase) ?? bindings[0]

    if (!rail.active || currentBaseLayerId !== base.layerId) {
      // (Re)start on the new base. rail.start passes the current composite, so set
      // it first; then re-bind every non-base source.
      rail.setCompositeAll(compositeFrom(layers))
      const baseSource = baseSourceFor(base.layerId)
      if (!baseSource) return
      try {
        await rail.start(baseSource, targetResolution)
      } catch (e) {
        get().log('Pipeline start failed: ' + (e instanceof Error ? e.message : e))
        return
      }
      currentBaseLayerId = base.layerId
      boundSource.clear()
      boundSource.set(base.layerId, base.source)
      // Bump the epoch: the new backend has a fresh preview element to re-parent.
      set({ active: true, previewEpoch: get().previewEpoch + 1 })
      for (const b of bindings) {
        if (b.layerId === base.layerId) continue
        rail.setLayerSource(b.layerId, b.kind, b.source)
        boundSource.set(b.layerId, b.source)
      }
      return
    }

    // Running, same base → reconcile composite + the source bindings by diffing.
    rail.setCompositeAll(compositeFrom(layers))
    for (const b of bindings) {
      if (boundSource.get(b.layerId) !== b.source) {
        rail.setLayerSource(b.layerId, b.kind, b.source)
        boundSource.set(b.layerId, b.source)
      }
    }
    for (const layerId of [...boundSource.keys()]) {
      if (layerId === currentBaseLayerId) continue
      if (!bindings.some((b) => b.layerId === layerId)) {
        rail.setLayerSource(layerId, 'video', null)
        boundSource.delete(layerId)
      }
    }
  }

  /** Release a clip's live source (camera clone / screen / pooled video / etc.). */
  function releaseClip(clip: Clip): void {
    if (clip.kind === 'camera') {
      releaseCamera(clip.id)
      clipStreams.delete(clip.id)
    } else if (clip.kind === 'screen') {
      clipStreams.get(clip.id)?.getTracks().forEach((t) => {
        try {
          t.stop()
        } catch {
          /* already stopped */
        }
      })
      clipStreams.delete(clip.id)
    } else if (clip.kind === 'video') {
      videoDetachers.get(clip.id)?.()
      videoDetachers.delete(clip.id)
      releaseVideoSource(clip.id)
    }
  }

  /** Ensure a layer's activeCellId points at a clip-bearing cell (or null) and
   *  its kind/role reflect the active clip. `preferCellId` is made active when it
   *  holds a clip (used when a dragged clip lands in it). */
  function normalizeLayer(l: Layer, preferCellId?: string): Layer {
    let activeCellId =
      preferCellId && l.cells.some((c) => c.id === preferCellId && c.clip) ? preferCellId : l.activeCellId
    if (!l.cells.find((c) => c.id === activeCellId)?.clip) activeCellId = l.cells.find((c) => c.clip)?.id ?? null
    const active = l.cells.find((c) => c.id === activeCellId)?.clip ?? null
    return { ...l, activeCellId, kind: active?.kind ?? null, role: active ? clipMeta(active.kind).role : null }
  }

  /** Put a clip into a cell + make it the layer's active clip + select it. */
  function placeClip(layerId: LayerId, cellId: string, clip: Clip): void {
    set((s) => ({
      layers: s.layers.map((l) =>
        l.id !== layerId
          ? l
          : {
              ...l,
              kind: l.kind ?? clip.kind,
              role: l.role ?? clipMeta(clip.kind).role,
              activeCellId: cellId,
              cells: l.cells.map((c) => (c.id === cellId ? { ...c, clip } : c)),
            },
      ),
      selectedClipId: clip.id,
    }))
  }

  function attachVideoListeners(clipId: ClipId, el: HTMLVideoElement): void {
    const patch = (p: Partial<VideoClipState>) =>
      set((s) => ({ videoState: { ...s.videoState, [clipId]: { ...s.videoState[clipId], ...p } } }))
    const onTime = () => patch({ currentTime: el.currentTime })
    const onDur = () => patch({ duration: el.duration })
    const onPlay = () => patch({ playing: true })
    const onPause = () => patch({ playing: false })
    el.addEventListener('timeupdate', onTime)
    el.addEventListener('durationchange', onDur)
    el.addEventListener('play', onPlay)
    el.addEventListener('pause', onPause)
    videoDetachers.set(clipId, () => {
      el.removeEventListener('timeupdate', onTime)
      el.removeEventListener('durationchange', onDur)
      el.removeEventListener('play', onPlay)
      el.removeEventListener('pause', onPause)
    })
  }

  // Restore a saved composition (structure only; sources re-acquire on activate)
  // or start with one empty layer.
  const initialLayers = loadComposition() ?? [newEmptyLayer('Layer 1')]

  // Fetch the server's output resolution early (fire-and-forget; never blocks —
  // the /healthz proxy hangs with no server) so the first start pins to it; if a
  // clip is added before it resolves, re-pin the running pipeline once.
  void getHealthz()
    .then((h) => {
      if (!h.resolution || h.resolution.height <= 0 || h.resolution.width <= 0) return
      targetResolution = { width: h.resolution.width, height: h.resolution.height }
      if (rail.active) {
        currentBaseLayerId = null
        void syncPipeline()
      }
    })
    .catch(() => {})

  return {
    layers: initialLayers,
    selectedClipId: null,
    active: false,
    previewEpoch: 0,
    devices: [],
    videoState: {},
    feedbackAvailable: false,

    drawMode: 'off',
    drawColor: '#ffffff',
    drawSize: 6,

    layoutLayer: null,
    cropMode: false,

    markerEnabled: false,
    markerLandmark: 15,
    markerColor: '#ff3c3c',
    markerSize: 32,
    markerTrail: false,
    markerTrailLen: 20,
    poseStatus: '',

    log: () => {},
    setLogger(fn) {
      set({ log: fn })
    },

    async refreshCameras() {
      try {
        const devs = await navigator.mediaDevices.enumerateDevices()
        const cams = devs
          .filter((d) => d.kind === 'videoinput')
          .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }))
        set({ devices: cams })
      } catch (e) {
        get().log('Camera enumeration error: ' + (e instanceof Error ? e.message : e))
      }
    },

    addLayer() {
      const n = get().layers.length + 1
      set((s) => ({ layers: [newEmptyLayer(`Layer ${n}`), ...s.layers] }))
    },

    clearComposition() {
      for (const layer of get().layers) {
        for (const cell of layer.cells) if (cell.clip) releaseClip(cell.clip)
      }
      rail.stop()
      boundSource.clear()
      currentBaseLayerId = null
      set({
        layers: [newEmptyLayer('Layer 1')],
        selectedClipId: null,
        videoState: {},
        active: false,
        layoutLayer: null,
        drawMode: 'off',
      })
    },

    async removeLayer(layerId) {
      const layer = layerById(get().layers, layerId)
      if (!layer) return
      for (const cell of layer.cells) if (cell.clip) releaseClip(cell.clip)
      set((s) => {
        const removedClipIds = new Set(layer.cells.map((c) => c.clip?.id).filter(Boolean) as string[])
        const videoState = { ...s.videoState }
        for (const id of removedClipIds) delete videoState[id]
        const layers = s.layers.filter((l) => l.id !== layerId)
        const selectedClipId =
          s.selectedClipId && removedClipIds.has(s.selectedClipId) ? null : s.selectedClipId
        return {
          layers,
          videoState,
          selectedClipId,
          layoutLayer: s.layoutLayer === layerId ? null : s.layoutLayer,
        }
      })
      await syncPipeline()
    },

    moveLayer(layerId, dir) {
      const layers = get().layers
      const i = layers.findIndex((l) => l.id === layerId)
      const j = i + dir
      if (i < 0 || j < 0 || j >= layers.length) return
      const next = layers.slice()
      ;[next[i], next[j]] = [next[j], next[i]]
      set({ layers: next })
      rail.setCompositeAll(compositeFrom(next))
    },

    addCell(layerId) {
      set((s) => ({
        layers: s.layers.map((l) =>
          l.id !== layerId ? l : { ...l, cells: [...l.cells, { id: freshCell(), clip: null }] },
        ),
      }))
    },

    async removeClip(layerId, cellId) {
      const layer = layerById(get().layers, layerId)
      const cell = layer?.cells.find((c) => c.id === cellId)
      const clip = cell?.clip
      if (clip) releaseClip(clip)
      set((s) => ({
        layers: s.layers.map((l) => {
          if (l.id !== layerId) return l
          const cells = l.cells.map((c) => (c.id === cellId ? { ...c, clip: null } : c))
          const wasActive = l.activeCellId === cellId
          const hasOtherClip = cells.some((c) => c.clip)
          return {
            ...l,
            cells,
            activeCellId: wasActive ? cells.find((c) => c.clip)?.id ?? null : l.activeCellId,
            kind: hasOtherClip ? l.kind : null,
            role: hasOtherClip ? l.role : null,
          }
        }),
        videoState: clip ? omit(s.videoState, clip.id) : s.videoState,
        selectedClipId: clip && s.selectedClipId === clip.id ? null : s.selectedClipId,
      }))
      await syncPipeline()
    },

    async activateCell(layerId, cellId) {
      const layer = layerById(get().layers, layerId)
      const cell = layer?.cells.find((c) => c.id === cellId)
      const clip = cell?.clip
      set((s) => ({
        layers: s.layers.map((l) => (l.id === layerId ? { ...l, activeCellId: cellId } : l)),
        selectedClipId: clip?.id ?? s.selectedClipId,
      }))
      // Re-acquire a camera/screen source that isn't live yet (e.g. a clip
      // restored from a previous session). Video/image need a re-pick (needsFile).
      if (clip && !clipStreams.get(clip.id)) {
        try {
          if (clip.kind === 'camera') {
            clipStreams.set(clip.id, await acquireCamera(clip.id, clip.deviceId || ''))
          } else if (clip.kind === 'screen') {
            const getDisplay = navigator.mediaDevices?.getDisplayMedia?.bind(navigator.mediaDevices)
            if (getDisplay) {
              const stream = await getDisplay({ video: true, audio: false })
              clipStreams.set(clip.id, stream)
              stream.getVideoTracks()[0]?.addEventListener('ended', () => void get().removeClip(layerId, cellId))
            }
          }
        } catch (e) {
          get().log('Re-activate failed: ' + (e instanceof Error ? e.message : e))
        }
      }
      await syncPipeline()
    },

    async moveClip(fromLayerId, fromCellId, toLayerId, toCellId) {
      if (fromLayerId === toLayerId && fromCellId === toCellId) return
      const moving = layerById(get().layers, fromLayerId)?.cells.find((c) => c.id === fromCellId)?.clip
      if (!moving) return
      set((s) => {
        const toClip = layerById(s.layers, toLayerId)?.cells.find((c) => c.id === toCellId)?.clip ?? null
        const swapped = s.layers.map((l) => {
          let cells = l.cells
          if (l.id === fromLayerId) cells = cells.map((c) => (c.id === fromCellId ? { ...c, clip: toClip } : c))
          if (l.id === toLayerId) cells = cells.map((c) => (c.id === toCellId ? { ...c, clip: moving } : c))
          return cells === l.cells ? l : { ...l, cells }
        })
        const fixed = swapped.map((l) =>
          l.id === toLayerId ? normalizeLayer(l, toCellId) : l.id === fromLayerId ? normalizeLayer(l) : l,
        )
        return { layers: fixed, selectedClipId: moving.id }
      })
      await syncPipeline()
    },

    selectClip(id) {
      set({ selectedClipId: id })
    },

    async fillCellCamera(layerId, cellId, deviceId) {
      const { log } = get()
      const clip = makeClip('camera', 'Camera')
      clip.deviceId = deviceId
      try {
        const stream = await acquireCamera(clip.id, deviceId)
        clipStreams.set(clip.id, stream)
        // Label the clip with the actual device name (track.label).
        const name = stream.getVideoTracks()[0]?.label
        if (name) clip.label = name
      } catch (e) {
        log('Camera failed: ' + (e instanceof Error ? e.message : e))
        return
      }
      placeClip(layerId, cellId, clip)
      await get().refreshCameras()
      await syncPipeline()
    },

    async fillCellVideo(layerId, cellId, file) {
      const { log } = get()
      const clip = makeClip('video', file.name)
      clip.file = file
      try {
        const src = acquireVideoSource(clip.id)
        const meta = await src.load(file)
        src.setLoop(true)
        src.setRate(1)
        attachVideoListeners(clip.id, src.el)
        set((s) => ({
          videoState: {
            ...s.videoState,
            [clip.id]: {
              name: file.name,
              meta: `${meta.width}×${meta.height}, ${fmtTime(meta.duration)}`,
              duration: meta.duration,
              currentTime: 0,
              playing: true,
              loop: true,
              rate: 1,
            },
          },
        }))
      } catch (e) {
        log('Video load failed: ' + (e instanceof Error ? e.message : e))
        return
      }
      placeClip(layerId, cellId, clip)
      await syncPipeline()
    },

    async fillCellScreen(layerId, cellId) {
      const { log } = get()
      const getDisplay = navigator.mediaDevices?.getDisplayMedia?.bind(navigator.mediaDevices)
      if (!getDisplay) {
        log('Screen share unavailable in this browser.')
        return
      }
      const clip = makeClip('screen', 'Screen')
      let stream: MediaStream
      try {
        stream = await getDisplay({ video: true, audio: false })
      } catch (e) {
        log('Screen share cancelled: ' + (e instanceof Error ? e.message : e))
        return
      }
      clipStreams.set(clip.id, stream)
      stream.getVideoTracks()[0]?.addEventListener('ended', () => void get().removeClip(layerId, cellId))
      placeClip(layerId, cellId, clip)
      await syncPipeline()
    },

    async fillCellFeedback(layerId, cellId) {
      const clip = makeClip('feedback', 'Feedback')
      placeClip(layerId, cellId, clip)
      await syncPipeline()
    },

    async fillCellEffect(layerId, cellId, kind) {
      const clip = makeClip(kind, clipMeta(kind).label)
      clip.effectConfig = defaultEffectConfig(kind)
      placeClip(layerId, cellId, clip)
      await syncPipeline()
    },

    setEffectConfig(clipId, patch) {
      const found = findClip(get().layers, clipId)
      if (!found) return
      const merged = { ...found.clip.effectConfig, ...patch }
      set((s) => ({
        layers: s.layers.map((l) => ({
          ...l,
          cells: l.cells.map((c) =>
            c.clip?.id === clipId ? { ...c, clip: { ...c.clip, effectConfig: merged } } : c,
          ),
        })),
      }))
      if (activeClip(found.layer)?.id === clipId) {
        rail.setComposite({ op: 'patch', layers: [{ id: found.layer.id, effectConfig: merged }] })
      }
    },

    async setClipDevice(clipId, deviceId) {
      const found = findClip(get().layers, clipId)
      if (!found || found.clip.kind !== 'camera') return
      try {
        const stream = await acquireCamera(clipId, deviceId)
        clipStreams.set(clipId, stream)
        const name = stream.getVideoTracks()[0]?.label
        set((s) => ({
          layers: s.layers.map((l) => ({
            ...l,
            cells: l.cells.map((c) =>
              c.clip?.id === clipId ? { ...c, clip: { ...c.clip, deviceId, label: name || c.clip.label } } : c,
            ),
          })),
        }))
        await syncPipeline()
      } catch (e) {
        get().log('Camera switch failed: ' + (e instanceof Error ? e.message : e))
      }
    },

    setClipMirror(clipId, on) {
      const found = findClip(get().layers, clipId)
      if (!found) return
      set((s) => ({
        layers: s.layers.map((l) => ({
          ...l,
          cells: l.cells.map((c) => (c.clip?.id === clipId ? { ...c, clip: { ...c.clip, mirror: on } } : c)),
        })),
      }))
      // If this clip is the layer's active one, push the mirror change live.
      if (activeClip(found.layer)?.id === clipId) {
        rail.setComposite({ op: 'patch', layers: [{ id: found.layer.id, mirror: on }] })
      }
    },

    setLayerOpacity(layerId, opacity) {
      set((s) => ({ layers: s.layers.map((l) => (l.id === layerId ? { ...l, opacity } : l)) }))
      rail.setComposite({ op: 'patch', layers: [{ id: layerId, opacity }] })
    },
    setLayerBlend(layerId, blend) {
      set((s) => ({ layers: s.layers.map((l) => (l.id === layerId ? { ...l, blend } : l)) }))
      rail.setComposite({ op: 'patch', layers: [{ id: layerId, blend }] })
    },

    toggleVideoPlay(clipId) {
      const el = acquireVideoSource(clipId).el
      if (el.paused) {
        if (el.ended && !get().videoState[clipId]?.loop) el.currentTime = 0
        void el.play().catch((e) => get().log('Video play failed: ' + e))
      } else {
        el.pause()
      }
    },
    seekVideo(clipId, t) {
      acquireVideoSource(clipId).seek(t)
      set((s) => ({ videoState: { ...s.videoState, [clipId]: { ...s.videoState[clipId], currentTime: t } } }))
    },
    setVideoLoop(clipId, on) {
      acquireVideoSource(clipId).setLoop(on)
      set((s) => ({ videoState: { ...s.videoState, [clipId]: { ...s.videoState[clipId], loop: on } } }))
    },
    setVideoRate(clipId, r) {
      acquireVideoSource(clipId).setRate(r)
      set((s) => ({ videoState: { ...s.videoState, [clipId]: { ...s.videoState[clipId], rate: r } } }))
    },

    setLayoutLayer(id) {
      if (id) set({ layoutLayer: id, drawMode: 'off' })
      else set({ layoutLayer: null })
    },
    setCropMode(on) {
      set({ cropMode: on })
    },
    setLayerTransform(id, transform) {
      const layer = layerById(get().layers, id)
      const clip = layer ? activeClip(layer) : null
      set((s) => ({
        layers: s.layers.map((l) => {
          if (l.id !== id) return l
          if (clip) {
            return {
              ...l,
              cells: l.cells.map((c) =>
                c.clip?.id === clip.id ? { ...c, clip: { ...c.clip, transform: transform ?? undefined } } : c,
              ),
            }
          }
          return { ...l, transform: transform ?? undefined }
        }),
      }))
      rail.setComposite({ op: 'patch', layers: [{ id, transform: transform ?? undefined }] })
    },

    attachFeedback(stream) {
      const ok = !!stream && stream.getVideoTracks().length > 0
      feedbackStream = ok ? stream : null
      set({ feedbackAvailable: ok })
      void syncPipeline()
    },

    setDrawMode(mode) {
      set({ drawMode: mode, ...(mode !== 'off' ? { layoutLayer: null } : {}) })
      rail.configureDraw({ erase: mode === 'eraser' })
    },
    setDrawColor(c) {
      set({ drawColor: c })
      rail.configureDraw({ color: c })
    },
    setDrawSize(n) {
      set({ drawSize: n })
      rail.configureDraw({ size: n })
    },
    clearDrawing() {
      rail.clearDrawing()
    },

    async setMarkerEnabled(on) {
      set({ markerEnabled: on, poseStatus: on ? 'loading pose model...' : 'marker: OFF' })
      rail.configureMarker({ enabled: on })
      if (on) {
        try {
          await inputVision.acquire('marker', { face: false, pose: true })
          set({ poseStatus: 'marker: ON' })
        } catch {
          set({ poseStatus: 'pose load error', markerEnabled: false })
          rail.configureMarker({ enabled: false })
        }
      } else {
        void inputVision.release('marker')
      }
    },
    setMarkerLandmark(n) {
      set({ markerLandmark: n })
      rail.configureMarker({ landmark: n })
    },
    setMarkerColor(c) {
      set({ markerColor: c })
      rail.configureMarker({ color: c })
    },
    setMarkerSize(n) {
      set({ markerSize: n })
      rail.configureMarker({ size: n })
    },
    setMarkerTrail(on) {
      set({ markerTrail: on })
      rail.configureMarker({ trail: on })
    },
    setMarkerTrailLen(n) {
      set({ markerTrailLen: n })
      rail.configureMarker({ trailLen: n })
    },
  }
})

// Persist the composition structure (debounced) whenever the grid changes.
let saveTimer: ReturnType<typeof setTimeout> | undefined
usePipelineStore.subscribe((state, prev) => {
  if (state.layers === prev.layers) return
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => saveComposition(usePipelineStore.getState().layers), 500)
})

let cellN = 0
function freshCell(): string {
  cellN += 1
  return `cell-x-${cellN}`
}

function omit<T extends Record<string, unknown>>(obj: T, key: string): T {
  const next = { ...obj }
  delete next[key]
  return next
}
