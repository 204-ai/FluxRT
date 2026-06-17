// Input pipeline state: sources (camera + video file), mirror, compositing,
// marker, draw. Drives the module-level Rail and VideoFileSource; holds only
// renderable state.

import { create } from 'zustand'
import {
  acquireCamera,
  acquireVideoSource,
  inputVision,
  rail,
  releaseCamera,
  releaseVideoSource,
  videoSource,
} from './runtime'
import { getHealthz } from '../lib/api'
import type { BaseSource, BlendMode, ClipId, LayerId, LayerTransform } from '../pipeline/core/types'
import { CAMERA_LAYER, hasVideoTrack, VIDEO_LAYER } from '../pipeline/core/types'

// Stable key for the seeded camera clip's stream in the runtime camera pool
// (one webcam in the seeded model; generic per-clip keys arrive in G3).
const SEED_CAMERA = 'seed-camera'
import type { Layer } from './layerModel'
import { activeClip, freshId, layerById, makeClip, seedLayers } from './layerModel'

/** Transient transport state for an ADDED (overlay) video clip, keyed by clip
 *  id. The seeded video layer keeps the singular video* fields above; extra
 *  layers each track their own playback here so the pool stays decoupled. */
interface ExtraVideoState {
  name: string
  duration: number
  currentTime: number
  playing: boolean
  loop: boolean
  rate: number
}

// Per-added-clip <video> event detachers, mirroring the seeded layer's
// listeners — kept module-level so a removed layer never strands handlers.
const extraDetachers = new Map<string, () => void>()

// Live getDisplayMedia streams for screen-share layers, keyed by layer id —
// stopped on layer removal so the browser's "sharing" indicator clears.
const screenStreams = new Map<string, MediaStream>()

/** Immutable per-layer patch by id (the store holds layers as an ordered list). */
function patchLayer(layers: Layer[], id: LayerId, patch: Partial<Layer>): Layer[] {
  return layers.map((l) => (l.id === id ? { ...l, ...patch } : l))
}

export type DrawMode = 'off' | 'brush' | 'eraser'

interface CameraDevice {
  deviceId: string
  label: string
}

interface PipelineState {
  camEnabled: boolean
  devices: CameraDevice[]
  deviceId: string
  mirror: boolean
  active: boolean

  videoLoaded: boolean
  videoName: string
  videoMeta: string
  videoDuration: number
  videoCurrentTime: number
  videoPlaying: boolean
  videoLoop: boolean
  videoRate: number

  // The compositing stack, ordered front → back (top row frontmost). Holds the
  // per-layer mix + geometry + cells/clips. Seeded with the three legacy layers;
  // dynamic add/remove/reorder lands in P2.
  layers: Layer[]
  // Transport for ADDED (overlay) video layers, keyed by their clip id.
  extraVideo: Record<string, ExtraVideoState>
  // The feedback layer is live only while a remote output stream is connected.
  feedbackAvailable: boolean

  drawMode: DrawMode
  drawColor: string
  drawSize: number

  // OBS-style layer framing (move/resize/crop). `layoutLayer` is the layer being
  // edited — null means the layout overlay is off (it doubles as mode + selection).
  layoutLayer: LayerId | null
  // Within layout: false = move/resize the frame, true = crop the content edges.
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
  enableCam(): Promise<void>
  disableCam(): Promise<void>
  setDevice(deviceId: string): Promise<void>
  refreshCameras(): Promise<void>
  setMirror(on: boolean): void
  startPipeline(): Promise<void>
  stopPipeline(): void

  loadVideoFile(file: File): Promise<void>
  unloadVideo(): Promise<void>
  toggleVideoPlay(): void
  seekVideo(t: number): void
  setVideoLoop(on: boolean): void
  setVideoRate(r: number): void

  setLayerOpacity(id: LayerId, opacity: number): void
  setLayerBlend(id: LayerId, blend: BlendMode): void
  /** Move a layer up (-1, toward the front) or down (+1) in the z-order; the
   *  composite re-stacks live (no restart). */
  moveLayer(id: LayerId, dir: -1 | 1): void
  /** Add a new video layer (front) from a file. With a running pipeline this
   *  hot-adds a live overlay (its own pooled <video> source) with no restart;
   *  with nothing running it falls back to loading the seeded video layer. */
  addVideoLayer(file: File): Promise<void>
  /** Remove an ADDED layer: drop its source + composite slot + pooled state.
   *  (The seeded camera/video/feedback layers are not removed here.) */
  removeLayer(id: LayerId): Promise<void>
  toggleExtraVideoPlay(clipId: string): void
  seekExtraVideo(clipId: string, t: number): void
  setExtraVideoLoop(clipId: string, on: boolean): void
  setExtraVideoRate(clipId: string, r: number): void
  /** Add a screen-share layer (getDisplayMedia) as a live overlay — proves a
   *  new mediastream clip kind needs only a registry entry + an acquire (no
   *  pipeline/worker change). */
  addScreenLayer(): Promise<void>

  // The clip selected for the detail pane + highlight (null = none).
  selectedClipId: ClipId | null
  selectClip(id: ClipId | null): void
  /** Wire the remote output stream in/out as the feedback layer (null = clear). */
  attachFeedback(stream: MediaStream | null): void

  /** Select a layer to frame (move/resize/crop), or null to close the overlay.
   *  Opening the overlay turns drawing off (they share the preview surface). */
  setLayoutLayer(id: LayerId | null): void
  setCropMode(on: boolean): void
  /** Set a layer's framing transform live (null restores legacy cover-fit). */
  setLayerTransform(id: LayerId, transform: LayerTransform | null): void

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

// Transport listeners on the file <video> — attached on load, detached on
// unload, so a swapped store logger never leaves stale handlers behind.
let detachVideoListeners: (() => void) | null = null

export const usePipelineStore = create<PipelineState>((set, get) => {
  /** Restart the rail to match the current source set (camera/video). Throws
   *  on start failure (e.g. getUserMedia) — callers decide how to recover. */
  async function restartSources(): Promise<void> {
    rail.stop()
    if (!get().camEnabled && !get().videoLoaded) {
      get().stopPipeline()
      // Nothing left to analyze — drop the marker's vision worker.
      if (get().markerEnabled) {
        set({ markerEnabled: false, poseStatus: '' })
        rail.configureMarker({ enabled: false })
        void inputVision.release('marker')
      }
      return
    }
    set({ active: false })
    await get().startPipeline()
  }

  function attachVideoListeners(): void {
    const el = videoSource.el
    const onTime = () => set({ videoCurrentTime: el.currentTime })
    const onDuration = () => set({ videoDuration: el.duration })
    const onPlay = () => set({ videoPlaying: true })
    const onPause = () => set({ videoPlaying: false })
    el.addEventListener('timeupdate', onTime)
    el.addEventListener('durationchange', onDuration)
    el.addEventListener('play', onPlay)
    el.addEventListener('pause', onPause)
    detachVideoListeners = () => {
      el.removeEventListener('timeupdate', onTime)
      el.removeEventListener('durationchange', onDuration)
      el.removeEventListener('play', onPlay)
      el.removeEventListener('pause', onPause)
      detachVideoListeners = null
    }
  }

  // Seed once so the default selection (camera clip) matches the seeded layers.
  const initialLayers = seedLayers()

  return {
    camEnabled: false,
    devices: [],
    deviceId: '',
    mirror: false,
    active: false,

    videoLoaded: false,
    videoName: '',
    videoMeta: '',
    videoDuration: 0,
    videoCurrentTime: 0,
    videoPlaying: false,
    videoLoop: true,
    videoRate: 1,

    layers: initialLayers,
    extraVideo: {},
    selectedClipId: activeClip(initialLayers[0])?.id ?? null,
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

    async enableCam() {
      const { log } = get()
      if (!navigator.mediaDevices?.getUserMedia) {
        log('getUserMedia unavailable — needs HTTPS or a secure-origin allowlist.')
        set({ camEnabled: false })
        return
      }
      try {
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        tmp.getTracks().forEach((t) => t.stop())
      } catch (e) {
        log('Camera permission denied: ' + (e instanceof Error ? e.message : e))
        set({ camEnabled: false })
        return
      }
      // Default to mirrored (selfie) view when the user enables their camera.
      set({ camEnabled: true, mirror: true })
      rail.setMirror(true)
      await get().refreshCameras()
      try {
        await restartSources()
      } catch (e) {
        log('Camera start failed: ' + (e instanceof Error ? e.message : e))
        set({ camEnabled: false, mirror: false })
        rail.setMirror(false)
        // Fall back to a video-only pipeline if a file is still loaded.
        if (get().videoLoaded) await restartSources().catch(() => {})
      }
    },

    async disableCam() {
      set({ camEnabled: false, mirror: false })
      rail.setMirror(false)
      releaseCamera(SEED_CAMERA)
      await restartSources().catch((e) =>
        get().log('Pipeline restart failed: ' + (e instanceof Error ? e.message : e)),
      )
    },

    async setDevice(deviceId) {
      set({ deviceId })
      if (!get().camEnabled || !rail.active) return
      // Hot-swap the camera track in place — keeps the output stream alive
      // instead of restarting the whole pipeline (which froze the output).
      try {
        const stream = await acquireCamera(SEED_CAMERA, deviceId)
        rail.setLayerSource(CAMERA_LAYER, 'camera', stream)
      } catch (e) {
        get().log('Camera switch failed: ' + (e instanceof Error ? e.message : e))
        // Fall back to a full restart if the in-place swap couldn't acquire.
        await restartSources().catch(() => {})
      }
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

    setMirror(on) {
      set({ mirror: on })
      rail.setMirror(on)
    },

    async startPipeline() {
      const { deviceId, camEnabled, videoLoaded, log } = get()
      // Crop the input to the server's output aspect ratio so the input preview
      // (and the frames sent upstream) match the model's framing — no distortion
      // or letterbox when the camera/video aspect differs from the output.
      let targetAspect: number | null = null
      try {
        const h = await getHealthz()
        if (h.resolution && h.resolution.height > 0) {
          targetAspect = h.resolution.width / h.resolution.height
        }
      } catch {
        /* server resolution unknown — fall back to the source aspect */
      }
      // Acquire + build the BASE source (the store owns acquisition now). Camera
      // is base when enabled; otherwise the seeded video file.
      let base: BaseSource
      if (camEnabled) {
        const stream = await acquireCamera(SEED_CAMERA, deviceId)
        base = { layerId: CAMERA_LAYER, kind: 'camera', stream, videoEl: null }
      } else if (videoLoaded) {
        base = { layerId: VIDEO_LAYER, kind: 'video', stream: null, videoEl: videoSource.el }
      } else {
        throw new Error('no input source')
      }
      const { label } = await rail.start(base, targetAspect)
      log('Input pipeline started: ' + label)
      set({ active: true })
      // Camera is the base → attach the seeded video file as an overlay (both on).
      if (camEnabled && videoLoaded) rail.setLayerSource(VIDEO_LAYER, 'video', videoSource.el)
      // Re-bind any added (overlay) video layers onto the freshly-built backend
      // so a source-set restart doesn't drop them — their mix already rode along
      // in rail.composite (op:'add'); only the live source needs re-attaching.
      for (const layer of get().layers) {
        const clip = activeClip(layer)
        if (clip && get().extraVideo[clip.id]) {
          rail.setLayerSource(layer.id, 'video', acquireVideoSource(clip.id).el)
        } else if (clip?.kind === 'screen') {
          const s = screenStreams.get(layer.id)
          if (s) rail.setLayerSource(layer.id, 'screen', s)
        }
      }
    },

    stopPipeline() {
      rail.stop()
      releaseCamera(SEED_CAMERA)
      set({ active: false, drawMode: 'off', layoutLayer: null })
    },

    async loadVideoFile(file) {
      const { log, videoLoop, videoRate } = get()
      const active = get().active
      try {
        // Case 1 — a clip is already feeding a running pipeline: swap the
        // <video> src in place and keep the SAME captured track / output stream
        // alive (no rail restart, no WebRTC renegotiation, no black frame).
        if (active && get().videoLoaded) {
          const meta = await videoSource.swapSource(file)
          videoSource.setLoop(videoLoop)
          videoSource.setRate(videoRate)
          // Re-feed the running pipeline the element's NEW captured track so the
          // output (and WebRTC) keep flowing; the old track ended on the src swap.
          rail.swapVideoSource(videoSource.el)
          // Listeners stay bound to the same element across the swap.
          set({
            videoName: file.name,
            videoMeta: `${meta.width}×${meta.height}, ${fmtTime(meta.duration)}`,
            videoDuration: meta.duration,
            videoCurrentTime: 0,
            videoPlaying: true,
          })
          log(
            `Video swapped: ${file.name} (${meta.width}x${meta.height}, ${fmtTime(meta.duration)}) — pipeline kept live`,
          )
          return
        }

        // First load of this clip onto the shared <video> element.
        const meta = await videoSource.load(file)
        detachVideoListeners?.()
        attachVideoListeners()
        videoSource.setLoop(videoLoop)
        videoSource.setRate(videoRate)
        set({
          videoLoaded: true,
          videoName: file.name,
          videoMeta: `${meta.width}×${meta.height}, ${fmtTime(meta.duration)}`,
          videoDuration: meta.duration,
          videoCurrentTime: 0,
          videoPlaying: true,
        })

        // Case 2 — pipeline already running (a camera): hot-ADD the clip as an
        // overlay layer instead of restarting, so the output stream / WebRTC
        // track stays alive (no stall). The worker creates the overlay valve on
        // the first swap-video even though it was started camera-only.
        if (active) {
          rail.swapVideoSource(videoSource.el)
          log(
            `Video added: ${file.name} (${meta.width}x${meta.height}, ${fmtTime(meta.duration)}) — composited live over the camera`,
          )
          return
        }

        // Case 3 — nothing running yet: start the pipeline with this clip.
        log(`Video loaded: ${file.name} (${meta.width}x${meta.height}, ${fmtTime(meta.duration)})`)
        await restartSources()
      } catch (e) {
        log('Video load failed: ' + (e instanceof Error ? e.message : e))
      }
    },

    async unloadVideo() {
      // Hot-remove: the camera is still feeding a running pipeline → drop the
      // overlay layer in place and keep the SAME output stream alive (mirror of
      // the hot-ADD in loadVideoFile). Otherwise (video-only / stopped) fall
      // back to a source-set restart.
      const hotRemove = get().active && get().camEnabled
      detachVideoListeners?.()
      videoSource.unload()
      set({
        videoLoaded: false,
        videoName: '',
        videoMeta: '',
        videoDuration: 0,
        videoCurrentTime: 0,
        videoPlaying: false,
      })
      if (hotRemove) {
        rail.clearVideoSource()
        get().log('Video removed — camera kept live')
        return
      }
      await restartSources().catch((e) =>
        get().log('Pipeline restart failed: ' + (e instanceof Error ? e.message : e)),
      )
    },

    toggleVideoPlay() {
      const el = videoSource.el
      if (el.paused) {
        if (el.ended && !get().videoLoop) videoSource.seek(0)
        void videoSource.play().catch((e) => get().log('Video play failed: ' + e))
      } else {
        videoSource.pause()
      }
    },

    seekVideo(t) {
      videoSource.seek(t)
      set({ videoCurrentTime: t })
    },

    setVideoLoop(on) {
      set({ videoLoop: on })
      videoSource.setLoop(on)
    },

    setVideoRate(r) {
      set({ videoRate: r })
      videoSource.setRate(r)
    },

    setLayerOpacity(id, opacity) {
      set((s) => ({ layers: patchLayer(s.layers, id, { opacity }) }))
      rail.setComposite({ op: 'patch', layers: [{ id, opacity }] })
    },
    setLayerBlend(id, blend) {
      set((s) => ({ layers: patchLayer(s.layers, id, { blend }) }))
      rail.setComposite({ op: 'patch', layers: [{ id, blend }] })
    },
    moveLayer(id, dir) {
      const layers = get().layers
      const i = layers.findIndex((l) => l.id === id)
      const j = i + dir
      if (i < 0 || j < 0 || j >= layers.length) return
      const next = layers.slice()
      ;[next[i], next[j]] = [next[j], next[i]]
      set({ layers: next })
      // Re-stack the live composite to match — z-order changes immediately.
      rail.setComposite({ op: 'reorder', order: next.map((l) => l.id) })
    },

    async addVideoLayer(file) {
      const { log } = get()
      // Nothing running yet → load into the seeded video layer (keeps the
      // proven base/start path); extra layers are always live overlays.
      if (!get().active) {
        await get().loadVideoFile(file)
        return
      }
      try {
        const clip = makeClip('video', file.name)
        const layerId = freshId('layer')
        const cellId = freshId('cell')
        const src = acquireVideoSource(clip.id)
        const meta = await src.load(file)
        src.setLoop(true)
        src.setRate(1)
        // Per-clip transport listeners (mirror of the seeded layer's).
        const el = src.el
        const onTime = () =>
          set((s) => ({
            extraVideo: {
              ...s.extraVideo,
              [clip.id]: { ...s.extraVideo[clip.id], currentTime: el.currentTime },
            },
          }))
        const onDur = () =>
          set((s) => ({
            extraVideo: { ...s.extraVideo, [clip.id]: { ...s.extraVideo[clip.id], duration: el.duration } },
          }))
        const onPlay = () =>
          set((s) => ({
            extraVideo: { ...s.extraVideo, [clip.id]: { ...s.extraVideo[clip.id], playing: true } },
          }))
        const onPause = () =>
          set((s) => ({
            extraVideo: { ...s.extraVideo, [clip.id]: { ...s.extraVideo[clip.id], playing: false } },
          }))
        el.addEventListener('timeupdate', onTime)
        el.addEventListener('durationchange', onDur)
        el.addEventListener('play', onPlay)
        el.addEventListener('pause', onPause)
        extraDetachers.set(clip.id, () => {
          el.removeEventListener('timeupdate', onTime)
          el.removeEventListener('durationchange', onDur)
          el.removeEventListener('play', onPlay)
          el.removeEventListener('pause', onPause)
        })

        const layer: Layer = {
          id: layerId,
          name: file.name,
          kind: 'video',
          role: 'source',
          opacity: 1,
          blend: 'normal',
          cells: [{ id: cellId, clip }],
          activeCellId: cellId,
        }
        set((s) => ({
          layers: [layer, ...s.layers],
          extraVideo: {
            ...s.extraVideo,
            [clip.id]: {
              name: file.name,
              duration: meta.duration,
              currentTime: 0,
              playing: true,
              loop: true,
              rate: 1,
            },
          },
        }))
        // Add the layer to the live composite (frontmost) then hot-bind its
        // source — both in place, no restart, output track stays alive.
        rail.setComposite({ op: 'add', layer: { id: layerId, opacity: 1, blend: 'normal', mirror: false }, index: 0 })
        rail.setLayerSource(layerId, 'video', el)
        log(`Video layer added: ${file.name} (${meta.width}x${meta.height}, ${fmtTime(meta.duration)})`)
      } catch (e) {
        log('Add video layer failed: ' + (e instanceof Error ? e.message : e))
      }
    },

    async removeLayer(id) {
      const layer = layerById(get().layers, id)
      if (!layer) return
      const clip = activeClip(layer)
      // Drop the live source + composite slot in place (output stays alive).
      rail.setLayerSource(id, clip?.kind ?? 'video', null)
      rail.setComposite({ op: 'remove', id })
      if (clip) {
        extraDetachers.get(clip.id)?.()
        extraDetachers.delete(clip.id)
        releaseVideoSource(clip.id)
      }
      const screen = screenStreams.get(id)
      if (screen) {
        screen.getTracks().forEach((t) => {
          try {
            t.stop()
          } catch {
            /* already stopped */
          }
        })
        screenStreams.delete(id)
      }
      set((s) => {
        const extraVideo = { ...s.extraVideo }
        if (clip) delete extraVideo[clip.id]
        const layers = s.layers.filter((l) => l.id !== id)
        // Re-select the frontmost remaining clip if the selected one is gone.
        const selectedClipId =
          clip && s.selectedClipId === clip.id
            ? (layers[0] ? activeClip(layers[0]) : null)?.id ?? null
            : s.selectedClipId
        return { layers, extraVideo, selectedClipId }
      })
      if (get().layoutLayer === id) set({ layoutLayer: null })
    },

    toggleExtraVideoPlay(clipId) {
      const el = acquireVideoSource(clipId).el
      if (el.paused) {
        if (el.ended && !get().extraVideo[clipId]?.loop) el.currentTime = 0
        void el.play().catch((e) => get().log('Video play failed: ' + e))
      } else {
        el.pause()
      }
    },
    seekExtraVideo(clipId, t) {
      acquireVideoSource(clipId).seek(t)
      set((s) => ({
        extraVideo: { ...s.extraVideo, [clipId]: { ...s.extraVideo[clipId], currentTime: t } },
      }))
    },
    setExtraVideoLoop(clipId, on) {
      acquireVideoSource(clipId).setLoop(on)
      set((s) => ({ extraVideo: { ...s.extraVideo, [clipId]: { ...s.extraVideo[clipId], loop: on } } }))
    },
    setExtraVideoRate(clipId, r) {
      acquireVideoSource(clipId).setRate(r)
      set((s) => ({ extraVideo: { ...s.extraVideo, [clipId]: { ...s.extraVideo[clipId], rate: r } } }))
    },

    async addScreenLayer() {
      const { log } = get()
      if (!get().active) {
        log('Start the camera or a video before adding a screen layer.')
        return
      }
      const getDisplay = navigator.mediaDevices?.getDisplayMedia?.bind(navigator.mediaDevices)
      if (!getDisplay) {
        log('Screen share unavailable in this browser.')
        return
      }
      let stream: MediaStream
      try {
        stream = await getDisplay({ video: true, audio: false })
      } catch (e) {
        log('Screen share cancelled: ' + (e instanceof Error ? e.message : e))
        return
      }
      const clip = makeClip('screen', 'Screen')
      const layerId = freshId('layer')
      const cellId = freshId('cell')
      const layer: Layer = {
        id: layerId,
        name: 'Screen',
        kind: 'screen',
        role: 'source',
        opacity: 1,
        blend: 'normal',
        cells: [{ id: cellId, clip }],
        activeCellId: cellId,
      }
      screenStreams.set(layerId, stream)
      // The user can stop sharing from the browser chrome — drop the layer then.
      stream.getVideoTracks()[0]?.addEventListener('ended', () => void get().removeLayer(layerId))
      set((s) => ({ layers: [layer, ...s.layers], selectedClipId: clip.id }))
      rail.setComposite({ op: 'add', layer: { id: layerId, opacity: 1, blend: 'normal', mirror: false }, index: 0 })
      rail.setLayerSource(layerId, 'screen', stream)
      log('Screen layer added')
    },

    selectClip(id) {
      set({ selectedClipId: id })
    },
    attachFeedback(stream) {
      const ok = hasVideoTrack(stream)
      // Contain any failure (e.g. track.clone / MSTP) so wiring the feedback
      // layer can never break the remote-track handler or the live output.
      try {
        rail.setFeedback(ok ? stream : null)
        set({ feedbackAvailable: ok })
      } catch (e) {
        get().log('Feedback layer setup failed: ' + (e instanceof Error ? e.message : e))
        set({ feedbackAvailable: false })
      }
    },

    setLayoutLayer(id) {
      // Drawing and framing share the preview surface — entering one exits the
      // other so pointer drags are never ambiguous.
      if (id) set({ layoutLayer: id, drawMode: 'off' })
      else set({ layoutLayer: null })
    },
    setCropMode(on) {
      set({ cropMode: on })
    },
    setLayerTransform(id, transform) {
      set((s) => ({ layers: patchLayer(s.layers, id, { transform: transform ?? undefined }) }))
      // Carry the value explicitly (incl. undefined) so the patch resets the
      // compositor layer back to legacy cover-fit when cleared.
      rail.setComposite({ op: 'patch', layers: [{ id, transform: transform ?? undefined }] })
    },

    setDrawMode(mode) {
      // Drawing and framing are mutually exclusive on the shared preview.
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
