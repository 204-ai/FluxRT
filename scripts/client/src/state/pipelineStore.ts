// Input pipeline state: sources (camera + video file), mirror, compositing,
// marker, draw. Drives the module-level Rail and VideoFileSource; holds only
// renderable state.

import { create } from 'zustand'
import { inputVision, rail, videoSource } from './runtime'
import type { BlendMode, LayerOrder } from '../pipeline/core/types'

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

  compositeOrder: LayerOrder
  compositeOpacity: number
  compositeBlend: BlendMode

  drawMode: DrawMode
  drawColor: string
  drawSize: number

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

  setCompositeOrder(order: LayerOrder): void
  setCompositeOpacity(opacity: number): void
  setCompositeBlend(blend: BlendMode): void

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

    compositeOrder: 'camera-over',
    compositeOpacity: 0.5,
    compositeBlend: 'normal',

    drawMode: 'off',
    drawColor: '#ffffff',
    drawSize: 6,

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
      await restartSources().catch((e) =>
        get().log('Pipeline restart failed: ' + (e instanceof Error ? e.message : e)),
      )
    },

    async setDevice(deviceId) {
      set({ deviceId })
      if (!get().camEnabled || !rail.active) return
      try {
        await restartSources()
      } catch (e) {
        get().log('Camera switch failed: ' + (e instanceof Error ? e.message : e))
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
      const { label } = await rail.start({
        deviceId: deviceId || null,
        camera: camEnabled,
        videoEl: videoLoaded ? videoSource.el : null,
      })
      log('Input pipeline started: ' + label)
      set({ active: true })
    },

    stopPipeline() {
      rail.stop()
      set({ active: false, drawMode: 'off' })
    },

    async loadVideoFile(file) {
      const { log, videoLoop, videoRate } = get()
      // Hot-swap: a clip is already feeding a running pipeline → swap the
      // <video> src in place and keep the SAME captured track / output stream
      // alive (no rail restart, no WebRTC renegotiation, no black frame). The
      // source set (camera on/off) is unchanged here, so only the file differs.
      const hotSwap = get().videoLoaded && get().active
      try {
        if (hotSwap) {
          const meta = await videoSource.swapSource(file)
          videoSource.setLoop(videoLoop)
          videoSource.setRate(videoRate)
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
        log(`Video loaded: ${file.name} (${meta.width}x${meta.height}, ${fmtTime(meta.duration)})`)
        await restartSources()
      } catch (e) {
        log('Video load failed: ' + (e instanceof Error ? e.message : e))
      }
    },

    async unloadVideo() {
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

    setCompositeOrder(order) {
      set({ compositeOrder: order })
      rail.setComposite({ order })
    },
    setCompositeOpacity(opacity) {
      set({ compositeOpacity: opacity })
      rail.setComposite({ opacity })
    },
    setCompositeBlend(blend) {
      set({ compositeBlend: blend })
      rail.setComposite({ blend })
    },

    setDrawMode(mode) {
      set({ drawMode: mode })
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
