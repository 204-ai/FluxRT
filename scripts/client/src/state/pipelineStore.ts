// Input pipeline state: camera, mirror, marker, draw. Drives the module-level
// Rail; holds only renderable state.

import { create } from 'zustand'
import { inputVision, rail } from './runtime'

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
  showInputPreview: boolean

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
  disableCam(): void
  setDevice(deviceId: string): Promise<void>
  refreshCameras(): Promise<void>
  setMirror(on: boolean): void
  setShowInputPreview(on: boolean): void
  startPipeline(): Promise<void>
  stopPipeline(): void

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

export const usePipelineStore = create<PipelineState>((set, get) => ({
  camEnabled: false,
  devices: [],
  deviceId: '',
  mirror: false,
  active: false,
  showInputPreview: false,

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
      await get().startPipeline()
    } catch (e) {
      log('Camera start failed: ' + (e instanceof Error ? e.message : e))
      set({ camEnabled: false })
    }
  },

  disableCam() {
    set({
      camEnabled: false,
      mirror: false,
      markerEnabled: false,
      poseStatus: '',
    })
    rail.setMirror(false)
    rail.configureMarker({ enabled: false })
    void inputVision.release('marker')
    get().stopPipeline()
  },

  async setDevice(deviceId) {
    set({ deviceId })
    if (!get().camEnabled || !rail.active) return
    rail.stop()
    try {
      await get().startPipeline()
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

  setShowInputPreview(on) {
    set({ showInputPreview: on })
  },

  async startPipeline() {
    const { deviceId, log } = get()
    const { label } = await rail.start(deviceId || null)
    log('Camera pipeline started: ' + label)
    // Auto-enable the side-by-side input preview on the Output tab.
    set({ active: true, showInputPreview: true })
  },

  stopPipeline() {
    rail.stop()
    set({ active: false, showInputPreview: false, drawMode: 'off' })
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
}))
