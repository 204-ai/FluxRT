// MediaPipe engine — ported from sense-human/src/vision/visionEngine.ts.
// Differences: runs in a worker, takes ImageBitmap input with caller-owned
// timestamps (the rail's tap controls cadence), models/wasm are vendored
// locally (LAN demos must not depend on a CDN), and face/pose are
// individually toggleable (the hand marker only needs pose).

import {
  FilesetResolver,
  FaceLandmarker,
  PoseLandmarker,
  ImageSegmenter,
} from '@mediapipe/tasks-vision'
import { analyze, MovementTracker } from './analyze'
import type { SegmentationMask, VisionResult } from './types'

export interface EngineConfig {
  wasmBase: string
  faceModel: string
  poseModel: string
  segmentModel: string
  face: boolean
  pose: boolean
  segment: boolean
}

export const DEFAULT_ENGINE_PATHS = {
  wasmBase: '/mediapipe/wasm',
  faceModel: '/mediapipe/models/face_landmarker.task',
  poseModel: '/mediapipe/models/pose_landmarker_lite.task',
  segmentModel: '/mediapipe/models/selfie_segmenter.tflite',
}

/**
 * MediaPipe's wasm loader uses importScripts(), which module workers don't
 * support — the glue then loads module-scoped and the global ModuleFactory
 * check fails ("ModuleFactory not set.", google-ai-edge/mediapipe#5257).
 * Preload the glue with indirect eval so its `var ModuleFactory` lands on
 * the worker global scope. MediaPipe also CLEARS self.ModuleFactory after
 * every task creation, so the cached factory must be restored before EACH
 * createFromOptions call, not just once. SIMD build first, nosimd fallback.
 */
let cachedModuleFactory: unknown = null

async function ensureWasmGlue(wasmBase: string): Promise<void> {
  const g = globalThis as Record<string, unknown>
  if (!cachedModuleFactory) {
    for (const name of ['vision_wasm_internal.js', 'vision_wasm_nosimd_internal.js']) {
      try {
        const res = await fetch(`${wasmBase}/${name}`)
        if (!res.ok) continue
        const src = await res.text()
        ;(0, eval)(src) // indirect eval: global scope, sets self.ModuleFactory
        if (g.ModuleFactory) {
          cachedModuleFactory = g.ModuleFactory
          break
        }
      } catch {
        /* try next variant */
      }
    }
  }
  if (cachedModuleFactory) g.ModuleFactory = cachedModuleFactory
}

export class VisionEngine {
  private face: FaceLandmarker | null = null
  private pose: PoseLandmarker | null = null
  private seg: ImageSegmenter | null = null
  private tracker = new MovementTracker()
  private fpsWindow: number[] = []
  private lastTs = 0

  async init(cfg: EngineConfig, onStatus: (msg: string) => void): Promise<void> {
    onStatus('loading vision runtime…')
    await ensureWasmGlue(cfg.wasmBase)
    const fileset = await FilesetResolver.forVisionTasks(cfg.wasmBase)

    if (cfg.face) {
      onStatus('loading face landmarker…')
      await ensureWasmGlue(cfg.wasmBase)
      this.face = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: cfg.faceModel, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      })
    }
    if (cfg.pose) {
      onStatus('loading pose landmarker…')
      await ensureWasmGlue(cfg.wasmBase)
      this.pose = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: cfg.poseModel, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
      })
    }
    if (cfg.segment) {
      onStatus('loading image segmenter…')
      await ensureWasmGlue(cfg.wasmBase)
      this.seg = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: cfg.segmentModel, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      })
    }
    onStatus('')
  }

  detect(input: ImageBitmap, tsMs: number): VisionResult | null {
    // detectForVideo timestamps must increase monotonically.
    if (tsMs <= this.lastTs) tsMs = this.lastTs + 1
    this.lastTs = tsMs

    const t0 = performance.now()
    const faceResult = this.face ? this.face.detectForVideo(input, tsMs) : null
    const poseResult = this.pose ? this.pose.detectForVideo(input, tsMs) : null
    const segmentation = this.seg ? this.runSegment(input, tsMs) : null
    const inferenceMs = performance.now() - t0

    const now = performance.now()
    this.fpsWindow.push(now)
    while (this.fpsWindow.length > 0 && now - this.fpsWindow[0] > 1000) this.fpsWindow.shift()

    return {
      analysis: analyze(faceResult, poseResult, this.tracker, this.fpsWindow.length, inferenceMs),
      faceLandmarks: faceResult?.faceLandmarks ?? [],
      poseLandmarks: poseResult?.landmarks ?? [],
      segmentation,
      tsMs,
    }
  }

  /** Copy the foreground-confidence mask into a plain Uint8Array, then close
   *  the result so the GPU-backed masks are freed every frame (MediaPipe holds
   *  them until close() — leaking them stalls the GPU). The selfie segmenter
   *  emits a single confidence mask where higher = more likely foreground. */
  private runSegment(input: ImageBitmap, tsMs: number): SegmentationMask | null {
    const res = this.seg!.segmentForVideo(input, tsMs)
    try {
      const mask = res.confidenceMasks?.[0]
      if (!mask) return null
      const f = mask.getAsFloat32Array()
      const data = new Uint8Array(f.length)
      for (let i = 0; i < f.length; i++) data[i] = (f[i] * 255) | 0
      return { data, width: mask.width, height: mask.height }
    } finally {
      res.close()
    }
  }

  close(): void {
    this.face?.close()
    this.pose?.close()
    this.seg?.close()
    this.face = null
    this.pose = null
    this.seg = null
  }
}
