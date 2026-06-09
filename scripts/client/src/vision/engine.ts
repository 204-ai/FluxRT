// MediaPipe engine — ported from sense-human/src/vision/visionEngine.ts.
// Differences: runs in a worker, takes ImageBitmap input with caller-owned
// timestamps (the rail's tap controls cadence), models/wasm are vendored
// locally (LAN demos must not depend on a CDN), and face/pose are
// individually toggleable (the hand marker only needs pose).

import {
  FilesetResolver,
  FaceLandmarker,
  PoseLandmarker,
} from '@mediapipe/tasks-vision'
import { analyze, MovementTracker } from './analyze'
import type { VisionResult } from './types'

export interface EngineConfig {
  wasmBase: string
  faceModel: string
  poseModel: string
  face: boolean
  pose: boolean
}

export const DEFAULT_ENGINE_PATHS = {
  wasmBase: '/mediapipe/wasm',
  faceModel: '/mediapipe/models/face_landmarker.task',
  poseModel: '/mediapipe/models/pose_landmarker_lite.task',
}

export class VisionEngine {
  private face: FaceLandmarker | null = null
  private pose: PoseLandmarker | null = null
  private tracker = new MovementTracker()
  private fpsWindow: number[] = []
  private lastTs = 0

  async init(cfg: EngineConfig, onStatus: (msg: string) => void): Promise<void> {
    onStatus('loading vision runtime…')
    const fileset = await FilesetResolver.forVisionTasks(cfg.wasmBase)

    if (cfg.face) {
      onStatus('loading face landmarker…')
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
      this.pose = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: cfg.poseModel, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numPoses: 1,
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
    const inferenceMs = performance.now() - t0

    const now = performance.now()
    this.fpsWindow.push(now)
    while (this.fpsWindow.length > 0 && now - this.fpsWindow[0] > 1000) this.fpsWindow.shift()

    return {
      analysis: analyze(faceResult, poseResult, this.tracker, this.fpsWindow.length, inferenceMs),
      faceLandmarks: faceResult?.faceLandmarks ?? [],
      poseLandmarks: poseResult?.landmarks ?? [],
      tsMs,
    }
  }

  close(): void {
    this.face?.close()
    this.pose?.close()
    this.face = null
    this.pose = null
  }
}
