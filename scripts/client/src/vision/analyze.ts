// Blendshapes/landmarks → semantic features. Ported from
// sense-human/src/vision/analyze.ts; MovementTracker is instance-scoped so
// each engine (input vs output source) tracks its own motion history.

import type {
  FaceLandmarkerResult,
  PoseLandmarkerResult,
  NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import type {
  Attention,
  BodyFeatures,
  Expression,
  FaceFeatures,
  HeadPose,
  HumanAnalysis,
} from './types'

function blendshapeMap(result: FaceLandmarkerResult): Map<string, number> {
  const map = new Map<string, number>()
  const categories = result.faceBlendshapes?.[0]?.categories ?? []
  for (const c of categories) map.set(c.categoryName, c.score)
  return map
}

/**
 * Extract yaw/pitch/roll (degrees) from the 4x4 facial transformation matrix.
 * Matrix data is column-major; rotation R decomposed as Ry(yaw)·Rx(pitch)·Rz(roll).
 */
function headPoseFromMatrix(d: number[] | Float32Array): HeadPose {
  const clamp = (v: number) => Math.max(-1, Math.min(1, v))
  const deg = (v: number) => (v * 180) / Math.PI
  // column-major: r[row][col] = d[col*4 + row]
  const r10 = d[1], r11 = d[5], r12 = d[9]
  const r02 = d[8], r22 = d[10]
  return {
    yaw: deg(Math.atan2(r02, r22)),
    pitch: deg(Math.asin(clamp(-r12))),
    roll: deg(Math.atan2(r10, r11)),
  }
}

function classifyExpression(bs: Map<string, number>): { expression: Expression; score: number } {
  const g = (n: string) => bs.get(n) ?? 0
  const smile = (g('mouthSmileLeft') + g('mouthSmileRight')) / 2
  const frown = (g('mouthFrownLeft') + g('mouthFrownRight') + g('browDownLeft') + g('browDownRight')) / 4
  const surprise = (g('browInnerUp') + g('eyeWideLeft') + g('eyeWideRight') + g('jawOpen')) / 4
  const squint = (g('eyeSquintLeft') + g('eyeSquintRight')) / 2
  const talking = g('jawOpen') * (1 - surprise)

  const candidates: Array<[Expression, number]> = [
    ['happy', smile],
    ['frowning', frown],
    ['surprised', surprise],
    ['squinting', squint * 0.8],
    ['talking', talking * 0.7],
  ]
  candidates.sort((a, b) => b[1] - a[1])
  const [expression, score] = candidates[0]
  if (score < 0.25) return { expression: 'neutral', score: 1 - score }
  return { expression, score }
}

function classifyAttention(head: HeadPose, blinking: boolean): Attention {
  const off = Math.abs(head.yaw) > 25 || head.pitch < -20 || head.pitch > 25
  if (off) return 'looking away'
  if (Math.abs(head.yaw) > 15 || blinking) return 'distracted'
  return 'engaged'
}

export function analyzeFace(result: FaceLandmarkerResult | null): FaceFeatures | null {
  if (!result || result.faceLandmarks.length === 0) return null
  const bs = blendshapeMap(result)
  const g = (n: string) => bs.get(n) ?? 0

  const matrixData = result.facialTransformationMatrixes?.[0]?.data
  const headPose: HeadPose = matrixData
    ? headPoseFromMatrix(matrixData)
    : { yaw: 0, pitch: 0, roll: 0 }

  const eyeBlinkLeft = g('eyeBlinkLeft')
  const eyeBlinkRight = g('eyeBlinkRight')
  const blinking = eyeBlinkLeft > 0.5 && eyeBlinkRight > 0.5
  const { expression, score } = classifyExpression(bs)

  const topBlendshapes = [...bs.entries()]
    .filter(([name]) => name !== '_neutral')
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([name, s]) => ({ name, score: s }))

  return {
    expression,
    expressionScore: score,
    smile: (g('mouthSmileLeft') + g('mouthSmileRight')) / 2,
    jawOpen: g('jawOpen'),
    browRaise: g('browInnerUp'),
    eyeBlinkLeft,
    eyeBlinkRight,
    blinking,
    headPose,
    attention: classifyAttention(headPose, blinking),
    topBlendshapes,
  }
}

// Pose landmark indices (BlazePose 33-point topology)
const NOSE = 0
const L_SHOULDER = 11, R_SHOULDER = 12
const L_WRIST = 15, R_WRIST = 16
const L_HIP = 23, R_HIP = 24

export class MovementTracker {
  private prev: NormalizedLandmark[] | null = null
  private energy = 0

  update(landmarks: NormalizedLandmark[]): number {
    if (this.prev && this.prev.length === landmarks.length) {
      let sum = 0
      for (let i = 0; i < landmarks.length; i++) {
        sum += Math.hypot(landmarks[i].x - this.prev[i].x, landmarks[i].y - this.prev[i].y)
      }
      const instant = sum / landmarks.length
      this.energy = this.energy * 0.9 + instant * 0.1
    }
    this.prev = landmarks
    return this.energy
  }

  reset() {
    this.prev = null
    this.energy = 0
  }
}

export function analyzeBody(
  result: PoseLandmarkerResult | null,
  tracker: MovementTracker,
): BodyFeatures | null {
  if (!result || result.landmarks.length === 0) {
    tracker.reset()
    return null
  }
  const lm = result.landmarks[0]
  const visible = (i: number) => (lm[i].visibility ?? 1) > 0.5

  const leftHandRaised = visible(L_WRIST) && visible(L_SHOULDER) && lm[L_WRIST].y < lm[L_SHOULDER].y
  const rightHandRaised = visible(R_WRIST) && visible(R_SHOULDER) && lm[R_WRIST].y < lm[R_SHOULDER].y

  const shoulderMidX = (lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2
  const hipMidX = visible(L_HIP) && visible(R_HIP) ? (lm[L_HIP].x + lm[R_HIP].x) / 2 : shoulderMidX
  const lean = shoulderMidX - hipMidX
  // landmarks are in source coords: +x = person's left side
  const leaning = lean > 0.04 ? 'right' : lean < -0.04 ? 'left' : 'centered'

  const shoulderTilt =
    (Math.atan2(lm[R_SHOULDER].y - lm[L_SHOULDER].y, lm[R_SHOULDER].x - lm[L_SHOULDER].x) * 180) /
      Math.PI -
    180

  const shoulderWidth = Math.abs(lm[L_SHOULDER].x - lm[R_SHOULDER].x)
  const noseToShoulderY = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2 - lm[NOSE].y
  let posture: BodyFeatures['posture'] = 'unknown'
  if (visible(NOSE) && visible(L_SHOULDER) && visible(R_SHOULDER) && shoulderWidth > 0.05) {
    // head sinking toward shoulder line relative to shoulder width = slouch
    posture = noseToShoulderY / shoulderWidth < 0.45 ? 'slouching' : 'upright'
  }

  const movementEnergy = tracker.update(lm)
  const activity: BodyFeatures['activity'] =
    movementEnergy < 0.002 ? 'still'
    : movementEnergy < 0.008 ? 'calm'
    : movementEnergy < 0.02 ? 'active'
    : 'very active'

  return {
    leftHandRaised,
    rightHandRaised,
    leaning,
    shoulderTilt: ((shoulderTilt % 360) + 540) % 360 - 180,
    movementEnergy,
    activity,
    posture,
  }
}

export function analyze(
  face: FaceLandmarkerResult | null,
  pose: PoseLandmarkerResult | null,
  tracker: MovementTracker,
  fps: number,
  inferenceMs: number,
): HumanAnalysis {
  const faceFeatures = analyzeFace(face)
  const bodyFeatures = analyzeBody(pose, tracker)
  return {
    present: faceFeatures !== null || bodyFeatures !== null,
    face: faceFeatures,
    body: bodyFeatures,
    fps,
    inferenceMs,
  }
}
