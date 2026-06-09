// Ported from sense-human/src/vision/types.ts. Landmarks are kept as plain
// serializable arrays so results cross the vision-worker boundary intact.

export interface Landmark {
  x: number
  y: number
  z?: number
  visibility?: number
}

export interface HeadPose {
  /** degrees; + = looking to their left (camera right) */
  yaw: number
  /** degrees; + = looking up */
  pitch: number
  /** degrees; + = head tilted */
  roll: number
}

export type Expression =
  | 'neutral'
  | 'happy'
  | 'surprised'
  | 'frowning'
  | 'squinting'
  | 'talking'

export type Attention = 'engaged' | 'looking away' | 'distracted'

export type ActivityLevel = 'still' | 'calm' | 'active' | 'very active'

export interface FaceFeatures {
  expression: Expression
  /** 0..1 confidence of the dominant expression */
  expressionScore: number
  smile: number
  jawOpen: number
  browRaise: number
  eyeBlinkLeft: number
  eyeBlinkRight: number
  blinking: boolean
  headPose: HeadPose
  attention: Attention
  topBlendshapes: Array<{ name: string; score: number }>
}

export interface BodyFeatures {
  leftHandRaised: boolean
  rightHandRaised: boolean
  leaning: 'left' | 'right' | 'centered'
  shoulderTilt: number
  /** smoothed per-frame landmark displacement, 0..~1 */
  movementEnergy: number
  activity: ActivityLevel
  posture: 'upright' | 'slouching' | 'unknown'
}

export interface HumanAnalysis {
  present: boolean
  face: FaceFeatures | null
  body: BodyFeatures | null
  fps: number
  inferenceMs: number
}

/** Result payload posted from the vision worker per processed frame. */
export interface VisionResult {
  analysis: HumanAnalysis
  faceLandmarks: Landmark[][]
  poseLandmarks: Landmark[][]
  tsMs: number
}
