// Display-only overlay renderer — face mesh, pose skeleton, tracking box +
// expression label. Ported from sense-human/src/vision/draw.ts. Detection runs
// on the already-mirrored composite (input) / un-mirrored remote video (output),
// so landmarks are already in the space the overlay renders into — drawn 1:1
// with no flip.

import {
  DrawingUtils,
  FaceLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import type { Landmark, VisionResult } from './types'

const FACE_MESH_COLOR = 'rgba(0, 255, 170, 0.18)'
const FACE_FEATURE_COLOR = 'rgba(0, 255, 170, 0.85)'
const POSE_COLOR = 'rgba(80, 170, 255, 0.9)'
const BOX_COLOR = 'rgba(0, 255, 170, 0.9)'

export function drawOverlay(canvas: HTMLCanvasElement, result: VisionResult): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const utils = new DrawingUtils(ctx)

  // worker results always carry z; the serializable type just marks it optional
  for (const lms of result.poseLandmarks) {
    const landmarks = lms as NormalizedLandmark[]
    utils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      color: POSE_COLOR,
      lineWidth: 3,
    })
    utils.drawLandmarks(landmarks, { color: POSE_COLOR, radius: 3, lineWidth: 1 })
  }

  for (const lms of result.faceLandmarks) {
    const landmarks = lms as NormalizedLandmark[]
    utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: FACE_MESH_COLOR,
      lineWidth: 0.5,
    })
    utils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_CONTOURS, {
      color: FACE_FEATURE_COLOR,
      lineWidth: 1.5,
    })
  }

  const face = result.faceLandmarks[0]
  if (face) drawFaceBox(ctx, face, canvas.width, canvas.height, result)
}

function drawFaceBox(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  w: number,
  h: number,
  result: VisionResult,
): void {
  let minX = 1, minY = 1, maxX = 0, maxY = 0
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
  }
  const pad = 0.03
  const x = (minX - pad) * w
  const y = (minY - pad) * h
  const bw = (maxX - minX + pad * 2) * w
  const bh = (maxY - minY + pad * 2) * h

  ctx.strokeStyle = BOX_COLOR
  ctx.lineWidth = 2
  const corner = Math.min(bw, bh) * 0.18
  for (const [cx, cy, dx, dy] of [
    [x, y, 1, 1],
    [x + bw, y, -1, 1],
    [x, y + bh, 1, -1],
    [x + bw, y + bh, -1, -1],
  ] as const) {
    ctx.beginPath()
    ctx.moveTo(cx + dx * corner, cy)
    ctx.lineTo(cx, cy)
    ctx.lineTo(cx, cy + dy * corner)
    ctx.stroke()
  }

  const face = result.analysis.face
  if (face) {
    const label = `${face.expression} ${(face.expressionScore * 100).toFixed(0)}%`
    ctx.font = 'bold 16px ui-monospace, monospace'
    const metrics = ctx.measureText(label)
    const ly = Math.max(22, y - 10)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.65)'
    ctx.fillRect(x, ly - 18, metrics.width + 12, 24)
    ctx.fillStyle = BOX_COLOR
    ctx.fillText(label, x + 6, ly)
  }
}
