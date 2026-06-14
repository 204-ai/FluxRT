// Vendors the MediaPipe WASM runtime + task models into public/ so the
// client works on a LAN with no internet (CDN is a liability for demos).
// Run once after `yarn install`: `yarn vendor`.
import { cp, mkdir } from 'node:fs/promises'
import { createWriteStream, existsSync } from 'node:fs'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const wasmSrc = path.join(root, 'node_modules/@mediapipe/tasks-vision/wasm')
const wasmDst = path.join(root, 'public/mediapipe/wasm')
const modelsDst = path.join(root, 'public/mediapipe/models')

const MODELS = {
  'face_landmarker.task':
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
  'pose_landmarker_lite.task':
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}

await mkdir(wasmDst, { recursive: true })
await mkdir(modelsDst, { recursive: true })
await cp(wasmSrc, wasmDst, { recursive: true })
console.log('wasm copied ->', wasmDst)

for (const [name, url] of Object.entries(MODELS)) {
  const dst = path.join(modelsDst, name)
  if (existsSync(dst)) {
    console.log('exists, skipping', name)
    continue
  }
  const res = await fetch(url)
  if (!res.ok) throw new Error(`download failed ${name}: ${res.status}`)
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dst))
  console.log('downloaded', name)
}
