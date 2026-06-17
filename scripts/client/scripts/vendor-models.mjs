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

// onnxruntime-web WebGPU (jsep) runtime — served from /onnx/ (ort.env.wasm.wasmPaths).
const ortSrc = path.join(root, 'node_modules/onnxruntime-web/dist')
const ortDst = path.join(root, 'public/onnx')
const ORT_FILES = ['ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.jsep.mjs']

// Depth Anything V2 small — fp16 ONNX (~50MB) for the WebGPU depth pass.
const depthDst = path.join(root, 'public/models/depth')
const DEPTH_MODELS = {
  'depth_anything_v2_vits_fp16.onnx':
    'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx',
}

await mkdir(wasmDst, { recursive: true })
await mkdir(modelsDst, { recursive: true })
await cp(wasmSrc, wasmDst, { recursive: true })
console.log('wasm copied ->', wasmDst)

// ort jsep runtime
await mkdir(ortDst, { recursive: true })
for (const name of ORT_FILES) {
  await cp(path.join(ortSrc, name), path.join(ortDst, name))
}
console.log('ort jsep copied ->', ortDst)

await mkdir(depthDst, { recursive: true })

async function download(name, url, dst) {
  const out = path.join(dst, name)
  if (existsSync(out)) {
    console.log('exists, skipping', name)
    return
  }
  const res = await fetch(url)
  if (!res.ok) throw new Error(`download failed ${name}: ${res.status}`)
  await pipeline(Readable.fromWeb(res.body), createWriteStream(out))
  console.log('downloaded', name)
}

for (const [name, url] of Object.entries(MODELS)) await download(name, url, modelsDst)
for (const [name, url] of Object.entries(DEPTH_MODELS)) await download(name, url, depthDst)
