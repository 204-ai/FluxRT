// Depth Anything V2 (small) depth estimation via onnxruntime-web on WebGPU.
//
// IMPORTANT (verified against ort-web source + issue microsoft/onnxruntime#26107):
// ort-web does NOT let you reuse an external GPUDevice — it always derives its
// OWN device from the adapter, and WebGPU forbids using device-A resources on
// device-B. So we share only the ADAPTER (capability consistency) and accept
// that ort runs on its own device. The depth map crosses back to the compositor
// device via ONE GPU->CPU readback (getData) of a small map (~392x392), at the
// reduced cadence the worker drives — NOT per composite frame.
//
// ort is lazy-imported so the (multi-MB) runtime is only fetched when depth is
// actually enabled; with the flag off it never loads. Model + ort wasm are
// vendored under public/ (no CDN — LAN demos), like the MediaPipe assets.

import type { InferenceSession, Tensor } from 'onnxruntime-web'
import { getGpuContext } from './gpu'

const DIM = 392 // 28*14 — a patch-multiple (DINOv2 patch size 14); fixed to avoid recompiles
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]
const MODEL_URL = '/models/depth/depth_anything_v2_vits_fp16.onnx'

/** IEEE half (Uint16 bits) → float32. */
function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15
  const e = (h & 0x7c00) >> 10
  const f = h & 0x03ff
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}

export interface DepthResult {
  data: Uint8Array // normalized 0..255, 0=far 1(255)=near
  w: number
  h: number
}

export class DepthSession {
  private session: InferenceSession | null = null
  private inputName = ''
  private outputName = ''
  private pre = new OffscreenCanvas(DIM, DIM)
  private preCtx = this.pre.getContext('2d', { willReadFrequently: true })
  private input = new Float32Array(3 * DIM * DIM)
  private inputTensor: Tensor | null = null

  /** Load the model on WebGPU (sharing the compositor's adapter). Returns null
   *  if WebGPU is unavailable or anything fails — depth then simply stays off. */
  static async create(): Promise<DepthSession | null> {
    const gpu = await getGpuContext()
    if (!gpu) return null
    try {
      const ort = await import('onnxruntime-web/webgpu')
      ort.env.wasm.wasmPaths = '/onnx/' // vendored jsep wasm (mirrors /mediapipe/wasm)
      ort.env.wasm.numThreads = 1 // WebGPU EP needs no threads → no COOP/COEP requirement
      // ort ignores env.webgpu.device (#26107); the adapter is the only lever, and
      // ort makes its own device from it regardless. Set it for capability parity.
      ort.env.webgpu.adapter = gpu.adapter as unknown as GPUAdapter
      const s = new DepthSession()
      s.session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer', // output stays on ORT's device during run
      })
      s.inputName = s.session.inputNames[0] // expect 'pixel_values'
      s.outputName = s.session.outputNames[0] // expect 'predicted_depth'
      s.inputTensor = new ort.Tensor('float32', s.input, [1, 3, DIM, DIM])
      console.info('[depth] session ready', { in: s.inputName, out: s.outputName })
      return s
    } catch (e) {
      console.error('[depth] session create failed:', e)
      return null
    }
  }

  /** Run one inference on a source frame/bitmap → normalized 8-bit depth map.
   *  Crosses the ORT-device → CPU boundary once via getData (required: ORT's
   *  device != the compositor's). null on failure. */
  async run(src: VideoFrame | ImageBitmap): Promise<DepthResult | null> {
    if (!this.session || !this.inputTensor || !this.preCtx) return null
    try {
      // CPU preprocess: cover-fit into DIM×DIM, RGB, NCHW planar, ImageNet-normalize.
      this.preCtx.drawImage(src as CanvasImageSource, 0, 0, DIM, DIM)
      const px = this.preCtx.getImageData(0, 0, DIM, DIM).data
      const plane = DIM * DIM
      for (let i = 0; i < plane; i++) {
        this.input[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0]
        this.input[plane + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1]
        this.input[2 * plane + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2]
      }
      const outputs = await this.session.run({ [this.inputName]: this.inputTensor })
      const out = outputs[this.outputName]
      const raw = (await out.getData(true)) as Float32Array | Uint16Array
      const [h, w] = out.dims.slice(-2) as [number, number]
      // predicted_depth is unbounded relative inverse depth → per-frame min-max.
      const n = w * h
      const u8 = new Uint8Array(n)
      const isHalf = raw instanceof Uint16Array
      let mn = Infinity
      let mx = -Infinity
      const depth = new Float32Array(n)
      for (let i = 0; i < n; i++) {
        const v = isHalf ? halfToFloat(raw[i]) : (raw[i] as number)
        depth[i] = v
        if (v < mn) mn = v
        if (v > mx) mx = v
      }
      const inv = mx > mn ? 255 / (mx - mn) : 0
      for (let i = 0; i < n; i++) u8[i] = (depth[i] - mn) * inv
      return { data: u8, w, h }
    } catch (e) {
      console.error('[depth] run failed:', e)
      return null
    }
  }

  dispose(): void {
    void this.session?.release()
    this.session = null
    this.inputTensor = null
  }
}
