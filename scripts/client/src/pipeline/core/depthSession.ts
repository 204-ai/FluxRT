// Depth Anything V2 (small) depth estimation via onnxruntime-web on WebGPU.
//
// Runs inside a DEDICATED depth worker (depth.worker.ts), continuously
// (drop-and-replace) and off the compositing thread — mirroring the smooth
// transformers.js webgpu-realtime-depth demo: depth-only execution, as fast as
// the GPU allows. ort owns its own GPUDevice (it can't reuse an external one —
// microsoft/onnxruntime#26107), so the depth map crosses back to the compositor
// via a small GPU->CPU readback (getData) per inference.
//
// Input size is configurable (the demo's "Image size" slider, default 518). It
// MUST be a multiple of 14 (DINOv2 patch size). Bigger = sharper depth + slower
// (e.g. ~504-518 ≈ the realtime demo's quality at ~7fps; 280 is faster + blurrier).

import type { InferenceSession, Tensor } from 'onnxruntime-web'

type Ort = typeof import('onnxruntime-web/webgpu')

const DEFAULT_DIM = 518 // 37*14 — matches the model's native config + the demo's quality
const MEAN = [0.485, 0.456, 0.406]
const STD = [0.229, 0.224, 0.225]
const MODEL_URL = '/models/depth/depth_anything_v2_vits_fp16.onnx'

/** Snap to a multiple of 14, clamped to a sane range. */
export function snapDim(n: number): number {
  return Math.max(140, Math.min(700, Math.round(n / 14) * 14))
}

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
  data: Uint8Array // normalized 0..255, 0=far 255=near
  w: number
  h: number
}

export class DepthSession {
  private session: InferenceSession | null = null
  private ort: Ort | null = null
  private inputName = ''
  private outputName = ''
  private dim = DEFAULT_DIM
  private pre!: OffscreenCanvas
  private preCtx!: OffscreenCanvasRenderingContext2D
  private input!: Float32Array
  private inputTensor!: Tensor
  // Running depth range (EMA) so the normalized brightness doesn't jump per frame.
  private runMin = 0
  private runMax = 0
  private haveRange = false

  /** Load the model on WebGPU (ort owns its own device). Returns null if WebGPU
   *  is unavailable or anything fails — depth then simply stays off. */
  static async create(dim = DEFAULT_DIM): Promise<DepthSession | null> {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null
    try {
      const ort = await import('onnxruntime-web/webgpu')
      ort.env.wasm.wasmPaths = '/onnx/' // vendored jsep wasm (mirrors /mediapipe/wasm)
      ort.env.wasm.numThreads = 1 // WebGPU EP needs no threads → no COOP/COEP requirement
      const s = new DepthSession()
      s.ort = ort
      s.session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['webgpu'],
        preferredOutputLocation: 'gpu-buffer',
      })
      s.inputName = s.session.inputNames[0] // expect 'pixel_values'
      s.outputName = s.session.outputNames[0] // expect 'predicted_depth'
      s.setSize(dim)
      console.info('[depth] session ready', { in: s.inputName, out: s.outputName, dim: s.dim })
      return s
    } catch (e) {
      console.error('[depth] session create failed:', e)
      return null
    }
  }

  /** (Re)allocate the preprocess canvas + input tensor for a new input size. */
  setSize(dim: number): void {
    const d = snapDim(dim)
    if (this.pre && d === this.dim) return
    this.dim = d
    this.pre = new OffscreenCanvas(d, d)
    this.preCtx = this.pre.getContext('2d', { willReadFrequently: true })!
    this.input = new Float32Array(3 * d * d)
    if (this.ort) this.inputTensor = new this.ort.Tensor('float32', this.input, [1, 3, d, d])
  }

  /** Run one inference on a source frame/bitmap → normalized 8-bit depth map.
   *  Crosses the ORT-device → CPU boundary once via getData. null on failure. */
  async run(src: VideoFrame | ImageBitmap, mirror = false): Promise<DepthResult | null> {
    if (!this.session || !this.inputTensor || !this.preCtx) return null
    try {
      const dim = this.dim
      // CPU preprocess: cover-fit into dim×dim, RGB, NCHW planar, ImageNet-normalize.
      // Apply the source layer's selfie mirror so the depth map aligns L-R with the
      // mirrored composite (depth runs on the raw base, which is pre-mirror).
      if (mirror) {
        this.preCtx.save()
        this.preCtx.translate(dim, 0)
        this.preCtx.scale(-1, 1)
        this.preCtx.drawImage(src as CanvasImageSource, 0, 0, dim, dim)
        this.preCtx.restore()
      } else {
        this.preCtx.drawImage(src as CanvasImageSource, 0, 0, dim, dim)
      }
      const px = this.preCtx.getImageData(0, 0, dim, dim).data
      const plane = dim * dim
      for (let i = 0; i < plane; i++) {
        this.input[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0]
        this.input[plane + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1]
        this.input[2 * plane + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2]
      }
      const outputs = await this.session.run({ [this.inputName]: this.inputTensor })
      const out = outputs[this.outputName]
      const raw = (await out.getData(true)) as Float32Array | Uint16Array
      const [h, w] = out.dims.slice(-2) as [number, number]
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
      // Running min/max (EMA): per-frame min-max makes brightness flicker; smoothing
      // the range stabilizes it without per-pixel temporal lag.
      if (!this.haveRange) {
        this.runMin = mn
        this.runMax = mx
        this.haveRange = true
      } else {
        const a = 0.15
        this.runMin += (mn - this.runMin) * a
        this.runMax += (mx - this.runMax) * a
      }
      const lo = this.runMin
      const inv = this.runMax > lo ? 255 / (this.runMax - lo) : 0
      for (let i = 0; i < n; i++) {
        const t = (depth[i] - lo) * inv
        u8[i] = t < 0 ? 0 : t > 255 ? 255 : t
      }
      return { data: u8, w, h }
    } catch (e) {
      console.error('[depth] run failed:', e)
      return null
    }
  }

  dispose(): void {
    void this.session?.release()
    this.session = null
    this.inputTensor = null as unknown as Tensor
  }
}
