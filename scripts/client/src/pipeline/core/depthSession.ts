// Depth Anything V2 (small) depth estimation via onnxruntime-web on WebGPU.
//
// Runs inside a DEDICATED depth worker (depth.worker.ts), continuously
// (drop-and-replace) and off the compositing thread — mirroring the smooth
// transformers.js webgpu-realtime-depth demo: depth-only execution, as fast as
// the GPU allows, nothing else competing. ort owns its own GPUDevice (it can't
// reuse an external one — microsoft/onnxruntime#26107), so the depth map crosses
// back to the compositor via a small GPU->CPU readback (getData) per inference.
//
// ort is lazy-imported so the (multi-MB) runtime loads only when depth is on.
// Model + ort wasm are vendored under public/ (no CDN — LAN demos).

import type { InferenceSession, Tensor } from 'onnxruntime-web'

// 20*14 — a patch-multiple (DINOv2 patch size 14). Smaller = far fewer ViT tokens
// = much faster inference (tokens scale with (DIM/14)²) and cheaper CPU preprocess.
const DIM = 280
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
  // Running depth range (EMA) so the normalized brightness doesn't jump per frame.
  private runMin = 0
  private runMax = 0
  private haveRange = false

  /** Load the model on WebGPU (ort owns its own device). Returns null if WebGPU
   *  is unavailable or anything fails — depth then simply stays off. */
  static async create(): Promise<DepthSession | null> {
    if (typeof navigator === 'undefined' || !navigator.gpu) return null
    try {
      const ort = await import('onnxruntime-web/webgpu')
      ort.env.wasm.wasmPaths = '/onnx/' // vendored jsep wasm (mirrors /mediapipe/wasm)
      ort.env.wasm.numThreads = 1 // WebGPU EP needs no threads → no COOP/COEP requirement
      // Do NOT hand ort our adapter: the compositor already consumed it
      // (a GPUAdapter creates exactly one device), so ort's requestDevice() would
      // throw "adapter is consumed". Let ort request its OWN adapter+device — the
      // depth map crosses back to the compositor device via the getData readback
      // regardless (ort's device != the compositor's, by WebGPU design). The gpu
      // probe above is only a "is WebGPU usable at all" gate.
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
      // Running min/max (EMA): per-frame min-max makes the depth brightness jump
      // frame-to-frame (visible flicker); smoothing the range stabilizes it.
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
      // No per-pixel temporal smoothing: running continuously (off the composite
      // thread) keeps the map fresh, like the realtime demo. The range EMA above
      // is enough to stop brightness flicker. (The old Uint8 EMA also truncated
      // fractional steps → systematic dark drift; removed.)
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
