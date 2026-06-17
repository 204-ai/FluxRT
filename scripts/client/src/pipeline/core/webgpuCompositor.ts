// WebGPU compositor — a GPU-resident alternative to the 2D-canvas `Compositor`,
// behind the SAME public surface (setComposite / drawComposite / bus) so the
// streams pipeline worker can swap to it behind a capability probe. Each layer's
// frame stays on the GPU: a live VideoFrame is imported zero-copy via
// importExternalTexture and composited into an rgba16float accumulator, blitted
// to a GPUCanvasContext whose canvas feeds `new VideoFrame(canvas)` → MSTG.
//
// Compositing is a ping-pong over two rgba16float accumulators, back-to-front.
// At each stack position we either:
//   • SOURCE layer — Pass A renders the transformed frame into a scratch texture
//     (opacity as alpha); Pass B blends (accumulator, scratch) → other
//     accumulator. screen/multiply/difference are computed in WGSL.
//   • EFFECT layer 'shader' — a full-screen pass applies a CSS-style colour
//     filter (hue-rotate/saturate/invert/grayscale/brightness/contrast/sepia,
//     composed into one 4×4 matrix) to the accumulator (everything below),
//     mixed by the layer opacity.
//
// NOT YET PORTED: the `blur` filter (separable Gaussian), and the `marker` /
// `drawLayer` vector effects (best done as a 2D overlay texture). Those effect
// layers are skipped here. Non-VideoFrame layer sources are also skipped.

import { AnalyzerBus } from './bus'
import type {
  BlendMode,
  CanvasEffect,
  Composite,
  CompositeOp,
  Ctx2D,
  EffectInit,
  FrameInfo,
  LayerId,
  LayerOptions,
} from './types'
import { applyCompositeOp, defaultComposite, layerDrawRects } from './types'
import { createEffect } from '../effects/registry'
import { depthModeNum } from '../effects/depthStub'
import type { GpuContext } from './gpu'
import type { FrameMap } from './compositor'

const UNIFORM_STRIDE = 256 // ≥ minUniformBufferOffsetAlignment; one slice per layer
const LAYER_UNIFORM_BYTES = 48 // 3×vec4<f32>
const FX_UNIFORM_BYTES = 80 // mat4x4<f32> + vec4<f32>
const BLUR_UNIFORM_BYTES = 32 // 2×vec4<f32>
const MAX_LAYERS = 16
const ACCUM_FORMAT: GPUTextureFormat = 'rgba16float'

const BLEND_NUM: Record<BlendMode, number> = { normal: 0, screen: 1, multiply: 2, difference: 3 }
// Column-major identity — fed to the colour-matrix uniform when a shader is
// blur-only (no colour change), so the fxmix opacity field is still valid.
const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

const WGSL = /* wgsl */ `
struct Layer {
  dst: vec4<f32>,    // ndc x0, ytop, x1, ybot
  uv:  vec4<f32>,    // u0, v0, u1, v1
  params: vec4<f32>, // opacity, mirror(0/1), blendMode, _
};
struct Fx {
  cmat: mat4x4<f32>, // colour matrix (3×3 + bias in last column) applied to vec4(rgb,1)
  params: vec4<f32>, // opacity, _, _, _
};
@group(0) @binding(0) var<uniform> L: Layer;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var srcTex: texture_external;
@group(0) @binding(3) var accumTex: texture_2d<f32>;
@group(0) @binding(4) var layerTex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> FX: Fx;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

// ---- Pass A: transformed layer (external texture) → scratch -----------------
@vertex
fn layer_vs(@builtin(vertex_index) vid: u32) -> VSOut {
  let xs = array<f32, 4>(L.dst.x, L.dst.z, L.dst.x, L.dst.z);
  let ys = array<f32, 4>(L.dst.y, L.dst.y, L.dst.w, L.dst.w);
  var u0 = L.uv.x; var u1 = L.uv.z;
  if (L.params.y > 0.5) { let t = u0; u0 = u1; u1 = t; } // selfie mirror = flip U
  let us = array<f32, 4>(u0, u1, u0, u1);
  let vs = array<f32, 4>(L.uv.y, L.uv.y, L.uv.w, L.uv.w);
  var o: VSOut;
  o.pos = vec4<f32>(xs[vid], ys[vid], 0.0, 1.0);
  o.uv = vec2<f32>(us[vid], vs[vid]);
  return o;
}

@fragment
fn layer_fs(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSampleBaseClampToEdge(srcTex, samp, in.uv);
  return vec4<f32>(c.rgb, L.params.x); // alpha carries this layer's opacity (its quad mask)
}

// ---- full-screen vertex (blend / fx / blit) ---------------------------------
@vertex
fn fullscreen_vs(@builtin(vertex_index) vid: u32) -> VSOut {
  let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4<f32>(p[vid], 0.0, 1.0);
  o.uv = vec2<f32>(p[vid].x * 0.5 + 0.5, 0.5 - p[vid].y * 0.5); // y-flip → uv(0,0)=top-left
  return o;
}

// ---- Pass B: blend(accum, layerScratch) → accum' ----------------------------
fn blend_rgb(mode: f32, b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
  if (mode < 0.5) { return s; }                          // normal
  if (mode < 1.5) { return 1.0 - (1.0 - b) * (1.0 - s); } // screen
  if (mode < 2.5) { return b * s; }                       // multiply
  return abs(b - s);                                      // difference
}

@fragment
fn blend_fs(in: VSOut) -> @location(0) vec4<f32> {
  let b = textureSampleLevel(accumTex, samp, in.uv, 0.0);
  let s = textureSampleLevel(layerTex, samp, in.uv, 0.0);
  let blended = blend_rgb(L.params.z, b.rgb, s.rgb);
  return vec4<f32>(mix(b.rgb, blended, s.a), 1.0); // s.a = layer opacity × quad coverage
}

// ---- effect layer: colour filter over the accumulator (everything below) ----
@fragment
fn fx_fs(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(accumTex, samp, in.uv, 0.0);
  let f = (FX.cmat * vec4<f32>(c.rgb, 1.0)).rgb;
  return vec4<f32>(mix(c.rgb, clamp(f, vec3<f32>(0.0), vec3<f32>(1.0)), FX.params.x), 1.0);
}

// ---- blur (separable Gaussian) + full-strength colour + mix ----------------
struct Blur { p0: vec4<f32>, p1: vec4<f32> }; // p0: texelX,texelY,dirX,dirY ; p1.x: radius(px)
@group(0) @binding(6) var<uniform> BL: Blur;
@group(0) @binding(7) var blurredTex: texture_2d<f32>;

// colour matrix at full strength (no opacity mix) — feeds the blur chain
@fragment
fn fxfull_fs(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(accumTex, samp, in.uv, 0.0);
  let f = (FX.cmat * vec4<f32>(c.rgb, 1.0)).rgb;
  return vec4<f32>(clamp(f, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// one separable-Gaussian direction over accumTex (binding 3)
@fragment
fn blur_fs(in: VSOut) -> @location(0) vec4<f32> {
  let texel = BL.p0.xy;
  let dir = BL.p0.zw;
  let radius = BL.p1.x;
  let sigma = max(radius * 0.5, 1.0);
  let r = i32(ceil(radius));
  var sum = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i = -r; i <= r; i = i + 1) {
    let fi = f32(i);
    let w = exp(-(fi * fi) / (2.0 * sigma * sigma));
    sum = sum + textureSampleLevel(accumTex, samp, in.uv + dir * texel * fi, 0.0).rgb * w;
    wsum = wsum + w;
  }
  return vec4<f32>(sum / wsum, 1.0);
}

// mix original (binding 3) with the fully-filtered result (binding 7) by opacity
@fragment
fn fxmix_fs(in: VSOut) -> @location(0) vec4<f32> {
  let orig = textureSampleLevel(accumTex, samp, in.uv, 0.0).rgb;
  let filt = textureSampleLevel(blurredTex, samp, in.uv, 0.0).rgb;
  return vec4<f32>(mix(orig, filt, FX.params.x), 1.0);
}

// ---- global effects overlay (marker / drawLayer) over the accumulator -------
@group(0) @binding(8) var overlayTex: texture_2d<f32>;
@fragment
fn overlay_fs(in: VSOut) -> @location(0) vec4<f32> {
  let b = textureSampleLevel(accumTex, samp, in.uv, 0.0).rgb;
  let o = textureSampleLevel(overlayTex, samp, in.uv, 0.0); // straight alpha
  return vec4<f32>(mix(b, o.rgb, o.a), 1.0);
}

// ---- depth effect (samples a GPU depth map produced by the ort-web session) -
struct Depth { params: vec4<f32> }; // strength, mode(0 fog/1 replace/2 mask), near, far
@group(0) @binding(9) var depthTex: texture_2d<f32>;
@group(0) @binding(10) var<uniform> DU: Depth;
@fragment
fn depth_fs(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(accumTex, samp, in.uv, 0.0).rgb;
  let d = textureSampleLevel(depthTex, samp, in.uv, 0.0).r; // 0=far, 1=near
  let m = DU.params.y;
  var outc = mix(vec3<f32>(0.05, 0.06, 0.1), c, d);  // mode 0 fog: far → fog colour
  if (m >= 0.5 && m < 1.5) { outc = vec3<f32>(d); }   // mode 1 replace: show depth map
  else if (m >= 1.5) { outc = c * step(0.4, d); }     // mode 2 mask: keep near, cut far
  return vec4<f32>(mix(c, outc, DU.params.x), 1.0);
}

// ---- depth temporal ease: glide the displayed depth toward the latest raw map
// so it doesn't step between (low-fps) inferences. binding 3 = previous smoothed,
// binding 9 = latest raw; output r8.
@fragment
fn depth_ease_fs(in: VSOut) -> @location(0) vec4<f32> {
  let prev = textureSampleLevel(accumTex, samp, in.uv, 0.0).r;
  let target = textureSampleLevel(depthTex, samp, in.uv, 0.0).r;
  return vec4<f32>(mix(prev, target, 0.18), 0.0, 0.0, 1.0);
}

// ---- final blit: accum → canvas --------------------------------------------
@fragment
fn blit_fs(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(textureSampleLevel(accumTex, samp, in.uv, 0.0).rgb, 1.0);
}
`

function isVideoFrame(src: unknown): src is VideoFrame {
  return typeof VideoFrame !== 'undefined' && src instanceof VideoFrame
}

// --- CSS colour-filter → 3×3 matrix + bias ---------------------------------
type Mat3 = { m: number[]; b: number[] } // m: row-major 3×3, b: rgb bias
const IDENT: Mat3 = { m: [1, 0, 0, 0, 1, 0, 0, 0, 1], b: [0, 0, 0] }

function mul3(a: number[], b: number[]): number[] {
  const r = new Array(9).fill(0)
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]
  return r
}
function mulVec3(m: number[], v: number[]): number[] {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ]
}

/** Parse a single CSS filter function into a colour transform. Unsupported
 *  functions (blur, opacity, drop-shadow, …) return identity. */
function filterMatrix(fn: string, arg: string): Mat3 {
  const num = (def: number) => {
    const t = arg.trim()
    if (t === '') return def
    if (t.endsWith('%')) return parseFloat(t) / 100
    return parseFloat(t)
  }
  switch (fn) {
    case 'brightness': {
      const v = num(1)
      return { m: [v, 0, 0, 0, v, 0, 0, 0, v], b: [0, 0, 0] }
    }
    case 'contrast': {
      const v = num(1)
      const o = 0.5 * (1 - v)
      return { m: [v, 0, 0, 0, v, 0, 0, 0, v], b: [o, o, o] }
    }
    case 'invert': {
      const v = num(1)
      const s = 1 - 2 * v
      return { m: [s, 0, 0, 0, s, 0, 0, 0, s], b: [v, v, v] }
    }
    case 'saturate': {
      const s = num(1)
      return {
        m: [
          0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s,
          0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s,
          0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s,
        ],
        b: [0, 0, 0],
      }
    }
    case 'grayscale': {
      const g = Math.min(1, num(1))
      const k = 1 - g
      const lr = 0.2126, lg = 0.7152, lb = 0.0722
      return {
        m: [
          k + g * lr, g * lg, g * lb,
          g * lr, k + g * lg, g * lb,
          g * lr, g * lg, k + g * lb,
        ],
        b: [0, 0, 0],
      }
    }
    case 'sepia': {
      const s = Math.min(1, num(1))
      const k = 1 - s
      const sm = [0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131]
      const id = IDENT.m
      return { m: sm.map((v, i) => k * id[i] + s * v), b: [0, 0, 0] }
    }
    case 'hue-rotate': {
      let a = parseFloat(arg) || 0
      if (arg.includes('turn')) a *= 360
      else if (arg.includes('rad')) a = (a * 180) / Math.PI
      const r = (a * Math.PI) / 180
      const c = Math.cos(r)
      const n = Math.sin(r)
      return {
        m: [
          0.213 + c * 0.787 - n * 0.213, 0.715 - c * 0.715 - n * 0.715, 0.072 - c * 0.072 + n * 0.928,
          0.213 - c * 0.213 + n * 0.143, 0.715 + c * 0.285 + n * 0.14, 0.072 - c * 0.072 - n * 0.283,
          0.213 - c * 0.213 - n * 0.787, 0.715 - c * 0.715 + n * 0.715, 0.072 + c * 0.928 + n * 0.072,
        ],
        b: [0, 0, 0],
      }
    }
    default:
      return IDENT // blur / opacity / unknown — not a colour matrix
  }
}

/** Compose a CSS filter string into one 4×4 colour matrix (column-major, ready
 *  for writeBuffer): out = M·vec4(rgb,1). Returns null for 'none'/empty. */
function colorMatrix(filter: string): Float32Array | null {
  if (!filter || filter === 'none') return null
  let m = IDENT.m
  let b = IDENT.b
  const re = /([a-z-]+)\(([^)]*)\)/g
  let hit: RegExpExecArray | null
  let any = false
  while ((hit = re.exec(filter))) {
    const f = filterMatrix(hit[1], hit[2])
    m = mul3(f.m, m) // apply this filter AFTER the accumulated ones
    b = mulVec3(f.m, b).map((v, i) => v + f.b[i])
    any = true
  }
  if (!any) return null
  // Column-major 4×4: cols 0-2 hold the 3×3 (row i, col j → m[i*3+j]); col 3 = bias.
  return new Float32Array([
    m[0], m[3], m[6], 0,
    m[1], m[4], m[7], 0,
    m[2], m[5], m[8], 0,
    b[0], b[1], b[2], 1,
  ])
}

const MAX_BLUR_PX = 32

/** Extract the blur(Npx) radius from a CSS filter string, clamped. 0 = none. */
function blurRadius(filter: string): number {
  const m = /blur\(\s*([\d.]+)\s*px\s*\)/.exec(filter)
  if (!m) return 0
  return Math.min(MAX_BLUR_PX, Math.max(0, parseFloat(m[1]) || 0))
}

export class WebGpuCompositor {
  readonly bus = new AnalyzerBus()
  composite: Composite = defaultComposite()

  private device: GPUDevice
  private context: GPUCanvasContext
  private canvasFormat: GPUTextureFormat
  private sampler: GPUSampler
  private layerUniforms: GPUBuffer
  private fxUniforms: GPUBuffer
  private blurUniforms: GPUBuffer
  private depthUniforms: GPUBuffer
  private scratch = new Float32Array(12)
  private fxScratch = new Float32Array(20)
  private blurScratch = new Float32Array(8)
  private depthScratch = new Float32Array(4)
  private diag = 0
  // GPU depth map (r8unorm) written by setDepthData from the ort-web session;
  // undefined until the first inference lands (depth layer reuse-last / skips).
  private depthDataTex: GPUTexture | null = null
  private depthDataView: GPUTextureView | null = null
  private depthW = 0
  private depthH = 0
  // Ping-pong smoothed depth (eased toward the raw map each frame) so the depth
  // effect glides between inferences instead of stepping at the model's low fps.
  private depthSmooth: [GPUTexture, GPUTexture] | null = null
  private depthSmoothViews: [GPUTextureView, GPUTextureView] | null = null
  private depthSmoothCur = 0
  // Cache the composed colour matrix + blur radius per effect layer; recompute
  // only when its filter string changes (keyed by layer id, not string, so an
  // animated hue-rotate doesn't grow the map unboundedly).
  private fxCache = new Map<LayerId, { filter: string; matrix: Float32Array | null; blur: number }>()
  // Global effects (marker / drawLayer) run on a 2D overlay canvas, uploaded as a
  // texture and composited over the final accumulator — reuses the CanvasEffect code.
  private effects: CanvasEffect[] = []
  private overlayCanvas: OffscreenCanvas
  private overlayCtx: Ctx2D | null

  private layerLayout: GPUBindGroupLayout
  private blendLayout: GPUBindGroupLayout
  private fxLayout: GPUBindGroupLayout
  private blurLayout: GPUBindGroupLayout
  private mixLayout: GPUBindGroupLayout
  private overlayLayout: GPUBindGroupLayout
  private depthLayout: GPUBindGroupLayout
  private depthEaseLayout: GPUBindGroupLayout
  private blitLayout: GPUBindGroupLayout
  private layerPipeline: GPURenderPipeline
  private blendPipeline: GPURenderPipeline
  private fxPipeline: GPURenderPipeline
  private fxFullPipeline: GPURenderPipeline
  private blurPipeline: GPURenderPipeline
  private mixPipeline: GPURenderPipeline
  private overlayPipeline: GPURenderPipeline
  private depthPipeline: GPURenderPipeline
  private depthEasePipeline: GPURenderPipeline
  private blitPipeline: GPURenderPipeline

  private accumA!: GPUTexture
  private accumB!: GPUTexture
  private layerTex!: GPUTexture
  // Stable views for the persistent targets, created once (the canvas view must
  // still be taken per-frame from getCurrentTexture).
  private accumAView!: GPUTextureView
  private accumBView!: GPUTextureView
  private layerTexView!: GPUTextureView
  private overlayTex!: GPUTexture
  private overlayTexView!: GPUTextureView

  constructor(
    gpu: GpuContext,
    canvas: OffscreenCanvas,
    private width: number,
    private height: number,
  ) {
    this.device = gpu.device
    this.canvasFormat = gpu.canvasFormat
    const ctx = canvas.getContext('webgpu')
    if (!ctx) throw new Error('webgpu canvas context unavailable')
    this.context = ctx
    this.context.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' })

    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.layerUniforms = this.device.createBuffer({
      size: UNIFORM_STRIDE * MAX_LAYERS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.fxUniforms = this.device.createBuffer({
      size: UNIFORM_STRIDE * MAX_LAYERS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Two slices per effect layer (horizontal + vertical blur direction).
    this.blurUniforms = this.device.createBuffer({
      size: UNIFORM_STRIDE * MAX_LAYERS * 2,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.depthUniforms = this.device.createBuffer({
      size: UNIFORM_STRIDE * MAX_LAYERS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // 2D canvas the global effects (marker/drawLayer) render onto each frame.
    this.overlayCanvas = new OffscreenCanvas(width, height)
    this.overlayCtx = this.overlayCanvas.getContext('2d')

    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT
    const F = GPUShaderStage.FRAGMENT
    this.layerLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, externalTexture: {} },
      ],
    })
    this.blendLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        { binding: 4, visibility: F, texture: { sampleType: 'float' } },
      ],
    })
    this.fxLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        { binding: 5, visibility: F, buffer: { type: 'uniform' } },
      ],
    })
    this.blurLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        { binding: 6, visibility: F, buffer: { type: 'uniform' } },
      ],
    })
    this.mixLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        { binding: 5, visibility: F, buffer: { type: 'uniform' } },
        { binding: 7, visibility: F, texture: { sampleType: 'float' } },
      ],
    })
    this.overlayLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        { binding: 8, visibility: F, texture: { sampleType: 'float' } },
      ],
    })
    this.depthLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
        { binding: 9, visibility: F, texture: { sampleType: 'float' } },
        { binding: 10, visibility: F, buffer: { type: 'uniform' } },
      ],
    })
    this.depthEaseLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } }, // prev smoothed
        { binding: 9, visibility: F, texture: { sampleType: 'float' } }, // latest raw
      ],
    })
    this.blitLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
      ],
    })

    const module = this.device.createShaderModule({ code: WGSL })
    const pipe = (
      layout: GPUBindGroupLayout,
      vs: string,
      fs: string,
      format: GPUTextureFormat,
      topology: GPUPrimitiveTopology,
    ): GPURenderPipeline =>
      this.device.createRenderPipeline({
        label: fs, // so an "invalid pipeline" error names the offending pass
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format }] },
        primitive: { topology },
      })
    this.layerPipeline = pipe(this.layerLayout, 'layer_vs', 'layer_fs', ACCUM_FORMAT, 'triangle-strip')
    this.blendPipeline = pipe(this.blendLayout, 'fullscreen_vs', 'blend_fs', ACCUM_FORMAT, 'triangle-list')
    this.fxPipeline = pipe(this.fxLayout, 'fullscreen_vs', 'fx_fs', ACCUM_FORMAT, 'triangle-list')
    this.fxFullPipeline = pipe(this.fxLayout, 'fullscreen_vs', 'fxfull_fs', ACCUM_FORMAT, 'triangle-list')
    this.blurPipeline = pipe(this.blurLayout, 'fullscreen_vs', 'blur_fs', ACCUM_FORMAT, 'triangle-list')
    this.mixPipeline = pipe(this.mixLayout, 'fullscreen_vs', 'fxmix_fs', ACCUM_FORMAT, 'triangle-list')
    this.overlayPipeline = pipe(this.overlayLayout, 'fullscreen_vs', 'overlay_fs', ACCUM_FORMAT, 'triangle-list')
    this.depthPipeline = pipe(this.depthLayout, 'fullscreen_vs', 'depth_fs', ACCUM_FORMAT, 'triangle-list')
    // Ease target is rgba16float (the proven render format) — r8unorm-as-target
    // produced an invalid pipeline on the target adapter. Depth lives in .r.
    this.depthEasePipeline = pipe(this.depthEaseLayout, 'fullscreen_vs', 'depth_ease_fs', ACCUM_FORMAT, 'triangle-list')
    this.blitPipeline = pipe(this.blitLayout, 'fullscreen_vs', 'blit_fs', this.canvasFormat, 'triangle-list')

    this.allocTargets()
  }

  private allocTargets(): void {
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    const size = { width: this.width, height: this.height }
    this.accumA = this.device.createTexture({ size, format: ACCUM_FORMAT, usage })
    this.accumB = this.device.createTexture({ size, format: ACCUM_FORMAT, usage })
    this.layerTex = this.device.createTexture({ size, format: ACCUM_FORMAT, usage })
    this.accumAView = this.accumA.createView()
    this.accumBView = this.accumB.createView()
    this.layerTexView = this.layerTex.createView()
    // rgba8 overlay (effects render to a 2D canvas, uploaded here). Needs
    // RENDER_ATTACHMENT for copyExternalImageToTexture's destination.
    this.overlayTex = this.device.createTexture({
      size,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.overlayTexView = this.overlayTex.createView()
  }

  /** Stable view for one of the two ping-pong accumulators. */
  private accumView(tex: GPUTexture): GPUTextureView {
    return tex === this.accumA ? this.accumAView : this.accumBView
  }

  setComposite(value: Composite | CompositeOp): void {
    if (Array.isArray(value)) this.composite = value.map((l) => ({ ...l }))
    else applyCompositeOp(this.composite, value)
  }

  setEffects(inits: EffectInit[]): void {
    this.disposeEffects()
    this.effects = inits.map((e) => createEffect(e.name, e.config))
    for (const e of this.effects) e.init?.(this.width, this.height)
  }
  configureEffect(name: string, patch: Record<string, unknown>): void {
    this.effects.find((e) => e.name === name)?.configure(patch)
  }
  effectMessage(name: string, data: unknown): void {
    this.effects.find((e) => e.name === name)?.message?.(data)
  }
  // Effect LAYERS (shader) take no out-of-band messages; only global effects do.
  effectLayerMessage(_layerId: LayerId, _data: unknown): void {}

  private writeLayer(slot: number, frame: VideoFrame, opts: LayerOptions, mirror: boolean, blend: BlendMode): boolean {
    const w = frame.displayWidth
    const h = frame.displayHeight
    const r = layerDrawRects(this.width, this.height, w, h, opts.transform)
    if (!r) return false
    const W = this.width
    const H = this.height
    this.scratch[0] = (r.dx / W) * 2 - 1
    this.scratch[1] = 1 - (r.dy / H) * 2
    this.scratch[2] = ((r.dx + r.dw) / W) * 2 - 1
    this.scratch[3] = 1 - ((r.dy + r.dh) / H) * 2
    this.scratch[4] = r.sx / w
    this.scratch[5] = r.sy / h
    this.scratch[6] = (r.sx + r.sw) / w
    this.scratch[7] = (r.sy + r.sh) / h
    this.scratch[8] = opts.opacity
    this.scratch[9] = mirror ? 1 : 0
    this.scratch[10] = BLEND_NUM[blend] ?? 0
    this.scratch[11] = 0
    this.device.queue.writeBuffer(this.layerUniforms, slot * UNIFORM_STRIDE, this.scratch)
    return true
  }

  private writeFx(slot: number, matrix: Float32Array, opacity: number): void {
    this.fxScratch.set(matrix, 0) // 16 floats
    this.fxScratch[16] = opacity
    this.fxScratch[17] = 0
    this.fxScratch[18] = 0
    this.fxScratch[19] = 0
    this.device.queue.writeBuffer(this.fxUniforms, slot * UNIFORM_STRIDE, this.fxScratch)
  }

  private writeBlur(slot: number, dirX: number, dirY: number, radius: number): void {
    this.blurScratch[0] = 1 / this.width
    this.blurScratch[1] = 1 / this.height
    this.blurScratch[2] = dirX
    this.blurScratch[3] = dirY
    this.blurScratch[4] = radius
    this.blurScratch[5] = 0
    this.blurScratch[6] = 0
    this.blurScratch[7] = 0
    this.device.queue.writeBuffer(this.blurUniforms, slot * UNIFORM_STRIDE, this.blurScratch)
  }

  /** Upload a new depth map (from the ort-web DepthSession) onto the GPU depth
   *  texture, (re)creating it if the dims changed. Called at the depth cadence
   *  (~5fps), not per composite frame; between calls the depth layer reuses it. */
  setDepthData(data: Uint8Array, w: number, h: number): void {
    const size = { width: w, height: h }
    if (!this.depthDataTex || this.depthW !== w || this.depthH !== h) {
      this.depthDataTex?.destroy()
      this.depthSmooth?.forEach((t) => t.destroy())
      this.depthDataTex = this.device.createTexture({
        size,
        format: 'r8unorm',
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.depthDataView = this.depthDataTex.createView()
      // Ping-pong smoothed pair, rgba16float (proven render-target format; depth
      // is carried in .r). Not seeded — the ease ramps up from the raw map over a
      // few frames (a brief fade-in on first depth, acceptable for a live effect).
      const su = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      const a = this.device.createTexture({ size, format: ACCUM_FORMAT, usage: su })
      const b = this.device.createTexture({ size, format: ACCUM_FORMAT, usage: su })
      this.depthSmooth = [a, b]
      this.depthSmoothViews = [a.createView(), b.createView()]
      this.depthSmoothCur = 0
      this.depthW = w
      this.depthH = h
    }
    this.device.queue.writeTexture({ texture: this.depthDataTex }, data, { bytesPerRow: w, rowsPerImage: h }, size)
  }

  private writeDepth(slot: number, strength: number, mode: number): void {
    this.depthScratch[0] = strength
    this.depthScratch[1] = mode
    this.depthScratch[2] = 0
    this.depthScratch[3] = 1
    this.device.queue.writeBuffer(this.depthUniforms, slot * UNIFORM_STRIDE, this.depthScratch)
  }

  /** One full-screen pass: pipeline + bind group entries → target view. */
  private fsPass(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    pipeline: GPURenderPipeline,
    layout: GPUBindGroupLayout,
    entries: GPUBindGroupEntry[],
  ): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store' }],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, this.device.createBindGroup({ layout, entries }))
    pass.draw(3)
    pass.end()
  }

  /** Composite back-to-front into the accumulator (ping-pong) then blit to the
   *  canvas. Source layers blend their frame; 'shader' effect layers run a colour
   *  filter over everything below. All external-texture imports are synchronous
   *  within this call (their textures are valid only for the current task). */
  drawComposite(frames: FrameMap, tsMs: number): void {
    const encoder = this.device.createCommandEncoder()
    let readTex = this.accumA
    let writeTex = this.accumB
    const layerView = this.layerTexView
    const layerUni = (slot: number): GPUBufferBinding => ({
      buffer: this.layerUniforms,
      offset: slot * UNIFORM_STRIDE,
      size: LAYER_UNIFORM_BYTES,
    })
    const fxUni = (slot: number): GPUBufferBinding => ({
      buffer: this.fxUniforms,
      offset: slot * UNIFORM_STRIDE,
      size: FX_UNIFORM_BYTES,
    })
    const blurUni = (slot: number): GPUBufferBinding => ({
      buffer: this.blurUniforms,
      offset: slot * UNIFORM_STRIDE,
      size: BLUR_UNIFORM_BYTES,
    })
    const depthUni = (slot: number): GPUBufferBinding => ({
      buffer: this.depthUniforms,
      offset: slot * UNIFORM_STRIDE,
      size: 16,
    })

    // Seed the accumulator with opaque black.
    encoder
      .beginRenderPass({
        colorAttachments: [
          { view: this.accumView(readTex), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
        ],
      })
      .end()

    // Temporal-ease the depth map once per frame (only when a depth layer is
    // active + data has arrived): smoothNext = mix(smoothPrev, raw, 0.18), so the
    // depth effect glides between low-fps inferences instead of stepping.
    if (
      this.depthDataView &&
      this.depthSmoothViews &&
      this.composite.some((l) => l.effectName === 'depth' && l.opacity > 0)
    ) {
      const prev = this.depthSmoothViews[this.depthSmoothCur]
      const next = this.depthSmoothViews[this.depthSmoothCur ^ 1]
      this.fsPass(encoder, next, this.depthEasePipeline, this.depthEaseLayout, [
        { binding: 1, resource: this.sampler },
        { binding: 3, resource: prev },
        { binding: 9, resource: this.depthDataView },
      ])
      this.depthSmoothCur ^= 1
    }

    let layerSlot = 0
    let fxSlot = 0
    let blurSlot = 0
    let depthSlot = 0
    let drawn = 0
    // Composite is ordered front→back; iterate in reverse to draw back-to-front.
    for (let i = this.composite.length - 1; i >= 0; i--) {
      const layer = this.composite[i]
      if (layer.opacity <= 0) continue

      if (layer.effectName) {
        if (layer.effectName === 'depth') {
          // No depth data yet → skip (reuse-last is free: the texture persists).
          if (!this.depthDataView || depthSlot >= MAX_LAYERS) continue
          const dcfg = (layer.effectConfig ?? {}) as Record<string, unknown>
          this.writeDepth(depthSlot, layer.opacity, depthModeNum(dcfg.mode))
          this.fsPass(encoder, this.accumView(writeTex), this.depthPipeline, this.depthLayout, [
            { binding: 1, resource: this.sampler },
            { binding: 3, resource: this.accumView(readTex) },
            // Sample the eased (smoothed) depth, not the raw map.
            { binding: 9, resource: this.depthSmoothViews?.[this.depthSmoothCur] ?? this.depthDataView },
            { binding: 10, resource: depthUni(depthSlot) },
          ])
          ;[readTex, writeTex] = [writeTex, readTex]
          depthSlot++
          drawn++
          continue
        }
        if (layer.effectName !== 'shader' || fxSlot >= MAX_LAYERS) continue // 'shader'/'depth' ported
        const filter = (layer.effectConfig?.filter as string | undefined) ?? 'none'
        let cached = this.fxCache.get(layer.id)
        if (!cached || cached.filter !== filter) {
          cached = { filter, matrix: colorMatrix(filter), blur: blurRadius(filter) }
          this.fxCache.set(layer.id, cached)
        }
        const { matrix, blur } = cached
        if (!matrix && blur <= 0) continue // none / unsupported → no change
        // opacity (params.x) is the EFFECT STRENGTH: fx_fs / fxmix_fs both
        // mix(original, filtered, opacity) — 0 = off, 1 = full.
        this.writeFx(fxSlot, matrix ?? IDENTITY_MAT4, layer.opacity)

        if (blur <= 0) {
          // Colour-only — one pass, opacity mix built in.
          this.fsPass(encoder, this.accumView(writeTex), this.fxPipeline, this.fxLayout, [
            { binding: 1, resource: this.sampler },
            { binding: 3, resource: this.accumView(readTex) },
            { binding: 5, resource: fxUni(fxSlot) },
          ])
          ;[readTex, writeTex] = [writeTex, readTex]
        } else {
          // colour(full) → blurH → blurV → mix(original, blurred, opacity).
          // readTex ("below") is only ever READ here, so it survives to the mix.
          let colorView = this.accumView(readTex)
          if (matrix) {
            this.fsPass(encoder, this.layerTexView, this.fxFullPipeline, this.fxLayout, [
              { binding: 1, resource: this.sampler },
              { binding: 3, resource: this.accumView(readTex) },
              { binding: 5, resource: fxUni(fxSlot) },
            ])
            colorView = this.layerTexView
          }
          this.writeBlur(blurSlot, 1, 0, blur)
          this.fsPass(encoder, this.accumView(writeTex), this.blurPipeline, this.blurLayout, [
            { binding: 1, resource: this.sampler },
            { binding: 3, resource: colorView },
            { binding: 6, resource: blurUni(blurSlot) },
          ])
          blurSlot++
          this.writeBlur(blurSlot, 0, 1, blur)
          this.fsPass(encoder, this.layerTexView, this.blurPipeline, this.blurLayout, [
            { binding: 1, resource: this.sampler },
            { binding: 3, resource: this.accumView(writeTex) },
            { binding: 6, resource: blurUni(blurSlot) },
          ])
          blurSlot++
          this.fsPass(encoder, this.accumView(writeTex), this.mixPipeline, this.mixLayout, [
            { binding: 1, resource: this.sampler },
            { binding: 3, resource: this.accumView(readTex) },
            { binding: 5, resource: fxUni(fxSlot) },
            { binding: 7, resource: this.layerTexView },
          ])
          ;[readTex, writeTex] = [writeTex, readTex]
        }
        fxSlot++
        drawn++
        continue
      }

      const src = frames[layer.id]
      if (!src || !isVideoFrame(src)) continue // only live VideoFrame layers for now
      if (layerSlot >= MAX_LAYERS) continue
      if (!this.writeLayer(layerSlot, src, layer, layer.mirror, layer.blend)) continue
      const extTex = this.device.importExternalTexture({ source: src })
      // Pass A: transformed layer → scratch (cleared transparent).
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            { view: layerView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          ],
        })
        pass.setPipeline(this.layerPipeline)
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: this.layerLayout,
            entries: [
              { binding: 0, resource: layerUni(layerSlot) },
              { binding: 1, resource: this.sampler },
              { binding: 2, resource: extTex },
            ],
          }),
        )
        pass.draw(4)
        pass.end()
      }
      // Pass B: blend(read, scratch) → write, then ping-pong.
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: this.accumView(writeTex), loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(this.blendPipeline)
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: this.blendLayout,
            entries: [
              { binding: 0, resource: layerUni(layerSlot) },
              { binding: 1, resource: this.sampler },
              { binding: 3, resource: this.accumView(readTex) },
              { binding: 4, resource: layerView },
            ],
          }),
        )
        pass.draw(3)
        pass.end()
      }
      ;[readTex, writeTex] = [writeTex, readTex]
      layerSlot++
      drawn++
    }

    // Global effects (marker/drawLayer) → 2D overlay → upload → composite on top
    // of the finished accumulator. Matches the 2D Compositor, which runs the
    // global effect chain last.
    if (this.effects.length && this.overlayCtx) {
      const octx = this.overlayCtx
      octx.clearRect(0, 0, this.width, this.height)
      const info: FrameInfo = { width: this.width, height: this.height, tsMs }
      for (const e of this.effects) e.render(octx, info, this.bus)
      this.device.queue.copyExternalImageToTexture(
        { source: this.overlayCanvas },
        { texture: this.overlayTex, premultipliedAlpha: false },
        { width: this.width, height: this.height },
      )
      this.fsPass(encoder, this.accumView(writeTex), this.overlayPipeline, this.overlayLayout, [
        { binding: 1, resource: this.sampler },
        { binding: 3, resource: this.accumView(readTex) },
        { binding: 8, resource: this.overlayTexView },
      ])
      ;[readTex, writeTex] = [writeTex, readTex]
    }

    if (this.diag < 2) {
      this.diag++
      console.info(
        '[webgpu] frame',
        this.diag,
        '| layers:',
        this.composite.map((l) => `${l.id}:${l.effectName ? `fx(${l.effectName})` : l.blend}@${l.opacity}`).join(' '),
        '| drawn:',
        drawn,
        '| frames:',
        Object.keys(frames)
          .map((k) => `${k}:${isVideoFrame(frames[k]) ? 'vf' : frames[k] ? 'other' : 'none'}`)
          .join(' '),
      )
    }

    // Blit the final accumulator to the canvas.
    const blit = encoder.beginRenderPass({
      colorAttachments: [
        { view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store' },
      ],
    })
    blit.setPipeline(this.blitPipeline)
    blit.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.blitLayout,
        entries: [
          { binding: 1, resource: this.sampler },
          { binding: 3, resource: this.accumView(readTex) },
        ],
      }),
    )
    blit.draw(3)
    blit.end()

    this.device.queue.submit([encoder.finish()])
  }

  disposeEffects(): void {
    for (const e of this.effects) e.dispose?.()
    this.effects = []
  }

  dispose(): void {
    this.disposeEffects()
    this.layerUniforms.destroy()
    this.fxUniforms.destroy()
    this.blurUniforms.destroy()
    this.depthUniforms.destroy()
    this.accumA.destroy()
    this.accumB.destroy()
    this.layerTex.destroy()
    this.overlayTex.destroy()
    this.depthDataTex?.destroy()
    this.depthSmooth?.forEach((t) => t.destroy())
  }
}
