// WebGPU compositor — a GPU-resident alternative to the 2D-canvas `Compositor`,
// behind the SAME public surface (setComposite / drawComposite / bus) so the
// streams pipeline worker can swap to it behind a capability probe. It keeps
// each layer's frame on the GPU: a live VideoFrame is imported zero-copy via
// importExternalTexture and composited into an rgba16float accumulator, which is
// then blitted to a GPUCanvasContext whose canvas feeds `new VideoFrame(canvas)`
// → MSTG — no pixel returns to the CPU on the hot path.
//
// Blending: every blend mode goes through a ping-pong over two rgba16float
// accumulators so screen/multiply/difference (which read the backdrop) are
// correct, not just NORMAL. Per layer, two passes: (A) render the transformed
// layer into a scratch texture; (B) full-screen blend of (accumulator, scratch)
// into the other accumulator. Geometry reuses types.ts (`layerDrawRects`) so it
// stays pixel-identical to the 2D backend.
//
// NOT YET IMPLEMENTED (tracked): effect LAYERS (shader/marker/drawLayer) — WGSL
// ports / overlay-texture upload; non-VideoFrame layer sources (retained
// ImageBitmap, depth texture). Those are skipped here today.

import { AnalyzerBus } from './bus'
import type { BlendMode, Composite, CompositeOp, EffectInit, LayerId, LayerOptions } from './types'
import { applyCompositeOp, defaultComposite, layerDrawRects } from './types'
import type { GpuContext } from './gpu'
import type { FrameMap } from './compositor'

const UNIFORM_STRIDE = 256 // ≥ minUniformBufferOffsetAlignment; one slice per layer
const UNIFORM_BYTES = 48 // 3×vec4<f32> payload actually used
const MAX_LAYERS = 16
const ACCUM_FORMAT: GPUTextureFormat = 'rgba16float'

const BLEND_NUM: Record<BlendMode, number> = { normal: 0, screen: 1, multiply: 2, difference: 3 }

const WGSL = /* wgsl */ `
struct Layer {
  dst: vec4<f32>,    // ndc x0, ytop, x1, ybot
  uv:  vec4<f32>,    // u0, v0, u1, v1
  params: vec4<f32>, // opacity, mirror(0/1), blendMode, _
};
@group(0) @binding(0) var<uniform> L: Layer;

// ---- Pass A: transformed layer (external texture) → scratch -----------------
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var srcTex: texture_external;

struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

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

// ---- full-screen helpers ----------------------------------------------------
struct FSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn fullscreen_vs(@builtin(vertex_index) vid: u32) -> FSOut {
  let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: FSOut;
  o.pos = vec4<f32>(p[vid], 0.0, 1.0);
  o.uv = vec2<f32>(p[vid].x * 0.5 + 0.5, 0.5 - p[vid].y * 0.5); // y-flip → uv(0,0)=top-left
  return o;
}

// ---- Pass B: blend(accum, layerScratch) → accum' ----------------------------
@group(0) @binding(3) var accumTex: texture_2d<f32>;
@group(0) @binding(4) var layerTex: texture_2d<f32>;

fn blend_rgb(mode: f32, b: vec3<f32>, s: vec3<f32>) -> vec3<f32> {
  if (mode < 0.5) { return s; }                          // normal
  if (mode < 1.5) { return 1.0 - (1.0 - b) * (1.0 - s); } // screen
  if (mode < 2.5) { return b * s; }                       // multiply
  return abs(b - s);                                      // difference
}

@fragment
fn blend_fs(in: FSOut) -> @location(0) vec4<f32> {
  let b = textureSampleLevel(accumTex, samp, in.uv, 0.0);
  let s = textureSampleLevel(layerTex, samp, in.uv, 0.0);
  let blended = blend_rgb(L.params.z, b.rgb, s.rgb);
  return vec4<f32>(mix(b.rgb, blended, s.a), 1.0); // s.a = layer opacity × quad coverage
}

// ---- final blit: accum → canvas --------------------------------------------
@fragment
fn blit_fs(in: FSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(textureSampleLevel(accumTex, samp, in.uv, 0.0).rgb, 1.0);
}
`

function isVideoFrame(src: unknown): src is VideoFrame {
  return typeof VideoFrame !== 'undefined' && src instanceof VideoFrame
}

export class WebGpuCompositor {
  readonly bus = new AnalyzerBus()
  composite: Composite = defaultComposite()

  private device: GPUDevice
  private context: GPUCanvasContext
  private canvasFormat: GPUTextureFormat
  private sampler: GPUSampler
  private uniforms: GPUBuffer
  private scratch = new Float32Array(12)
  private diag = 0

  private layerLayout: GPUBindGroupLayout
  private blendLayout: GPUBindGroupLayout
  private blitLayout: GPUBindGroupLayout
  private layerPipeline: GPURenderPipeline
  private blendPipeline: GPURenderPipeline
  private blitPipeline: GPURenderPipeline

  private accumA!: GPUTexture
  private accumB!: GPUTexture
  private layerTex!: GPUTexture

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
    // alphaMode 'opaque': output must never be semi-transparent (upstream encodes
    // alpha as black and breaks screen/multiply) — matches the 2D opaque base.
    this.context.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' })

    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.uniforms = this.device.createBuffer({
      size: UNIFORM_STRIDE * MAX_LAYERS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

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
    this.blitLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
      ],
    })

    const module = this.device.createShaderModule({ code: WGSL })
    this.layerPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layerLayout] }),
      vertex: { module, entryPoint: 'layer_vs' },
      fragment: { module, entryPoint: 'layer_fs', targets: [{ format: ACCUM_FORMAT }] },
      primitive: { topology: 'triangle-strip' },
    })
    this.blendPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.blendLayout] }),
      vertex: { module, entryPoint: 'fullscreen_vs' },
      fragment: { module, entryPoint: 'blend_fs', targets: [{ format: ACCUM_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    })
    this.blitPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.blitLayout] }),
      vertex: { module, entryPoint: 'fullscreen_vs' },
      fragment: { module, entryPoint: 'blit_fs', targets: [{ format: this.canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    })

    this.allocTargets()
  }

  private allocTargets(): void {
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    const size = { width: this.width, height: this.height }
    this.accumA = this.device.createTexture({ size, format: ACCUM_FORMAT, usage })
    this.accumB = this.device.createTexture({ size, format: ACCUM_FORMAT, usage })
    this.layerTex = this.device.createTexture({ size, format: ACCUM_FORMAT, usage })
  }

  setComposite(value: Composite | CompositeOp): void {
    if (Array.isArray(value)) this.composite = value.map((l) => ({ ...l }))
    else applyCompositeOp(this.composite, value)
  }

  // Effect layers / global effects not yet ported to WGSL — no-ops with matching
  // signatures so the worker drives this and the 2D Compositor identically.
  setEffects(_inits: EffectInit[]): void {}
  configureEffect(_name: string, _patch: Record<string, unknown>): void {}
  effectMessage(_name: string, _data: unknown): void {}
  effectLayerMessage(_layerId: LayerId, _data: unknown): void {}

  /** Pack one layer's NDC dest rect + source UV sub-rect + opacity/mirror/blend
   *  into its uniform slice. Returns false when not drawable. */
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
    this.device.queue.writeBuffer(this.uniforms, slot * UNIFORM_STRIDE, this.scratch)
    return true
  }

  /** Composite back-to-front into the accumulator (ping-pong) then blit to the
   *  canvas. All external-texture imports happen synchronously within this call
   *  (their textures are valid only for the current task). */
  drawComposite(frames: FrameMap, _tsMs: number): void {
    type Drawn = { slot: number; extTex: GPUExternalTexture }
    const draws: Drawn[] = []
    for (let i = this.composite.length - 1; i >= 0 && draws.length < MAX_LAYERS; i--) {
      const layer = this.composite[i]
      if (layer.effectName) continue // effect layers not yet ported
      if (layer.opacity <= 0) continue
      const src = frames[layer.id]
      if (!src || !isVideoFrame(src)) continue // only live VideoFrame layers for now
      const slot = draws.length
      if (!this.writeLayer(slot, src, layer, layer.mirror, layer.blend)) continue
      draws.push({ slot, extTex: this.device.importExternalTexture({ source: src }) })
    }

    if (this.diag < 2) {
      this.diag++
      console.info(
        '[webgpu] frame',
        this.diag,
        '| layers:',
        this.composite.map((l) => `${l.id}:${l.effectName ? 'fx' : l.blend}@${l.opacity}`).join(' '),
        '| drawn:',
        draws.length,
        '| frames:',
        Object.keys(frames)
          .map((k) => `${k}:${isVideoFrame(frames[k]) ? 'vf' : frames[k] ? 'other' : 'none'}`)
          .join(' '),
      )
    }

    const uni = (slot: number): GPUBufferBinding => ({
      buffer: this.uniforms,
      offset: slot * UNIFORM_STRIDE,
      size: UNIFORM_BYTES,
    })
    const encoder = this.device.createCommandEncoder()
    let readTex = this.accumA
    let writeTex = this.accumB
    const layerView = this.layerTex.createView()

    // Seed the accumulator with opaque black.
    encoder
      .beginRenderPass({
        colorAttachments: [
          { view: readTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
        ],
      })
      .end()

    for (const d of draws) {
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
              { binding: 0, resource: uni(d.slot) },
              { binding: 1, resource: this.sampler },
              { binding: 2, resource: d.extTex },
            ],
          }),
        )
        pass.draw(4)
        pass.end()
      }
      // Pass B: blend(read, scratch) → write, then ping-pong.
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: writeTex.createView(), loadOp: 'clear', storeOp: 'store' }],
        })
        pass.setPipeline(this.blendPipeline)
        pass.setBindGroup(
          0,
          this.device.createBindGroup({
            layout: this.blendLayout,
            entries: [
              { binding: 0, resource: uni(d.slot) },
              { binding: 1, resource: this.sampler },
              { binding: 3, resource: readTex.createView() },
              { binding: 4, resource: layerView },
            ],
          }),
        )
        pass.draw(3)
        pass.end()
      }
      const t = readTex
      readTex = writeTex
      writeTex = t
    }

    // Blit the final accumulator to the canvas.
    const canvasView = this.context.getCurrentTexture().createView()
    const blit = encoder.beginRenderPass({
      colorAttachments: [{ view: canvasView, loadOp: 'clear', storeOp: 'store' }],
    })
    blit.setPipeline(this.blitPipeline)
    blit.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.blitLayout,
        entries: [
          { binding: 1, resource: this.sampler },
          { binding: 3, resource: readTex.createView() },
        ],
      }),
    )
    blit.draw(3)
    blit.end()

    this.device.queue.submit([encoder.finish()])
  }

  disposeEffects(): void {}

  dispose(): void {
    this.uniforms.destroy()
    this.accumA.destroy()
    this.accumB.destroy()
    this.layerTex.destroy()
  }
}
