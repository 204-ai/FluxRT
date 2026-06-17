// WebGPU compositor — a GPU-resident alternative to the 2D-canvas `Compositor`,
// behind the SAME public surface (setComposite / drawComposite / bus) so the
// streams pipeline worker can swap to it behind a capability probe without other
// changes. It keeps each layer's frame on the GPU: a live VideoFrame is imported
// zero-copy via importExternalTexture and drawn as a textured quad into a
// GPUCanvasContext; the output canvas then feeds `new VideoFrame(canvas)` → MSTG,
// never touching the CPU on the hot path.
//
// SCOPE (initial): the common path only — back-to-front layers with opacity,
// per-layer OBS transform/crop and selfie mirror, NORMAL blend (src-over), over
// an opaque-black base. The geometry reuses types.ts (`layerDrawRects`) so it
// stays pixel-identical to the 2D backend and the same transform tests guard it.
//
// NOT YET IMPLEMENTED (tracked for the next steps):
//   • screen/multiply/difference blend — these read the destination, so they
//     need an rgba16float accumulator with ping-pong, not fixed-function blend.
//   • effect LAYERS (shader/marker/drawLayer) — WGSL ports / overlay-texture
//     upload; today they are skipped here (drawn only by the 2D backend).
//   • non-VideoFrame layer sources (retained ImageBitmap/canvas, depth texture).
// Until those land this compositor is constructed but NOT swapped into the
// worker, so the 2D path remains the shipping behavior.

import { AnalyzerBus } from './bus'
import type { Composite, CompositeOp, LayerOptions } from './types'
import { applyCompositeOp, defaultComposite, layerDrawRects } from './types'
import type { GpuContext } from './gpu'
import type { FrameMap } from './compositor'

/** Uniform stride per layer — 12 floats (3×vec4) of payload, padded up to the
 *  256-byte minimum uniform-buffer offset alignment so each layer binds its own
 *  slice of one buffer. */
const UNIFORM_STRIDE = 256
const MAX_LAYERS = 16

const WGSL = /* wgsl */ `
struct Layer {
  dst: vec4<f32>,    // ndc x0, ytop, x1, ybot
  uv:  vec4<f32>,    // u0, v0, u1, v1
  params: vec4<f32>, // opacity, mirror(0/1), _, _
};
@group(0) @binding(0) var<uniform> L: Layer;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var tex: texture_external;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
  // 4-vertex triangle strip covering the layer's destination rect.
  let xs = array<f32, 4>(L.dst.x, L.dst.z, L.dst.x, L.dst.z);
  let ys = array<f32, 4>(L.dst.y, L.dst.y, L.dst.w, L.dst.w);
  var u0 = L.uv.x;
  var u1 = L.uv.z;
  if (L.params.y > 0.5) { let t = u0; u0 = u1; u1 = t; } // selfie mirror = flip U
  let us = array<f32, 4>(u0, u1, u0, u1);
  let vs = array<f32, 4>(L.uv.y, L.uv.y, L.uv.w, L.uv.w);
  var o: VSOut;
  o.pos = vec4<f32>(xs[vid], ys[vid], 0.0, 1.0);
  o.uv = vec2<f32>(us[vid], vs[vid]);
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let c = textureSampleBaseClampToEdge(tex, samp, in.uv);
  // Straight alpha from opacity; the configured src-over blend does the mix.
  return vec4<f32>(c.rgb, L.params.x);
}
`

/** Fixed-function src-over (premultiplied-style straight alpha). Only NORMAL is
 *  representable this way; other modes need a dst-read pass (see header). */
const SRC_OVER: GPUBlendState = {
  color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
}

function isVideoFrame(src: unknown): src is VideoFrame {
  return typeof VideoFrame !== 'undefined' && src instanceof VideoFrame
}

export class WebGpuCompositor {
  readonly bus = new AnalyzerBus()
  composite: Composite = defaultComposite()

  private device: GPUDevice
  private context: GPUCanvasContext
  private pipeline: GPURenderPipeline
  private bindLayout: GPUBindGroupLayout
  private sampler: GPUSampler
  private uniforms: GPUBuffer
  private scratch = new Float32Array(12) // one layer's payload

  constructor(
    gpu: GpuContext,
    canvas: OffscreenCanvas,
    private width: number,
    private height: number,
  ) {
    this.device = gpu.device
    const ctx = canvas.getContext('webgpu')
    if (!ctx) throw new Error('webgpu canvas context unavailable')
    this.context = ctx
    // alphaMode 'opaque': the output must never be semi-transparent (upstream
    // encodes alpha as black and breaks screen/multiply) — matches the 2D base.
    this.context.configure({ device: this.device, format: gpu.canvasFormat, alphaMode: 'opaque' })

    this.bindLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, externalTexture: {} },
      ],
    })
    const module = this.device.createShaderModule({ code: WGSL })
    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.bindLayout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: gpu.canvasFormat, blend: SRC_OVER }] },
      primitive: { topology: 'triangle-strip' },
    })
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.uniforms = this.device.createBuffer({
      size: UNIFORM_STRIDE * MAX_LAYERS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
  }

  /** Replace the whole composite (init) or apply a structural/mix op. Mirrors
   *  the 2D Compositor so the worker drives both identically. */
  setComposite(value: Composite | CompositeOp): void {
    if (Array.isArray(value)) this.composite = value.map((l) => ({ ...l }))
    else applyCompositeOp(this.composite, value)
  }

  // Effect layers / global effects are not yet ported to WGSL — no-ops so the
  // worker's existing calls are safe against this backend. See header TODO.
  setEffects(): void {}
  configureEffect(): void {}
  effectMessage(): void {}
  effectLayerMessage(): void {}

  /** Pack one layer's destination NDC rect + source UV sub-rect + opacity/mirror
   *  into `scratch` and upload it to this layer's uniform slice. Returns false
   *  when the layer is not drawable (offscreen / zero area). */
  private writeLayer(slot: number, frame: VideoFrame, opts: LayerOptions, mirror: boolean): boolean {
    const w = frame.displayWidth
    const h = frame.displayHeight
    const r = layerDrawRects(this.width, this.height, w, h, opts.transform)
    if (!r) return false
    const W = this.width
    const H = this.height
    // Canvas px (y-down) → NDC (y-up).
    this.scratch[0] = (r.dx / W) * 2 - 1 // x0
    this.scratch[1] = 1 - (r.dy / H) * 2 // ytop
    this.scratch[2] = ((r.dx + r.dw) / W) * 2 - 1 // x1
    this.scratch[3] = 1 - ((r.dy + r.dh) / H) * 2 // ybot
    this.scratch[4] = r.sx / w // u0
    this.scratch[5] = r.sy / h // v0
    this.scratch[6] = (r.sx + r.sw) / w // u1
    this.scratch[7] = (r.sy + r.sh) / h // v1
    this.scratch[8] = opts.opacity
    this.scratch[9] = mirror ? 1 : 0
    this.scratch[10] = 0
    this.scratch[11] = 0
    this.device.queue.writeBuffer(this.uniforms, slot * UNIFORM_STRIDE, this.scratch)
    return true
  }

  /** Composite the stack back-to-front into the canvas in one render pass. Each
   *  live VideoFrame layer is imported zero-copy and drawn as a quad. Layers with
   *  no frame, effect layers, or non-NORMAL blends are skipped here (TODO). */
  drawComposite(frames: FrameMap, _tsMs: number): void {
    type Drawn = { extTex: GPUExternalTexture; slot: number }
    const draws: Drawn[] = []

    // Back-to-front (composite is ordered front→back). Build uniforms + import
    // external textures BEFORE the pass; external textures are valid only within
    // this task, which is fine — drawComposite runs synchronously in the loop.
    for (let i = this.composite.length - 1; i >= 0 && draws.length < MAX_LAYERS; i--) {
      const layer = this.composite[i]
      if (layer.effectName) continue // effect layers not yet ported
      if (layer.opacity <= 0) continue
      const src = frames[layer.id]
      if (!src || !isVideoFrame(src)) continue // only live VideoFrame layers for now
      // NOTE: layer.blend other than 'normal' (screen/multiply/difference) is
      // drawn as NORMAL here until the dst-read ping-pong pass lands — present
      // but not yet the correct blend. See header TODO.
      const slot = draws.length
      if (!this.writeLayer(slot, src, layer, layer.mirror)) continue
      draws.push({ extTex: this.device.importExternalTexture({ source: src }), slot })
    }

    const view = this.context.getCurrentTexture().createView()
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
    })
    pass.setPipeline(this.pipeline)
    for (const d of draws) {
      const bind = this.device.createBindGroup({
        layout: this.bindLayout,
        entries: [
          { binding: 0, resource: { buffer: this.uniforms, offset: d.slot * UNIFORM_STRIDE, size: 48 } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: d.extTex },
        ],
      })
      pass.setBindGroup(0, bind)
      pass.draw(4)
    }
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  disposeEffects(): void {}

  dispose(): void {
    this.uniforms.destroy()
  }
}
