// Framework-agnostic media-rail core types. No React, no DOM assumptions
// beyond canvas — the same effect code runs on the main thread (canvas
// backend) and inside the pipeline worker (WebCodecs streams backend).

import type { ClipKind } from './clipKinds'
export type { ClipKind } from './clipKinds'

export type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

export interface FrameInfo {
  width: number
  height: number
  tsMs: number
}

/** Latest-value store analyzers publish into; effects read synchronously at render time. */
export interface BusReader {
  get<T>(key: string): T | undefined
}

/**
 * A composited layer in the effect chain. Rendered every frame, in order,
 * on top of the base video frame. Must be constructable in a worker:
 * OffscreenCanvas-compatible ops only, no DOM access.
 */
export interface CanvasEffect<Cfg = Record<string, unknown>> {
  readonly name: string
  config: Cfg
  configure(patch: Partial<Cfg>): void
  init?(width: number, height: number): void
  render(ctx: Ctx2D, info: FrameInfo, bus: BusReader): void
  /** Out-of-band messages (e.g. draw strokes) forwarded from the main thread. */
  message?(data: unknown): void
  dispose?(): void
}

export type RailBackendKind = 'streams' | 'canvas'

export type BlendMode = 'normal' | 'screen' | 'multiply' | 'difference'

// ---------------------------------------------------------------------------
// Dynamic layer/clip model (Resolume-style). The compositing stack is an
// ORDERED list of layers, front → back (the panel's top row is frontmost).
// Each layer carries its mix (opacity/blend), geometry (transform) and a
// per-layer selfie flip (mirror) — there is no fixed camera/video/feedback
// set. Which clip currently feeds a layer is tracked separately (the worker /
// backend hold the live frames keyed by layer id); the compositor only needs
// the per-layer RENDER options below.
// ---------------------------------------------------------------------------

/** Opaque per-layer instance id (was the closed 'camera'|'video'|'feedback'
 *  union). Stable across a layer's lifetime. */
export type LayerId = string
/** Opaque per-clip instance id, stable across activation/deactivation. */
export type ClipId = string

/** Per-layer mix: every layer blends onto the accumulated result beneath it
 *  with its own opacity + blend — not just the top one. (The geometry/render
 *  payload the compositor consumes; clip identity lives in the store/worker.) */
export interface LayerOptions {
  /** 0..1 */
  opacity: number
  blend: BlendMode
  /** OBS-style framing — move/resize/crop. Absent = legacy cover-fit (the
   *  source center-cropped to fill the canvas), so an untouched layer is
   *  unchanged. Materialized to identity on first edit. */
  transform?: LayerTransform
}

/** The render payload for ONE layer the compositor draws: its mix + geometry +
 *  per-layer selfie flip, tagged with the layer id so frames can be matched to
 *  it. `mirror` replaces the old global camera-only flip (it now belongs to the
 *  layer, seeded from the active clip's kind). */
export interface LayerRender extends LayerOptions {
  id: LayerId
  mirror: boolean
}

/** The whole compositing stack, ORDERED front → back (top row first). */
export type Composite = LayerRender[]

/** A structural / mix edit to the composite. A plain Record patch can't express
 *  order, insert or delete, so structure is mutated through explicit ops. */
export type LayerMixPatch = { id: LayerId } & Partial<Omit<LayerRender, 'id'>>
export type CompositeOp =
  | { op: 'patch'; layers: LayerMixPatch[] }
  | { op: 'add'; layer: LayerRender; index?: number }
  | { op: 'remove'; id: LayerId }
  | { op: 'reorder'; order: LayerId[] }

// --- Migration seed -------------------------------------------------------
// Stable ids for the three legacy layers. The store/UI still seed exactly these
// during P0/P1 so behavior is byte-identical; once the store is dynamic
// (P1) and add/remove lands (P2) these are just the default starting layers.
export const CAMERA_LAYER: LayerId = 'camera'
export const VIDEO_LAYER: LayerId = 'video'
export const FEEDBACK_LAYER: LayerId = 'feedback'

/** The kind each seeded layer's first clip is — used to seed `mirror` and to
 *  drive backend source binding during migration. */
export const SEED_LAYER_KIND: Record<LayerId, ClipKind> = {
  [CAMERA_LAYER]: 'camera',
  [VIDEO_LAYER]: 'video',
  [FEEDBACK_LAYER]: 'feedback',
}

/** Legacy seed id list (front → back). Retained so migration-era UI
 *  (TransformOverlay) and the store can enumerate the default layers; replaced
 *  by reading `layers[]` once the store is dynamic. */
export const LAYER_IDS = [CAMERA_LAYER, VIDEO_LAYER, FEEDBACK_LAYER] as const

/** Fresh default mix — the three legacy layers, every layer fully visible
 *  (opacity 1, normal blend), camera mirrored. Returns a new array each call so
 *  each holder (compositor, rail) mutates its own copy. */
export function defaultComposite(): Composite {
  return [
    { id: CAMERA_LAYER, opacity: 1, blend: 'normal', mirror: false },
    { id: VIDEO_LAYER, opacity: 1, blend: 'normal', mirror: false },
    { id: FEEDBACK_LAYER, opacity: 1, blend: 'normal', mirror: false },
  ]
}

/** Apply a structural/mix op to a composite in place. Patch looks each layer up
 *  by id (a flat Object.assign would clobber order); add/remove/reorder splice
 *  the ordered list. Unknown ids are ignored (a patch for a removed layer is a
 *  no-op, not a crash). */
export function applyCompositeOp(target: Composite, op: CompositeOp): void {
  switch (op.op) {
    case 'patch':
      for (const p of op.layers) {
        const layer = target.find((l) => l.id === p.id)
        if (!layer) continue
        // Carry every field explicitly (incl. transform: undefined → reset to
        // legacy cover-fit) so a patch that clears framing actually clears it.
        const dst = layer as unknown as Record<string, unknown>
        const src = p as unknown as Record<string, unknown>
        for (const k of Object.keys(p)) {
          if (k === 'id') continue
          dst[k] = src[k]
        }
      }
      break
    case 'add': {
      if (target.some((l) => l.id === op.layer.id)) break
      const i = op.index ?? 0
      target.splice(Math.max(0, Math.min(i, target.length)), 0, op.layer)
      break
    }
    case 'remove': {
      const i = target.findIndex((l) => l.id === op.id)
      if (i >= 0) target.splice(i, 1)
      break
    }
    case 'reorder': {
      const byId = new Map(target.map((l) => [l.id, l]))
      const next: Composite = []
      for (const id of op.order) {
        const l = byId.get(id)
        if (l) {
          next.push(l)
          byId.delete(id)
        }
      }
      // Any layer not named in the order keeps its relative position at the end.
      for (const l of target) if (byId.has(l.id)) next.push(l)
      target.length = 0
      target.push(...next)
      break
    }
  }
}

/**
 * OBS-style per-layer placement, all normalized to the OUTPUT canvas in [0..1].
 * The source is first cover-fit to fill the canvas (the legacy baseline);
 * `frame` then re-places that full content as a rect on the canvas, and `crop`
 * trims fractions off each edge of the content (clipped within the frame).
 * Identity (frame = full canvas, crop = 0) renders identically to the legacy
 * cover-fit path — an untouched layer is visually unchanged.
 */
export interface LayerTransform {
  /** Placement of the full (uncropped) cover-fit content, normalized 0..1. */
  frame: { x: number; y: number; w: number; h: number }
  /** Fractions trimmed off each content edge, 0..1 (left+right<1, top+bottom<1). */
  crop: { left: number; top: number; right: number; bottom: number }
}

/** Fresh identity transform — fills the canvas, no crop (== legacy cover-fit). */
export function identityTransform(): LayerTransform {
  return { frame: { x: 0, y: 0, w: 1, h: 1 }, crop: { left: 0, top: 0, right: 0, bottom: 0 } }
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v)

/** Cover-fit (center-crop) destination rect: scale the source up to fill the
 *  W×H canvas, centered — never stretched, never letterboxed. */
export function coverRect(
  W: number,
  H: number,
  w: number,
  h: number,
): { dx: number; dy: number; dw: number; dh: number } {
  if (w <= 0 || h <= 0) return { dx: 0, dy: 0, dw: W, dh: H }
  const scale = Math.max(W / w, H / h)
  const dw = w * scale
  const dh = h * scale
  return { dx: (W - dw) / 2, dy: (H - dh) / 2, dw, dh }
}

/** Visible (destination) rect of a transformed layer, normalized 0..1 — the
 *  frame inset by the crop. This is the on-screen box the UI manipulates. */
export function layerDestRect(t: LayerTransform): { x: number; y: number; w: number; h: number } {
  const f = t.frame
  const cl = clamp01(t.crop.left)
  const cr = clamp01(t.crop.right)
  const ct = clamp01(t.crop.top)
  const cb = clamp01(t.crop.bottom)
  return {
    x: f.x + cl * f.w,
    y: f.y + ct * f.h,
    w: Math.max(0, 1 - cl - cr) * f.w,
    h: Math.max(0, 1 - ct - cb) * f.h,
  }
}

export interface DrawRects {
  sx: number
  sy: number
  sw: number
  sh: number
  dx: number
  dy: number
  dw: number
  dh: number
}

/** Source + destination pixel rects for ctx.drawImage, given the canvas W×H,
 *  the source w×h and an optional transform. Returns null when nothing is
 *  drawable. With no transform this is the legacy cover-fit (whole source →
 *  cover rect, clipped by the canvas). */
export function layerDrawRects(
  W: number,
  H: number,
  w: number,
  h: number,
  t?: LayerTransform,
): DrawRects | null {
  if (w <= 0 || h <= 0 || W <= 0 || H <= 0) return null
  const cover = coverRect(W, H, w, h)
  if (!t) return { sx: 0, sy: 0, sw: w, sh: h, dx: cover.dx, dy: cover.dy, dw: cover.dw, dh: cover.dh }

  const cl = clamp01(t.crop.left)
  const cr = clamp01(t.crop.right)
  const ct = clamp01(t.crop.top)
  const cb = clamp01(t.crop.bottom)
  // Crop fractions are over the cover content, whose visible canvas span is the
  // full canvas — map each fraction back to source px through the cover inverse.
  const srcX = (frac: number) => ((frac * W - cover.dx) / cover.dw) * w
  const srcY = (frac: number) => ((frac * H - cover.dy) / cover.dh) * h
  const sx = srcX(cl)
  const sw = srcX(1 - cr) - sx
  const sy = srcY(ct)
  const sh = srcY(1 - cb) - sy

  const d = layerDestRect(t)
  const dx = d.x * W
  const dy = d.y * H
  const dw = d.w * W
  const dh = d.h * H
  if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return null
  return { sx, sy, sw, sh, dx, dy, dw, dh }
}

/** True when a stream carries at least one video track usable as a layer. */
export function hasVideoTrack(stream: MediaStream | null | undefined): boolean {
  return !!stream && stream.getVideoTracks().length > 0
}

/**
 * The single BASE source a rail starts on — it sets the canvas dims, drives the
 * frame cadence, and is the vision-tap source. Acquired by the store (no
 * getUserMedia inside the rail) and identified by its layer id (not hardcoded to
 * a camera/video slot). Every other clip hot-attaches after start via
 * RailBackend.setLayerSource. A mediastream base (camera/screen/feedback) sets
 * `stream`; an element base (video file) sets `videoEl`.
 */
export interface BaseSource {
  layerId: LayerId
  kind: ClipKind
  stream: MediaStream | null
  videoEl: HTMLVideoElement | null
}

export interface SourceSet {
  base: BaseSource
}

export interface RailStartOptions {
  width: number
  height: number
  fps: number
  composite: Composite
  effects: EffectInit[]
}

export interface EffectInit {
  name: string
  config: Record<string, unknown>
}

/** Callback receiving sampled COMPOSITE frames (already mirrored when a layer's
 *  mirror is on) for analysis. */
export type TapCallback = (frame: ImageBitmap, tsMs: number) => void

/** A live source for one layer handed to a backend: a MediaStream (camera /
 *  feedback) or an <video> element (file). null removes the layer's source. */
export type LayerSourceInput = MediaStream | HTMLVideoElement | null

export interface RailBackend {
  readonly kind: RailBackendKind
  /** Element to mount as the live preview (canvas or <video>). */
  readonly previewEl: HTMLCanvasElement | HTMLVideoElement
  readonly outputStream: MediaStream
  start(opts: RailStartOptions, sources: SourceSet): Promise<void>
  stop(): void
  /** Apply a structural/mix op to the composite (patch/add/remove/reorder).
   *  Hot — never restarts; mirror is just a per-layer field in a patch. */
  setComposite(op: CompositeOp): void
  /** Bind (or clear, with null source) one layer's live source IN PLACE — the
   *  generalized hot add/swap/remove that keeps the output track alive (no
   *  restart, no WebRTC renegotiation). `kind` tells the backend how to cross
   *  the source in (mediastream / clone / element). */
  setLayerSource(id: LayerId, kind: ClipKind, source: LayerSourceInput): void
  configureEffect(name: string, patch: Record<string, unknown>): void
  effectMessage(name: string, data: unknown): void
  busPush(key: string, value: unknown): void
  /** Sample source frames at most every `intervalMs`; null disables the tap. */
  setTap(intervalMs: number, cb: TapCallback | null): void
  snapshot(type?: string): Promise<Blob>
}
