// Framework-agnostic media-rail core types. No React, no DOM assumptions
// beyond canvas — the same effect code runs on the main thread (canvas
// backend) and inside the pipeline worker (WebCodecs streams backend).

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

/** The compositing layers in stack order, front → back — the panel stacks them
 *  top to bottom in this order. Camera is frontmost; feedback (the previous
 *  output fed back in) is the backmost field the front layers paint over. Single
 *  source of truth: the layer type, default factory, and merge loop all derive
 *  from this, so adding/reordering a layer is one edit. */
export const LAYER_IDS = ['camera', 'video', 'feedback'] as const
export type LayerId = (typeof LAYER_IDS)[number]

/** Per-layer mix: every layer blends onto the accumulated result beneath it
 *  with its own opacity + blend — not just the top one. */
export interface LayerOptions {
  /** 0..1 */
  opacity: number
  blend: BlendMode
  /** OBS-style framing — move/resize/crop. Absent = legacy cover-fit (the
   *  source center-cropped to fill the canvas), so an untouched layer is
   *  unchanged. Materialized to identity on first edit. */
  transform?: LayerTransform
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

/** Mix settings for the whole stack. Layers are drawn back-to-front (feedback,
 *  then video, then camera) over an opaque base, so the panel's top row is on top. */
export type CompositeOptions = Record<LayerId, LayerOptions>

/** A live tweak to one or more layers — each layer's fields are optional. */
export type CompositePatch = Partial<Record<LayerId, Partial<LayerOptions>>>

/** Fresh default mix — every layer fully visible (opacity 1, normal blend), the
 *  standard layers-panel default. Returns a new object each call so each holder
 *  (compositor, rail, store) mutates its own copy. */
export function defaultComposite(): CompositeOptions {
  return {
    camera: { opacity: 1, blend: 'normal' },
    video: { opacity: 1, blend: 'normal' },
    feedback: { opacity: 1, blend: 'normal' },
  }
}

/** Apply a patch in place, per layer — a flat Object.assign would drop the
 *  untouched field (a blend-only patch must keep the layer's opacity). */
export function mergeComposite(target: CompositeOptions, patch: CompositePatch): void {
  for (const id of LAYER_IDS) {
    const p = patch[id]
    if (p) Object.assign(target[id], p)
  }
}

/** True when a stream carries at least one video track usable as a layer. */
export function hasVideoTrack(stream: MediaStream | null | undefined): boolean {
  return !!stream && stream.getVideoTracks().length > 0
}

/**
 * Input layers for a rail session. At least one must be non-null. The camera
 * layer (when present) is the base: it sets canvas dims, drives the frame
 * cadence, and is the vision-tap source.
 */
export interface SourceSet {
  cameraStream: MediaStream | null
  videoEl: HTMLVideoElement | null
}

export interface RailStartOptions {
  width: number
  height: number
  fps: number
  mirrored: boolean
  composite: CompositeOptions
  effects: EffectInit[]
}

export interface EffectInit {
  name: string
  config: Record<string, unknown>
}

/** Callback receiving sampled COMPOSITE frames (already mirrored when the
 *  camera mirror is on) for analysis. */
export type TapCallback = (frame: ImageBitmap, tsMs: number) => void

export interface RailBackend {
  readonly kind: RailBackendKind
  /** Element to mount as the live preview (canvas or <video>). */
  readonly previewEl: HTMLCanvasElement | HTMLVideoElement
  readonly outputStream: MediaStream
  start(opts: RailStartOptions, sources: SourceSet): Promise<void>
  stop(): void
  setMirror(on: boolean): void
  setComposite(patch: CompositePatch): void
  /** Set (or clear, with null) the feedback layer — the remote output stream
   *  fed back in. Hot add/remove in place, like the video overlay; no restart. */
  setFeedback(stream: MediaStream | null): void
  configureEffect(name: string, patch: Record<string, unknown>): void
  effectMessage(name: string, data: unknown): void
  busPush(key: string, value: unknown): void
  /** Sample source frames at most every `intervalMs`; null disables the tap. */
  setTap(intervalMs: number, cb: TapCallback | null): void
  /** Hot-swap the video-file input in place — re-feed the element's
   *  (re-captured) video track WITHOUT recreating the output track, so the
   *  WebRTC stream keeps flowing. */
  swapVideo(videoEl: HTMLVideoElement): void
  /** Hot-swap the camera input in place — feed the new device's track without
   *  recreating the output track (no pipeline restart / WebRTC renegotiation). */
  swapCamera(cameraStream: MediaStream): void
  /** Hot-REMOVE the video-file overlay in place — drop the overlay layer while
   *  the camera keeps feeding, WITHOUT recreating the output track (mirror of
   *  swapVideo). No-op when there is no overlay. */
  clearVideo(): void
  snapshot(type?: string): Promise<Blob>
}
