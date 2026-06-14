// Framework-agnostic media-rail core types. No React, no DOM assumptions
// beyond canvas — the same effect code runs on the main thread (canvas
// backend) and inside the pipeline worker (WebCodecs streams backend).

export type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

export interface FrameInfo {
  width: number
  height: number
  tsMs: number
  /** True when the base frame was drawn horizontally flipped (selfie view). */
  mirrored: boolean
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
export type LayerOrder = 'camera-over' | 'video-over'

export interface CompositeOptions {
  order: LayerOrder
  /** Opacity of the top layer, 0..1. */
  opacity: number
  blend: BlendMode
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
  setComposite(patch: Partial<CompositeOptions>): void
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
