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

export interface RailStartOptions {
  deviceId: string | null
  width: number
  height: number
  fps: number
  mirrored: boolean
  effects: EffectInit[]
}

export interface EffectInit {
  name: string
  config: Record<string, unknown>
}

/** Callback receiving sampled source frames (pre-mirror) for analysis. */
export type TapCallback = (frame: ImageBitmap, tsMs: number) => void

export interface RailBackend {
  readonly kind: RailBackendKind
  /** Element to mount as the live preview (canvas or <video>). */
  readonly previewEl: HTMLCanvasElement | HTMLVideoElement
  readonly outputStream: MediaStream
  start(opts: RailStartOptions, raw: MediaStream): Promise<void>
  stop(): void
  setMirror(on: boolean): void
  configureEffect(name: string, patch: Record<string, unknown>): void
  effectMessage(name: string, data: unknown): void
  busPush(key: string, value: unknown): void
  /** Sample source frames at most every `intervalMs`; null disables the tap. */
  setTap(intervalMs: number, cb: TapCallback | null): void
  snapshot(type?: string): Promise<Blob>
}
