// Background removal / replacement, composited as an effect layer. Reads the
// foreground matte the image segmenter publishes on the bus (key
// 'segmentation') — same decoupled path as the pose marker, so detection
// cadence (~15Hz) is independent of the compositing frame rate.
//
// Ordering: this effect must run BEFORE marker/drawLayer so those render on
// top of the replaced background. It rebuilds the whole frame each pass:
//   1. snapshot the composited subject (camera + video, already drawn)
//   2. paint the new background (blurred copy / solid color / transparent)
//   3. cut the snapshot to the foreground via the mask alpha
//   4. draw the cut-out subject back on top
//
// The matte arrives at model resolution (256x256); scaling it up to the canvas
// with smoothing on feathers the edge for free. The mask is in the SAME
// (already-mirrored) space as the composite — it's segmented from the tapped
// composite frame — so it's applied 1:1 with no flip (see marker.ts).

import type { BusReader, CanvasEffect, Ctx2D, FrameInfo } from '../core/types'
import type { SegmentationMask } from '../../vision/types'

export type BackgroundMode = 'blur' | 'color' | 'transparent'

export interface BackgroundConfig {
  enabled: boolean
  mode: BackgroundMode
  /** Gaussian blur radius (px) for 'blur' mode. */
  blur: number
  /** Fill color for 'color' mode. */
  color: string
  /** Flip foreground/background — safety valve if the model's mask reads inverted. */
  invert: boolean
}

export function createBackgroundEffect(
  initial?: Partial<BackgroundConfig>,
): CanvasEffect<BackgroundConfig> {
  // Snapshot of the composited subject, reused each frame.
  let snap: OffscreenCanvas | null = null
  let sctx: OffscreenCanvasRenderingContext2D | null = null
  // Small canvas holding the matte as an alpha channel; rebuilt only when a
  // new mask arrives (bus overwrites the reference) or its config changes.
  let maskCanvas: OffscreenCanvas | null = null
  let mctx: OffscreenCanvasRenderingContext2D | null = null
  let lastMask: SegmentationMask | null = null
  let lastInvert = false
  let W = 0
  let H = 0

  function rebuildMask(seg: SegmentationMask, invert: boolean): void {
    if (!maskCanvas || maskCanvas.width !== seg.width || maskCanvas.height !== seg.height) {
      maskCanvas = new OffscreenCanvas(seg.width, seg.height)
      mctx = maskCanvas.getContext('2d')
    }
    if (!mctx) return
    const img = mctx.createImageData(seg.width, seg.height)
    const d = img.data
    for (let i = 0; i < seg.data.length; i++) {
      const a = invert ? 255 - seg.data[i] : seg.data[i]
      const j = i * 4
      d[j] = 255
      d[j + 1] = 255
      d[j + 2] = 255
      d[j + 3] = a
    }
    mctx.putImageData(img, 0, 0)
  }

  return {
    name: 'backgroundRemoval',
    config: { enabled: false, mode: 'blur', blur: 12, color: '#00ff00', invert: false, ...initial },
    configure(patch) {
      Object.assign(this.config, patch)
    },
    init(width: number, height: number) {
      W = width
      H = height
      snap = new OffscreenCanvas(W, H)
      sctx = snap.getContext('2d')
      lastMask = null
    },
    render(ctx: Ctx2D, info: FrameInfo, bus: BusReader) {
      const o = this.config
      if (!o.enabled || !snap || !sctx) return
      const seg = bus.get<SegmentationMask>('segmentation')
      // No matte yet (segmenter still loading / between samples): pass the
      // frame through untouched rather than blanking it.
      if (!seg) return
      if (seg !== lastMask || o.invert !== lastInvert) {
        rebuildMask(seg, o.invert)
        lastMask = seg
        lastInvert = o.invert
      }
      if (!maskCanvas) return

      const w = info.width
      const h = info.height

      // 1. Snapshot the already-composited subject.
      sctx.globalCompositeOperation = 'copy'
      sctx.filter = 'none'
      sctx.drawImage(ctx.canvas, 0, 0, w, h)

      // 2. Paint the background onto the live frame.
      if (o.mode === 'blur') {
        ctx.save()
        ctx.filter = `blur(${o.blur}px)`
        ctx.drawImage(snap, 0, 0, w, h)
        ctx.restore()
      } else if (o.mode === 'color') {
        ctx.save()
        ctx.fillStyle = o.color
        ctx.fillRect(0, 0, w, h)
        ctx.restore()
      } else {
        ctx.clearRect(0, 0, w, h)
      }

      // 3. Cut the snapshot down to the foreground (mask scaled up, feathered).
      sctx.globalCompositeOperation = 'destination-in'
      sctx.imageSmoothingEnabled = true
      sctx.drawImage(maskCanvas, 0, 0, w, h)
      sctx.globalCompositeOperation = 'source-over'

      // 4. Draw the cut-out subject over the new background.
      ctx.drawImage(snap, 0, 0, w, h)
    },
    dispose() {
      snap = null
      sctx = null
      maskCanvas = null
      mctx = null
      lastMask = null
    },
  }
}
