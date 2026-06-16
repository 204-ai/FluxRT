// OBS-style framing overlay for the input composition. When a layer is picked
// for layout (pipelineStore.layoutLayer), this draws an interactive bounding
// box over the preview: drag the body to MOVE, drag a handle to RESIZE the
// frame (Shift locks aspect), or switch to CROP mode to trim the content edges.
// Everything is normalized to the output canvas [0..1] and written straight to
// the layer transform, so the live composite (and the frames sent upstream)
// update as you drag — no restart.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'
import { rail } from '../../state/runtime'
import type { LayerId, LayerTransform } from '../../pipeline/core/types'
import { identityTransform, layerDestRect, LAYER_IDS } from '../../pipeline/core/types'

type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w'
const HANDLES: Handle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
const LAYER_LABEL: Record<LayerId, string> = { camera: 'Camera', video: 'Video', feedback: 'Feedback' }

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v)

/** Measured pixel rect of the preview element, relative to our container. */
interface Stage {
  left: number
  top: number
  width: number
  height: number
}
function toPx(r: { x: number; y: number; w: number; h: number }, s: Stage) {
  return {
    left: s.left + r.x * s.width,
    top: s.top + r.y * s.height,
    width: r.w * s.width,
    height: r.h * s.height,
  }
}

/** Resize the FRAME (the full content placement). */
function resizeFrame(
  start: LayerTransform,
  handle: Handle,
  dnx: number,
  dny: number,
  lockAspect: boolean,
): LayerTransform {
  const MIN = 0.05
  const s = start.frame
  const right = s.x + s.w
  const bottom = s.y + s.h
  let x = s.x
  let y = s.y
  let w = s.w
  let h = s.h
  if (handle.includes('w')) {
    x = Math.min(s.x + dnx, right - MIN)
    w = right - x
  }
  if (handle.includes('e')) w = Math.max(MIN, s.w + dnx)
  if (handle.includes('n')) {
    y = Math.min(s.y + dny, bottom - MIN)
    h = bottom - y
  }
  if (handle.includes('s')) h = Math.max(MIN, s.h + dny)
  // Corner + Shift: keep the original aspect (undistorted content), re-anchoring
  // the opposite corner so the dragged corner follows the pointer.
  if (lockAspect && handle.length === 2) {
    const ratio = s.w / s.h || 1
    h = w / ratio
    if (handle.includes('n')) y = bottom - h
  }
  return { frame: { x, y, w, h }, crop: { ...start.crop } }
}

function moveFrame(start: LayerTransform, dnx: number, dny: number): LayerTransform {
  const { w, h } = start.frame
  // Keep at least 10% of the frame on-canvas so it can't be lost off-screen.
  const x = clamp(start.frame.x + dnx, 0.1 - w, 0.9)
  const y = clamp(start.frame.y + dny, 0.1 - h, 0.9)
  return { frame: { x, y, w, h }, crop: { ...start.crop } }
}

/** Trim the content edges (frame fixed) — the visible region shrinks within it. */
function cropEdit(start: LayerTransform, handle: Handle, dnx: number, dny: number): LayerTransform {
  const MIN = 0.02
  const fw = start.frame.w || 1
  const fh = start.frame.h || 1
  const c = { ...start.crop }
  if (handle.includes('w')) c.left = clamp(start.crop.left + dnx / fw, 0, 1 - start.crop.right - MIN)
  if (handle.includes('e')) c.right = clamp(start.crop.right - dnx / fw, 0, 1 - start.crop.left - MIN)
  if (handle.includes('n')) c.top = clamp(start.crop.top + dny / fh, 0, 1 - start.crop.bottom - MIN)
  if (handle.includes('s')) c.bottom = clamp(start.crop.bottom - dny / fh, 0, 1 - start.crop.top - MIN)
  return { frame: { ...start.frame }, crop: c }
}

export function TransformOverlay() {
  const layoutLayer = usePipelineStore((s) => s.layoutLayer)
  const cropMode = usePipelineStore((s) => s.cropMode)
  const active = usePipelineStore((s) => s.active)
  const layers = usePipelineStore((s) => s.layers)
  const camEnabled = usePipelineStore((s) => s.camEnabled)
  const videoLoaded = usePipelineStore((s) => s.videoLoaded)
  const feedbackAvailable = usePipelineStore((s) => s.feedbackAvailable)
  const setLayoutLayer = usePipelineStore((s) => s.setLayoutLayer)
  const setCropMode = usePipelineStore((s) => s.setCropMode)

  const containerRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)
  const [stage, setStage] = useState<Stage | null>(null)

  const present: Record<LayerId, boolean> = {
    camera: camEnabled,
    video: videoLoaded,
    feedback: feedbackAvailable,
  }
  const presentLayers = LAYER_IDS.filter((id) => present[id])

  // Measure the preview element's rect relative to our container so the box
  // tracks the actual displayed media (works maximized/letterboxed too).
  const measure = useCallback(() => {
    const cont = containerRef.current
    const prev = rail.previewEl
    if (!cont || !prev) return
    const cr = cont.getBoundingClientRect()
    const pr = prev.getBoundingClientRect()
    if (pr.width === 0 || pr.height === 0) return
    setStage({ left: pr.left - cr.left, top: pr.top - cr.top, width: pr.width, height: pr.height })
  }, [])

  // The `active` dep is load-bearing: rail.previewEl is not React state and gets
  // a fresh identity whenever Rail.start() builds a new backend. A full restart
  // toggles active (overlay remounts → observer rebinds to the new element);
  // hot-swaps keep the same backend/previewEl. If a future change rebuilds the
  // backend without toggling active, re-bind the observer to the element here.
  useLayoutEffect(() => {
    measure()
    const cont = containerRef.current
    const prev = rail.previewEl
    const ro = new ResizeObserver(measure)
    if (cont) ro.observe(cont)
    if (prev) ro.observe(prev)
    window.addEventListener('resize', measure)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [measure, layoutLayer, active])

  // End any in-flight drag when the framed layer changes or the overlay
  // unmounts — its captured layer id would otherwise go stale.
  useEffect(() => () => cleanupRef.current?.(), [layoutLayer])

  // If the framed layer disappears (e.g. video unloaded) switch to another
  // present layer or close the overlay.
  useEffect(() => {
    if (!layoutLayer) return
    if (!active || !present[layoutLayer]) setLayoutLayer(presentLayers[0] ?? null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutLayer, active, camEnabled, videoLoaded, feedbackAvailable])

  const beginDrag = (handle: Handle | 'move') => (e: React.PointerEvent) => {
    if (!stage || !layoutLayer) return
    e.preventDefault()
    e.stopPropagation()
    // Tear down any in-flight drag (e.g. a second touch) before starting a new
    // one, so its window listeners can't leak.
    cleanupRef.current?.()
    const layer = layoutLayer
    const crop = cropMode
    const startTransform = layers[layer]?.transform ?? identityTransform()
    const start: LayerTransform = {
      frame: { ...startTransform.frame },
      crop: { ...startTransform.crop },
    }
    const startX = e.clientX
    const startY = e.clientY
    const st = stage

    const onMove = (ev: PointerEvent) => {
      const dnx = (ev.clientX - startX) / st.width
      const dny = (ev.clientY - startY) / st.height
      let next: LayerTransform
      if (handle === 'move') next = moveFrame(start, dnx, dny)
      else if (crop) next = cropEdit(start, handle, dnx, dny)
      else next = resizeFrame(start, handle, dnx, dny, ev.shiftKey)
      usePipelineStore.getState().setLayerTransform(layer, next)
    }
    const onUp = () => cleanupRef.current?.()
    cleanupRef.current = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
      cleanupRef.current = null
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
  }

  if (!layoutLayer || !active) return <div ref={containerRef} className="layout-overlay" />

  const transform = layers[layoutLayer]?.transform ?? identityTransform()
  const dest = layerDestRect(transform)
  // In crop mode the handles act on the visible (dest) rect; in move mode on the
  // full frame. The other rect is shown faintly for context.
  const activeRect = cropMode ? dest : transform.frame
  const ghostRect = cropMode ? transform.frame : dest
  const activePx = stage ? toPx(activeRect, stage) : null
  const ghostPx = stage ? toPx(ghostRect, stage) : null

  return (
    <div ref={containerRef} className="layout-overlay">
      {/* faint context rect (frame while cropping, crop region while moving) */}
      {ghostPx && <div className="layout-ghost" style={ghostPx} />}

      {/* the active, editable box */}
      {activePx && (
        <div className={'layout-box' + (cropMode ? ' crop' : '')} style={activePx}>
          <div className="layout-move" onPointerDown={beginDrag('move')} />
          {HANDLES.map((h) => (
            <div key={h} className={'layout-handle h-' + h} onPointerDown={beginDrag(h)} />
          ))}
        </div>
      )}

      {/* mini toolbar */}
      <div className="layout-bar" onPointerDown={(e) => e.stopPropagation()}>
        {presentLayers.length > 1 &&
          presentLayers.map((id) => (
            <button
              key={id}
              className={'tool' + (id === layoutLayer ? ' active' : '')}
              title={`Frame the ${LAYER_LABEL[id]} layer`}
              onClick={() => setLayoutLayer(id)}
            >
              {LAYER_LABEL[id]}
            </button>
          ))}
        <span className="layout-sep" />
        <button
          className={'tool' + (!cropMode ? ' active' : '')}
          title="Move / resize the layer (Shift locks aspect)"
          aria-pressed={!cropMode}
          onClick={() => setCropMode(false)}
        >
          ✥ Move
        </button>
        <button
          className={'tool' + (cropMode ? ' active' : '')}
          title="Crop the layer's edges"
          aria-pressed={cropMode}
          onClick={() => setCropMode(true)}
        >
          ⌗ Crop
        </button>
        <button
          className="tool"
          title="Reset framing (fill, no crop)"
          onClick={() => usePipelineStore.getState().setLayerTransform(layoutLayer, null)}
        >
          ⟲ Reset
        </button>
        <button className="tool" title="Done framing" onClick={() => setLayoutLayer(null)}>
          ✓ Done
        </button>
      </div>
    </div>
  )
}
