// Hosts the pipeline-owned preview element (canvas or <video>) and binds
// freehand-draw pointer events. The element is created outside React and
// re-parented between the Input tab and the Output-tab split slot — identity
// must be preserved (it backs captureStream / snapshots), so we append/remove
// rather than render it.

import { useEffect, useRef } from 'react'
import { rail } from '../../state/runtime'
import { usePipelineStore } from '../../state/pipelineStore'

export function CanvasHost({ holds }: { holds: boolean }) {
  const slotRef = useRef<HTMLDivElement>(null)
  const active = usePipelineStore((s) => s.active)
  // Re-parents the (new) preview element whenever the backend is rebuilt.
  const previewEpoch = usePipelineStore((s) => s.previewEpoch)

  useEffect(() => {
    const slot = slotRef.current
    // union type (canvas | video) loses addEventListener overloads — widen
    const el = rail.previewEl as HTMLElement | null
    if (!slot || !el || !holds || !active) return

    slot.appendChild(el)

    const toNorm = (e: PointerEvent) => {
      const r = el.getBoundingClientRect()
      return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height }
    }
    const down = (e: PointerEvent) => {
      if (usePipelineStore.getState().drawMode === 'off') return
      el.setPointerCapture(e.pointerId)
      const p = toNorm(e)
      rail.beginStroke(p.x, p.y)
    }
    const move = (e: PointerEvent) => {
      if (usePipelineStore.getState().drawMode === 'off') return
      const p = toNorm(e)
      rail.moveStroke(p.x, p.y)
    }
    const up = () => rail.endStroke()

    el.addEventListener('pointerdown', down)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    el.addEventListener('pointercancel', up)
    el.addEventListener('pointerleave', up)

    return () => {
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
      el.removeEventListener('pointercancel', up)
      el.removeEventListener('pointerleave', up)
      if (el.parentElement === slot) slot.removeChild(el)
    }
  }, [holds, active, previewEpoch])

  return <div ref={slotRef} className="canvas-host" />
}
