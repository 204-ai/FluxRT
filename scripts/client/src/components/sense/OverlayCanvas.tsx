// Absolutely-positioned overlay canvas drawing the sense results (face mesh,
// skeleton, tracking box) over a target source. Display-only — never baked
// into the outbound stream. Subscribes imperatively; zero React re-renders.

import { useEffect, useRef } from 'react'
import { drawOverlay } from '../../vision/draw'
import { onSenseResult, useSenseStore, type SenseSource } from '../../state/senseStore'
import { usePipelineStore } from '../../state/pipelineStore'

export function OverlayCanvas({ source }: { source: SenseSource }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const enabled = useSenseStore((s) => s.enabled)
  const activeSource = useSenseStore((s) => s.source)
  const show = enabled && activeSource === source

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !show) return
    return onSenseResult((r) => {
      const parent = canvas.parentElement
      if (!parent) return
      // Match the displayed size; the renderer works in canvas pixels.
      const w = parent.clientWidth
      const h = parent.clientHeight
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
      const mirrored = source === 'input' && usePipelineStore.getState().mirror
      drawOverlay(canvas, r, mirrored)
    })
  }, [show, source])

  useEffect(() => {
    if (!show && canvasRef.current) {
      const c = canvasRef.current
      c.getContext('2d')?.clearRect(0, 0, c.width, c.height)
    }
  }, [show])

  return <canvas ref={canvasRef} className="sense-overlay" style={{ display: show ? 'block' : 'none' }} />
}
