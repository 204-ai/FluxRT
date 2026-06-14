// Hand-marker styling as a collapsible popover (like the drawing toolbar),
// shown as a 🎨 button next to the landmark selector. Reads the pipeline store.

import { useState } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'

export function MarkerStyle() {
  const p = usePipelineStore()
  const [open, setOpen] = useState(false)

  return (
    <span className="marker-style">
      <button
        className="icon-btn"
        title="Marker style"
        aria-label="Marker style"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        🎨
      </button>
      {open && (
        <div className="marker-style-pop">
          <label>
            color{' '}
            <input type="color" value={p.markerColor} onChange={(e) => p.setMarkerColor(e.target.value)} />
          </label>
          <label>
            size{' '}
            <input
              type="range"
              min={6}
              max={120}
              step={1}
              value={p.markerSize}
              onChange={(e) => p.setMarkerSize(+e.target.value)}
            />
            <span className="dim">{p.markerSize}px</span>
          </label>
          <label>
            <input type="checkbox" checked={p.markerTrail} onChange={(e) => p.setMarkerTrail(e.target.checked)} /> Trail
          </label>
          <label>
            length{' '}
            <input
              type="range"
              min={4}
              max={80}
              step={1}
              value={p.markerTrailLen}
              onChange={(e) => p.setMarkerTrailLen(+e.target.value)}
            />
            <span className="dim">{p.markerTrailLen}</span>
          </label>
        </div>
      )}
    </span>
  )
}
