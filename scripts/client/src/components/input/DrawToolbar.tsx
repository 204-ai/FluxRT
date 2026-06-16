// Floating, collapsible drawing toolbar shown over the input preview. Minimised
// to a single pencil button; expands in place. It is absolutely positioned
// inside #inputView (.overlay-anchor), so expanding it never resizes or reflows
// the preview. Drives the shared input draw state in the pipeline store.

import { useState } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'

export function DrawToolbar() {
  const p = usePipelineStore()
  const [open, setOpen] = useState(false)

  if (!open) {
    return (
      <button
        className={'draw-fab viewer-chrome' + (p.drawMode !== 'off' ? ' active' : '')}
        title="Drawing tools"
        aria-label="Open drawing tools"
        onClick={() => {
          // Opening the toolbar selects the pencil so you can draw immediately.
          setOpen(true)
          p.setDrawMode('brush')
        }}
      >
        ✏️
      </button>
    )
  }

  return (
    <div className="draw-widget viewer-chrome" role="toolbar" aria-label="Drawing tools">
      <button className="tool" title="Collapse" aria-label="Collapse drawing tools" onClick={() => setOpen(false)}>
        ✕
      </button>
      <button
        className={'tool' + (p.drawMode === 'brush' ? ' active' : '')}
        title="Brush"
        aria-label="Brush"
        aria-pressed={p.drawMode === 'brush'}
        onClick={() => p.setDrawMode(p.drawMode === 'brush' ? 'off' : 'brush')}
      >
        ✏️
      </button>
      <button
        className={'tool' + (p.drawMode === 'eraser' ? ' active' : '')}
        title="Eraser"
        aria-label="Eraser"
        aria-pressed={p.drawMode === 'eraser'}
        onClick={() => p.setDrawMode(p.drawMode === 'eraser' ? 'off' : 'eraser')}
      >
        🧽
      </button>
      <input
        type="color"
        value={p.drawColor}
        title="Color"
        aria-label="Brush color"
        onChange={(e) => p.setDrawColor(e.target.value)}
      />
      <input
        type="range"
        min={1}
        max={60}
        step={1}
        value={p.drawSize}
        title="Brush size"
        aria-label="Brush size"
        onChange={(e) => p.setDrawSize(+e.target.value)}
      />
      <span className="dim">{p.drawSize}</span>
      <button className="tool" title="Clear drawing" aria-label="Clear drawing" onClick={() => p.clearDrawing()}>
        🗑️
      </button>
    </div>
  )
}
