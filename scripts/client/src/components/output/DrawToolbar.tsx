// GIMP-style edit toolbar shown next to the input preview in split mode.
// Shares the centralized draw state with the Input-tab controls.

import { usePipelineStore } from '../../state/pipelineStore'

export function DrawToolbar() {
  const p = usePipelineStore()
  return (
    <div className="tool-bar">
      <button
        className={'tool' + (p.drawMode === 'brush' ? ' active' : '')}
        title="Brush"
        onClick={() => p.setDrawMode(p.drawMode === 'brush' ? 'off' : 'brush')}
      >
        ✏️
      </button>
      <button
        className={'tool' + (p.drawMode === 'eraser' ? ' active' : '')}
        title="Eraser"
        onClick={() => p.setDrawMode(p.drawMode === 'eraser' ? 'off' : 'eraser')}
      >
        🧽
      </button>
      <input type="color" value={p.drawColor} title="Color" onChange={(e) => p.setDrawColor(e.target.value)} />
      <input
        type="range"
        min={1}
        max={60}
        step={1}
        value={p.drawSize}
        title="Brush size"
        onChange={(e) => p.setDrawSize(+e.target.value)}
      />
      <span className="dim" style={{ textAlign: 'center' }}>
        {p.drawSize}
      </span>
      <button className="tool" title="Clear drawing" onClick={() => p.clearDrawing()}>
        🗑️
      </button>
    </div>
  )
}
