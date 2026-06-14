// Background removal controls: toggle the segmenter, pick how the background
// is replaced (blur / solid color / transparent), and tune the active mode.
// Reads the pipeline store; changes apply live (no pipeline restart).

import { usePipelineStore } from '../../state/pipelineStore'
import type { BackgroundMode } from '../../pipeline/effects/backgroundRemoval'

const MODES: BackgroundMode[] = ['blur', 'color', 'transparent']

export function BackgroundSection() {
  const p = usePipelineStore()
  // Only meaningful once a camera/video is feeding the rail.
  if (!p.active) return null

  return (
    <div className="src-panel">
      <div className="src-title">Background</div>
      <div className="controls src-row">
        <label className="dim" title="Replace the background using on-device segmentation">
          <input
            type="checkbox"
            checked={p.bgEnabled}
            onChange={(e) => void p.setBgEnabled(e.target.checked)}
          />{' '}
          Remove
        </label>
        <select
          className="device-pick"
          value={p.bgMode}
          disabled={!p.bgEnabled}
          onChange={(e) => p.setBgMode(e.target.value as BackgroundMode)}
        >
          {MODES.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        {p.bgMode === 'blur' && (
          <label className="dim">
            blur{' '}
            <input
              type="range"
              min={2}
              max={40}
              step={1}
              value={p.bgBlur}
              disabled={!p.bgEnabled}
              onChange={(e) => p.setBgBlur(+e.target.value)}
            />
            <span className="dim">{p.bgBlur}px</span>
          </label>
        )}
        {p.bgMode === 'color' && (
          <label className="dim">
            color{' '}
            <input
              type="color"
              value={p.bgColor}
              disabled={!p.bgEnabled}
              onChange={(e) => p.setBgColor(e.target.value)}
            />
          </label>
        )}
        <label className="dim" title="Flip foreground/background if the matte reads inverted">
          <input
            type="checkbox"
            checked={p.bgInvert}
            disabled={!p.bgEnabled}
            onChange={(e) => p.setBgInvert(e.target.checked)}
          />{' '}
          Invert
        </label>
      </div>
      <span className="dim">{p.bgStatus || 'segments you from the background on-device'}</span>
    </div>
  )
}
