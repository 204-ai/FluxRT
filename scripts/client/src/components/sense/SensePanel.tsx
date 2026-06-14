// Human sensing panel: "sense human" (detection / expression / behavior, can
// drive the prompt) + the hand marker. Both run on the SAME shared input vision
// engine (inputVision) and always sense the composite input — there is no
// input/output source selector.

import { useSenseStore, type SenseOverlay } from '../../state/senseStore'
import { usePipelineStore } from '../../state/pipelineStore'
import { InfoPanel } from './InfoPanel'
import { MarkerStyle } from './MarkerStyle'

const LANDMARKS: Array<[number, string]> = [
  [15, 'Left wrist'],
  [16, 'Right wrist'],
  [19, 'Left index'],
  [20, 'Right index'],
  [0, 'Nose'],
  [11, 'Left shoulder'],
  [12, 'Right shoulder'],
]

export function SensePanel() {
  const enabled = useSenseStore((s) => s.enabled)
  const overlay = useSenseStore((s) => s.overlay)
  const status = useSenseStore((s) => s.status)
  const analysis = useSenseStore((s) => s.analysis)
  const setEnabled = useSenseStore((s) => s.setEnabled)
  const setOverlay = useSenseStore((s) => s.setOverlay)
  const camActive = usePipelineStore((s) => s.active)
  const p = usePipelineStore()

  return (
    <section className="sense-panel panel-box">
      <div className="section-label">Human sensing</div>

      <div className="controls">
        <label title="Detect a person and analyse expression / behavior from the input">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => void setEnabled(e.target.checked)}
            disabled={!camActive}
          />{' '}
          Sense human (detection, expression, behavior){camActive ? '' : ' — enable camera'}
        </label>
        {enabled && (
          <label className="dim" title="How the sense visualisation appears over the input preview">
            show{' '}
            <select value={overlay} onChange={(e) => setOverlay(e.target.value as SenseOverlay)}>
              <option value="overlay">overlay</option>
              <option value="only">only</option>
              <option value="off">don't show</option>
            </select>
          </label>
        )}
        <span className="dim">{status}</span>
      </div>
      {/* In overlay/only mode the metrics show as a dense overlay over the
          preview (see MetricsOverlay); only show them in-panel when hidden.
          The "drive prompt from sense" controls live at the bottom of the
          prompt panel (ComposeControls). */}
      {enabled && overlay === 'off' && <InfoPanel analysis={analysis} />}

      <div className="section-label">Hand marker</div>
      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={p.markerEnabled}
            disabled={!p.active}
            onChange={(e) => void p.setMarkerEnabled(e.target.checked)}
          />{' '}
          Enable
        </label>
        <label>
          landmark{' '}
          <select value={p.markerLandmark} onChange={(e) => p.setMarkerLandmark(+e.target.value)}>
            {LANDMARKS.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <MarkerStyle />
        <span className="dim">{p.poseStatus}</span>
      </div>
    </section>
  )
}
