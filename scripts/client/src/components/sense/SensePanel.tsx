// Human sensing panel: "sense human" (detection / expression / behavior, can
// drive the prompt) + the hand marker. Both run on the SAME shared input vision
// engine (inputVision) and always sense the composite input — there is no
// input/output source selector.

import { useSenseStore, type SenseOverlay } from '../../state/senseStore'
import { usePipelineStore } from '../../state/pipelineStore'
import { InfoPanel } from './InfoPanel'

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
      {enabled && <ComposeControls />}
      {enabled && <InfoPanel analysis={analysis} />}

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
        <label>
          color <input type="color" value={p.markerColor} onChange={(e) => p.setMarkerColor(e.target.value)} />
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
        </label>
        <span className="dim" style={{ minWidth: 28 }}>
          {p.markerSize}px
        </span>
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
        </label>
        <span className="dim" style={{ minWidth: 28 }}>
          {p.markerTrailLen}
        </span>
        <span className="dim">{p.poseStatus}</span>
      </div>
    </section>
  )
}

function ComposeControls() {
  const composeEnabled = useSenseStore((s) => s.composeEnabled)
  const composeTheme = useSenseStore((s) => s.composeTheme)
  const composeMinGapSecs = useSenseStore((s) => s.composeMinGapSecs)
  const composeKey = useSenseStore((s) => s.composeKey)
  const composePrompt = useSenseStore((s) => s.composePrompt)
  const setComposeEnabled = useSenseStore((s) => s.setComposeEnabled)
  const setComposeTheme = useSenseStore((s) => s.setComposeTheme)
  const setComposeMinGap = useSenseStore((s) => s.setComposeMinGap)

  return (
    <div className="controls">
      <label title="Compose a FLUX prompt from emotion (valence/arousal/gaze) + gesture slots; send when the combination changes">
        <input
          type="checkbox"
          checked={composeEnabled}
          onChange={(e) => setComposeEnabled(e.target.checked)}
        />{' '}
        🎭 Drive prompt from sense
      </label>
      <label>
        theme{' '}
        <select value={composeTheme} onChange={(e) => setComposeTheme(e.target.value as 'natural' | 'glitch')}>
          <option value="natural">🌿 natural — painterly, weather, flora</option>
          <option value="glitch">📺 glitch — mosaic, chrome, phosphor</option>
        </select>
      </label>
      <label className="dim">
        min gap{' '}
        <input
          type="number"
          min={1}
          style={{ width: 54 }}
          value={composeMinGapSecs}
          onChange={(e) => setComposeMinGap(+e.target.value)}
        />{' '}
        s
      </label>
      {composeEnabled && (
        <span className="dim compose-readout">
          {composeKey ? `slots: ${composeKey}` : 'waiting for detection…'}
          {composePrompt ? ` → ${composePrompt.slice(0, 90)}…` : ''}
        </span>
      )}
    </div>
  )
}
