// Sense feature panel: enable toggle, source selector, live analysis readout.

import { useSenseStore } from '../../state/senseStore'
import { usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import { InfoPanel } from './InfoPanel'

export function SensePanel() {
  const { enabled, source, status, analysis, setEnabled, setSource } = useSenseStore()
  const camActive = usePipelineStore((s) => s.active)
  const connected = useSessionStore((s) => s.connected)

  const inputDisabled = !camActive
  const outputDisabled = !connected

  return (
    <section className="sense-panel">
      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => void setEnabled(e.target.checked)}
            disabled={!camActive && !connected}
          />{' '}
          Sense human (detection, expression, behavior)
        </label>
        <label>
          source{' '}
          <select
            value={source}
            onChange={(e) => void setSource(e.target.value as 'input' | 'output')}
          >
            <option value="input" disabled={inputDisabled}>
              camera input{inputDisabled ? ' (enable camera)' : ''}
            </option>
            <option value="output" disabled={outputDisabled}>
              AI output{outputDisabled ? ' (connect first)' : ''}
            </option>
          </select>
        </label>
        <span className="dim">{status}</span>
      </div>
      {enabled && <ComposeControls />}
      {enabled && <InfoPanel analysis={analysis} />}
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
