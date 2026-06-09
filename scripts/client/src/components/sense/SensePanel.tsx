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
      {enabled && <InfoPanel analysis={analysis} />}
    </section>
  )
}
