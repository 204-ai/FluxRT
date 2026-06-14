// "Drive prompt from sense" controls — compose toggle, theme, min gap, and the
// live readout. Lives at the bottom of the prompt panel (shown when sensing is
// enabled) since it builds/sends the prompt. Reads the shared sense store.

import { useSenseStore } from '../../state/senseStore'

export function ComposeControls() {
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
