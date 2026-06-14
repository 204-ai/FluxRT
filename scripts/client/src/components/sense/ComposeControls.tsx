// "Drive prompt from sense" — a single compact inline row at the bottom of the
// prompt panel (shown when sensing is enabled). Reads the shared sense store.

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
    <div className="controls compose-row">
      <label title="Compose a FLUX prompt from emotion (valence/arousal/gaze) + gesture slots; send when the combination changes">
        <input
          type="checkbox"
          checked={composeEnabled}
          onChange={(e) => setComposeEnabled(e.target.checked)}
        />{' '}
        🎭 Drive from sense
      </label>
      <select
        value={composeTheme}
        title="Theme"
        onChange={(e) => setComposeTheme(e.target.value as 'natural' | 'glitch')}
      >
        <option value="natural">🌿 natural</option>
        <option value="glitch">📺 glitch</option>
      </select>
      <label className="dim" title="Minimum seconds between prompt updates">
        gap{' '}
        <input
          type="number"
          min={1}
          value={composeMinGapSecs}
          onChange={(e) => setComposeMinGap(+e.target.value)}
        />{' '}
        s
      </label>
      {composeEnabled && (
        <span className="dim compose-readout" title={composePrompt || composeKey}>
          {composeKey ? `slots: ${composeKey}` : 'waiting…'}
          {composePrompt ? ` → ${composePrompt}` : ''}
        </span>
      )}
    </div>
  )
}
