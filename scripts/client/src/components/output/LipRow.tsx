import { useState } from 'react'
import { setLipTransfer } from '../../lib/api'
import { useSessionStore } from '../../state/sessionStore'

export function LipRow() {
  const lipEnabled = useSessionStore((s) => s.lipEnabled)
  const lipActive = useSessionStore((s) => s.lipActive)
  const [busy, setBusy] = useState(false)

  const label = !lipEnabled
    ? 'lipsync: unavailable (add lip_transfer to config)'
    : lipActive
      ? 'lipsync: ON'
      : 'lipsync: OFF'

  return (
    <div className="controls">
      <label>
        <input
          type="checkbox"
          disabled={!lipEnabled || busy}
          checked={lipActive}
          onChange={async (e) => {
            const on = e.target.checked
            setBusy(true)
            try {
              const j = await setLipTransfer(on)
              useSessionStore.getState().setLip(true, j.lip_active)
            } catch (err) {
              useSessionStore
                .getState()
                .logLine('Lip transfer toggle failed: ' + (err instanceof Error ? err.message : err))
            } finally {
              setBusy(false)
            }
          }}
        />{' '}
        Lip transfer
      </label>
      <span className="dim">{label}</span>
    </div>
  )
}
