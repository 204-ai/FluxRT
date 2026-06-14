// Dense, collapsible detection-metrics overlay shown over the input preview
// when the sense overlay is visible ('overlay' / 'only'). Reuses InfoPanel.

import { useState } from 'react'
import { useSenseStore } from '../../state/senseStore'
import { InfoPanel } from './InfoPanel'

export function MetricsOverlay() {
  const analysis = useSenseStore((s) => s.analysis)
  const [open, setOpen] = useState(true)

  if (!open) {
    return (
      <button
        className="metrics-fab"
        title="Show detection metrics"
        aria-label="Show detection metrics"
        onClick={() => setOpen(true)}
      >
        📊
      </button>
    )
  }

  return (
    <div className="metrics-overlay" role="region" aria-label="Detection metrics">
      <div className="metrics-head">
        <span className="dim">metrics</span>
        <button
          className="metrics-x"
          title="Collapse metrics"
          aria-label="Collapse metrics"
          onClick={() => setOpen(false)}
        >
          ✕
        </button>
      </div>
      <InfoPanel analysis={analysis} />
    </div>
  )
}
