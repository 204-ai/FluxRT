// Dense, collapsible saved-prompts + ratings toolbar pinned to the top-right of
// the output Stage. Collapsed to a small ⭐ pill; expands into a compact panel.
// Absolutely positioned inside .remote-wrap (.overlay-anchor) so opening it
// never resizes/reflows the output video. Mirrors the input DrawToolbar pattern.

import { useState } from 'react'
import { ratingLabel } from '../../lib/features'
import { usePromptStore } from '../../state/promptStore'

function RatingSelect({ axis }: { axis: 'style' | 'tracking' | 'stability' }) {
  const value = usePromptStore((s) =>
    axis === 'style' ? s.rateStyle : axis === 'tracking' ? s.rateTracking : s.rateStability,
  )
  const setRating = usePromptStore((s) => s.setRating)
  return (
    <select value={value} onChange={(e) => setRating(axis, +e.target.value)}>
      <option value={0}>–</option>
      {[1, 2, 3, 4, 5].map((i) => (
        <option key={i} value={i}>
          {i}
        </option>
      ))}
    </select>
  )
}

export function RatingOverlay() {
  const p = usePromptStore()
  const [open, setOpen] = useState(false)

  if (!open) {
    return (
      <button
        className={'rate-fab' + (p.loopRunning ? ' active' : '')}
        title="Saved prompts & ratings"
        aria-label="Open saved prompts and ratings"
        onClick={() => setOpen(true)}
      >
        ⭐ {p.savedPrompts.length}
      </button>
    )
  }

  return (
    <div className="rate-widget" role="region" aria-label="Saved prompts and ratings">
      <div className="rate-head">
        <span className="dim">⭐ Saved ({p.savedPrompts.length})</span>
        <button className="rate-x" title="Collapse" aria-label="Collapse" onClick={() => setOpen(false)}>
          ✕
        </button>
      </div>
      <select
        className="rate-list"
        value=""
        onChange={(e) => {
          const i = parseInt(e.target.value, 10)
          if (!isNaN(i)) p.applySaved(i)
        }}
      >
        <option value="">pick a saved prompt…</option>
        {p.savedPrompts.map((e, i) => (
          <option key={i} value={i}>
            {`${ratingLabel(e)}  ${e.prompt.length > 70 ? e.prompt.slice(0, 70) + '…' : e.prompt}`}
          </option>
        ))}
      </select>
      <div className="rate-grid">
        <label className="dim" title="PROMPTING.md scale, 1–5">
          sty
          <RatingSelect axis="style" />
        </label>
        <label className="dim" title="Does the output follow your movement?">
          trk
          <RatingSelect axis="tracking" />
        </label>
        <label className="dim" title="Does the look hold steady frame to frame?">
          stb
          <RatingSelect axis="stability" />
        </label>
      </div>
      <div className="rate-actions">
        <button className="rate-btn" title="Save current prompt with these ratings" onClick={() => void p.saveCurrent()}>
          ♥
        </button>
        <button className="rate-btn" title="Remove current prompt from saved" onClick={() => void p.deleteCurrent()}>
          🗑
        </button>
        <label className="dim" title="Autoloop delay (seconds)">
          ↻
          <input type="number" min={2} value={p.loopDelay} onChange={(e) => p.setLoopDelay(+e.target.value)} />s
        </label>
        <button
          className={'rate-btn' + (p.loopRunning ? ' active' : '')}
          title="Cycle through the saved prompts in order"
          onClick={() => p.toggleLoop()}
        >
          {p.loopRunning ? '⏸' : '▶'}
        </button>
      </div>
      {p.savedStatus && <div className="dim rate-status">{p.savedStatus}</div>}
    </div>
  )
}
