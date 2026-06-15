// Simple prompt scorecard pinned to the top-right of the output stage.
// Collapsed to a ⭐ button; expands into a compact card of star-bar ratings
// (style / tracking / stability) + save/delete. The saved-prompt picker and
// the autoplay player live below the prompt input (PromptPlayer), not here.
// Absolutely positioned inside .remote-wrap so opening it never reflows the video.

import { useState } from 'react'
import { usePromptStore } from '../../state/promptStore'

function StarRating({ axis }: { axis: 'style' | 'tracking' | 'stability' }) {
  const value = usePromptStore((s) =>
    axis === 'style' ? s.rateStyle : axis === 'tracking' ? s.rateTracking : s.rateStability,
  )
  const setRating = usePromptStore((s) => s.setRating)
  return (
    <span className="stars" role="radiogroup" aria-label={`${axis} rating`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <button
          key={i}
          type="button"
          className={'star' + (i <= value ? ' on' : '')}
          title={`${i} / 5`}
          aria-label={`${axis} ${i} of 5`}
          onClick={() => setRating(axis, value === i ? 0 : i)}
        >
          ★
        </button>
      ))}
    </span>
  )
}

export function RatingOverlay() {
  const p = usePromptStore()
  const [open, setOpen] = useState(false)

  if (!open) {
    return (
      <button className="rate-fab viewer-chrome" title="Rate this prompt" aria-label="Open scorecard" onClick={() => setOpen(true)}>
        ⭐
      </button>
    )
  }

  return (
    <div className="rate-widget viewer-chrome" role="region" aria-label="Prompt scorecard">
      <div className="rate-head">
        <span className="dim">Score</span>
        <button className="rate-x" title="Collapse" aria-label="Collapse" onClick={() => setOpen(false)}>
          ✕
        </button>
      </div>
      <div className="rate-card">
        <label className="dim" title="PROMPTING.md scale, 1–5">
          <span>style</span>
          <StarRating axis="style" />
        </label>
        <label className="dim" title="Does the output follow your movement?">
          <span>tracking</span>
          <StarRating axis="tracking" />
        </label>
        <label className="dim" title="Does the look hold steady frame to frame?">
          <span>stability</span>
          <StarRating axis="stability" />
        </label>
      </div>
      <div className="rate-actions">
        <button className="rate-btn" title="Save current prompt with these ratings" onClick={() => void p.saveCurrent()}>
          ♥ Save
        </button>
        <button className="rate-btn" title="Remove current prompt from saved" onClick={() => void p.deleteCurrent()}>
          🗑
        </button>
      </div>
      {p.savedStatus && <div className="dim rate-status">{p.savedStatus}</div>}
    </div>
  )
}
