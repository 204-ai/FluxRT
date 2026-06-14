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

export function SavedPromptsRow() {
  const p = usePromptStore()
  return (
    <div className="ref">
      <select
        style={{ flex: '1 1 300px' }}
        value=""
        onChange={(e) => {
          const i = parseInt(e.target.value, 10)
          if (!isNaN(i)) p.applySaved(i)
        }}
      >
        <option value="">{`⭐ Saved prompts (${p.savedPrompts.length})…`}</option>
        {p.savedPrompts.map((e, i) => (
          <option key={i} value={i}>
            {`${ratingLabel(e)}  ${e.prompt.length > 90 ? e.prompt.slice(0, 90) + '…' : e.prompt}`}
          </option>
        ))}
      </select>
      <label className="dim" title="PROMPTING.md scale, 1–5">
        style <RatingSelect axis="style" />
      </label>
      <label className="dim" title="Does the output follow your movement?">
        tracking <RatingSelect axis="tracking" />
      </label>
      <label className="dim" title="Does the look hold steady frame to frame?">
        stability <RatingSelect axis="stability" />
      </label>
      <button title="Save the current prompt with these ratings" onClick={() => void p.saveCurrent()}>
        ♥ Save
      </button>
      <button title="Remove the current prompt from saved" onClick={() => void p.deleteCurrent()}>
        🗑
      </button>
      <label className="dim">
        every{' '}
        <input
          type="number"
          min={2}
          style={{ width: 54 }}
          value={p.loopDelay}
          onChange={(e) => p.setLoopDelay(+e.target.value)}
        />{' '}
        s
      </label>
      <button className={p.loopRunning ? 'active' : ''} title="Cycle through the saved prompts in order" onClick={() => p.toggleLoop()}>
        {p.loopRunning ? '⏸ Stop' : '▶ Loop'}
      </button>
      <span className="dim">{p.savedStatus}</span>
    </div>
  )
}
