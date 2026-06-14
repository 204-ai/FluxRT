// Saved-prompt picker + autoplay "player", shown directly below the prompt
// input. Picking a saved prompt applies it; the player cycles through the
// saved prompts on a delay. Ratings live in the scorecard (RatingOverlay).

import { ratingLabel } from '../../lib/features'
import { usePromptStore } from '../../state/promptStore'

export function PromptPlayer() {
  const p = usePromptStore()
  return (
    <div className="controls prompt-player">
      <select
        className="saved-pick"
        value=""
        onChange={(e) => {
          const i = parseInt(e.target.value, 10)
          if (!isNaN(i)) p.applySaved(i)
        }}
      >
        <option value="">{`⭐ Saved prompts (${p.savedPrompts.length})…`}</option>
        {p.savedPrompts.map((e, i) => (
          <option key={i} value={i}>
            {`${ratingLabel(e)}  ${e.prompt.length > 80 ? e.prompt.slice(0, 80) + '…' : e.prompt}`}
          </option>
        ))}
      </select>
      <button
        className={p.loopRunning ? 'active' : ''}
        title="Play through the saved prompts in order"
        onClick={() => p.toggleLoop()}
      >
        {p.loopRunning ? '⏸ Stop' : '▶ Play'}
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
      <span className="dim">{p.savedStatus}</span>
    </div>
  )
}
