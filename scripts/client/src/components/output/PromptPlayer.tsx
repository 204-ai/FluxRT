// Saved-prompt picker + autoplay player, on one compact line below the prompt
// input. Picking a saved prompt applies it; ▶/⏸ cycles through them on a delay.

import { ratingLabel } from '../../lib/features'
import { usePromptStore } from '../../state/promptStore'
import { useFileDrop } from '../../lib/useFileDrop'

export function PromptPlayer() {
  const p = usePromptStore()
  const { open, inputProps } = useFileDrop((f) => void p.loadPromptsFromFile(f))
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
        <option value="">{`⭐ Saved (${p.savedPrompts.length})…`}</option>
        {p.savedPrompts.map((e, i) => (
          <option key={i} value={i}>
            {`${ratingLabel(e)}  ${e.prompt.length > 80 ? e.prompt.slice(0, 80) + '…' : e.prompt}`}
          </option>
        ))}
      </select>
      <button
        className="icon-btn"
        title="Load prompts from a file — one prompt per line, or a JSON array"
        aria-label="Load prompts from a file"
        onClick={open}
      >
        📂
      </button>
      <input {...inputProps} accept=".txt,.json,text/plain,application/json" hidden />
      <button
        className="icon-btn"
        title="Apply a random saved prompt"
        aria-label="Shuffle: apply a random saved prompt"
        onClick={() => p.shuffleSelect()}
      >
        🔀
      </button>
      <button
        className={'icon-btn' + (p.loopRunning ? ' on' : '')}
        title={p.loopRunning ? 'Stop autoplay' : 'Play through saved prompts'}
        aria-label={p.loopRunning ? 'Stop autoplay' : 'Play saved prompts'}
        onClick={() => p.toggleLoop()}
      >
        {p.loopRunning ? '⏸' : '▶'}
      </button>
      <input
        className="loop-delay"
        type="number"
        min={2}
        title="Autoplay delay (seconds)"
        value={p.loopDelay}
        onChange={(e) => p.setLoopDelay(+e.target.value)}
      />
      <span className="dim sp-status">{p.savedStatus}</span>
    </div>
  )
}
