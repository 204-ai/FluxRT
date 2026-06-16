// Saved-prompt picker + autoplay player. Line 1: pick / load / shuffle / play.
// Line 2: rating filters (👎/👍/❤️ from the localStorage triage) + a small info
// message. Filters only narrow the dropdown; shuffle/autoplay still walk the
// full saved list. Each option is prefixed with its triage verdict, if any.

import { useState } from 'react'
import { ratingLabel } from '../../lib/features'
import { usePromptStore } from '../../state/promptStore'
import { useRatingStore } from '../../state/ratingStore'
import { useFileDrop } from '../../lib/useFileDrop'
import type { Verdict } from '../../lib/ratings'

const VERDICT_ICON: Record<Verdict, string> = { skip: '👎', like: '👍', love: '❤️' }
const FILTERS: { key: Verdict | 'all'; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'skip', label: 'Disliked' },
  { key: 'like', label: 'Liked' },
  { key: 'love', label: 'Loved' },
]

export function PromptPlayer() {
  const p = usePromptStore()
  const ratings = useRatingStore((s) => s.ratings)
  const [filter, setFilter] = useState<Verdict | 'all'>('all')
  const { open, inputProps } = useFileDrop((f) => void p.loadPromptsFromFile(f))

  const verdictOf = (prompt: string): Verdict | undefined => ratings[prompt.trim()]?.verdict
  // Keep each prompt's original index so applySaved(i) / the autoloop stay valid
  // even when the list is filtered down.
  const visible = p.savedPrompts
    .map((e, i) => ({ e, i }))
    .filter(({ e }) => filter === 'all' || verdictOf(e.prompt) === filter)
  const ratedCount = p.savedPrompts.filter((e) => verdictOf(e.prompt)).length

  const info =
    (filter === 'all'
      ? `${p.savedPrompts.length} saved · ${ratedCount} rated`
      : `${visible.length} of ${p.savedPrompts.length} ${VERDICT_ICON[filter]}`) +
    (p.savedStatus ? ` · ${p.savedStatus}` : '')

  return (
    <div className="prompt-player-wrap">
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
          {visible.map(({ e, i }) => {
            const vi = verdictOf(e.prompt)
            const text = e.prompt.length > 80 ? e.prompt.slice(0, 80) + '…' : e.prompt
            // Every line leads with a rating icon: the triage verdict, or ○ when unrated.
            return (
              <option key={i} value={i}>
                {`${vi ? VERDICT_ICON[vi] : '○'} ${ratingLabel(e)}  ${text}`}
              </option>
            )
          })}
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
      </div>
      <div className="sp-info">
        <span className="sp-filters" role="group" aria-label="Filter saved prompts by rating">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              className={'sp-filter' + (filter === f.key ? ' on' : '')}
              title={`Show ${f.label.toLowerCase()}`}
              aria-pressed={filter === f.key}
              onClick={() => setFilter((cur) => (cur === f.key ? 'all' : f.key))}
            >
              {f.key === 'all' ? 'All' : VERDICT_ICON[f.key]}
            </button>
          ))}
        </span>
        <span className="dim sp-status">{info}</span>
      </div>
    </div>
  )
}
