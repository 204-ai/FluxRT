// Quick triage bar over the output video: skip / like / love the current
// prompt, plus an export-to-JSON button. Bottom-center, revealed with the other
// viewer chrome on hover/tap (the .viewer-chrome class). Persists to
// localStorage via ratingStore — separate from the server-backed star scorecard.

import { usePromptStore } from '../../state/promptStore'
import { useRatingStore } from '../../state/ratingStore'
import type { Verdict } from '../../lib/ratings'

const VERDICTS: { v: Verdict; icon: string; label: string }[] = [
  { v: 'skip', icon: '👎', label: 'Dislike' },
  { v: 'like', icon: '👍', label: 'Like' },
  { v: 'love', icon: '❤️', label: 'Love' },
]

export function RatingBar() {
  const prompt = usePromptStore((s) => s.prompt)
  const rate = useRatingStore((s) => s.rate)
  const exportJson = useRatingStore((s) => s.exportJson)
  // Highlight the verdict already set for this exact prompt.
  const current = useRatingStore((s) => s.ratings[prompt.trim()]?.verdict ?? null)

  if (!prompt.trim()) return null

  return (
    <div className="rate-bar viewer-chrome" role="group" aria-label="Rate this prompt">
      {VERDICTS.map(({ v, icon, label }) => (
        <button
          key={v}
          className={'rate-pill' + (current === v ? ' on' : '')}
          title={label}
          aria-label={label}
          aria-pressed={current === v}
          onClick={() => rate(v)}
        >
          {icon}
        </button>
      ))}
      <button className="rate-pill rate-export" title="Export ratings as JSON" aria-label="Export ratings as JSON" onClick={() => exportJson()}>
        ⬇
      </button>
    </div>
  )
}
