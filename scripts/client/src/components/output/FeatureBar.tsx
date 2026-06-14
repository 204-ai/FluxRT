// Facial-feature builder bar: grouped facial dropdown (resets to placeholder
// so another can be added), persistent style slot, randomize, reset.

import { FEATURE_ORDER, FEATURES, STYLES } from '../../lib/features'
import { usePromptStore } from '../../state/promptStore'

export function FeatureBar() {
  const styleSelection = usePromptStore((s) => s.styleSelection)
  const applyFeature = usePromptStore((s) => s.applyFeature)
  const randomize = usePromptStore((s) => s.randomize)
  const resetFeatures = usePromptStore((s) => s.resetFeatures)

  return (
    <div className="controls" id="featBar">
      <select
        className="feat"
        value=""
        onChange={(e) => {
          const opt = e.target.selectedOptions[0]
          if (!opt?.value) return
          applyFeature(opt.dataset.feat as (typeof FEATURE_ORDER)[number], opt.value)
        }}
      >
        <option value="">🙂 Add facial feature…</option>
        {FEATURE_ORDER.map((k) => (
          <optgroup key={k} label={`${FEATURES[k].emoji} ${FEATURES[k].label}`}>
            {FEATURES[k].opts.map((p) => (
              <option key={p} value={p} data-feat={k}>
                {p}
              </option>
            ))}
          </optgroup>
        ))}
      </select>

      <select className="feat" value={styleSelection} onChange={(e) => applyFeature('style', e.target.value)}>
        <option value="">🎨 Style…</option>
        {STYLES.map((p) => (
          <option key={p} value={p}>
            {p.replace(/^in the style of /, '')}
          </option>
        ))}
      </select>

      <button className="icon-btn" title="Random facial features + style" aria-label="Randomize" onClick={() => randomize()}>
        🎲
      </button>
      <button className="icon-btn" title="Clear features, restore default prompt" aria-label="Reset" onClick={() => resetFeatures()}>
        ↺
      </button>
    </div>
  )
}
