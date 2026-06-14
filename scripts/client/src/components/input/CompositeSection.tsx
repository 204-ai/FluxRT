// Compositing "mixer" shown between the camera and video panels when both are
// live: a vertical opacity fader (camera over video) with the blend mode as an
// icon toolbar button below it. Changes apply live (no pipeline restart).

import { usePipelineStore } from '../../state/pipelineStore'
import type { BlendMode } from '../../pipeline/core/types'

const BLENDS: BlendMode[] = ['normal', 'screen', 'multiply', 'difference']
const BLEND_SHORT: Record<BlendMode, string> = {
  normal: 'nrm',
  screen: 'scr',
  multiply: 'mul',
  difference: 'dif',
}

export function CompositeSection() {
  const p = usePipelineStore()
  if (!(p.camEnabled && p.videoLoaded && p.active)) return null

  const pct = Math.round(p.compositeOpacity * 100)
  const cycleBlend = () => {
    const i = BLENDS.indexOf(p.compositeBlend)
    p.setCompositeBlend(BLENDS[(i + 1) % BLENDS.length])
  }

  return (
    <div className="mixer" title="Camera / video mix">
      <input
        className="mixer-slider"
        type="range"
        min={0}
        max={100}
        step={1}
        value={pct}
        aria-label="Camera-over-video opacity"
        onChange={(e) => p.setCompositeOpacity(+e.target.value / 100)}
      />
      <span className="dim mixer-val">{pct}%</span>
      <button
        className="icon-btn"
        title={`Blend: ${p.compositeBlend} (click to change)`}
        aria-label="Blend mode"
        onClick={cycleBlend}
      >
        {BLEND_SHORT[p.compositeBlend]}
      </button>
    </div>
  )
}
