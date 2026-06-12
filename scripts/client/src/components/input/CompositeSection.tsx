// Two-layer compositing controls — shown only when camera + video are both
// live. Changes apply live (no pipeline restart).

import { usePipelineStore } from '../../state/pipelineStore'
import type { BlendMode, LayerOrder } from '../../pipeline/core/types'

const BLENDS: BlendMode[] = ['normal', 'screen', 'multiply', 'difference']

export function CompositeSection() {
  const p = usePipelineStore()
  if (!(p.camEnabled && p.videoLoaded && p.active)) return null

  return (
    <>
      <div className="section-label">Compositing</div>
      <div className="controls composite-controls">
        <label>
          layers{' '}
          <select
            value={p.compositeOrder}
            onChange={(e) => p.setCompositeOrder(e.target.value as LayerOrder)}
          >
            <option value="camera-over">camera over video</option>
            <option value="video-over">video over camera</option>
          </select>
        </label>
        <label>
          opacity{' '}
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={Math.round(p.compositeOpacity * 100)}
            onChange={(e) => p.setCompositeOpacity(+e.target.value / 100)}
          />
        </label>
        <span className="dim" style={{ minWidth: 32 }}>
          {Math.round(p.compositeOpacity * 100)}%
        </span>
        <label>
          blend{' '}
          <select
            value={p.compositeBlend}
            onChange={(e) => p.setCompositeBlend(e.target.value as BlendMode)}
          >
            {BLENDS.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </label>
      </div>
    </>
  )
}
