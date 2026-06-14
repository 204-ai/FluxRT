import { Stage } from './Stage'
import { PromptEditor } from './PromptEditor'
import { PromptPlayer } from './PromptPlayer'
import { FeatureBar } from './FeatureBar'
import { ReferencePanel } from './ReferencePanel'
import { ComfyRow } from './ComfyRow'
import { ComposeControls } from '../sense/ComposeControls'
import { useSenseStore } from '../../state/senseStore'

export function OutputTab({ active }: { active: boolean }) {
  const senseEnabled = useSenseStore((s) => s.enabled)
  return (
    <section className={'tab-panel' + (active ? ' active' : '')}>
      <Stage />

      {/* Prompt box — prompt input with the reference image inline beside it. */}
      <section className="panel-box">
        <div className="section-label">Prompt</div>
        <div className="controls prompt-row">
          <PromptEditor />
          <ReferencePanel />
        </div>
        <PromptPlayer />
        <FeatureBar />
        {senseEnabled && <ComposeControls />}
      </section>

      {/* ComfyUI box. */}
      <section className="panel-box">
        <div className="section-label">ComfyUI</div>
        <ComfyRow />
      </section>
    </section>
  )
}
