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

      {/* Prompt box — everything that builds/sends the prompt. */}
      <section className="panel-box">
        <div className="section-label">Prompt</div>
        <div className="controls">
          <PromptEditor />
        </div>
        <PromptPlayer />
        <FeatureBar />
        {senseEnabled && <ComposeControls />}
      </section>

      {/* Reference image + ComfyUI box. */}
      <section className="panel-box">
        <div className="section-label">Reference &amp; ComfyUI</div>
        <ReferencePanel />
        <ComfyRow />
      </section>
    </section>
  )
}
