import { Stage } from './Stage'
import { PromptEditor } from './PromptEditor'
import { PromptPlayer } from './PromptPlayer'
import { FeatureBar } from './FeatureBar'
import { ReferencePanel } from './ReferencePanel'
import { ComfyRow } from './ComfyRow'

export function OutputTab({ active }: { active: boolean }) {
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
