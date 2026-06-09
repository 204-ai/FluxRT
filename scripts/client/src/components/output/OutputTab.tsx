import { Stage } from './Stage'
import { SessionControls } from './SessionControls'
import { PromptEditor } from './PromptEditor'
import { FeatureBar } from './FeatureBar'
import { SavedPromptsRow } from './SavedPromptsRow'
import { ReferencePanel } from './ReferencePanel'
import { ComfyRow } from './ComfyRow'
import { LipRow } from './LipRow'

export function OutputTab({ active }: { active: boolean }) {
  return (
    <section className={'tab-panel' + (active ? ' active' : '')}>
      <Stage />
      <SessionControls />
      <div className="controls" style={{ alignItems: 'flex-start' }}>
        <PromptEditor />
        <ReferencePanel />
      </div>
      <FeatureBar />
      <SavedPromptsRow />
      <ComfyRow />
      <LipRow />
    </section>
  )
}
