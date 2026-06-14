import { useEffect, useRef } from 'react'
import { useSessionStore } from '../../state/sessionStore'
import { usePromptStore } from '../../state/promptStore'
import { registerFocusable } from '../../state/focusRegistry'

export function SessionControls() {
  const seed = usePromptStore((s) => s.seed)
  const steps = usePromptStore((s) => s.steps)
  const setSeed = usePromptStore((s) => s.setSeed)
  const setSteps = usePromptStore((s) => s.setSteps)

  const seedRef = useRef<HTMLInputElement>(null)
  const stepsRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    registerFocusable('seed', seedRef.current)
    registerFocusable('steps', stepsRef.current)
    return () => {
      registerFocusable('seed', null)
      registerFocusable('steps', null)
    }
  }, [])

  useEffect(() => {
    const onUnload = () => useSessionStore.getState().stop()
    window.addEventListener('beforeunload', onUnload)
    return () => window.removeEventListener('beforeunload', onUnload)
  }, [])

  return (
    <div className="controls">
      <label>
        seed{' '}
        <input
          ref={seedRef}
          type="number"
          value={seed}
          onChange={(e) => usePromptStore.setState({ seed: e.target.value })}
          onBlur={(e) => setSeed(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && setSeed((e.target as HTMLInputElement).value)}
        />
      </label>
      <label>
        steps{' '}
        <input
          ref={stepsRef}
          type="number"
          min={1}
          max={8}
          value={steps}
          onChange={(e) => usePromptStore.setState({ steps: e.target.value })}
          onBlur={(e) => setSteps(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && setSteps((e.target as HTMLInputElement).value)}
        />
      </label>
    </div>
  )
}
