import { useEffect, useRef } from 'react'
import { useSessionStore } from '../state/sessionStore'
import { usePromptStore } from '../state/promptStore'
import { registerFocusable } from '../state/focusRegistry'

export function Header() {
  const status = useSessionStore((s) => s.status)
  const statusCls = useSessionStore((s) => s.statusCls)
  const perf = useSessionStore((s) => s.perf)
  const starting = useSessionStore((s) => s.starting)
  const connected = useSessionStore((s) => s.connected)
  const start = useSessionStore((s) => s.start)
  const stop = useSessionStore((s) => s.stop)
  const seed = usePromptStore((s) => s.seed)
  const steps = usePromptStore((s) => s.steps)
  const setSeed = usePromptStore((s) => s.setSeed)
  const setSteps = usePromptStore((s) => s.setSteps)

  const canStart = !starting && !connected && status !== 'connecting...'

  const seedRef = useRef<HTMLInputElement>(null)
  const stepsRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const poll = () => void useSessionStore.getState().pollPerf()
    poll()
    const t = setInterval(poll, 1000)
    return () => clearInterval(t)
  }, [])

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
    <header>
      <h1>FluxRT WebRTC</h1>
      <span id="status" className={statusCls}>
        {status}
      </span>
      <button
        className="icon-btn start"
        aria-label="Start session"
        title="Start"
        disabled={!canStart}
        onClick={() => void start()}
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
          <path d="M4 3l9 5-9 5z" />
        </svg>
      </button>
      <button
        className="icon-btn stop"
        aria-label="Stop session"
        title="Stop"
        disabled={!connected}
        onClick={() => stop()}
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
          <rect x="3" y="3" width="10" height="10" rx="1" />
        </svg>
      </button>
      <label className="nav-field" title="Generation seed">
        seed
        <input
          ref={seedRef}
          type="number"
          value={seed}
          onChange={(e) => usePromptStore.setState({ seed: e.target.value })}
          onBlur={(e) => setSeed(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && setSeed((e.target as HTMLInputElement).value)}
        />
      </label>
      <label className="nav-field" title="Diffusion steps (1–8)">
        steps
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
      <span id="fpsBar">
        {`pipe ${perf.pipe} (×interp ${perf.interp})  ·  recv ${perf.recv}  ·  proc ${perf.proc}  ·  vram ${perf.vram}`}
      </span>
    </header>
  )
}
