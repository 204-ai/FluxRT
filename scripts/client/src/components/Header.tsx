import { useEffect, useRef, useState } from 'react'
import { useSessionStore } from '../state/sessionStore'
import { usePromptStore } from '../state/promptStore'
import { registerFocusable } from '../state/focusRegistry'
import { setLipTransfer } from '../lib/api'
import { serverBase, setServerBase } from '../lib/serverBase'

export function Header() {
  const status = useSessionStore((s) => s.status)
  const statusCls = useSessionStore((s) => s.statusCls)
  const starting = useSessionStore((s) => s.starting)
  const connected = useSessionStore((s) => s.connected)
  const start = useSessionStore((s) => s.start)
  const stop = useSessionStore((s) => s.stop)
  const lipEnabled = useSessionStore((s) => s.lipEnabled)
  const lipActive = useSessionStore((s) => s.lipActive)
  const seed = usePromptStore((s) => s.seed)
  const steps = usePromptStore((s) => s.steps)
  const setSeed = usePromptStore((s) => s.setSeed)
  const setSteps = usePromptStore((s) => s.setSteps)
  const morph = usePromptStore((s) => s.morph)
  const setMorph = usePromptStore((s) => s.setMorph)
  const [lipBusy, setLipBusy] = useState(false)
  const [server, setServer] = useState(serverBase())

  const canStart = !starting && !connected && status !== 'connecting...'

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

  const toggleLip = async (on: boolean) => {
    setLipBusy(true)
    try {
      const j = await setLipTransfer(on)
      useSessionStore.getState().setLip(true, j.lip_active)
    } catch (err) {
      useSessionStore
        .getState()
        .logLine('Lip transfer toggle failed: ' + (err instanceof Error ? err.message : err))
    } finally {
      setLipBusy(false)
    }
  }

  return (
    <header>
      <h1>FluxRT WebRTC</h1>
      <span id="status" className={statusCls}>
        {status}
      </span>

      <label
        className="nav-field"
        title="Backend server URL — blank = same origin (dev proxy / bundled). e.g. https://my-backend:8765. Set before Start; cross-origin needs CORS on the backend."
      >
        server
        <input
          type="text"
          placeholder="same origin"
          value={server}
          disabled={connected || starting}
          onChange={(e) => setServer(e.target.value)}
          onBlur={(e) => setServerBase(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && setServerBase((e.target as HTMLInputElement).value)}
          style={{ width: 170 }}
        />
      </label>

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
      <button
        className={'icon-btn morph' + (morph ? ' on' : '')}
        aria-label="Toggle slerp prompt morphing"
        aria-pressed={morph}
        title={
          morph
            ? 'Slerp morphing ON — prompt changes morph smoothly'
            : 'Slerp morphing OFF — prompt changes swap instantly'
        }
        onClick={() => setMorph(!morph)}
      >
        🌀
      </button>
      <button
        className={'icon-btn lip' + (lipActive ? ' on' : '')}
        aria-label="Toggle lipsync"
        aria-pressed={lipActive}
        disabled={!lipEnabled || lipBusy}
        title={
          lipEnabled
            ? lipActive
              ? 'Lipsync ON — click to disable'
              : 'Lipsync OFF — click to enable'
            : 'Lipsync unavailable (add lip_transfer to config)'
        }
        onClick={() => void toggleLip(!lipActive)}
      >
        👄
      </button>
    </header>
  )
}
