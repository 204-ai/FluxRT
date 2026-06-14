import { useEffect } from 'react'
import { useSessionStore } from '../state/sessionStore'

export function Header() {
  const status = useSessionStore((s) => s.status)
  const statusCls = useSessionStore((s) => s.statusCls)
  const perf = useSessionStore((s) => s.perf)
  const starting = useSessionStore((s) => s.starting)
  const connected = useSessionStore((s) => s.connected)
  const start = useSessionStore((s) => s.start)
  const stop = useSessionStore((s) => s.stop)

  const canStart = !starting && !connected && status !== 'connecting...'

  useEffect(() => {
    const poll = () => void useSessionStore.getState().pollPerf()
    poll()
    const t = setInterval(poll, 1000)
    return () => clearInterval(t)
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
      <span id="fpsBar">
        {`pipe ${perf.pipe} (×interp ${perf.interp})  ·  recv ${perf.recv}  ·  proc ${perf.proc}  ·  vram ${perf.vram}`}
      </span>
    </header>
  )
}
