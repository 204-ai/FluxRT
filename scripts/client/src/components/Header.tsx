import { useEffect } from 'react'
import { useSessionStore } from '../state/sessionStore'

export function Header() {
  const status = useSessionStore((s) => s.status)
  const statusCls = useSessionStore((s) => s.statusCls)
  const perf = useSessionStore((s) => s.perf)

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
      <span id="fpsBar">
        {`pipe ${perf.pipe} (×interp ${perf.interp})  ·  recv ${perf.recv}  ·  proc ${perf.proc}  ·  vram ${perf.vram}`}
      </span>
    </header>
  )
}
