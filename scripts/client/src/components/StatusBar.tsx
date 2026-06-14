// Fixed footer status bar: pipeline stats (moved from the navbar) + a one-line
// log that expands to ~5 lines when the bar is clicked.

import { useEffect, useRef, useState } from 'react'
import { useSessionStore } from '../state/sessionStore'

export function StatusBar() {
  const perf = useSessionStore((s) => s.perf)
  const logText = useSessionStore((s) => s.logText)
  const [expanded, setExpanded] = useState(false)
  const logRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    const poll = () => void useSessionStore.getState().pollPerf()
    poll()
    const t = setInterval(poll, 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logText, expanded])

  return (
    <footer
      className={'statusbar' + (expanded ? ' expanded' : '')}
      onClick={() => setExpanded((v) => !v)}
      title={expanded ? 'Click to collapse the log' : 'Click to expand the log'}
    >
      <div className="statusbar-stats">
        {`pipe ${perf.pipe} (×interp ${perf.interp})  ·  recv ${perf.recv}  ·  proc ${perf.proc}  ·  vram ${perf.vram}`}
      </div>
      <pre className="statusbar-log" ref={logRef}>
        {logText}
      </pre>
    </footer>
  )
}
