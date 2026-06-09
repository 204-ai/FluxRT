import { useEffect, useRef } from 'react'
import { useSessionStore } from '../state/sessionStore'

export function LogPanel() {
  const logText = useSessionStore((s) => s.logText)
  const ref = useRef<HTMLPreElement>(null)

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [logText])

  return (
    <pre className="log" ref={ref}>
      {logText}
    </pre>
  )
}
