import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { Header } from './components/Header'
import { LogPanel } from './components/LogPanel'
import { InputTab } from './components/input/InputTab'
import { OutputTab } from './components/output/OutputTab'
import { SensePanel } from './components/sense/SensePanel'
import { useSessionStore } from './state/sessionStore'

let booted = false

export default function App() {
  const [leftPct, setLeftPct] = useState(50)
  const splitRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  useEffect(() => {
    // module-level guard: StrictMode double-mount must not double-boot
    if (booted) return
    booted = true
    void useSessionStore.getState().boot()
  }, [])

  // Draggable divider between the input and output panes.
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const el = splitRef.current
      if (!dragging.current || !el) return
      const rect = el.getBoundingClientRect()
      const pct = ((e.clientX - rect.left) / rect.width) * 100
      setLeftPct(Math.min(80, Math.max(20, pct)))
    }
    const onUp = () => {
      if (!dragging.current) return
      dragging.current = false
      document.body.style.userSelect = ''
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [])

  return (
    <>
      <Header />
      {/* Split view: input sources left, generated output right. Both panels
          stay mounted; drag the divider to resize. */}
      <div className="split-view" ref={splitRef} style={{ '--split-left': `${leftPct}%` } as CSSProperties}>
        <InputTab active />
        <div
          className="split-divider"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize input/output panels"
          onPointerDown={() => {
            dragging.current = true
            document.body.style.userSelect = 'none'
          }}
        />
        <OutputTab active />
      </div>
      <SensePanel />
      <LogPanel />
    </>
  )
}
