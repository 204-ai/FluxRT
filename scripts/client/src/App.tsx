import { useEffect } from 'react'
import { Header } from './components/Header'
import { LogPanel } from './components/LogPanel'
import { InputTab } from './components/input/InputTab'
import { OutputTab } from './components/output/OutputTab'
import { SensePanel } from './components/sense/SensePanel'
import { useSessionStore } from './state/sessionStore'

let booted = false

export default function App() {
  useEffect(() => {
    // module-level guard: StrictMode double-mount must not double-boot
    if (booted) return
    booted = true
    void useSessionStore.getState().boot()
  }, [])

  return (
    <>
      <Header />
      {/* Split view: input sources on the left, generated output on the right.
          Both panels stay mounted — the Output panel owns the remote <video>,
          the Input panel owns the camera. Replaces the legacy tab switcher. */}
      <div className="split-view">
        <InputTab active />
        <OutputTab active />
      </div>
      <SensePanel />
      <LogPanel />
    </>
  )
}
