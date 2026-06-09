import { useEffect } from 'react'
import { Header } from './components/Header'
import { Tabs } from './components/Tabs'
import { LogPanel } from './components/LogPanel'
import { InputTab } from './components/input/InputTab'
import { OutputTab } from './components/output/OutputTab'
import { SensePanel } from './components/sense/SensePanel'
import { useSessionStore } from './state/sessionStore'

let booted = false

export default function App() {
  const activeTab = useSessionStore((s) => s.activeTab)

  useEffect(() => {
    // module-level guard: StrictMode double-mount must not double-boot
    if (booted) return
    booted = true
    void useSessionStore.getState().boot()
  }, [])

  return (
    <>
      <Header />
      <Tabs />
      {/* Both tabs stay mounted (display toggled via CSS) — the Output tab owns
          the remote <video> that must survive tab switches, the Input tab owns
          camera controls. Mirrors the legacy tab-panel behavior. */}
      <OutputTab active={activeTab === 'output'} />
      <InputTab active={activeTab === 'input'} />
      <SensePanel />
      <LogPanel />
    </>
  )
}
