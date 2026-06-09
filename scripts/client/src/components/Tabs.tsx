import { useSessionStore } from '../state/sessionStore'

export function Tabs() {
  const activeTab = useSessionStore((s) => s.activeTab)
  const setTab = useSessionStore((s) => s.setTab)
  return (
    <nav className="tabs">
      <button className={'tab-btn' + (activeTab === 'input' ? ' active' : '')} onClick={() => setTab('input')}>
        Input
      </button>
      <button className={'tab-btn' + (activeTab === 'output' ? ' active' : '')} onClick={() => setTab('output')}>
        Output
      </button>
    </nav>
  )
}
