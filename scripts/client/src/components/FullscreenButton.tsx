// Small corner button that maximizes its viewer to fill the app viewport — NOT
// the OS screen. It does not use the native Fullscreen API; instead the parent
// pins the .overlay-anchor over the page via the `viewport-max` class. Kept
// controlled (maximized + onToggle from the parent) so the state can drive the
// anchor's className. Esc exits while maximized.

import { useEffect } from 'react'

export function FullscreenButton({
  label,
  maximized,
  onToggle,
}: {
  label: string
  maximized: boolean
  onToggle: () => void
}) {
  useEffect(() => {
    if (!maximized) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onToggle()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [maximized, onToggle])

  return (
    <button
      className={'fs-fab viewer-chrome' + (maximized ? ' active' : '')}
      title={maximized ? 'Exit fullscreen' : `Fullscreen ${label}`}
      aria-label={maximized ? 'Exit fullscreen' : `Fullscreen ${label}`}
      aria-pressed={maximized}
      onClick={onToggle}
    >
      ⛶
    </button>
  )
}
