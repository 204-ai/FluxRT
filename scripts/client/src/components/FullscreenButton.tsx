// Small corner button that toggles native fullscreen on its .overlay-anchor.
// Dropped inside the input preview (#inputView) or the output stage
// (.remote-wrap); clicking expands that container to fullscreen, clicking
// again (or Esc) restores it. Re-uses the existing element so the live
// canvas/video and its overlays keep streaming — no re-parenting.

import { useEffect, useRef, useState } from 'react'

export function FullscreenButton({ label }: { label: string }) {
  const btnRef = useRef<HTMLButtonElement>(null)
  const [isFs, setIsFs] = useState(false)

  useEffect(() => {
    const onChange = () => {
      const anchor = btnRef.current?.closest('.overlay-anchor') ?? null
      setIsFs(!!anchor && document.fullscreenElement === anchor)
    }
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const toggle = () => {
    const anchor = btnRef.current?.closest('.overlay-anchor') as HTMLElement | null
    if (!anchor) return
    // Both calls reject if the gesture is stale; swallow rather than throwing an
    // unhandled rejection. Guard requestFullscreen explicitly so unsupported
    // browsers no-op instead of relying on optional-chaining subtleties.
    if (document.fullscreenElement === anchor) {
      void document.exitFullscreen?.().catch(() => {})
    } else if (anchor.requestFullscreen) {
      void anchor.requestFullscreen().catch(() => {})
    }
  }

  return (
    <button
      ref={btnRef}
      className={'fs-fab viewer-chrome' + (isFs ? ' active' : '')}
      title={isFs ? 'Exit fullscreen' : `Fullscreen ${label}`}
      aria-label={isFs ? 'Exit fullscreen' : `Fullscreen ${label}`}
      onClick={toggle}
    >
      ⛶
    </button>
  )
}
