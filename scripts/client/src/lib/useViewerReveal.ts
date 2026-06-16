// Reveals a viewer's floating chrome (the icon overlays) on touch interaction.
// Desktop reveals via CSS :hover / :focus-within; touch devices don't hover, so
// a pointerdown — a tap, or the start of a draw stroke — flips a `shown` flag
// that the CSS `.controls-shown` rule keys off, then auto-hides after a short
// idle so the viewer stays clean. Mouse input is ignored (CSS :hover handles
// it). Revealing on any touch-down (not just a clean tap) deliberately avoids a
// tap-vs-drag race with freehand drawing: the stroke proceeds untouched and the
// controls simply flash in, so the otherwise-hidden draw toolbar stays reachable.
//
// The idle timer is refreshed on move/up as well as down, so a long draw stroke
// (which fires a single pointerdown then a run of pointermoves under capture)
// keeps the chrome visible instead of hiding it out from under the user.

import { useCallback, useEffect, useRef, useState, type PointerEvent } from 'react'

export function useViewerReveal(idleMs = 3000) {
  const [shown, setShown] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Mirrors `shown` so the per-move keep-alive can skip redundant setState.
  const shownRef = useRef(false)

  const hide = useCallback(() => {
    shownRef.current = false
    setShown(false)
  }, [])

  // Reveal (if needed) and (re)start the idle countdown from this interaction.
  const keepAlive = useCallback(
    (e: PointerEvent) => {
      if (e.pointerType === 'mouse') return // desktop reveals via CSS :hover
      if (!shownRef.current) {
        shownRef.current = true
        setShown(true)
      }
      if (timer.current) clearTimeout(timer.current)
      timer.current = setTimeout(hide, idleMs)
    },
    [hide, idleMs],
  )

  // Drop any pending hide timer on unmount.
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])

  return {
    shown,
    /** Spread onto the .overlay-anchor to drive touch reveal + idle auto-hide. */
    pointerProps: { onPointerDown: keepAlive, onPointerMove: keepAlive, onPointerUp: keepAlive },
  }
}
