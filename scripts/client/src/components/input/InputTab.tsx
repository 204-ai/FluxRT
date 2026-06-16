// Input tab: sources (camera + video file), compositing, input preview + draw.
// (Hand marker + human sensing live together in the Human-sensing panel.)

import { useEffect, useState } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import { CanvasHost } from './CanvasHost'
import { LayerStack } from './LayerStack'
import { TransformOverlay } from './TransformOverlay'
import { FullscreenButton } from '../FullscreenButton'
import { OverlayCanvas } from '../sense/OverlayCanvas'
import { DrawToolbar } from './DrawToolbar'
import { SensePanel } from '../sense/SensePanel'
import { MetricsOverlay } from '../sense/MetricsOverlay'
import { useSenseStore } from '../../state/senseStore'
import { useViewerReveal } from '../../lib/useViewerReveal'

export function InputTab({ active }: { active: boolean }) {
  const p = usePipelineStore()
  const senseEnabled = useSenseStore((s) => s.enabled)
  const senseOverlay = useSenseStore((s) => s.overlay)
  const senseOnly = senseEnabled && senseOverlay === 'only'
  const reveal = useViewerReveal()
  const [maximized, setMaximized] = useState(false)

  // Don't stay maximized over the empty-state message if the preview stops.
  useEffect(() => {
    if (!p.active) setMaximized(false)
  }, [p.active])

  useEffect(() => {
    p.setLogger((m) => useSessionStore.getState().logLine(m))
    const onDeviceChange = () => {
      if (usePipelineStore.getState().camEnabled) void usePipelineStore.getState().refreshCameras()
    }
    navigator.mediaDevices?.addEventListener?.('devicechange', onDeviceChange)
    return () => navigator.mediaDevices?.removeEventListener?.('devicechange', onDeviceChange)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <section className={'tab-panel' + (active ? ' active' : '')}>
      <div
        id="inputView"
        className={
          'overlay-anchor' +
          (senseOnly ? ' sense-only' : '') +
          (reveal.shown ? ' controls-shown' : '') +
          (maximized ? ' viewport-max' : '')
        }
        {...reveal.pointerProps}
      >
        {!p.active && (
          <div className="dim" style={{ padding: 24 }}>
            Enable your camera or load a video to preview &amp; draw on the input.
          </div>
        )}
        <CanvasHost holds={p.active} />
        {p.active && senseOverlay !== 'off' && <OverlayCanvas source="input" />}
        {p.active && <TransformOverlay />}
        {p.active && <DrawToolbar />}
        {p.active && (
          <FullscreenButton label="input" maximized={maximized} onToggle={() => setMaximized((v) => !v)} />
        )}
        {p.active && senseEnabled && senseOverlay !== 'off' && <MetricsOverlay />}
      </div>

      <LayerStack />

      <SensePanel />
    </section>
  )
}
