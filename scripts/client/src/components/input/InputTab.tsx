// Input tab: sources (camera + video file), compositing, input preview + draw.
// (Hand marker + human sensing live together in the Human-sensing panel.)

import { useEffect, useState } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import { CanvasHost } from './CanvasHost'
import { FullscreenButton } from '../FullscreenButton'
import { CompositeSection } from './CompositeSection'
import { VideoSourceSection } from './VideoSourceSection'
import { OverlayCanvas } from '../sense/OverlayCanvas'
import { DrawToolbar } from './DrawToolbar'
import { SensePanel } from '../sense/SensePanel'
import { MetricsOverlay } from '../sense/MetricsOverlay'
import { useSenseStore } from '../../state/senseStore'
import { useViewerReveal } from '../../lib/useViewerReveal'

export function InputTab({ active }: { active: boolean }) {
  const p = usePipelineStore()
  const inputRole = useSessionStore((s) => s.inputRole)
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

  const roleLabel =
    inputRole === 'you'
      ? 'input: you (steering)'
      : inputRole === 'peer'
        ? 'input: peer (other client)'
        : 'input: server'

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
        {p.active && <DrawToolbar />}
        {p.active && (
          <FullscreenButton label="input" maximized={maximized} onToggle={() => setMaximized((v) => !v)} />
        )}
        {p.active && senseEnabled && senseOverlay !== 'off' && <MetricsOverlay />}
      </div>

      <div className="source-panels">
        <div className="src-panel">
          <div className="src-title">Camera</div>
          <div className="controls src-row">
            {/* Selecting a camera auto-enables it; "Camera off" disables. */}
            <select
              className="device-pick"
              value={p.camEnabled ? p.deviceId || 'default' : ''}
              onChange={(e) => {
                const v = e.target.value
                if (v === '') {
                  void p.disableCam()
                  return
                }
                const id = v === 'default' ? '' : v
                usePipelineStore.setState({ deviceId: id })
                if (p.camEnabled) void p.setDevice(id)
                else void p.enableCam()
              }}
            >
              <option value="">Camera off</option>
              <option value="default">Default camera</option>
              {p.devices.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label}
                </option>
              ))}
            </select>
            <label className="dim" title="Mirror the camera (selfie view)">
              <input
                type="checkbox"
                checked={p.mirror}
                disabled={!p.camEnabled}
                onChange={(e) => p.setMirror(e.target.checked)}
              />{' '}
              Mirror
            </label>
          </div>
          <span className="dim">{roleLabel}</span>
        </div>
        <CompositeSection />
        <VideoSourceSection />
      </div>

      <SensePanel />
    </section>
  )
}
