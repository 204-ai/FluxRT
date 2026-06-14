// Input tab: sources (camera + video file), compositing, input preview +
// draw, hand marker.

import { useEffect } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import { CanvasHost } from './CanvasHost'
import { CompositeSection } from './CompositeSection'
import { VideoSourceSection } from './VideoSourceSection'
import { OverlayCanvas } from '../sense/OverlayCanvas'
import { DrawToolbar } from './DrawToolbar'

const LANDMARKS: Array<[number, string]> = [
  [15, 'Left wrist'],
  [16, 'Right wrist'],
  [19, 'Left index'],
  [20, 'Right index'],
  [0, 'Nose'],
  [11, 'Left shoulder'],
  [12, 'Right shoulder'],
]

export function InputTab({ active }: { active: boolean }) {
  const p = usePipelineStore()
  const inputRole = useSessionStore((s) => s.inputRole)

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
      <div id="inputView" className="overlay-anchor">
        {!p.active && (
          <div className="dim" style={{ padding: 24 }}>
            Enable your camera or load a video to preview &amp; draw on the input.
          </div>
        )}
        <CanvasHost holds={p.active} />
        {p.active && <OverlayCanvas source="input" />}
        {p.active && <DrawToolbar />}
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
        <VideoSourceSection />
      </div>

      <CompositeSection />

      <div className="section-label">Hand marker</div>
      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={p.markerEnabled}
            disabled={!p.active}
            onChange={(e) => void p.setMarkerEnabled(e.target.checked)}
          />{' '}
          Enable
        </label>
        <label>
          landmark{' '}
          <select value={p.markerLandmark} onChange={(e) => p.setMarkerLandmark(+e.target.value)}>
            {LANDMARKS.map(([v, label]) => (
              <option key={v} value={v}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          color <input type="color" value={p.markerColor} onChange={(e) => p.setMarkerColor(e.target.value)} />
        </label>
        <label>
          size{' '}
          <input
            type="range"
            min={6}
            max={120}
            step={1}
            value={p.markerSize}
            onChange={(e) => p.setMarkerSize(+e.target.value)}
          />
        </label>
        <span className="dim" style={{ minWidth: 28 }}>
          {p.markerSize}px
        </span>
        <label>
          <input type="checkbox" checked={p.markerTrail} onChange={(e) => p.setMarkerTrail(e.target.checked)} /> Trail
        </label>
        <label>
          length{' '}
          <input
            type="range"
            min={4}
            max={80}
            step={1}
            value={p.markerTrailLen}
            onChange={(e) => p.setMarkerTrailLen(+e.target.value)}
          />
        </label>
        <span className="dim" style={{ minWidth: 28 }}>
          {p.markerTrailLen}
        </span>
        <span className="dim">{p.poseStatus}</span>
      </div>
    </section>
  )
}
