// Input tab: camera enable/picker/mirror, input preview + draw, hand marker.

import { useEffect } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import { CanvasHost } from './CanvasHost'
import { OverlayCanvas } from '../sense/OverlayCanvas'

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
  const showInStage = useSessionStore((s) => s.activeTab) === 'output' && p.showInputPreview

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
      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={p.camEnabled}
            onChange={(e) => (e.target.checked ? void p.enableCam() : p.disableCam())}
          />{' '}
          Use my camera as input
        </label>
        <select
          style={{ flex: '1 1 220px' }}
          disabled={!p.camEnabled}
          value={p.deviceId}
          onChange={(e) => void p.setDevice(e.target.value)}
        >
          {p.devices.length === 0 ? (
            <option value="">— pick a camera —</option>
          ) : (
            <option value="">Default camera</option>
          )}
          {p.devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label}
            </option>
          ))}
        </select>
        <label>
          <input
            type="checkbox"
            checked={p.mirror}
            disabled={!p.camEnabled}
            onChange={(e) => p.setMirror(e.target.checked)}
          />{' '}
          Mirror input
        </label>
        <span className="dim">{roleLabel}</span>
      </div>

      <div className="section-label">Input preview &amp; draw</div>
      <div id="inputView" className="overlay-anchor">
        {!p.active && (
          <div className="dim" style={{ padding: 24 }}>
            Enable your camera to preview &amp; draw on the input.
          </div>
        )}
        <CanvasHost holds={p.active && !showInStage} />
        {p.active && !showInStage && <OverlayCanvas source="input" />}
      </div>
      <div className="controls">
        <label>
          <input
            type="checkbox"
            disabled={!p.active}
            checked={p.drawMode !== 'off'}
            onChange={(e) => p.setDrawMode(e.target.checked ? 'brush' : 'off')}
          />{' '}
          Draw on input
        </label>
        <label>
          color{' '}
          <input
            type="color"
            value={p.drawColor}
            disabled={!p.active}
            onChange={(e) => p.setDrawColor(e.target.value)}
          />
        </label>
        <label>
          size{' '}
          <input
            type="range"
            min={1}
            max={60}
            step={1}
            value={p.drawSize}
            disabled={!p.active}
            onChange={(e) => p.setDrawSize(+e.target.value)}
          />
        </label>
        <span className="dim" style={{ minWidth: 28 }}>
          {p.drawSize}px
        </span>
        <button disabled={!p.active} onClick={() => p.clearDrawing()}>
          Clear drawing
        </button>
      </div>

      <div className="section-label">Hand marker</div>
      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={p.markerEnabled}
            disabled={!p.camEnabled}
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
