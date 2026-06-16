// Compact "layers" panel: camera, video and feedback stacked top→bottom (the
// top row is the frontmost layer). Each row carries its source control plus a
// minimal fader with the blend-mode button in front. Every layer blends down
// the stack with its own opacity + blend; changes apply live (no restart).

import { fmtTime, usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import type { BlendMode, LayerId } from '../../pipeline/core/types'
import { DropZone } from '../DropZone'

const BLENDS: BlendMode[] = ['normal', 'screen', 'multiply', 'difference']
const BLEND_SHORT: Record<BlendMode, string> = {
  normal: 'nrm',
  screen: 'scr',
  multiply: 'mul',
  difference: 'dif',
}
const RATES = [0.25, 0.5, 1, 1.5, 2]

/** The per-layer mix control: blend-mode button in front of a minimal fader. */
function LayerMix({ id, disabled }: { id: LayerId; disabled: boolean }) {
  const layer = usePipelineStore((s) => s.layers[id])
  const setLayerOpacity = usePipelineStore((s) => s.setLayerOpacity)
  const setLayerBlend = usePipelineStore((s) => s.setLayerBlend)
  const pct = Math.round(layer.opacity * 100)
  const cycleBlend = () => setLayerBlend(id, BLENDS[(BLENDS.indexOf(layer.blend) + 1) % BLENDS.length])

  return (
    <div className="layer-mix">
      <button
        className="icon-btn blend-btn"
        title={`Blend: ${layer.blend} (click to cycle)`}
        aria-label={`${id} blend mode`}
        disabled={disabled}
        onClick={cycleBlend}
      >
        {BLEND_SHORT[layer.blend]}
      </button>
      <input
        className="layer-fader"
        type="range"
        min={0}
        max={100}
        step={1}
        value={pct}
        disabled={disabled}
        aria-label={`${id} opacity`}
        onChange={(e) => setLayerOpacity(id, +e.target.value / 100)}
      />
      <span className="dim layer-pct">{pct}%</span>
    </div>
  )
}

function CameraRow() {
  const p = usePipelineStore()
  return (
    <div className="layer-row">
      <span className="layer-name">Camera</span>
      <div className="layer-src">
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
        <label className="dim mirror-lbl" title="Mirror the camera (selfie view)">
          <input
            type="checkbox"
            checked={p.mirror}
            disabled={!p.camEnabled}
            onChange={(e) => p.setMirror(e.target.checked)}
          />{' '}
          Mirror
        </label>
      </div>
      <LayerMix id="camera" disabled={!(p.camEnabled && p.active)} />
    </div>
  )
}

function VideoRow() {
  const p = usePipelineStore()
  return (
    <div className="layer-row video-row">
      <div className="layer-line">
        <span className="layer-name">Video</span>
        <div className="layer-src">
          <DropZone
            accept="video/*"
            label={p.videoLoaded ? p.videoName : 'Drop video or click'}
            onFile={(f) => void p.loadVideoFile(f)}
            title={
              p.videoLoaded ? `${p.videoName} — ${p.videoMeta}` : 'Drop a video file or click to choose'
            }
          />
          {p.videoLoaded && (
            <button
              className="icon-btn"
              title="Unload video"
              aria-label="Unload video"
              onClick={() => void p.unloadVideo()}
            >
              🗑
            </button>
          )}
        </div>
        <LayerMix id="video" disabled={!(p.videoLoaded && p.active)} />
      </div>
      {p.videoLoaded && (
        <div className="controls transport">
          <button
            className="icon-btn"
            title={p.videoPlaying ? 'Pause' : 'Play'}
            aria-label={p.videoPlaying ? 'Pause' : 'Play'}
            onClick={() => p.toggleVideoPlay()}
          >
            {p.videoPlaying ? '⏸' : '▶'}
          </button>
          <input
            className="seek"
            type="range"
            min={0}
            max={p.videoDuration || 0}
            step={0.1}
            value={p.videoCurrentTime}
            onChange={(e) => p.seekVideo(+e.target.value)}
          />
          <span className="time-readout">
            {fmtTime(p.videoCurrentTime)}/{fmtTime(p.videoDuration)}
          </span>
          <button
            className={'icon-btn' + (p.videoLoop ? ' on' : '')}
            title="Loop"
            aria-label="Loop"
            onClick={() => p.setVideoLoop(!p.videoLoop)}
          >
            🔁
          </button>
          <select
            className="rate"
            title="Playback speed"
            value={p.videoRate}
            onChange={(e) => p.setVideoRate(+e.target.value)}
          >
            {RATES.map((r) => (
              <option key={r} value={r}>
                {r}×
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  )
}

function FeedbackRow() {
  const active = usePipelineStore((s) => s.active)
  const feedbackAvailable = usePipelineStore((s) => s.feedbackAvailable)
  const live = feedbackAvailable && active
  return (
    <div className="layer-row">
      <span className="layer-name">Feedback</span>
      <div className="layer-src">
        <span className="dim feedback-hint">
          {live ? 'output → input loop' : 'live once the stream is running'}
        </span>
      </div>
      <LayerMix id="feedback" disabled={!live} />
    </div>
  )
}

export function LayerStack() {
  const inputRole = useSessionStore((s) => s.inputRole)
  const roleLabel =
    inputRole === 'you'
      ? 'input: you (steering)'
      : inputRole === 'peer'
        ? 'input: peer (other client)'
        : 'input: server'

  return (
    <div className="layer-stack">
      <CameraRow />
      <VideoRow />
      <FeedbackRow />
      <span className="dim layer-foot">{roleLabel}</span>
    </div>
  )
}
