// Detail / inspector pane for the SELECTED clip — kind-dispatched controls:
// camera device + mirror, video transport, feedback/screen status, plus framing.

import { fmtTime, usePipelineStore } from '../../state/pipelineStore'
import { findClip, type Clip } from '../../state/layerModel'
import { CLIP_ICON, clipMeta } from '../../pipeline/core/clipKinds'

const RATES = [0.25, 0.5, 1, 1.5, 2]

function FrameButton({ layerId }: { layerId: string }) {
  const layoutLayer = usePipelineStore((s) => s.layoutLayer)
  const setLayoutLayer = usePipelineStore((s) => s.setLayoutLayer)
  const on = layoutLayer === layerId
  return (
    <button
      className={'tool' + (on ? ' active' : '')}
      title="Frame this layer on the preview (move / resize / crop)"
      aria-pressed={on}
      onClick={() => setLayoutLayer(on ? null : layerId)}
    >
      ◳ Frame
    </button>
  )
}

function CameraDetail({ clip }: { clip: Clip }) {
  const devices = usePipelineStore((s) => s.devices)
  const setClipDevice = usePipelineStore((s) => s.setClipDevice)
  const setClipMirror = usePipelineStore((s) => s.setClipMirror)
  return (
    <div className="clip-detail-row">
      <select
        className="device-pick"
        value={clip.deviceId || 'default'}
        onChange={(e) => void setClipDevice(clip.id, e.target.value === 'default' ? '' : e.target.value)}
      >
        <option value="default">Default camera</option>
        {devices.map((d) => (
          <option key={d.deviceId} value={d.deviceId}>
            {d.label}
          </option>
        ))}
      </select>
      <label className="mirror-lbl" title="Mirror (selfie view)">
        <input type="checkbox" checked={clip.mirror} onChange={(e) => setClipMirror(clip.id, e.target.checked)} /> Mirror
      </label>
    </div>
  )
}

function VideoDetail({ clip }: { clip: Clip }) {
  const v = usePipelineStore((s) => s.videoState[clip.id])
  const toggle = usePipelineStore((s) => s.toggleVideoPlay)
  const seek = usePipelineStore((s) => s.seekVideo)
  const setLoop = usePipelineStore((s) => s.setVideoLoop)
  const setRate = usePipelineStore((s) => s.setVideoRate)
  if (!v) return <span className="dim">video unavailable</span>
  return (
    <div className="controls transport compact">
      <button className="icon-btn" title={v.playing ? 'Pause' : 'Play'} onClick={() => toggle(clip.id)}>
        {v.playing ? '⏸' : '▶'}
      </button>
      <input
        className="seek"
        type="range"
        min={0}
        max={v.duration || 0}
        step={0.1}
        value={v.currentTime}
        onChange={(e) => seek(clip.id, +e.target.value)}
      />
      <span className="time-readout">{fmtTime(v.currentTime)}</span>
      <button className={'icon-btn' + (v.loop ? ' on' : '')} title="Loop" onClick={() => setLoop(clip.id, !v.loop)}>
        🔁
      </button>
      <select className="rate" title="Speed" value={v.rate} onChange={(e) => setRate(clip.id, +e.target.value)}>
        {RATES.map((r) => (
          <option key={r} value={r}>
            {r}×
          </option>
        ))}
      </select>
    </div>
  )
}

function FeedbackDetail() {
  const live = usePipelineStore((s) => s.feedbackAvailable && s.active)
  return <span className="dim">{live ? 'output → input loop (live)' : 'waiting for the stream to run'}</span>
}

const SHADER_FILTERS: { label: string; value: string }[] = [
  { label: 'Hue rotate', value: 'hue-rotate(90deg)' },
  { label: 'Invert', value: 'invert(1)' },
  { label: 'Saturate', value: 'saturate(2)' },
  { label: 'Grayscale', value: 'grayscale(1)' },
  { label: 'Sepia', value: 'sepia(1)' },
  { label: 'Contrast', value: 'contrast(1.6)' },
  { label: 'Blur', value: 'blur(4px)' },
  { label: 'None', value: 'none' },
]

function ShaderDetail({ clip }: { clip: Clip }) {
  const setEffectConfig = usePipelineStore((s) => s.setEffectConfig)
  const filter = (clip.effectConfig?.filter as string) ?? 'none'
  return (
    <div className="clip-detail-row">
      <span className="dim">filter</span>
      <select className="device-pick" value={filter} onChange={(e) => setEffectConfig(clip.id, { filter: e.target.value })}>
        {SHADER_FILTERS.map((f) => (
          <option key={f.value} value={f.value}>
            {f.label}
          </option>
        ))}
      </select>
    </div>
  )
}

function ClipDetailBody({ clip }: { clip: Clip }) {
  if (clip.kind === 'camera') return <CameraDetail clip={clip} />
  if (clip.kind === 'video') return <VideoDetail clip={clip} />
  if (clip.kind === 'feedback') return <FeedbackDetail />
  if (clip.kind === 'screen') return <span className="dim">screen share — live</span>
  if (clip.kind === 'shader') return <ShaderDetail clip={clip} />
  return <span className="dim">no details for this clip</span>
}

export function ClipDetail() {
  const selectedClipId = usePipelineStore((s) => s.selectedClipId)
  const layers = usePipelineStore((s) => s.layers)
  const found = selectedClipId ? findClip(layers, selectedClipId) : null
  if (!found) return <div className="clip-detail empty dim">Select a clip to inspect it.</div>
  const { layer, clip } = found
  return (
    <div className="clip-detail">
      <div className="clip-detail-head">
        <span className="clip-detail-icon">{CLIP_ICON[clip.kind] ?? '◻'}</span>
        <span className="clip-detail-title" title={clip.label}>
          {clip.label}
        </span>
        <span className="dim clip-detail-kind">{clipMeta(clip.kind).label}</span>
      </div>
      <div className="clip-detail-body">
        <ClipDetailBody clip={clip} />
      </div>
      <div className="clip-detail-foot">
        <FrameButton layerId={layer.id} />
      </div>
    </div>
  )
}
