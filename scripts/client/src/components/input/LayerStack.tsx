// Compact "layers" panel: camera, video and feedback stacked top→bottom (the
// top row is the frontmost layer). Each row carries its source control plus a
// minimal fader with the blend-mode button in front. Every layer blends down
// the stack with its own opacity + blend; changes apply live (no restart).

import { fmtTime, usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import { activeClip, layerById, layerKind, type Layer } from '../../state/layerModel'
import type { BlendMode, LayerId } from '../../pipeline/core/types'
import { CAMERA_LAYER, FEEDBACK_LAYER, VIDEO_LAYER } from '../../pipeline/core/types'
import { DropZone } from '../DropZone'

// The three seeded layers are permanent; only ADDED layers are removable.
const SEEDED = new Set<LayerId>([CAMERA_LAYER, VIDEO_LAYER, FEEDBACK_LAYER])

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
  const layer = usePipelineStore((s) => layerById(s.layers, id))
  const setLayerOpacity = usePipelineStore((s) => s.setLayerOpacity)
  const setLayerBlend = usePipelineStore((s) => s.setLayerBlend)
  const layoutLayer = usePipelineStore((s) => s.layoutLayer)
  const setLayoutLayer = usePipelineStore((s) => s.setLayoutLayer)
  const framing = layoutLayer === id
  if (!layer) return null
  const pct = Math.round(layer.opacity * 100)
  const cycleBlend = () => setLayerBlend(id, BLENDS[(BLENDS.indexOf(layer.blend) + 1) % BLENDS.length])

  return (
    <div className="layer-mix">
      <button
        className={'icon-btn frame-btn' + (framing ? ' on' : '')}
        title="Frame this layer — move, resize & crop on the preview"
        aria-label={`${id} framing`}
        aria-pressed={framing}
        disabled={disabled}
        onClick={() => setLayoutLayer(framing ? null : id)}
      >
        ◳
      </button>
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

/** Transport for an ADDED (overlay) video layer — its own pooled <video>, so
 *  multiple video clips play independently. Mirrors VideoRow but reads per-clip
 *  state from extraVideo[clipId]. */
function ExtraVideoRow({ layerId, clipId }: { layerId: LayerId; clipId: string }) {
  const v = usePipelineStore((s) => s.extraVideo[clipId])
  const toggle = usePipelineStore((s) => s.toggleExtraVideoPlay)
  const seek = usePipelineStore((s) => s.seekExtraVideo)
  const setLoop = usePipelineStore((s) => s.setExtraVideoLoop)
  const setRate = usePipelineStore((s) => s.setExtraVideoRate)
  if (!v) return null
  return (
    <div className="layer-row video-row">
      <div className="layer-line">
        <span className="layer-name" title={v.name}>
          {v.name}
        </span>
        <LayerMix id={layerId} disabled={false} />
      </div>
      <div className="controls transport">
        <button
          className="icon-btn"
          title={v.playing ? 'Pause' : 'Play'}
          aria-label={v.playing ? 'Pause' : 'Play'}
          onClick={() => toggle(clipId)}
        >
          {v.playing ? '⏸' : '▶'}
        </button>
        <input
          className="seek"
          type="range"
          min={0}
          max={v.duration || 0}
          step={0.1}
          value={v.currentTime}
          onChange={(e) => seek(clipId, +e.target.value)}
        />
        <span className="time-readout">
          {fmtTime(v.currentTime)}/{fmtTime(v.duration)}
        </span>
        <button
          className={'icon-btn' + (v.loop ? ' on' : '')}
          title="Loop"
          aria-label="Loop"
          onClick={() => setLoop(clipId, !v.loop)}
        >
          🔁
        </button>
        <select className="rate" title="Playback speed" value={v.rate} onChange={(e) => setRate(clipId, +e.target.value)}>
          {RATES.map((r) => (
            <option key={r} value={r}>
              {r}×
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

/** The kind-specific body for one layer row. Seeded layers dispatch by their
 *  active clip's kind; added video layers render their own pooled transport.
 *  New clip kinds slot in here in P4 via a registry. */
function LayerBody({ layer }: { layer: Layer }) {
  const clip = activeClip(layer)
  if (!SEEDED.has(layer.id) && clip?.kind === 'video') {
    return <ExtraVideoRow layerId={layer.id} clipId={clip.id} />
  }
  const kind = layerKind(layer)
  if (kind === 'camera') return <CameraRow />
  if (kind === 'video') return <VideoRow />
  if (kind === 'feedback') return <FeedbackRow />
  return (
    <div className="layer-row">
      <span className="layer-name">Layer</span>
      <span className="dim layer-src">empty — add a clip</span>
    </div>
  )
}

/** One row in the stack: reorder handles + the kind body + (for added layers) a
 *  remove button. Clicking the row selects its active clip for the detail pane.
 *  The body already carries the per-layer mix (LayerMix). */
function LayerRow({ layer, index, count }: { layer: Layer; index: number; count: number }) {
  const moveLayer = usePipelineStore((s) => s.moveLayer)
  const removeLayer = usePipelineStore((s) => s.removeLayer)
  const selectClip = usePipelineStore((s) => s.selectClip)
  const selectedClipId = usePipelineStore((s) => s.selectedClipId)
  const clip = activeClip(layer)
  const selected = !!clip && clip.id === selectedClipId
  const removable = !SEEDED.has(layer.id)
  return (
    <div
      className={'layer-row-wrap' + (selected ? ' sel' : '')}
      onClick={() => clip && selectClip(clip.id)}
    >
      <div className="layer-reorder">
        <button
          className="icon-btn"
          title="Move layer forward (up the stack)"
          aria-label="Move layer up"
          disabled={index === 0}
          onClick={() => moveLayer(layer.id, -1)}
        >
          ▲
        </button>
        <button
          className="icon-btn"
          title="Move layer back (down the stack)"
          aria-label="Move layer down"
          disabled={index === count - 1}
          onClick={() => moveLayer(layer.id, 1)}
        >
          ▼
        </button>
      </div>
      <LayerBody layer={layer} />
      {removable && (
        <button
          className="icon-btn layer-remove"
          title="Remove this layer"
          aria-label="Remove layer"
          onClick={(e) => {
            e.stopPropagation()
            void removeLayer(layer.id)
          }}
        >
          ⌫
        </button>
      )}
    </div>
  )
}

export function LayerStack() {
  const layers = usePipelineStore((s) => s.layers)
  const active = usePipelineStore((s) => s.active)
  const addVideoLayer = usePipelineStore((s) => s.addVideoLayer)
  const addScreenLayer = usePipelineStore((s) => s.addScreenLayer)
  const inputRole = useSessionStore((s) => s.inputRole)
  const roleLabel =
    inputRole === 'you'
      ? 'input: you (steering)'
      : inputRole === 'peer'
        ? 'input: peer (other client)'
        : 'input: server'

  return (
    <div className="layer-stack">
      {layers.map((layer, i) => (
        <LayerRow key={layer.id} layer={layer} index={i} count={layers.length} />
      ))}
      <div className="layer-add">
        <DropZone
          accept="video/*"
          label="+ Video layer"
          title="Drop a video file or click — adds a new layer on top"
          onFile={(f) => void addVideoLayer(f)}
        />
        <button
          className="icon-btn add-screen"
          title={active ? 'Add a screen-share layer' : 'Start the stream first'}
          aria-label="Add screen layer"
          disabled={!active}
          onClick={() => void addScreenLayer()}
        >
          🖥 Screen
        </button>
      </div>
      <span className="dim layer-foot">{roleLabel}</span>
    </div>
  )
}
