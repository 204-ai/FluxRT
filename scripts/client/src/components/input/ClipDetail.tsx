// Detail / inspector pane for the SELECTED clip. Selecting a clip in the layer
// stack drives this; it shows the clip's kind-specific details + quick controls
// (transport for video, mirror for camera, framing for any). Dispatches by clip
// kind so a new kind (P4: screen; later: image/shader) is one more branch — or,
// eventually, a registry-provided editor.

import { fmtTime, usePipelineStore } from '../../state/pipelineStore'
import { findClip, type Clip } from '../../state/layerModel'
import { VIDEO_LAYER } from '../../pipeline/core/types'
import { CLIP_ICON, clipMeta } from '../../pipeline/core/clipKinds'

/** Frame-on-preview toggle (move/resize/crop the layer via TransformOverlay). */
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

/** Transport row driven by either the seeded video store fields or an added
 *  clip's extraVideo state — whichever backs this clip. */
function VideoDetail({ clip }: { clip: Clip }) {
  const isSeeded = useIsSeededVideoSelected()
  const seeded = usePipelineStore((s) => ({
    name: s.videoName,
    meta: s.videoMeta,
    duration: s.videoDuration,
    currentTime: s.videoCurrentTime,
    playing: s.videoPlaying,
    loaded: s.videoLoaded,
  }))
  const extra = usePipelineStore((s) => s.extraVideo[clip.id])
  const toggleSeeded = usePipelineStore((s) => s.toggleVideoPlay)
  const seekSeeded = usePipelineStore((s) => s.seekVideo)
  const toggleExtra = usePipelineStore((s) => s.toggleExtraVideoPlay)
  const seekExtra = usePipelineStore((s) => s.seekExtraVideo)

  if (isSeeded) {
    if (!seeded.loaded) return <span className="dim">No video loaded — drop one on the Video layer.</span>
    return (
      <>
        <div className="dim clip-detail-meta">{seeded.meta}</div>
        <div className="controls transport">
          <button className="icon-btn" title={seeded.playing ? 'Pause' : 'Play'} onClick={() => toggleSeeded()}>
            {seeded.playing ? '⏸' : '▶'}
          </button>
          <input
            className="seek"
            type="range"
            min={0}
            max={seeded.duration || 0}
            step={0.1}
            value={seeded.currentTime}
            onChange={(e) => seekSeeded(+e.target.value)}
          />
          <span className="time-readout">
            {fmtTime(seeded.currentTime)}/{fmtTime(seeded.duration)}
          </span>
        </div>
      </>
    )
  }
  if (!extra) return <span className="dim">Video unavailable.</span>
  return (
    <div className="controls transport">
      <button className="icon-btn" title={extra.playing ? 'Pause' : 'Play'} onClick={() => toggleExtra(clip.id)}>
        {extra.playing ? '⏸' : '▶'}
      </button>
      <input
        className="seek"
        type="range"
        min={0}
        max={extra.duration || 0}
        step={0.1}
        value={extra.currentTime}
        onChange={(e) => seekExtra(clip.id, +e.target.value)}
      />
      <span className="time-readout">
        {fmtTime(extra.currentTime)}/{fmtTime(extra.duration)}
      </span>
    </div>
  )
}

/** True when the selected clip is the seeded video layer's clip. */
function useIsSeededVideoSelected(): boolean {
  return usePipelineStore((s) => {
    const found = s.selectedClipId ? findClip(s.layers, s.selectedClipId) : null
    return found?.layer.id === VIDEO_LAYER
  })
}

function CameraDetail() {
  const mirror = usePipelineStore((s) => s.mirror)
  const camEnabled = usePipelineStore((s) => s.camEnabled)
  const setMirror = usePipelineStore((s) => s.setMirror)
  return (
    <label className="mirror-lbl" title="Mirror the camera (selfie view)">
      <input type="checkbox" checked={mirror} disabled={!camEnabled} onChange={(e) => setMirror(e.target.checked)} /> Mirror
    </label>
  )
}

function FeedbackDetail() {
  const live = usePipelineStore((s) => s.feedbackAvailable && s.active)
  return <span className="dim">{live ? 'output → input loop (live)' : 'waiting for the stream to run'}</span>
}

function ClipDetailBody({ clip }: { clip: Clip }) {
  if (clip.kind === 'camera') return <CameraDetail />
  if (clip.kind === 'video') return <VideoDetail clip={clip} />
  if (clip.kind === 'feedback') return <FeedbackDetail />
  if (clip.kind === 'screen') return <span className="dim">screen share — live overlay</span>
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
