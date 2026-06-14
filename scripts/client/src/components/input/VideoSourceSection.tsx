// Video-file input source: a minimal drop/upload zone + icon transport. Sits
// side-by-side with the camera; the transport spans the panel width.

import { useRef, useState } from 'react'
import { fmtTime, usePipelineStore } from '../../state/pipelineStore'

const RATES = [0.25, 0.5, 1, 1.5, 2]

export function VideoSourceSection() {
  const p = usePipelineStore()
  const fileInput = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  const pick = (file: File | undefined) => {
    if (file) void p.loadVideoFile(file)
    // Reset so re-picking the same file fires change again.
    if (fileInput.current) fileInput.current.value = ''
  }

  return (
    <div className="src-panel video-panel">
      <div className="src-title">Video</div>
      <div className="controls src-row">
        <input
          ref={fileInput}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <div
          className={'video-drop' + (drag ? ' drag' : '')}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            if (!drag) setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDrag(false)
            pick(e.dataTransfer.files?.[0])
          }}
          title={p.videoLoaded ? `${p.videoName} — ${p.videoMeta}` : 'Drop a video file or click to choose'}
        >
          {p.videoLoaded ? p.videoName : 'Drop video or click'}
        </div>
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
          <select className="rate" title="Playback speed" value={p.videoRate} onChange={(e) => p.setVideoRate(+e.target.value)}>
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
