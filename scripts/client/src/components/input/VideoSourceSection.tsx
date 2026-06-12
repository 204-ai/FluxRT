// Video-file input source: file picker, metadata, transport controls.

import { useRef } from 'react'
import { fmtTime, usePipelineStore } from '../../state/pipelineStore'

const RATES = [0.25, 0.5, 1, 1.5, 2]

export function VideoSourceSection() {
  const p = usePipelineStore()
  const fileInput = useRef<HTMLInputElement>(null)

  const pick = (file: File | undefined) => {
    if (file) void p.loadVideoFile(file)
    // Reset so re-picking the same file fires change again.
    if (fileInput.current) fileInput.current.value = ''
  }

  return (
    <>
      <div className="controls source-row">
        <span className={'source-badge' + (p.videoLoaded ? ' on' : '')}>video</span>
        <input
          ref={fileInput}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <button onClick={() => fileInput.current?.click()}>
          {p.videoLoaded ? 'Replace video…' : 'Load video…'}
        </button>
        {p.videoLoaded && (
          <>
            <span className="file-name" title={p.videoName}>
              {p.videoName}
            </span>
            <span className="dim">{p.videoMeta}</span>
            <button onClick={() => void p.unloadVideo()}>Unload</button>
          </>
        )}
        {!p.videoLoaded && <span className="dim">drive the input from an mp4/webm file</span>}
      </div>
      {p.videoLoaded && (
        <div className="controls transport">
          <button onClick={() => p.toggleVideoPlay()}>{p.videoPlaying ? '⏸' : '▶'}</button>
          <input
            type="range"
            min={0}
            max={p.videoDuration || 0}
            step={0.1}
            value={p.videoCurrentTime}
            onChange={(e) => p.seekVideo(+e.target.value)}
          />
          <span className="time-readout">
            {fmtTime(p.videoCurrentTime)} / {fmtTime(p.videoDuration)}
          </span>
          <label>
            <input
              type="checkbox"
              checked={p.videoLoop}
              onChange={(e) => p.setVideoLoop(e.target.checked)}
            />{' '}
            Loop
          </label>
          <label>
            speed{' '}
            <select value={p.videoRate} onChange={(e) => p.setVideoRate(+e.target.value)}>
              {RATES.map((r) => (
                <option key={r} value={r}>
                  {r}×
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
    </>
  )
}
