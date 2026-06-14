// Output stage: remote AI video + output sense overlay + a resolution readout.

import { useEffect, useRef, useState } from 'react'
import { setRemoteTrackHandler, useSessionStore } from '../../state/sessionStore'
import { outputVision } from '../../state/runtime'
import { OverlayCanvas } from '../sense/OverlayCanvas'
import { RatingOverlay } from './RatingOverlay'

export function Stage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const logLine = useSessionStore((s) => s.logLine)
  const connected = useSessionStore((s) => s.connected)
  const starting = useSessionStore((s) => s.starting)
  const status = useSessionStore((s) => s.status)
  const start = useSessionStore((s) => s.start)
  const stop = useSessionStore((s) => s.stop)
  const canStart = !starting && !connected && status !== 'connecting...'
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    // The received frame size IS the output resolution (already 2x when the
    // server's flow upscaler is on).
    const updateDims = () =>
      setDims(v.videoWidth && v.videoHeight ? { w: v.videoWidth, h: v.videoHeight } : null)
    setRemoteTrackHandler((stream) => {
      const has = stream.getTracks().length
      v.srcObject = has ? stream : null
      if (has) {
        // muted+autoplay normally suffices; surface refusals in the log
        // instead of failing silently with a black stage.
        v.play().catch((err) => logLine('Autoplay blocked: ' + err.message))
      } else {
        setDims(null)
      }
    })
    v.addEventListener('loadedmetadata', updateDims)
    v.addEventListener('resize', updateDims)
    outputVision.setVideo(v)
    return () => {
      setRemoteTrackHandler(() => {})
      v.removeEventListener('loadedmetadata', updateDims)
      v.removeEventListener('resize', updateDims)
      outputVision.setVideo(null)
    }
  }, [logLine])

  return (
    <div className="stage">
      <div className="remote-wrap overlay-anchor">
        <video id="v" ref={videoRef} autoPlay playsInline muted />
        <OverlayCanvas source="output" />
        <RatingOverlay />
        {connected ? (
          <button
            className="play-overlay pause"
            title="Stop stream"
            aria-label="Stop stream"
            onClick={() => stop()}
          >
            ⏸
          </button>
        ) : (
          <button
            className="play-overlay"
            title="Start stream"
            aria-label="Start stream"
            disabled={!canStart}
            onClick={() => void start()}
          >
            ▶
          </button>
        )}
        {dims && (
          <div className="res-badge" title="Output resolution">
            {dims.w}×{dims.h}
          </div>
        )}
      </div>
    </div>
  )
}
