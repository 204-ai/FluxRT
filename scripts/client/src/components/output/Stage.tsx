// Output stage: remote AI video + output sense overlay.

import { useEffect, useRef } from 'react'
import { setRemoteTrackHandler, useSessionStore } from '../../state/sessionStore'
import { outputVision } from '../../state/runtime'
import { OverlayCanvas } from '../sense/OverlayCanvas'
import { RatingOverlay } from './RatingOverlay'

export function Stage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const logLine = useSessionStore((s) => s.logLine)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    setRemoteTrackHandler((stream) => {
      v.srcObject = stream.getTracks().length ? stream : null
      if (stream.getTracks().length) {
        // muted+autoplay normally suffices; surface refusals in the log
        // instead of failing silently with a black stage.
        v.play().catch((err) => logLine('Autoplay blocked: ' + err.message))
      }
    })
    outputVision.setVideo(v)
    return () => {
      setRemoteTrackHandler(() => {})
      outputVision.setVideo(null)
    }
  }, [logLine])

  return (
    <div className="stage">
      <div className="remote-wrap overlay-anchor">
        <video id="v" ref={videoRef} autoPlay playsInline muted />
        <OverlayCanvas source="output" />
        <RatingOverlay />
      </div>
    </div>
  )
}
