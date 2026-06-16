// Output stage: remote AI video + output sense overlay. The output resolution
// is reported into the footer stats (sessionStore.outDims), not as an overlay.

import { useEffect, useRef, useState } from 'react'
import { setRemoteTrackHandler, useSessionStore } from '../../state/sessionStore'
import { outputVision } from '../../state/runtime'
import { OverlayCanvas } from '../sense/OverlayCanvas'
import { RatingOverlay } from './RatingOverlay'
import { RatingBar } from './RatingBar'
import { FullscreenButton } from '../FullscreenButton'
import { useViewerReveal } from '../../lib/useViewerReveal'

export function Stage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const logLine = useSessionStore((s) => s.logLine)
  const connected = useSessionStore((s) => s.connected)
  const starting = useSessionStore((s) => s.starting)
  const status = useSessionStore((s) => s.status)
  const start = useSessionStore((s) => s.start)
  const canStart = !starting && !connected && status !== 'connecting...'
  const reveal = useViewerReveal()
  const [maximized, setMaximized] = useState(false)

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    // The received frame size IS the output resolution (already 2x when the
    // server's flow upscaler is on); surfaced in the footer stats.
    const updateDims = () =>
      useSessionStore.setState({
        outDims: v.videoWidth && v.videoHeight ? `${v.videoWidth}×${v.videoHeight}` : '—',
      })
    setRemoteTrackHandler((stream) => {
      const has = stream.getTracks().length
      v.srcObject = has ? stream : null
      if (has) {
        // muted+autoplay normally suffices; surface refusals in the log
        // instead of failing silently with a black stage.
        v.play().catch((err) => logLine('Autoplay blocked: ' + err.message))
      } else {
        useSessionStore.setState({ outDims: '—' })
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
      <div
        className={'remote-wrap overlay-anchor' + (reveal.shown ? ' controls-shown' : '') + (maximized ? ' viewport-max' : '')}
        {...reveal.pointerProps}
      >
        <video id="v" ref={videoRef} autoPlay playsInline muted />
        <OverlayCanvas source="output" />
        <RatingOverlay />
        <RatingBar />
        <FullscreenButton label="output" maximized={maximized} onToggle={() => setMaximized((v) => !v)} />
        {/* Only the start (▶) button overlays the video, as the primary CTA;
            stopping is done from the header's Stop control. */}
        {!connected && (
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
      </div>
    </div>
  )
}
