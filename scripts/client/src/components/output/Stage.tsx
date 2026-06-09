// Output stage: remote AI video + (split mode) input preview + edit toolbar.

import { useEffect, useRef } from 'react'
import { setRemoteTrackHandler, useSessionStore } from '../../state/sessionStore'
import { usePipelineStore } from '../../state/pipelineStore'
import { outputVision } from '../../state/runtime'
import { CanvasHost } from '../input/CanvasHost'
import { OverlayCanvas } from '../sense/OverlayCanvas'
import { DrawToolbar } from './DrawToolbar'

export function Stage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const pipelineActive = usePipelineStore((s) => s.active)
  const showInputPreview = usePipelineStore((s) => s.showInputPreview)
  const activeTab = useSessionStore((s) => s.activeTab)
  const logLine = useSessionStore((s) => s.logLine)

  const split = pipelineActive && showInputPreview
  const holdsPreview = split && activeTab === 'output'

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
    <div className={'stage' + (split ? ' split' : '')}>
      {split && <DrawToolbar />}
      <div id="invSlot" className="overlay-anchor">
        <CanvasHost holds={holdsPreview} />
        {holdsPreview && <OverlayCanvas source="input" />}
      </div>
      <div className="remote-wrap overlay-anchor">
        <video id="v" ref={videoRef} autoPlay playsInline muted />
        <OverlayCanvas source="output" />
      </div>
    </div>
  )
}
