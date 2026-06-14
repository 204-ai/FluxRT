// Ambient declarations for mediacapture-transform (Chrome/Edge; not yet in
// lib.dom). Used only behind the capability probe in backends/detect.ts.

interface MediaStreamTrackProcessorInit {
  track: MediaStreamTrack
  maxBufferSize?: number
}

declare class MediaStreamTrackProcessor<T = VideoFrame> {
  constructor(init: MediaStreamTrackProcessorInit)
  readonly readable: ReadableStream<T>
}

interface MediaStreamTrackGeneratorInit {
  kind: 'video' | 'audio'
}

declare class MediaStreamTrackGenerator<T = VideoFrame> extends MediaStreamTrack {
  constructor(init: MediaStreamTrackGeneratorInit)
  readonly writable: WritableStream<T>
}

// Chrome/Firefox; not in lib.dom (Safari has no HTMLMediaElement.captureStream).
// Only called from the streams backend, which is Chrome/Edge-gated.
interface HTMLMediaElement {
  captureStream(): MediaStream
}
