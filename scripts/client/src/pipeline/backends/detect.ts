import type { RailBackendKind } from '../core/types'

/**
 * Streams backend needs window-context MediaStreamTrackProcessor + Generator
 * (Chrome/Edge). Safari exposes them worker-only (VideoTrackGenerator) and
 * Firefox not at all — both land on the canvas fallback.
 */
export function detectBackend(): RailBackendKind {
  if (
    typeof MediaStreamTrackProcessor !== 'undefined' &&
    typeof MediaStreamTrackGenerator !== 'undefined'
  ) {
    return 'streams'
  }
  return 'canvas'
}
