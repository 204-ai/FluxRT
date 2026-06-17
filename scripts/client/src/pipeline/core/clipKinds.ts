// Clip-kind registry — the table that makes camera / video / feedback "just
// three entries" instead of a hardcoded layer set. Mirrors the effect registry
// (effects/registry.ts) in spirit: per-kind knowledge (how the source crosses
// into a backend, whether it can drive cadence, default mirror) lives in ONE
// place, so a new kind (image, shader, ndi, generator) is a single entry plus —
// for generative kinds — a frame producer. Imported by both the main thread and
// the pipeline worker, so it must stay DOM-free.

export type ClipKind = string // open union; 'camera' | 'video' | 'feedback' today

/**
 * How a clip's live source is handed to a backend:
 *  - 'mediastream'       a MediaStream we may consume directly (camera).
 *  - 'mediastream-clone' a MediaStream we must clone before processing, because
 *                        the original is shared (the remote output → feedback;
 *                        one MediaStreamTrackProcessor is allowed per track).
 *  - 'element'           an <video>/<img> element drawn / captured each frame
 *                        (a video file; the element's playback state is owned
 *                        elsewhere and survives rail restarts).
 */
export type ClipSourceForm = 'mediastream' | 'mediastream-clone' | 'element'

export interface ClipKindMeta {
  kind: ClipKind
  label: string
  /** Selfie flip belongs to the kind (camera = true). The layer's active clip
   *  seeds its `mirror` from this. */
  mirrorable: boolean
  /** May this kind drive the worker wake loop / set canvas dims / be the vision
   *  tap base? camera & video = true; feedback = false (no own cadence). */
  canBeBase: boolean
  sourceForm: ClipSourceForm
  /** DropZone `accept` for cells that load media from a file. */
  accept?: string
}

export const CLIP_KINDS: Record<ClipKind, ClipKindMeta> = {
  camera: { kind: 'camera', label: 'Camera', mirrorable: true, canBeBase: true, sourceForm: 'mediastream' },
  video: { kind: 'video', label: 'Video', mirrorable: false, canBeBase: true, sourceForm: 'element', accept: 'video/*' },
  feedback: { kind: 'feedback', label: 'Feedback', mirrorable: false, canBeBase: false, sourceForm: 'mediastream-clone' },
  // A new mediastream kind reuses the camera source path verbatim — proof that a
  // kind is one registry entry + an acquire (getDisplayMedia), no pipeline change.
  screen: { kind: 'screen', label: 'Screen', mirrorable: false, canBeBase: true, sourceForm: 'mediastream' },
}

/** Per-kind glyph for compact cell chips. */
export const CLIP_ICON: Record<ClipKind, string> = {
  camera: '📷',
  video: '🎞',
  feedback: '🔁',
  screen: '🖥',
}

/** The kinds a user can ADD as a new layer/cell. (feedback is auto-wired from
 *  the remote output; camera is the seeded layer.) Ordered for the "+" menu. */
export const ADDABLE_CLIP_KINDS: ClipKind[] = ['video', 'screen']

export function clipMeta(kind: ClipKind): ClipKindMeta {
  const m = CLIP_KINDS[kind]
  if (!m) throw new Error('unknown clip kind: ' + kind)
  return m
}

/** Default mirror for a fresh clip of this kind (selfie flip for the camera). */
export function defaultMirror(kind: ClipKind): boolean {
  return CLIP_KINDS[kind]?.mirrorable ?? false
}
