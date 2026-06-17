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
export type ClipSourceForm = 'mediastream' | 'mediastream-clone' | 'element' | 'bitmap'

/** A clip is either a frame SOURCE (camera/video/…) or an EFFECT that transforms
 *  the composite of everything below it in the stack (draw/marker/shader). */
export type ClipRole = 'source' | 'effect'

/** How the store acquires a source clip's live media. */
export type ClipAcquire = 'getUserMedia' | 'getDisplayMedia' | 'file' | 'remote' | 'none'

export interface ClipKindMeta {
  kind: ClipKind
  role: ClipRole
  label: string
  /** Selfie flip belongs to the kind (camera = true). A fresh clip seeds its
   *  `mirror` from this. */
  mirrorable: boolean
  /** May this kind drive the worker wake loop / vision tap base? Frame-producing
   *  inputs (camera/video/screen) = true; feedback/effects = false. When no base
   *  is active the worker runs on a ticker. */
  canBeBase: boolean
  /** Source clips: how the live media crosses into a backend. Effect clips: none. */
  sourceForm?: ClipSourceForm
  /** Effect clips: the CanvasEffect registry name to instantiate. */
  effectName?: string
  /** How the store obtains the live source (source clips only). */
  acquire?: ClipAcquire
  /** DropZone `accept` for file-backed kinds. */
  accept?: string
}

export const CLIP_KINDS: Record<ClipKind, ClipKindMeta> = {
  camera: { kind: 'camera', role: 'source', label: 'Camera', mirrorable: true, canBeBase: true, sourceForm: 'mediastream', acquire: 'getUserMedia' },
  video: { kind: 'video', role: 'source', label: 'Video', mirrorable: false, canBeBase: true, sourceForm: 'element', acquire: 'file', accept: 'video/*' },
  feedback: { kind: 'feedback', role: 'source', label: 'Feedback', mirrorable: false, canBeBase: false, sourceForm: 'mediastream-clone', acquire: 'remote' },
  screen: { kind: 'screen', role: 'source', label: 'Screen', mirrorable: false, canBeBase: true, sourceForm: 'mediastream', acquire: 'getDisplayMedia' },
  // A still image — a static frame; can't drive cadence (the worker ticker
  // composites it). One registry entry + a bitmap source path.
  image: { kind: 'image', role: 'source', label: 'Image', mirrorable: false, canBeBase: false, sourceForm: 'bitmap', acquire: 'file', accept: 'image/*' },
  // Effect clips — interleaved into the back-to-front loop at their layer's
  // position, transforming everything composited below. Reuse the existing
  // CanvasEffect registry (effects/registry.ts) by name.
  draw: { kind: 'draw', role: 'effect', label: 'Draw', mirrorable: false, canBeBase: false, effectName: 'drawLayer', acquire: 'none' },
  marker: { kind: 'marker', role: 'effect', label: 'Marker', mirrorable: false, canBeBase: false, effectName: 'marker', acquire: 'none' },
  shader: { kind: 'shader', role: 'effect', label: 'Shader', mirrorable: false, canBeBase: false, effectName: 'shader', acquire: 'none' },
}

/** Per-kind glyph for compact cell chips. */
export const CLIP_ICON: Record<ClipKind, string> = {
  camera: '📷',
  video: '🎞',
  feedback: '🔁',
  screen: '🖥',
  image: '🖼',
  draw: '✏️',
  marker: '⌖',
  shader: '✨',
}

/** Source kinds offered in the cell picker. */
export const SOURCE_CLIP_KINDS: ClipKind[] = ['camera', 'video', 'feedback', 'screen', 'image']
/** Effect kinds offered in the cell picker (draw stays a global tool for now). */
export const EFFECT_CLIP_KINDS: ClipKind[] = ['marker', 'shader']
/** All kinds a user can pick when filling an empty cell. */
export const ADDABLE_CLIP_KINDS: ClipKind[] = [...SOURCE_CLIP_KINDS, ...EFFECT_CLIP_KINDS]

export function clipMeta(kind: ClipKind): ClipKindMeta {
  const m = CLIP_KINDS[kind]
  if (!m) throw new Error('unknown clip kind: ' + kind)
  return m
}

export function isEffectKind(kind: ClipKind): boolean {
  return CLIP_KINDS[kind]?.role === 'effect'
}

/** Default mirror for a fresh clip of this kind (selfie flip for the camera). */
export function defaultMirror(kind: ClipKind): boolean {
  return CLIP_KINDS[kind]?.mirrorable ?? false
}
