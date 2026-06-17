// Resolume-style layer / cell / clip model held by the pipeline store. A LAYER
// is a stack slot (mix + geometry); a CELL holds at most one CLIP; exactly one
// clip is active per layer. Clip kinds (camera | video | feedback | …) come
// from the clip-kind registry. This is the store/UI shape — the compositor only
// ever sees the resolved per-layer render options (LayerRender in core/types).
//
// P1 seeds the three legacy layers, one cell/clip each, so behavior is
// unchanged while the structure is now dynamic. add/remove/reorder (P2) and
// multi-cell grids (P3) build on this without a second migration.

import type { BlendMode, ClipId, ClipKind, LayerId, LayerTransform } from '../pipeline/core/types'
import { CAMERA_LAYER, VIDEO_LAYER, FEEDBACK_LAYER } from '../pipeline/core/types'
import type { ClipRole } from '../pipeline/core/clipKinds'
import { clipMeta, defaultMirror } from '../pipeline/core/clipKinds'

/** One clip occupying a cell. Holds only descriptors — live media handles
 *  (camera stream, <video> element, remote clone, effect instance) live in
 *  runtime maps keyed by clip id. */
export interface Clip {
  id: ClipId
  kind: ClipKind
  label: string
  /** Per-clip selfie flip; defaults from the kind (camera = true). */
  mirror: boolean
  /** Per-clip OBS framing (absent = cover-fit). */
  transform?: LayerTransform
  // --- kind-specific payloads (only the relevant one set) ---
  /** camera: chosen device (empty = default). */
  deviceId?: string
  /** video / image: the picked file (in-memory; not persisted — see needsFile). */
  file?: File
  /** True when a persisted file-backed clip needs the user to re-pick its file. */
  needsFile?: boolean
  /** effect: config for the CanvasEffect instance. */
  effectConfig?: Record<string, unknown>
}

/** A grid cell — empty (clip === null → the "+ add clip" slot) or holding one clip. */
export interface Cell {
  id: string
  clip: Clip | null
}

/** A compositing layer: a stack slot with its own mix and a track of cells, one
 *  of which is active. A layer is homogeneous: its `kind`/`role` are fixed by the
 *  first clip added; extra cells are the same kind. Empty layer = kind null. */
export interface Layer {
  id: LayerId
  name: string
  /** Fixed by the first clip; null until the layer has one. */
  kind: ClipKind | null
  role: ClipRole | null
  opacity: number
  blend: BlendMode
  /** OBS-style framing for this layer (absent = legacy cover-fit). Per-layer
   *  fallback; the active clip's own transform takes precedence when set. */
  transform?: LayerTransform
  cells: Cell[]
  /** Which cell's clip is live; null = layer muted (no active clip). */
  activeCellId: string | null
}

let n = 0
/** Monotonic id helper — avoids Date.now()/Math.random() (forbidden in some
 *  contexts) while staying unique within a session. */
export function freshId(prefix: string): string {
  n += 1
  return `${prefix}-${n}`
}

export function activeCell(layer: Layer): Cell | null {
  if (!layer.activeCellId) return null
  return layer.cells.find((c) => c.id === layer.activeCellId) ?? null
}

export function activeClip(layer: Layer): Clip | null {
  return activeCell(layer)?.clip ?? null
}

/** The kind of a layer's active clip (its "role"), or null when muted/empty. */
export function layerKind(layer: Layer): ClipKind | null {
  return activeClip(layer)?.kind ?? null
}

export function layerById(layers: Layer[], id: LayerId): Layer | undefined {
  return layers.find((l) => l.id === id)
}

/** Locate a clip (and its layer) anywhere in the stack by clip id. */
export function findClip(layers: Layer[], clipId: ClipId): { layer: Layer; clip: Clip } | null {
  for (const layer of layers) {
    for (const cell of layer.cells) {
      if (cell.clip?.id === clipId) return { layer, clip: cell.clip }
    }
  }
  return null
}

/** Build a fresh clip of a kind with a sensible default label + mirror. */
export function makeClip(kind: ClipKind, label?: string): Clip {
  return { id: freshId('clip'), kind, label: label ?? clipMeta(kind).label, mirror: defaultMirror(kind) }
}

/** A layer pre-seeded with a single clip of `kind` (active). */
export function makeLayer(kind: ClipKind, name: string, id?: LayerId): Layer {
  const clip = makeClip(kind, name)
  const cell: Cell = { id: freshId('cell'), clip }
  return {
    id: id ?? freshId('layer'),
    name,
    kind,
    role: clipMeta(kind).role,
    opacity: 1,
    blend: 'normal',
    cells: [cell],
    activeCellId: cell.id,
  }
}

/** A fresh empty layer: one empty cell, no kind yet (the grid's starting state
 *  and what [+ layer] appends). Click its empty cell to pick a kind. */
export function newEmptyLayer(name = 'Layer'): Layer {
  const cell: Cell = { id: freshId('cell'), clip: null }
  return {
    id: freshId('layer'),
    name,
    kind: null,
    role: null,
    opacity: 1,
    blend: 'normal',
    cells: [cell],
    activeCellId: null,
  }
}

/** The three legacy layers (front → back), with the stable migration ids so the
 *  pipeline composite, rail and store all agree during P0/P1. */
export function seedLayers(): Layer[] {
  return [
    makeLayer('camera', 'Camera', CAMERA_LAYER),
    makeLayer('video', 'Video', VIDEO_LAYER),
    makeLayer('feedback', 'Feedback', FEEDBACK_LAYER),
  ]
}
