// Composition persistence — saves the grid STRUCTURE (layers, cells, clip kinds,
// mix, blend, geometry, effect config, camera deviceId) to localStorage. Live
// media handles are never serialized: on load, camera/screen clips re-acquire on
// activation; video/image clips are flagged `needsFile` (a File can't be reopened
// from storage) and need a re-pick. Ids are regenerated on load so a restored
// composition never collides with this session's freshly-minted ids.

import type { BlendMode, ClipKind, LayerTransform } from '../pipeline/core/types'
import type { Cell, Clip, Layer } from './layerModel'
import { freshId } from './layerModel'
import { clipMeta } from '../pipeline/core/clipKinds'

const KEY = 'fluxrt.composition.v1'

interface ClipDoc {
  kind: ClipKind
  label: string
  mirror: boolean
  transform?: LayerTransform
  deviceId?: string
  effectConfig?: Record<string, unknown>
}
interface CellDoc {
  active: boolean
  clip: ClipDoc | null
}
interface LayerDoc {
  name: string
  kind: ClipKind | null
  opacity: number
  blend: BlendMode
  transform?: LayerTransform
  cells: CellDoc[]
}
interface CompositionDoc {
  version: 1
  layers: LayerDoc[]
}

function clipDoc(c: Clip): ClipDoc {
  return {
    kind: c.kind,
    label: c.label,
    mirror: c.mirror,
    transform: c.transform,
    deviceId: c.deviceId,
    effectConfig: c.effectConfig,
  }
}

export function saveComposition(layers: Layer[]): void {
  try {
    const doc: CompositionDoc = {
      version: 1,
      layers: layers.map((l) => ({
        name: l.name,
        kind: l.kind,
        opacity: l.opacity,
        blend: l.blend,
        transform: l.transform,
        cells: l.cells.map((c) => ({ active: c.id === l.activeCellId, clip: c.clip ? clipDoc(c.clip) : null })),
      })),
    }
    localStorage.setItem(KEY, JSON.stringify(doc))
  } catch {
    /* storage unavailable / quota — non-fatal */
  }
}

export function loadComposition(): Layer[] | null {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return null
    const doc = JSON.parse(raw) as CompositionDoc
    if (doc.version !== 1 || !Array.isArray(doc.layers) || doc.layers.length === 0) return null
    return doc.layers.map((ld) => {
      let activeCellId: string | null = null
      const cells: Cell[] = ld.cells.map((cd) => {
        const id = freshId('cell')
        if (cd.active) activeCellId = id
        if (!cd.clip) return { id, clip: null }
        const clip: Clip = {
          id: freshId('clip'),
          kind: cd.clip.kind,
          label: cd.clip.label,
          mirror: cd.clip.mirror,
          transform: cd.clip.transform,
          deviceId: cd.clip.deviceId,
          effectConfig: cd.clip.effectConfig,
          // video/image carry no File across reloads — flag for re-pick.
          needsFile: cd.clip.kind === 'video' || cd.clip.kind === 'image',
        }
        return { id, clip }
      })
      return {
        id: freshId('layer'),
        name: ld.name,
        kind: ld.kind,
        role: ld.kind ? clipMeta(ld.kind).role : null,
        opacity: ld.opacity,
        blend: ld.blend,
        transform: ld.transform,
        cells,
        activeCellId,
      }
    })
  } catch {
    return null
  }
}
