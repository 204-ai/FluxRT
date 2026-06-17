// Resolume-style composition grid: each LAYER is a row, each CLIP a cell. Click
// an empty cell to pick a source kind; click a clip cell to activate + select it
// (details + controls show in the ClipDetail pane). Add / remove / reorder
// layers; add / remove / swap clips. Layers are homogeneous — once a layer has a
// kind, its empty cells offer only that kind.

import { useRef, useState } from 'react'
import { usePipelineStore } from '../../state/pipelineStore'
import { useSessionStore } from '../../state/sessionStore'
import { activeClip, type Cell, type Layer } from '../../state/layerModel'
import type { BlendMode, ClipKind, LayerId } from '../../pipeline/core/types'
import { CLIP_ICON, clipMeta, isEffectKind } from '../../pipeline/core/clipKinds'

const BLENDS: BlendMode[] = ['normal', 'screen', 'multiply', 'difference']
const BLEND_SHORT: Record<BlendMode, string> = {
  normal: 'nrm',
  screen: 'scr',
  multiply: 'mul',
  difference: 'dif',
}
// Cell picker, grouped. (image: static-frame path pending; draw/marker: global
// tools for now.)
const PICK_SOURCES: ClipKind[] = ['camera', 'video', 'feedback', 'screen']
const PICK_EFFECTS: ClipKind[] = ['shader']
const ALL_PICKABLE: ClipKind[] = [...PICK_SOURCES, ...PICK_EFFECTS]

/** Per-layer mix: frame button + blend cycle + opacity fader. */
function LayerMix({ layer }: { layer: Layer }) {
  const setLayerOpacity = usePipelineStore((s) => s.setLayerOpacity)
  const setLayerBlend = usePipelineStore((s) => s.setLayerBlend)
  const layoutLayer = usePipelineStore((s) => s.layoutLayer)
  const setLayoutLayer = usePipelineStore((s) => s.setLayoutLayer)
  const active = usePipelineStore((s) => s.active)
  const framing = layoutLayer === layer.id
  const live = active && !!activeClip(layer)
  const pct = Math.round(layer.opacity * 100)
  const cycleBlend = () => setLayerBlend(layer.id, BLENDS[(BLENDS.indexOf(layer.blend) + 1) % BLENDS.length])
  return (
    <div className="layer-mix" onClick={(e) => e.stopPropagation()}>
      <button
        className={'icon-btn frame-btn' + (framing ? ' on' : '')}
        title="Frame this layer on the preview"
        aria-pressed={framing}
        disabled={!live}
        onClick={() => setLayoutLayer(framing ? null : layer.id)}
      >
        ◳
      </button>
      <button className="icon-btn blend-btn" title={`Blend: ${layer.blend}`} onClick={cycleBlend}>
        {BLEND_SHORT[layer.blend]}
      </button>
      <input
        className="layer-fader"
        type="range"
        min={0}
        max={100}
        step={1}
        value={pct}
        aria-label={`${layer.name} opacity`}
        onChange={(e) => setLayerOpacity(layer.id, +e.target.value / 100)}
      />
      <span className="dim layer-pct">{pct}%</span>
    </div>
  )
}

/** Grouped dropdown shown over an empty cell. Camera/feedback/screen/effects
 *  fill immediately; video opens a file picker. */
function KindPicker({ layerId, cellId, allowed }: { layerId: LayerId; cellId: string; allowed: ClipKind[] }) {
  const fillCamera = usePipelineStore((s) => s.fillCellCamera)
  const fillVideo = usePipelineStore((s) => s.fillCellVideo)
  const fillFeedback = usePipelineStore((s) => s.fillCellFeedback)
  const fillScreen = usePipelineStore((s) => s.fillCellScreen)
  const fillEffect = usePipelineStore((s) => s.fillCellEffect)
  const fileRef = useRef<HTMLInputElement>(null)
  const [open, setOpen] = useState(false)

  const pick = (kind: ClipKind) => {
    setOpen(false)
    if (kind === 'video') fileRef.current?.click()
    else if (kind === 'camera') void fillCamera(layerId, cellId, '')
    else if (kind === 'feedback') void fillFeedback(layerId, cellId)
    else if (kind === 'screen') void fillScreen(layerId, cellId)
    else if (isEffectKind(kind)) void fillEffect(layerId, cellId, kind)
  }

  const sources = PICK_SOURCES.filter((k) => allowed.includes(k))
  const effects = PICK_EFFECTS.filter((k) => allowed.includes(k))

  return (
    <div className="cell-add">
      <button
        className="clip-cell empty"
        title="Add a clip"
        onClick={(e) => {
          e.stopPropagation()
          setOpen((v) => !v)
        }}
      >
        +
      </button>
      {open && (
        <>
          <div className="kind-backdrop" onClick={() => setOpen(false)} />
          <div className="kind-menu" onClick={(e) => e.stopPropagation()}>
            {sources.length > 0 && <div className="kind-group">Sources</div>}
            {sources.map((k) => (
              <button key={k} className="kind-opt" onClick={() => pick(k)}>
                {CLIP_ICON[k] ?? '◻'} {clipMeta(k).label}
              </button>
            ))}
            {effects.length > 0 && <div className="kind-group">Effects</div>}
            {effects.map((k) => (
              <button key={k} className="kind-opt" onClick={() => pick(k)}>
                {CLIP_ICON[k] ?? '◻'} {clipMeta(k).label}
              </button>
            ))}
          </div>
        </>
      )}
      <input
        ref={fileRef}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void fillVideo(layerId, cellId, f)
          e.target.value = ''
        }}
      />
    </div>
  )
}

/** One grid cell — a clip chip (activate + select on click) or an empty picker. */
function ClipCell({ layer, cell }: { layer: Layer; cell: Cell }) {
  const selectedClipId = usePipelineStore((s) => s.selectedClipId)
  const activateCell = usePipelineStore((s) => s.activateCell)
  const removeClip = usePipelineStore((s) => s.removeClip)
  const clip = cell.clip

  if (!clip) {
    const allowed = layer.kind ? [layer.kind] : ALL_PICKABLE
    return <KindPicker layerId={layer.id} cellId={cell.id} allowed={allowed} />
  }
  const isActive = layer.activeCellId === cell.id
  const isSel = selectedClipId === clip.id
  return (
    <button
      className={'clip-cell' + (isActive ? ' active' : '') + (isSel ? ' sel' : '')}
      title={clip.label}
      onClick={(e) => {
        e.stopPropagation()
        void activateCell(layer.id, cell.id)
      }}
    >
      <span className="clip-cell-icon">{CLIP_ICON[clip.kind] ?? '◻'}</span>
      <span className="clip-cell-label">{clip.label}</span>
      <span
        className="clip-cell-x"
        title="Remove clip"
        onClick={(e) => {
          e.stopPropagation()
          void removeClip(layer.id, cell.id)
        }}
      >
        ⌫
      </span>
    </button>
  )
}

function LayerRow({ layer, index, count }: { layer: Layer; index: number; count: number }) {
  const moveLayer = usePipelineStore((s) => s.moveLayer)
  const removeLayer = usePipelineStore((s) => s.removeLayer)
  const addCell = usePipelineStore((s) => s.addCell)
  const selectClip = usePipelineStore((s) => s.selectClip)
  const clip = activeClip(layer)
  return (
    <div className="layer-row-wrap" onClick={() => clip && selectClip(clip.id)}>
      <div className="layer-reorder">
        <button className="icon-btn" title="Move up" disabled={index === 0} onClick={(e) => { e.stopPropagation(); moveLayer(layer.id, -1) }}>▲</button>
        <button className="icon-btn" title="Move down" disabled={index === count - 1} onClick={(e) => { e.stopPropagation(); moveLayer(layer.id, 1) }}>▼</button>
      </div>
      <div className="layer-grid-row">
        <div className="cell-track">
          {layer.cells.map((cell) => (
            <ClipCell key={cell.id} layer={layer} cell={cell} />
          ))}
          {layer.kind && (
            <button
              className="clip-cell empty add-cell"
              title="Add another clip"
              onClick={(e) => { e.stopPropagation(); addCell(layer.id) }}
            >
              +
            </button>
          )}
        </div>
        <LayerMix layer={layer} />
      </div>
      <button
        className="icon-btn layer-remove"
        title="Remove layer"
        onClick={(e) => { e.stopPropagation(); void removeLayer(layer.id) }}
      >
        ⌫
      </button>
    </div>
  )
}

export function LayerStack() {
  const layers = usePipelineStore((s) => s.layers)
  const addLayer = usePipelineStore((s) => s.addLayer)
  const inputRole = useSessionStore((s) => s.inputRole)
  const roleLabel =
    inputRole === 'you'
      ? 'input: you (steering)'
      : inputRole === 'peer'
        ? 'input: peer (other client)'
        : 'input: server'

  return (
    <div className="layer-stack">
      <div className="layer-stack-head">
        <span className="dim">COMPOSITION</span>
        <button className="icon-btn add-layer" title="Add a layer on top" onClick={() => addLayer()}>
          + Layer
        </button>
      </div>
      {layers.map((layer, i) => (
        <LayerRow key={layer.id} layer={layer} index={i} count={layers.length} />
      ))}
      <span className="dim layer-foot">{roleLabel}</span>
    </div>
  )
}
