# Layers Grid Redesign — fully generic clip grid

Supersedes the seeded camera/video/feedback model (P0–P4, already merged). New
target, per your spec:

- Start with **one layer, one empty cell**.
- **Click an empty cell → pick its kind** (camera / video / feedback / screen /
  image / … or an **effect**: draw / marker / shader).
- **Add / remove / reorder layers**; **add / remove / swap clips** within a layer.
- Layers are **homogeneous** ("more clips from the same selection type"): a
  layer's kind is fixed by its first clip; extra cells are the same kind.
- **N of every kind** (multiple cameras, videos, screens, feedback taps, …).
- Layout is a **grid**: each layer is a row, each clip a cell/column. Exactly one
  cell active per layer (the live one); selecting a cell shows it in the detail
  pane; the composite plays on the main preview.

What P0–P4 already gives us (reused, not rebuilt): the dynamic pipeline core —
`Composite` (ordered `LayerRender[]`), worker `Map<layerId,slot>` valves with a
`base` re-designation hook, `setLayerSource(id, kind, source)` /
`setComposite(op)`, the `clipKinds` registry, per-clip video pool, screen-share.
The redesign is mostly **store + UI + a few pipeline extensions**, not a
ground-up rewrite.

---

## 1. What has to change vs P0–P4

| Area | P0–P4 (now) | Redesign |
|---|---|---|
| Layers | seeded 3 (camera/video/feedback) | start with 1 empty layer; user builds the grid |
| Sources | camera = singleton in `rail` (getUserMedia inside `rail.start`); video pool; feedback singleton | **every clip owns its source**, acquired by the store; `rail` no longer owns getUserMedia |
| Base/cadence | camera or seeded video, fixed at start | **any frame-producing clip**; re-selected on removal; **rAF ticker fallback** when none (effect/image-only comps still render) |
| Start condition | camera enabled or video loaded | first frame-producing clip activated; can also start tickered (image/effect only) |
| Effects | `draw` + `marker` global, appended after all layers | **effect clips** interleaved at their layer's z-position (affect everything below) |
| Store | `camEnabled`/`videoLoaded`/`feedbackAvailable` + `video*` singleton fields | generic `layers[]` + per-clip source/transport maps; those flags become derived |
| UI | fixed rows w/ inline controls | **grid**: rows×cells, empty-cell kind picker, detail pane for the selected clip |

---

## 2. Data model

```ts
// pipeline/core/clipKinds.ts — extend the existing registry
export type ClipRole = 'source' | 'effect'
export interface ClipKindMeta {
  kind: ClipKind
  role: ClipRole                 // NEW: source produces frames; effect transforms the stack
  label: string
  mirrorable: boolean
  canBeBase: boolean             // can drive cadence/tap (frame-producing source)
  sourceForm?: 'mediastream' | 'mediastream-clone' | 'element' | 'bitmap'
  effectName?: string            // for role:'effect' → the CanvasEffect registry name
  acquire?: 'getUserMedia' | 'getDisplayMedia' | 'file' | 'remote' | 'none'
  accept?: string
}
// camera / video / feedback / screen (source) + image (source, bitmap) +
// draw / marker / shader (effect). One table entry each — that's the seam.
```

```ts
// state/layerModel.ts
export interface Clip {
  id: ClipId
  kind: ClipKind
  label: string
  mirror: boolean
  transform?: LayerTransform     // per-clip framing (moves off the layer)
  // source-kind payloads (only the relevant one is set):
  deviceId?: string              // camera
  file?: File                    // video / image
  // effect-kind payload:
  effectConfig?: Record<string, unknown>
}
export interface Cell { id: string; clip: Clip | null }   // null = empty "+ add"
export interface Layer {
  id: LayerId
  name: string
  kind: ClipKind | null          // fixed by the first clip; null = empty layer
  role: ClipRole | null
  opacity: number
  blend: BlendMode
  cells: Cell[]
  activeCellId: string | null
}
export const newEmptyLayer = (): Layer => ({ /* one empty cell, kind null */ })
```

Store no longer holds `camEnabled`/`videoLoaded`/`feedbackAvailable` as truth —
they become selectors derived from `layers[]`. Live media handles live in
runtime maps keyed by **clip id**:

```ts
// state/runtime.ts
cameraStreams: Map<ClipId, MediaStream>     // getUserMedia per camera clip
videoSources:  Map<ClipId, VideoFileSource> // per video clip (exists, generalized)
screenStreams: Map<ClipId, MediaStream>     // exists
feedbackClones:Map<ClipId, MediaStreamTrack>// remote-output clone per feedback clip
// images: Map<ClipId, ImageBitmap>          // later
```

---

## 3. Pipeline extensions (small, additive on P0)

1. **`rail` stops owning getUserMedia.** `rail.start(opts, base)` takes an
   already-acquired base — `{ layerId, kind, stream?, videoEl? }` — or **no base**
   (ticker mode). Dims already fixed to `targetAspect`, so base only supplies
   cadence + the vision tap. The store acquires every source (camera included)
   and passes/attaches them. `swapCameraDevice` moves to the store (acquire new
   stream → `setLayerSource`).
2. **Worker rAF/timer ticker fallback.** When the `baseLayerId` slot has no
   pending frame *and no base exists*, drive the compose loop on a ~30 fps timer
   so retained frames (feedback/screen/image) + effects keep compositing. (The
   `base` re-designation hook already exists from P0; add the ticker.)
3. **Interleaved effect layers.** The compositor's back-to-front loop becomes:
   for each layer in order — if it's a **source** layer, `drawLayer(frame)`; if
   it's an **effect** layer, run its `CanvasEffect.render(ctx, info, bus)` at that
   point (so it transforms everything composited *below* it). Effect layers carry
   no frame valve. `draw`/`marker` become effect kinds you can place; the legacy
   global draw/marker can stay until the UI migrates.
   - Compositor holds, per layer: either a frame (from `frames[id]`) or an
     effect instance. Effect instances are created from the registry by
     `effectName`, configured via `effectConfig`, addressed by layer id.

These are extensions of P0 contracts, not rewrites: `drawLayer`, geometry, valve
ownership/leak accounting, `setLayerSource`, `setComposite(op)` all stand.

---

## 4. Store: lifecycle & actions

Single reconciler keeps the rail in sync with `layers[]` so each action stays
simple and the WebRTC-alive invariant is centralized:

```ts
// desired = each layer's ACTIVE clip resolved to {layerId, kind, role, source}
async function syncPipeline() {
  const desired = activeBindings(get().layers)           // skips empty/effect-no-source
  const base = desired.find(d => canBeBase(d.kind) && d.source)
  if (!desired.length) { rail.stop(); set({active:false}); return }
  if (!rail.active || baseChanged(base)) {
    await rail.start(opts, base ?? null)                 // null = ticker mode
    rail.setComposite(replaceWith(compositeFrom(get().layers)))
    for (const d of desired) if (d !== base) bindSource(d) // setLayerSource
    for (const e of effectLayers()) rail.setComposite(addEffect(e))
  }
  set({ active: true })
}
```

Most edits are hot (no restart): add/remove/reorder layer → `setComposite(op)`;
activate/swap a cell → acquire (if needed) + `setLayerSource`; mix/transform →
`setComposite patch`. **Restart only when the base clip identity changes** (then
replay all bindings, exactly the P0 feedback/draw replay pattern, generalized).

Actions: `addLayer` / `removeLayer` / `moveLayer`; `addCell` / `removeCell`;
`fillCell(layerId, cellId, kind, payload)` (kind picker result → acquire source /
create effect → activate); `activateCell` (swap active clip); `selectClip`;
per-clip `setMirror` / transport / `setEffectConfig`; per-layer `opacity`/`blend`.
`refreshCameras` stays. `attachFeedback(remote)` fans the remote stream out to
every active feedback clip (clone per clip).

Base-removal: `syncPipeline` re-selects another frame-producing active clip, or
drops to ticker mode (effects/image still composite), or stops if nothing left.

---

## 5. Grid UI

```
┌───────────────────────────── COMPOSITION ───────────── [+ layer] ┐
│ row = layer            cells = clips (cols)        mix         │
│ ▲▼ Layer 3  [📷cam0●][📷cam1][ + ]      ◳ nrm ▓▓▓▓░ 80% [⌫]   │ front
│ ▲▼ Layer 2  [🎞clipA][🎞clipB●][ + ]    ◳ scr ▓▓▓▓▓100% [⌫]   │
│ ▲▼ Layer 1  [✏draw●][ + ]               ◳ nrm ▓▓░░░ 40% [⌫]   │ back
│ ▲▼ Layer 0  [ + add clip ]              (empty)               │
└────────────────────────────────────────────────────────────────┘
   ● = active cell   click empty cell → kind picker   [+ layer] appends empty
┌──────── main preview (live composite) ─────────┐ ┌── DETAIL: clipB ──┐
│                                                 │ │ 🎞 beach.mp4      │
│                                                 │ │ ▶ ──●── 0:12/0:42 │
└─────────────────────────────────────────────────┘ │ 🔁 1×  ☐ mirror   │
                                                     │ ◳ frame           │
                                                     └───────────────────┘
```

- **Empty cell** → a small kind picker (📷 Camera · 🎞 Video · 🔁 Feedback · 🖥
  Screen · ✏ Draw · … from `ADDABLE_CLIP_KINDS`). Picking acquires the source
  (camera → device list; video → file; screen → getDisplayMedia; effect →
  instantiate) and activates the cell. Once a layer has a kind, its other empty
  cells offer only that kind.
- **Cell** chip: kind glyph + label; active = green border; selected = ring.
  Click = activate + select. Right-edge ⌫ (hover) = remove clip.
- **Per-layer**: ▲▼ reorder, blend + opacity fader (reused `LayerMix`), ⌫ remove
  layer. **[+ layer]** appends an empty layer (one empty cell).
- **Detail pane** (selected clip): kind-specific editor (device picker, transport,
  mirror, effect params) + ◳ frame-on-preview. Registry-dispatched.
- Reuse existing CSS idioms (`icon-btn`, faders, `drop-zone`, dark theme); net-new
  = the cell-grid CSS.

---

## 6. Migration / phasing (each compiles + behavior visible)

- **G0 — registry + model**: add `role`/`effectName` to `clipKinds`; effect kinds
  (`draw`/`marker`); `newEmptyLayer`; `Clip` payload fields. No behavior change.
- **G1 — store generic sources**: per-clip source maps; move getUserMedia to the
  store; `rail.start(base)` generalized; `syncPipeline` reconciler. Seed still 3
  layers so existing UI keeps working through this step.
- **G2 — worker ticker + effect layers**: ticker fallback; interleaved effect
  rendering in the compositor; `setComposite` carries effect layers.
- **G3 — grid UI**: start with one empty layer; grid rows×cells; empty-cell kind
  picker; add/remove/swap layers & clips; detail pane; retire the seeded rows +
  global draw/marker toolbar (draw/marker become effect clips).
- **G4 — extra kinds**: `image` (static-frame worker path) + `shader` effect, as
  registry entries — proves the seam.
- **G5 — persistence**: `CompositionDoc` save/load to `localStorage` (§8), once
  the grid is stable.

Risk concentrates in **G1/G2** (base re-selection, ticker, per-clip camera
streams, effect interleave) — the parts I cannot runtime-test here. Each phase is
typecheck + unit-test gated; you smoke-test G1→G2 before the UI flips in G3.

---

## 7. Decisions (locked)

1. **Effect scope** = everything **below** the effect clip in the stack
   (Resolume-style). Wired by interleaving `CanvasEffect.render(ctx)` into the
   back-to-front loop at the effect layer's position — no per-layer buffers.
2. **Effect/image-only comps allowed** → the worker runs on a **~30 fps ticker**
   when no frame-producing clip is active (effect-over-feedback, still-image
   comps render). Ticker is the base fallback (§3.2).
3. **Camera duplication** = **clone one stream**: open a device once
   (`getUserMedia`), clone the track per camera clip on that device. Different
   devices open separately. Avoids second-open failures.
4. **Persistence** = **persist the layout** (§8): grid structure — layers, cells,
   kinds, mix, order, blend, transform, effect configs, camera deviceIds — to
   `localStorage`. Live sources are re-acquired on load (re-grant camera, re-pick
   files); a cell whose source can't auto-restore shows a "re-pick" affordance.
5. **Audio**: out of scope — video/screen audio tracks dropped, as today.

## 8. Persistence

A serializable `CompositionDoc` mirrors `layers[]` minus live handles:

```ts
interface ClipDoc { id; kind; label; mirror; transform?; deviceId?; effectConfig?; needsFile?: boolean }
interface CellDoc { id; clip: ClipDoc | null }
interface LayerDoc { id; name; kind; role; opacity; blend; cells: CellDoc[]; activeCellId }
interface CompositionDoc { version: 1; layers: LayerDoc[] }
```

- **Save**: debounced write to `localStorage` on any structural/mix change.
- **Load**: rebuild `layers[]`; auto-re-acquire camera (by `deviceId`) + screen
  (re-prompt) + feedback; **video/image cells flagged `needsFile`** render an
  empty "re-drop file" cell (a `File` can't be reopened from storage). Effects
  restore fully from `effectConfig`.
- A `version` field gates future migrations. Add **G5 — persistence** as the
  final phase (after the grid works in-memory), so save/load rides a stable
  model. Default to one empty layer when nothing is stored.
```
