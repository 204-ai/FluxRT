# Resolume-style Layers/Cells/Clips — Refactor Design

Refactor the WebRTC client's input "layers" from a fixed 3-layer stack
(`camera` / `video` / `feedback`) into a Resolume Arena–style model: a dynamic
vertical stack of **layers**, each a horizontal track of **cells**, each cell
holding a **clip**, exactly one clip active per layer. Add/remove/reorder layers
live; blend + opacity per layer; a bottom clip bar drives a detail pane; the
selected clip plays on the main preview. Minimal first version, future-proof.

---

## 1. Current state (analysis)

The pipeline is solid and already has the hard parts (worker-side WebCodecs
compositing, hot-swap without restart, OBS framing). The only real problem is
that **layer identity is hardcoded to source type**. One closed tuple drives
everything:

```ts
// pipeline/core/types.ts:43
export const LAYER_IDS = ['camera', 'video', 'feedback'] as const
export type LayerId = (typeof LAYER_IDS)[number]
export type CompositeOptions = Record<LayerId, LayerOptions>   // fixed 3 keys
```

### Coupling points (what blocks dynamic layers)

| Where | Coupling | Severity |
|---|---|---|
| `core/types.ts:43` | `LAYER_IDS` const-tuple **is** the layer type; `CompositeOptions`/`CompositePatch` are `Record<LayerId,…>`; `mergeComposite` loops the fixed tuple | blocker |
| `core/compositor.ts:98` | `drawComposite(camera, video, feedback, tsMs)` — **3 positional named args**, 3 hardcoded `this.composite.{feedback,video,camera}` draws | blocker |
| `workers/pipeline.worker.ts` | three named valves (base/overlay/feedback) + `baseIsCamera` branch + hand-derived `leakThreshold` | blocker |
| `backends/*` | `capturedStream` / `feedbackTrack` / `hiddenVideo`/`fileVideo`/`feedbackVideo` singletons | major |
| `state/pipelineStore.ts:35` | `layers: Record<LayerId,LayerOptions>` + flat `camEnabled` / `videoLoaded`/`videoName`/`videoPlaying`/… / `feedbackAvailable` all bound to the 3 fixed roles | major |
| `components/input/LayerStack.tsx` | hand-written `CameraRow` / `VideoRow` / `FeedbackRow` | major |
| `pipeline/videoFileSource.ts` + `runtime.ts:22` | **one** module-singleton `<video>` element — only one video clip can present at a time | major (the real constraint behind "many video clips") |

### Extension seams that already help

- **Effect registry** (`effects/registry.ts`, `name → factory`) — the ready
  template for a pluggable **ClipKind registry**.
- **`drawLayer(src, opts, mirror)`** (`compositor.ts:75`) is already
  clip-agnostic (takes any `CanvasImageSource | VideoFrame`) — it does **not**
  change.
- **Geometry** (`coverRect` / `layerDrawRects` / `layerDestRect` /
  `identityTransform`, `types.ts:74-157`) is pure, identity-free — unchanged.
- **Hot-swap machinery** (`swapVideo` / `swapCamera` / `clearVideo` /
  `setFeedback` + worker lazy-create-on-first-swap) already adds/removes a
  source **without** recreating the output track. Generalizing it is the whole
  game.
- **Restart-replay pattern** (`rail.ts:155-184` remembers `feedbackStream` +
  `drawHistory` and replays them after a forced restart) — the template for
  replaying *every* layer's binding.

### Invariants any refactor MUST preserve

1. **WebRTC stays alive on clip toggle** — activate/deactivate routes through
   add/remove/swap, **never** `rail.stop()/start()`. The MSTG / `captureStream`
   output track is never recreated.
2. **Worker / OffscreenCanvas-safe** compositor (no DOM in the worker path).
3. **Opaque `#000` base + back-to-front blend + cover-fit** — output is never
   semi-transparent (transparency encodes as black upstream, breaks
   screen/multiply).
4. **Single vision tap** samples the full **composite, post-mirror**, at cadence.
5. **Output dims** match the server aspect (`targetAspect`) — fixed at start.
6. **`previewEl` identity** — one `captureStream`-backed canvas; re-parent it,
   never spawn a second pipeline.

---

## 2. Target data model

Two registries mirroring `effects/registry.ts`, plus a dynamic ordered list.
**Mix (opacity/blend) lives on the layer; transform + mirror live on the clip**
(Resolume-correct, and it deletes the camera-only mirror special-case).

```ts
// pipeline/core/clipKinds.ts — makes camera/video/feedback "just 3 entries"
export type ClipKind = string                         // open; 'camera'|'video'|'feedback' today
export interface ClipKindMeta {
  kind: ClipKind
  label: string
  mirrorable: boolean        // camera=true (selfie flip belongs to the kind)
  canBeBase: boolean         // camera/video=true, feedback=false (no cadence/dims/tap)
  sourceForm: 'mediastream' | 'element' | 'mediastream-clone'
  accept?: string            // DropZone accept for media cells ('video/*')
  // future kinds add: needsVision?, generative? (render in worker), …
}
export const CLIP_KINDS: Record<ClipKind, ClipKindMeta> = {
  camera:   { kind:'camera',   label:'Camera',   mirrorable:true,  canBeBase:true,  sourceForm:'mediastream' },
  video:    { kind:'video',    label:'Video',    mirrorable:false, canBeBase:true,  sourceForm:'element', accept:'video/*' },
  feedback: { kind:'feedback', label:'Feedback', mirrorable:false, canBeBase:false, sourceForm:'mediastream-clone' },
}

// pipeline/core/types.ts — dynamic ordered model replacing LAYER_IDS/CompositeOptions
export type LayerId = string                          // opaque instance id (was a closed union)
export type ClipId  = string

export interface Clip {
  id: ClipId
  kind: ClipKind
  label: string                                       // file name / device label / "output loop"
  transform?: LayerTransform                          // per-clip OBS framing (reuse type as-is)
  mirror?: boolean                                    // default = clipMeta(kind).mirrorable
  config: Record<string, unknown>                     // deviceId | {loop,rate} | …
}
export interface Cell  { id: string; clip: Clip | null }   // clip:null = empty "+ add clip" cell
export interface Layer {
  id: LayerId
  name: string
  cells: Cell[]                                       // starts as [{clip:null}] per the requirement
  activeCellId: string | null                         // exactly one active clip per layer (null = muted)
  opacity: number                                     // was LayerOptions.opacity
  blend: BlendMode                                    // was LayerOptions.blend
}
export type Composite = Layer[]                        // ORDERED top(front) → bottom(back)

// Record can't express order/add/remove — use explicit ops keyed by opaque id:
export type CompositeOp =
  | { op:'patch';     layers: ({ id: LayerId } & Partial<Pick<Layer,'opacity'|'blend'>>)[] }
  | { op:'add';       layer: Layer; index?: number }
  | { op:'remove';    id: LayerId }
  | { op:'reorder';   order: LayerId[] }
  | { op:'setSource'; id: LayerId; clipId: ClipId | null }   // activate/deactivate a cell

export function defaultComposite(): Composite { /* seed 3 legacy layers during migration */ }
```

`LayerOptions {opacity, blend, transform}` is absorbed into `Layer` + `Clip`.
`coverRect`/`layerDrawRects`/`layerDestRect`/`identityTransform` unchanged.

---

## 3. Compositor / worker / rail plan

**`drawLayer` and all geometry stay byte-for-byte.** Only the orchestration
generalizes from "3 named things" to "an ordered list".

```ts
// compositor.ts — ordered loop replaces the 3 positional args
interface LayerDraw { src: Layer | null; opts: LayerOptions; mirror: boolean }
drawComposite(layers: LayerDraw[], tsMs: number): void {
  ctx.fillStyle = '#000'; ctx.fillRect(0,0,W,H)            // opaque base preserved
  for (let i = layers.length - 1; i >= 0; i--) {           // back → front
    const L = layers[i]
    if (L.src) this.drawLayer(L.src, L.opts, L.mirror)     // UNCHANGED drawLayer
  }
  for (const e of this.effects) e.render(ctx, info, this.bus)   // effects last, as today
}
```

This mirrors the existing effect-chain loop (`for (const e of this.effects)`) —
the proven pattern. `mirror` becomes per-`LayerDraw` (from the active clip),
removing the camera-only special case while keeping per-layer selfie flip.

- **`types.ts` `RailBackend`** — replace the five bespoke methods
  (`setFeedback`/`swapVideo`/`swapCamera`/`clearVideo`) with a generic set keyed
  by layer id: `setLayerSource(id, kind, source|null)`, `addLayer`,
  `removeLayer`, `reorderLayers`, `applyComposite(op)`. `SourceSet` becomes
  `Array<{layerId, kind, source}>` + an explicit `baseLayerId`.
- **Worker** — three named valves → `Map<layerId, {valve, retained}>` +
  `baseLayerId`. The `swap-video`/`swap-camera`/`clear-video`/`set-feedback`
  messages collapse into `layer-source` / `layer-add` / `layer-remove` /
  `layer-reorder`. `leakThreshold` recomputed on every add/remove (generalize
  `recomputeLeak`). The lazy-create-overlay-on-first-swap seam generalizes
  directly to lazy valve creation. Cleanup iterates `valves.values()`.
- **Backends** — `capturedStream`/`feedbackTrack`/`<video>` singletons become
  `Map<layerId, …>`. The streams backend already clones the feedback track
  (one MSTP per track) — reuse verbatim per layer.

### The one genuinely-new piece: cadence base decoupling

Today the camera **implicitly** drives the worker wake loop, sets canvas dims,
and is the vision tap (`worker.ts:114-116`, `baseIsCamera`). Decouple:

1. **Output dims fix to `targetAspect`/server resolution at start** (already
   computed, `rail.ts:131-135`) — so activating/deactivating a clip **never**
   changes dims and never forces a restart.
2. **Designate an explicit `baseLayerId`** = the topmost layer whose active clip
   has `canBeBase` (camera preferred, else video). If the base clip is
   deactivated, reselect another `canBeBase` layer; **if none exists, fall back
   to an internal rAF/ticker** so the loop never stalls. ← new code, small, but
   the highest-risk new logic. Without it, deactivating the loop-driving clip
   freezes the output.

Vision tap still samples the full composite post-mirror — **unchanged**.

---

## 4. Source lifecycle plan (keeping WebRTC alive)

The store holds only **descriptors**; live media handles live in runtime
singletons keyed by clip/layer id. Each clip kind resolves to a backend source
by `clipMeta(kind).sourceForm`:

- **camera** → a `getUserMedia` MediaStream (rail owns acquire/stop, as today).
  Device change stays the in-place `swapCameraDevice` (a full restart "froze the
  output").
- **video** → its **own** `HTMLVideoElement` via a per-clip `VideoFileSource`.
  The module singleton (`runtime.ts:22`) becomes `Map<clipId, VideoFileSource>`
  from the start (**locked: multi-video**) because **one element presents one
  clip** and ≥2 video clips may be live at once. Transport (play/seek/loop/rate)
  + listeners bind per instance — fixes the stale-`timeupdate`-handler risk
  (`pipelineStore.ts:130-147`).
- **feedback** (**locked: a normal clip kind, not a pinned layer**) → a clone of
  the remote output stream. The clip can sit in **any cell at any layer
  position**, and its z-order follows that layer (feedback-under or feedback-over
  other layers, the user's choice). `sessionStore.attachFeedback` generalizes
  from a fixed bottom slot to: find the layer whose active clip is
  `kind:'feedback'` and route the cloned remote track to that layer's valve;
  re-route on activation/move. The clip shows a **"waiting"** state until
  `feedbackAvailable` (the output→input loop only exists once the remote stream
  connects), but it is otherwise add/remove/reorderable like any clip. Guard
  preserved: a stale/ended remote track must never abort `start` — it just drops
  the feedback binding.

**Activation (`setSource` op)** routes through the generalized hot path,
**never** `rail.start()`. It reuses the proven `setFeedback`/`swapVideo`
lazy-add machinery: `rail.setLayerSource(id, kind, source)` → backend → worker
`layer-source` → lazy-create-or-swap that layer's valve. **Deactivation** →
`setLayerSource(…, null)` → worker drops the valve, closes pending + retained,
recomputes leak. The MSTG / `captureStream` output track stays alive — clip
toggles touch only input valves.

**Restart survival** (only on backend-kind or dims change): generalize the
`feedbackStream`+`drawHistory` replay to remember **every** layer's active
binding and re-establish each via `setLayerSource` after start, then push the
new output track through `onOutputTrack → session.replaceTrack` (no
renegotiation).

---

## 5. UI plan + layout

`LayerStack` becomes data-driven: `layers.map(LayerRow)` replaces the three
hand-written rows. Each row = name + drag/▲▼ reorder + a horizontally-scrolling
**cell track** + the **reused `LayerMix`** (frame btn ◳ + blend cycle + opacity
fader, re-parametrized off a `Layer` not a `LayerId` literal). Per-LAYER mix on
the right (Resolume-correct); per-CLIP controls (device/mirror/transport/framing)
in the **detail pane**. Clicking a cell **activates** (live) + **selects**
(highlight → detail pane + main preview). Reuse every existing CSS hook
(`.layer-stack`, `.layer-row`, `.layer-mix`, `.icon-btn.on` green, `.layer-fader`,
`.blend-btn`, `.frame-btn`, `.device-pick`, `DropZone`); net-new is one CSS grid
for the cell track. The main thumbnail **re-parents the one `previewEl`**, never
a second pipeline.

```
┌──────────────────────────── LAYERS ───────────────── [+ layer] ┐
│ ▲▼ Camera  [cam0●][cam1 ][ + ]      ◳ nrm ▓▓▓▓░ 80% [⌫]        │  top = frontmost
│ ▲▼ Video   [clipA ][clipB●][ + ]    ◳ scr ▓▓▓▓▓100% [⌫]        │  ● = active cell
│ ▲▼ Feedbk  [loop ●][ + ]            ◳ nrm ▓▓░░░ 40% [⌫]        │
└────────────────────────────────────────────────────────────────┘
        cell track scrolls →           per-layer mix (LayerMix)
┌──────── main preview (re-hosted previewEl, live composite) ─────┐
│                      [ ▶  ⟲  ⛶  ◳ ]                            │
└────────────────────────────────────────────────────────────────┘
┌──── CLIPS (bottom bar) ────┐ ┌──────── DETAIL: clipB (video) ──┐
│ [cam0][cam1][clipA]        │ │ src: beach.mp4  1280×720  0:42  │
│ [clipB◀sel][loop][ + ]     │ │ ▶ ───●──────── 0:12/0:42 🔁 1× │
└────────────────────────────┘ │ mirror ☐   framing [◳ on prev] │
   click clip → detail+preview └─────────────────────────────────┘
```

The clip bar is the selected layer's cells in one line for fast triggering; on
narrow widths it collapses into the row itself (left pane min ~240px).

---

## 6. Migration phases

Pipeline-first to de-risk the load-bearing hot path **before** any user-facing
change; each phase is independently reviewable, shippable, and invariant-safe.

| Phase | Goal | Key changes | Files | Ships |
|---|---|---|---|---|
| **P0** | Dynamic core, **identical behavior** | `clipKinds.ts` registry; `Layer[]`/`Cell`/`Clip`/`CompositeOp` types; `drawComposite`→ordered loop (+legacy 3-arg wrapper); worker valves → `Map`; backends → `Map`; rail generic methods (legacy 5 kept as shims). **Seed the same 3 default layers.** | `core/clipKinds.ts`, `core/types.ts`, `core/compositor.ts`, `workers/pipeline.worker.ts`, `backends/*`, `rail.ts` | Structural blocker gone, zero behavior change. Regression-test vs current 3-layer output + `leakThreshold`. |
| **P1** | Store → `layers[]`, behavior unchanged; **multi-video pool** | `pipelineStore`: `Record`+flat `video*`/`feedbackAvailable` → `layers: Layer[]` + `selectedClipId` + `framingLayerId`; `loadVideoFile`/`unloadVideo` re-keyed per layer (keep the 3-case hot-swap state machine); **`VideoFileSource` singleton → `Map<clipId,…>` now** (locked: multi-video) with per-source transport/listeners; `attachFeedback` generalized to route to the active-feedback-clip layer (locked: feedback-as-clip); `hasAnySource` selector for `sessionStore` gating. `LayerStack` still renders the seed rows, reading from `layers[]`. | `state/pipelineStore.ts`, `state/sessionStore.ts`, `state/runtime.ts`, `pipeline/videoFileSource.ts`, `components/input/TransformOverlay.tsx` | State dynamic-ready; multi-video + feedback-as-clip plumbing proven; unblocks UI. |
| **P2** | **Add/remove/reorder layers (headline)** | `LayerStack` maps `layers[]` → generic `LayerRow`; `[+ layer]` (seeds one empty cell), `[⌫]`, drag/▲▼ → `rail.addLayer`/`removeLayer`/`reorderLayers`; reused `LayerMix`. Still one cell per layer. | `components/input/LayerStack.tsx`, `app.css`, `pipelineStore.ts`, `rail.ts` | User builds an N-layer stack live, no restart — the headline feature. |
| **P3** | **Cells / clips / bar / detail / thumbnail** | Multi-cell rows; activate-one-per-layer via grid + clip bar; `selectedClipId` → detail pane + main thumbnail; per-clip controls (device/mirror/transport/framing) move into detail. (Video pool + feedback-as-clip already landed in P1.) | `components/input/LayerStack.tsx`, `InputTab.tsx`, `CanvasHost.tsx`, `pipelineStore.ts`, `app.css` | Full Resolume clip-matrix interaction. |
| **P4** | Extensible clip kinds | ClipKind registry siblings the effect registry; `image` (`<img>`/`ImageBitmap`), `shader`/`generator` (in-worker `render(ctx,info,bus)` reading the `AnalyzerBus`), `ndi`/screen-share (`mediastream`). Each = one registry entry + a detail-pane editor. No core/compositor change. | `core/clipKinds.ts`, `pipeline/clips/*`, `components/input/clipDetail/*` | New kinds with zero rework. |
| **P5** *(future)* | Columns / decks sequencing | A "column trigger" = `activeCellId` set to column `c` on every layer at once = N `setSource` ops. **UI-only** over existing `cells[]`. | `components/input/*`, `pipelineStore.ts` | Resolume deck launching. |

**Minimal shippable Resolume = P0→P3.** P0→P2 already deliver the dynamic layer
stack (the headline); P3 adds the requested cells/clip-bar/detail/thumbnail.
P4/P5 ride on a model that's already N-layer + grid-shaped, so they need no
rework.

---

## 7. Invariants preserved & how

- **WebRTC alive** — all clip ops route through add/remove/swap; output dims
  fixed to `targetAspect` so toggles never restart; restart only on
  backend/dims change, then `replaceTrack`.
- **Cadence never stalls** — explicit `baseLayerId` reselect + rAF-ticker
  fallback when no `canBeBase` clip is active.
- **No VideoFrame leaks** — `leakThreshold` recomputed on every add/remove;
  each valve closes pending+retained on supersede/remove/shutdown.
- **One `<video>` per clip** — `Map<clipId, VideoFileSource>` with isolated
  listener lifecycles (no stranded `timeupdate`/`play` handlers).
- **Worker-safe / opaque-base / back-to-front / cover-fit** — `drawLayer` +
  geometry unchanged; only the iteration generalizes.
- **Vision tap** — still samples full composite post-mirror at cadence;
  reordering/per-layer mirror must not change tap frame space (marker landmarks
  stay 1:1).
- **`previewEl` identity** — re-parent the one canvas into the thumbnail.

---

## 8. Future-proofing

New clip kinds are **one registry entry** (+ a `produce-a-frame` source for
generative kinds), exactly like adding an effect. `drawLayer` already takes any
`CanvasImageSource`, so `image`/`ndi` need zero render-path change;
`shader`/`generator` reuse the `CanvasEffect render(ctx,info,bus)` contract and
read vision via the `AnalyzerBus` with no wiring. Because `Layer` carries
`cells[]` + `activeCellId` from day one, **columns/decks are a pure UI/store
feature** (trigger column = N `setSource` ops) — no pipeline change ever. The
`CompositeOp` protocol (add/remove/reorder/patch/setSource by opaque id) already
expresses everything sequencing needs.

---

## 9. Decisions (locked)

1. **Multi-video from the start** — build the `Map<clipId, VideoFileSource>`
   element pool + per-source transport/listeners in **P1** (not deferred). Two
   video clips on different layers can be live at once.
2. **Feedback is a normal clip kind**, not a pinned layer. The remote output
   stream is routed to whichever layer holds the active `kind:'feedback'` clip,
   and its z-order follows that layer's position. `attachFeedback` re-routes on
   activation/move; the cell shows "waiting" until `feedbackAvailable`.
3. **Columns / deck sequencing → P5 (later).** Build `cells[]` + `activeCellId`
   now so it slots in free; no column-launch UI in the first pass.
4. **Draw + marker effects stay global** over the whole composite (no change) —
   not per-layer clips. Revisit only if needed.
5. **Transform / framing is per-clip** (carried on `Clip.transform`), replacing
   the current per-layer `layoutLayer`.

### Remaining unknowns (decide during P3, not blocking)

- Cell capacity per layer / horizontal scroll vs fixed columns in the compact
  pane (min ~240px).
- Whether a removed layer's loaded video clips are evicted from the pool
  immediately or cached for fast re-activation.
