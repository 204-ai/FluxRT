# Spec: deterministic frame-for-frame batch render (Route B + park)

Status: implemented (engine + server) · client UI pending · Owner: fluxrt

Implemented: `batch_mode` StreamProcessor (synchronous `submit_frame`, no scheduler,
`interpolation_exp=0`), `scripts/batch_render.py` (job manager + PyAV decode/encode),
`scripts/batch_routes.py` (`/batch/jobs` routes), wired in `run_webrtc.py` behind
`FLUXRT_BATCH_RENDER=1` with a best-effort VRAM preflight + `FLUXRT_BATCH_MAX_MB`
upload cap. GPU-free tests in `tests/`. Semantics note: output is 1:1 in COUNT and
reproducible run-to-run, and temporally coherent (frame N builds on the cache of the
prior frames) — not a per-frame-independent transform. Known caveat: the input video
is decoded eagerly into memory (bounded by the upload cap); stream-decoding is a
future improvement.

## Goal

Process a whole input video through the FluxRT pipeline and get **exactly one
diffused output frame per input frame** — deterministic (seed-stable), in order,
no dropped/duplicated/interpolated frames — and download it as an mp4.

This is impossible on the **live WebRTC path**: it is a realtime stream that
(1) drains input to latest (drops frames the pipeline can't keep up with —
`run_webrtc.py:320-354`), (2) free-runs the diffusion decoupled from input, and
(3) samples output at a fixed 30 fps with dup/skip (`FluxRTTrack`, `:364-395`).
So 1:1 needs a **non-realtime, dedicated** path.

## Hard constraint: do NOT reuse the live StreamProcessor (Route A)

The live `sp` is a single free-running pipeline with **one** shared input tensor,
**one** shared output tensor, **one** global `latest_rgb`, **one** global
prompt/seed/steps command queue, and **single-instance temporal caches**
(`UpdateController.cached_frame`, `ModelInferenceSubprocess.previous_frame`, RIFE
state). Any second caller driving it hijacks the live stream **every frame for the
whole job**:

- batch input overwrites the live input tensor (`run_webrtc.py:207-209`, one slot
  `stream_processor.py:25-31`);
- batch output is written to the single `latest_rgb` → **live viewers see batch
  frames** (`run_webrtc.py:389-390`, all peers read it);
- batch prompt/seed/steps mutate global params → live output renders with batch
  params (`run_webrtc.py:130-132`, `stream_processor.py:87-103`);
- temporal caches diff/blend batch-vs-live frames → ghosting at boundaries.

Pausing the live producer (`peer_input_active`) + time-slicing does **not** fix
this (param-swap latency ≥1 diffusion iter, temporal-cache pollution, drain-to-
latest still flashes batch to live). Conclusion: **separate instance only.**

## Architecture: a second, parked StreamProcessor

Add a **batch StreamProcessor** instance, separate from the live one. It has its
own subprocesses + shared memory + global params + temporal caches → **zero shared
state** with the live stream.

```
                    ┌─────────────────────────────┐
 live WebRTC  ─────▶│ live StreamProcessor (sp)    │──▶ FluxRTTrack (30fps, realtime)
                    └─────────────────────────────┘
                    ┌─────────────────────────────┐
 POST /batch  ─────▶│ batch StreamProcessor        │──▶ mp4 (1:1, deterministic)
 (video)            │  · interpolation_exp = 0     │
                    │  · synchronous single-step   │
                    │  · parked when no job         │
                    └─────────────────────────────┘
```

### Three required engine changes (the load-bearing work)

1. **Synchronous single-frame API.** Today the inference subprocess free-runs
   (`model_inference_subprocess.py:642-655`: unconditional `while running: read-
   latest → diffuse → write`) with no input gate, no frame-id, no handshake. Add a
   **batch/synchronous mode** to the subprocess: a request queue of `(seq, rgb)` →
   response queue of `(seq, rgb_out)`, processed strictly in order, one diffusion
   per request, blocking until that frame's output is produced. Expose on
   `StreamProcessor` as e.g. `submit_frame(rgb: np.ndarray) -> np.ndarray`
   (blocking) — used ONLY by the batch instance. The live free-run path is
   untouched (mode flag selected at construction).

2. **Disable interpolation for batch.** Construct the batch instance with
   `interpolation_exp = 0` → `batch_size = 2**0 = 1` (`output_scheduler_subprocess
   .py:26-27`, `model_inference_subprocess.py:360`). One diffused output per input,
   no RIFE tweening. (Batch does not need the output scheduler's fps pacing at all —
   in sync mode the response queue is the output.)

3. **Per-job temporal state.** Keep `cached_frame` / `previous_frame` / KV caches
   (they give frame-to-frame temporal coherence across the video — desirable). They
   are already per-instance, so no live pollution. **Reset them at job START**
   (fresh video) — reuse the existing `requires_reset` / prompt-embed reset path.
   Set prompt/seed/steps **once per job** before the first `submit_frame`.

### Park lifecycle (zero idle cost)

The inference subprocess loop has **no idle gate** — a started-but-jobless instance
still pegs the GPU every iteration (`model_inference_subprocess.py:642-655`). So:

- **Default: spawn-on-submit, teardown-on-complete.** Create + `start()` the batch
  instance when a job arrives; `stop()` (`running.value=False`, joins subprocesses)
  when it finishes/cancels. Idle → instance does not exist → **zero VRAM, zero GPU**.
  Cost: model load (~seconds) per job — acceptable for an offline feature.
- **Optional keep-warm** (frequent jobs): keep the instance resident but add an
  **input-changed/idle gate** to the loop (sleep when no pending request) so an idle
  warm instance frees GPU compute. Still holds VRAM. Make it a config flag
  (`batch_keep_warm: false` default).

## Server API (FastAPI, in or alongside `run_webrtc.py`)

Single job at a time (VRAM). Reject/queue concurrent submits.

| Method | Path | Body / Query | Returns |
|---|---|---|---|
| POST | `/batch/jobs` | multipart video file + `prompt,seed,steps,fps?` | `{job_id}` (202) |
| GET | `/batch/jobs/{id}` | — | `{state, frames_done, frames_total, fps, eta_s}` |
| GET | `/batch/jobs/{id}/result` | — | mp4 stream (200) when `state==done` |
| DELETE | `/batch/jobs/{id}` | — | cancel → `stop()` the instance |

`state ∈ {queued, loading, running, encoding, done, error, canceled}`.

### Server job flow
1. Decode input video → frames (pyav/ffmpeg), note its fps + frame count.
2. Spawn batch StreamProcessor (`interpolation_exp=0`), `start()`, wait `is_ready()`.
3. Reset temporal caches; `set_prompt/seed/steps` once.
4. For each input frame in order: `out = batch_sp.submit_frame(rgb)`; append; bump
   `frames_done`.
5. Encode outputs → mp4 (pyav/ffmpeg, **CFR at input fps**, H.264). Exactly
   `frames_total` output frames.
6. `stop()` the instance; mark `done`; expose result.
On error/cancel at any step: `stop()`, mark state, clean temp files.

### Batch-only deployment (single GPU)
The live model stays resident for the whole server lifetime, so a second full model
never co-fits on a 24 GB card. To render batch on one GPU, run a server in
**batch-only** mode — it skips the live pipeline entirely and gives the per-job
batch model the GPU to itself:
```
FLUXRT_BATCH_ONLY=1 python scripts/run_webrtc.py --config <cfg>   # or --batch-only
```
Batch-only implies batch enabled; `/offer` returns 503 and `/healthz` returns a
minimal `{ready:false, batch_only:true}`. Point a clip's serverBase at it for the
batch panel (its live connection won't come up — expected). For live + batch
together, use two GPUs / two servers.

### VRAM policy
Two full models (live + batch) may not co-fit. On submit, check free VRAM
(`get_reserved_memory` / torch). If a batch instance won't fit alongside the live
one:
- **Option 1 (default):** reject with `409 {detail: "stop the live stream to run a
  batch render"}` — surfaced in the client.
- **Option 2 (config):** auto-pause/stop the live stream for the job duration, then
  resume. Document the live interruption.

## Client UI (realtime-client)

New batch panel in the FluxRT detail (separate from the live record toggle in
PR #29). Job-style, not a live toggle:

```
FluxRT detail ▸ Config ▸ Batch render ▾
  [ ⬇ drop video · or 📂 choose ]
  prompt <current>   seed [52]  steps [2]   fps [source]
  [▶ Render frame-for-frame]
  ▓▓▓▓▓▓░░░░  142 / 360  39%  ~0:48 left        ← poll GET /batch/jobs/{id}
  ✓ fluxrt-render-<ts>.mp4   [download]          ← GET …/result
```

- POST the video + params → `job_id`; poll status (1–2s); show progress/eta.
- On `done`, download the mp4. On `409` (VRAM), show "stop live stream to render".
- One job at a time; disable submit while a job runs.

## Acceptance criteria

- Output frame count **== input frame count** (1:1; verify with ffprobe).
- Output is **CFR at input fps**; no dropped/duplicated/interpolated frames.
- **Deterministic**: same input+prompt+seed+steps → byte-stable (or
  pixel-identical) output across runs.
- **No live impact while idle**: with no batch job, live FPS unchanged (instance
  not resident / parked). Measure live `fps_pipeline` before/after.
- **No live state corruption while a batch runs** (separate instance): live
  prompt/seed/output unaffected (GPU contention/FPS drop is expected + acceptable,
  or avoided via the VRAM policy).
- Cancel + error paths always `stop()` the batch instance (no leaked subprocess /
  shared memory / VRAM).

## Out of scope / alternatives considered

- **Reuse live `sp` (Route A):** rejected — hijacks the live stream (see above).
- **Single-`sp` multi-stream refactor:** per-stream input/output tensor slots +
  per-slot params + per-stream temporal caches. Larger than Route B and touches the
  live hot path — not worth it for an offline feature.
- **Lockstep over free-run (no sync API):** fragile — no frame-id in the output
  tensor, timing-based polling races the free-run loop. The sync API is the clean
  version.

## Effort estimate

- Engine: synchronous single-frame mode in `model_inference_subprocess` +
  `StreamProcessor.submit_frame` + `interpolation_exp=0` construction + per-job
  reset — **M** (the sync mode is the bulk; the rest is wiring/config).
- Server: 4 endpoints + decode/encode + job manager + VRAM gate — **S–M**.
- Client: batch panel + polling — **S**.
