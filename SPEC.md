# SPEC — FluxRT webrtc server + batch render

## §G GOAL
input: ∀ time ≤1 peer drives pipeline input; owner death/leave → oldest waiter takes over seamless (≤~9s worst case, next-frame when graceful); transient blip ⊥ force full renegotiation (client grace); no-waiter owner ⊥ evicted on gap. [done T1-T8]
egress: WHEP endpoint — standard players (GStreamer whepsrc, browser WHEP libs, studio-world) watch FluxRT output; per-session resolution/fps knobs for bandwidth. `/offer` + ownership behavior identical.
ingress: WHIP endpoint — standard publishers (OBS WHIP, GStreamer whipsink, browser) feed pipeline input as ordinary ownership claimants; realtime-client protocol & contention semantics untouched.
batch: `--batch-only` render gets live-path speedups — GPU-side in/out, encode overlapped w/ render, small IPC, warm-up + heap freeze @ load, processor kept warm across jobs → 2nd back-to-back job first frame in seconds (not ~35s), steady fps ↑ vs baseline (1280×720→2560×1440 4 steps: ~1.0 fps, 72 fr job 107s). Baseline `perf/hot-path` @ 24ced69.

## §C CONSTRAINTS
- C1: InputOwnership sole mutator of owner/waiter/active state; policy fns pure, unit-testable w/o aiortc/GPU (existing module ethos).
- C2: c855950 guard FROZEN: ⊥ blind gap-evict owner. Gap-evict ONLY w/ takeover candidate present (§R1).
- C3: aiortc connectionState ∈ {new,connecting,connected,failed,closed}; ⊥ 'disconnected' server-side (§R2). Server logic ⊥ depend on 'disconnected' firing.
- C4: wire protocol `ctrlProtocol.ts` frozen — no new msg types (frame-gap needs none).
- C5: client scope = `engineSession.ts` only; `sessionStore.ts` untouched.
- C6: existing backoff `min(800·1.7^n, 6000)` & `waitServerReady` gating stay.
- C7: old test client `scripts/webrtc_test_client.html` ! keep working unmodified.
- C8: WHEP lives in `run_webrtc.py`, own section; no new deps; egress-only — ⊥ on("track")/on("datachannel") wiring on WHEP PCs; ported from sd-webrtc PR#9 pattern.
- C9: `/offer` path diff ! additive-only (`_rtc_config` already extracted — reuse as-is).
- C10: batch work ⊥ changes live output: `process_main`, `step_live`, `condition_from_input` behavior frozen; shared helpers may gain opt-in params (default = current behavior).
- C11: `ProcessorFactory` duck interface (`start/is_ready/set_*/submit_frame/stop`, opt `worker_alive`) kept → fake-processor tests `tests/test_batch_render.py` run unmodified; new methods (`reset`) probed via `getattr`, absent → cold path.
- C12: keep-warm default ON only under `--batch-only` / `FLUXRT_BATCH_ONLY=1`; live + batch VRAM ⊥ co-fit.
- C13: batch lip transfer stays disabled (`_batch_config`); `process_frame_with_pipeline` kept as lip fallback only, ⊥ optimized.
- C14: GPU kernels ⊥ bit-deterministic (`scripts/perf_ab.py` docstring) ∴ ∀ quality gate = LPIPS/PSNR vs noise floor (`perf_ab compare --noise`), ⊥ byte equality.
- C15: out of scope: multi-frame pipelining in child (submit n+1 before n returns), live-path changes, WebRTC egress.
- C16: ∀ batch change measured on RTX 5090, same server cfg, 72-fr 1080p clip (seed 52, steps 4, interp 0, flow upscaler on) before & after.

## §I INTERFACES
- fn: `owner_gap_should_release(gap_s: float, other_waiters: int) -> bool`  // pure, input_ownership.py; True iff gap_s ≥ OWNER_GAP_WITH_WAITER & other_waiters ≥ 1
- const: `OWNER_GAP_WITH_WAITER = 8.0`  // input_ownership.py
- method: `InputOwnership.num_other_waiters(pc) -> int`  // waiters excl. pc's own seq, under lock
- behavior: `_pump_owner_frames` gains gap supervision (recv wrapped in wait_for vs last-frame ts; claim time = initial ts) → policy fire → pump returns → existing `finally: release()` handoff path
- client: `engineSession.ts` — ice 'disconnected' → 3s grace timer → `scheduleRetry`; 'connected|completed' cancels; `pc.onconnectionstatechange === 'failed'` → `scheduleRetry`; `ch.onmessage` → `decodeCtrl` → per-clip inputRole via onStatus/store
- api: `/healthz` unchanged (`input_waiters`, `input_source` already exposed — e2e observability)
- api: `POST /whep` (Content-Type `application/sdp`, body=offer) → 201 + `Location: /whep/<uuid>` + answer SDP; 415 wrong ct; 400 empty/bad sdp
- api: `DELETE /whep/<uuid>` → 200 | 404; `PATCH /whep/<uuid>` → 405 (full ICE in answer, no trickle)
- api: `POST /whep?w=<px>&fps=<n>` — per-session output width (aspect kept, even-rounded, clamp [64, native]) & frame rate (clamp [1, 60], default 30)
- env: reuse `FLUXRT_STUN` / `FLUXRT_TURN_URL(+_USER/_PASS)` via `_rtc_config()`
- api: `POST /whip` (Content-Type `application/sdp`, offer w/ video) → 201 + `Location: /whip/<uuid>` + answer SDP; 415/400 same as WHEP; 503 when `sp is None` (like `/offer`)
- api: `DELETE /whip/<uuid>` → 200 (cancels consume task → ownership released via existing finally) | 404; `PATCH` → 405
- page: `GET /webrtc-test` → `scripts/webrtc-test.html` — unified test page, URL standard w/ sd-webrtc: mode whip+whep (2 PCs) | sendrecv `/offer` (1 PC, realtime-client style); view side|split|blend + mix; prompt bar → `POST /prompt`; capture @ healthz `resolution`; legacy `/test` stays (C7)
- fn: `ModelInferenceSubprocess.condition_from_rgb(frame_rgb) -> Tensor`  // batch twin of `condition_from_input`: RGB uint8 model-res → `input_pinned` → non_blocking upload → `input_lut` → (1,3,H,W) [-1,1]; no flip
- fn: `interpolate_frames(frame, bgr: bool = True)`  // batch passes `bgr=False` → uint8 RGB (N,H,W,3) straight off GPU; live default unchanged (C10)
- ipc: `request_queue` item ∈ `(seq, frame_rgb)` | `("reset", seq)` | `None`; reset → `response_queue.put((seq, []))` ack
- method: `StreamProcessor.reset(timeout)` / `ModelInferenceSubprocess.reset(timeout)`  // parent-side, blocks for ack; same death/timeout polling as `submit_frame`
- config: `batch_keep_warm: bool`  // default = batch-only mode (C12)
- config: `batch_keep_warm_idle_s: float`  // 0 = never idle-evict (default ?)
- config: `batch_encoder: "libx264" | "h264_nvenc"`  // default libx264; nvenc absent in PyAV build → libx264 + log line
- env: `FLUXRT_PROFILE=1` → batch per-N-frame span log: decode, parent crop, ipc out, render (CUDA events), quant+download, ipc back, encode
- error: `batch inference subprocess died during render (exitcode=<n>)` & same suffix on load-death msg
- api: `POST /batch/jobs` 409 iff prior job non-terminal | preflight fail; job `done` ⇒ slot free

## §R RESEARCH
id|topic|finding|src
R1|prior regression|c855950 blind 5s owner gap-evict froze healthy owners (keyframe/TURN settle, paused cam) → fixed by never-gap-evict; new policy conditions evict on waiter presence, not gap alone|FluxRT git history + input_ownership.py docstring
R2|aiortc states|connectionState ⊥ 'disconnected' — explicit `# NOTE: we do not have a 'disconnected' state`; states new/connecting/connected/failed/closed|github.com/aiortc/aiortc rtcpeerconnection.py __updateConnectionState
R3|dead-peer latency|aioice RFC7675 consent: CONSENT_INTERVAL=5, CONSENT_FAILURES=6, interval ×(0.8–1.2) → dead peer → close ≈25–35s|github.com/aiortc/aioice ice.py query_consent
R4|browser side|'disconnected' transient, may self-heal; escalation to 'failed' browser-dependent (~10s); engineSession.ts:136 comment relies on it|MDN RTCPeerConnection.connectionState
R5|per-session encode|aiortc encoder is per-RTCRtpSender (`self.__encoder` rtcrtpsender.py:104, built :308) → ∀ WHEP PC encodes independently ∴ per-session frame size = real bw+cpu lever, no shared-encode constraint|aiortc 1.14 rtcrtpsender.py
R6|encoder follows frame|encoder reconfigures when incoming frame dims change (vpx.py:198-213 codec.width/height ← frame) ∴ track emitting resized frames suffices — no encoder API needed|aiortc 1.14 codecs/vpx.py
R7|no SDP res knob|`imageattr` (RFC 6236) zero hits in aiortc → client ⊥ request resolution via SDP; WHEP draft has no media-param mechanism → custom query param on endpoint URL = legit server-local extension|aiortc 1.14 grep; draft-ietf-wish-whep
R8|bitrate auto-adapts|REMB feedback → `encoder.target_bitrate` (rtcrtpsender.py:284-290) → congestion already degrades bitrate, but resolution fixed → blockiness; resolution param = quality-preserving lever on top|aiortc 1.14 rtcrtpsender.py

## §V INVARIANTS
V1: ∀ time ≤1 owner (existing machine + tests)
V2: owner w/ 0 other waiters ⊥ gap-evicted — holds slot until terminal state / consent expiry backstop
V3: owner silent ≥ OWNER_GAP_WITH_WAITER & ≥1 other waiter → released ≤ threshold+poll(1s)+claim(≤1s) ≈ 10s worst; oldest waiter claims on its next frame
V4: handoff msgs: `input:you` → new owner only; `input:peer` broadcast (existing _input_notify path)
V5: waiter first-frame policy unchanged (WAITER_FIRST_FRAME_DEADLINE 25s, connected-waiter never evicted)
V6: client engine: 'disconnected' recovers ≤3s → same pc kept, ⊥ renegotiation; expires → teardown+retry w/ backoff; `closed=true` halts all retry
V7: client engine: inbound ctrl decoded; unknown msg → ignored, ⊥ throw; inputRole surfaced per clip
V8: `webrtc_test_client.html` connects & functions unmodified
V9: gap-yielded peer (pc alive) ! keep draining its track (aiortc decodes inbound RTP → unbounded queue) & rejoins as newest waiter ∴ can reclaim when slot frees; ⊥ alive undrained track
V10: valid SDP → 201 + `Location: /whep/<uuid>` + `application/sdp` answer; wrong ct → 415; empty/bad sdp → 400
V11: WHEP PC ⊥ on("track") | on("datachannel") → ⊥ touches InputOwnership | ctrl channels; sendrecv offer from viewer ⊥ reaches ingress
V12: ∀ WHEP session own `FluxRTTrack`; `latest_rgb` read only under `latest_lock`
V13: DELETE → close + deregister (404 unknown/repeat); PC state failed|closed → auto-clean (single-shot guard); WHEP PCs ∈ `pcs` → `_graceful_cleanup` covers shutdown
V14: `/offer` request/response + ownership semantics unchanged — WHEP code ⊥ writes state `/offer` reads except `pcs` membership
V15: `?w=` → resize INTER_AREA, aspect kept, even-rounded, clamp [64, native]; absent → native path identical (zero resize calls); `?fps=` clamp [1,60] default 30
V19: output rate == pipeline rate — tracks gate on `output_version`; `fps` = cap only; no new frame ≥1s → 1Hz keepalive repeat; wall-clock pts @ 90kHz (applies to `/offer` + `/whep` output alike)
V20: `output_version` bumped by `output_pump` per scheduler `frame_counter` tick (∴ interpolated frames published, out fps == fps_interpolated); pump active → `push_input_frame` ⊥ publishes latest_rgb (legacy sp w/o counter → input-rate fallback)
V21: shutdown bounded w/ live WHEP/WHIP: uvicorn `timeout_graceful_shutdown=3` (⊥ unbounded connection wait before lifespan) & consume tasks cancelled up front & per-pc close ≤3s & 15s watchdog
V16: `POST /whip` valid SDP → 201 + Location + `application/sdp`; 415 wrong ct; 400 empty/bad; 503 no pipeline
V17: WHIP publisher = ordinary ownership claimant — same `consume_peer_input(track, pc, ownership, _frame_sink, notify=_input_notify)` path as `/offer` ∴ V1-V5,V9 apply unchanged; WHIP PC gets empty `_fluxrt_channels` → `send_to_pc` no-op; ⊥ datachannel wiring
V18: realtime-client surfaces byte-identical — `/offer` route, ctrl vocabulary, `/healthz` fields untouched; WHIP claim visible to client only as existing `input:peer` broadcast (same as 2nd browser sender)
V22: batch out frames == in frames × 2**interp, submit order preserved, CFR fps = (job.fps | src fps | 25) × 2**interp — holds through encoder thread & streamed decode
V23: batch child (lip off) ⊥ PIL, ⊥ `process_frame_with_pipeline`, ⊥ float image GPU→CPU; only uint8 RGB leaves GPU (≈11 MB/fr @ 2560×1440, was ≈44 MB float32)
V24: batch output quality: new path vs baseline LPIPS/PSNR within noise floor (C14); frame count identical
V25: encoder queue bounded (maxsize ≤4) → parent RAM bounded; encoder exception → job `error` w/ msg; cancel → encoder thread drained & joined, partial file deleted (existing rule)
V26: input decode streamed ⊥ full frame list → parent RSS independent of clip length; `frames_total` = stream frame count, fallback = counted after decode (? progress % unknown until then)
V27: parent crops to model res via same `crop_maximal_rectangle(frame, h, w, area_downscale=cfg)` before `request_queue.put`; child crop kept as no-op guard ∴ child always gets model-res frames
V28: `compile_models` & `warmup` → `warm_up()` & `_freeze_heap()` finish before `proc_ready = True`; warm-up leaves caches as fresh boot ∴ first real frame time ≈ steady frame time (no compile on frame 1)
V29: warm reuse ⊥ observable carry-over: job on reused proc (after `reset`) vs same job on fresh proc within noise floor; `reset` clears update_controller cache, `pipe.spatial_cache`, `pipe._cond_latent_cache`, `previous_frame`, prompt-travel state; acked before first frame of new job
V30: reuse iff new job `interp` == proc build `interp`; else teardown + rebuild; reset/submit error → teardown (⊥ reuse suspect proc); shutdown | idle > `batch_keep_warm_idle_s` (>0) → teardown
V31: job `done` observable ⇒ `_active` cleared (teardown, if any, already finished) ∴ submit after polling `done` ⊥ 409; submit while non-terminal still 409 (existing tests)
V32: `FLUXRT_PROFILE` unset → batch path adds ⊥ CUDA events, ⊥ syncs, ⊥ per-frame log

## §T TASKS
id|status|task|cites
T1|x|`input_ownership.py`: add OWNER_GAP_WITH_WAITER + pure `owner_gap_should_release`|V2,V3
T2|x|`InputOwnership.num_other_waiters(pc)` under lock|V3,C1
T3|x|`_pump_owner_frames`: gap supervision → policy fire → return (release via existing finally)|V3,C2
T4|x|server tests: gap+waiter evicts & handoff; gap no-waiter holds; frame-delivering owner + waiter never evicted|V2,V3,V5
T5|x|`run_webrtc.py` comment fix re aiortc states (no behavior change)|R2
T6|x|`engineSession.ts`: 3s disconnected-grace + connectionstatechange 'failed' → retry|V6
T7|x|`engineSession.ts`: `ch.onmessage` → decodeCtrl → per-clip role|V7,C4
T8|x|client vitest: grace cancel/expiry/closed-halt; ctrl dispatch (fake RTCPeerConnection + fake timers)|V6,V7
T9|x|WHEP section in `run_webrtc.py`: `whep_sessions` registry, `POST /whep` (own PC + own `FluxRTTrack`, 201+Location+SDP), `DELETE`+404, `PATCH` 405, state cleanup, pcs join|V10,V11,V12,V13,V14
T10|x|smoke test `scripts/test_whep.py`: torch-free fluxrt shim (conftest pattern) + in-process uvicorn + aiortc client — 201/Location/answer, 2 concurrent viewers frames, 415/400, DELETE 200→404, auto-clean|V10,V12,V13
T11|x|`FluxRTTrack(fps, width=None)` — optional resize in recv; `/whep` parses `?w=`+`?fps=`|V15,R5,R6
T12|x|test: `?w=256` viewer receives 256px frames & concurrent native viewer unchanged; absurd `?w=`/`?fps=` clamped|V15
T13|x|WHIP section in `run_webrtc.py`: `whip_sessions` registry, `POST /whip` (ownership-claimant track wiring, consume-task cancel on close), `DELETE`+404, `PATCH` 405|V16,V17,V18
T14|x|`scripts/whip_test_client.html` + `GET /whip-client` route (mirror `/test`)|I.page
T15|x|smoke test `scripts/test_whip.py`: shim + fake sp + `_frame_sink` collector — publish → ownership active & frames collected; 2nd publisher joins/leaves → 1st still owner; DELETE → ownership released, repeat 404|V16,V17
T16|x|unify test pages → `GET /webrtc-test` (whip+whep | sendrecv toggle); drop `/whip-client` + `/whep-client` routes & files|I.page
T17|x|version-gated output pacing: `output_version` bump in `push_input_frame`, gated `FluxRTTrack.recv` w/ fps cap + 1Hz keepalive|V19
T18|~|batch profiling behind `FLUXRT_PROFILE`: spans decode / parent crop / ipc out / render (CUDA events) / quant+dl / ipc back / encode, log every N fr; record baseline table (72 & 255 fr 1080p→1440p, 576×320 steps 2/4/6/8)|V32,C16
T19|~|P1 child GPU I/O: `condition_from_rgb` + `process_frame_to_gpu` + `interpolate_frames(bgr=False)` in `process_main_batch`; drop `convert_np_to_torch` & CPU `[:, :, ::-1]` flip; lip-active → old path|V23,V22,C10,C13,I.fn
T20|.|P1 A/B: 72-fr clip old vs new + noise-floor run; LPIPS/PSNR, fps, peak VRAM|V24,C14,C16
T21|x|P2 `BatchJobManager._run`: encoder thread + bounded queue; streamed decode producer thread; tests w/ fake proc: order, count, encoder raise → error, cancel mid-job → threads joined & no file|V22,V25,V26,C11
T22|x|P2 opt `batch_encoder` = `h264_nvenc` probe + libx264 fallback|I.config
T23|x|P3 parent-side crop before `request_queue.put` (in `_run` | `StreamProcessor.submit_frame`); test: child receives model-res frame|V27
T24|~|P4 `process_main_batch`: `warm_up()` before `proc_ready` when `compile_models` & `warmup`; measure load vs first-frame shift|V28
T25|x|P5 `("reset", seq)` request + ack in child; `reset()` on subprocess & StreamProcessor; `BatchJobManager` keeps proc per `batch_keep_warm`, reuse rules, idle timer, shutdown teardown; tests w/ fake proc: 2 jobs same interp → 1 start, different interp → rebuild, error → teardown|V29,V30,C11,C12
T26|.|P5 GPU check: 2 back-to-back jobs — 2nd first frame seconds; warm vs fresh output within noise floor|V29,V24
T27|x|P6 set `done` after teardown & `_active` clear (finally ordering); test: submit immediately after poll sees `done` → 201|V31
T28|~|P7 exitcode in death errors; repro restart w/ `--set 'resolution={"width":1280,"height":720}'` → first job, capture child stderr; root cause → `/spec bug:`|I.error,V28
T29|.|report: per change load / first frame / steady fps (fr 2..N) / job total 72 & 255 fr / out count / LPIPS+PSNR vs baseline / noise floor / peak VRAM|C16,V24

## §B BUGS
id|date|cause|fix
B1|2026-07-06|gap-yield `return`ed w/ pc alive → nobody drains track; aiortc decodes regardless → queue grows ~12MB/s if paused cam resumes (caught in review, pre-merge)|V9
B2|2026-07-08|webrtc sampled output tensor once per INPUT frame → interpolated frames dropped, out fps pinned ≈ input fps (~30) ≠ fps_interpolated ∴ scheduler `frame_counter` + `output_pump` publishes per scheduled frame|V20
B3|2026-07-08|Ctrl+C hang w/ live WHEP/WHIP ? uvicorn default graceful shutdown waits unbounded on open connections (keep-alive /healthz polls) before lifespan runs; in-process repro exits 0.1s ∴ bounded `timeout_graceful_shutdown=3` + upfront consume-task cancel — verify on GPU box|V21
