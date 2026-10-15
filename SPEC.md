# SPEC — FluxRT webrtc server + batch render

## §G GOAL
input: ∀ time ≤1 peer drives pipeline input; owner death/leave → oldest waiter takes over seamless (≤~9s worst case, next-frame when graceful); transient blip ⊥ force full renegotiation (client grace); no-waiter owner ⊥ evicted on gap. [done T1-T8]
egress: WHEP endpoint — standard players (GStreamer whepsrc, browser WHEP libs, studio-world) watch FluxRT output; per-session resolution/fps knobs for bandwidth. `/offer` + ownership behavior identical.
ingress: WHIP endpoint — standard publishers (OBS WHIP, GStreamer whipsink, browser) feed pipeline input as ordinary ownership claimants; realtime-client protocol & contention semantics untouched.
batch: `--batch-only` render gets live-path speedups — GPU-side in/out, encode overlapped w/ render, small IPC, warm-up + heap freeze @ load, processor kept warm across jobs → 2nd back-to-back job first frame in seconds (not ~35s), steady fps ↑ vs baseline (1280×720→2560×1440 4 steps: ~1.0 fps, 72 fr job 107s). Baseline `perf/hot-path` @ 24ced69.
reference: ref image steerable on every path — per-batch-job ref, mid-render swap (live + batch), smooth ref travel; `--batch-only` /reference ⊥ dead end; ref feature off at boot ⇒ zero cost, on ⇒ cost within measured budget (C21).

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
- C17: `reference_image_resolution` boot-only (sets ref token count → compiled shapes) ∴ ∀ swapped/travelled ref resized to it; ⊥ runtime ref-res change.
- C18: no ref op issued → live & batch output behavior identical to pre-change (C10 spirit).
- C19: ref travel quality unproven (latent lerp may ghost) → T31/T32 experiments gate T35; ⊥ ship endpoint before go.
- C20: ctrl wire protocol frozen (C4) ∴ ref ops HTTP-only; `/healthz` fields untouched (V18) — reuse existing `reference_version`.
- C21: perf budget — `perf_ab` on RTX 4090 vs current main, same preset: ref off at boot → fps & peak VRAM ≡ baseline within run noise, output within LPIPS noise; ref set & static → ≡ today's ref cfg within noise; hard swap → ≤1 slow frame (1 full execute), ⊥ ~1s compile stall after warm-up; ref travel → mean fps over morph ≥75% steady & ≥ prompt travel same preset; batch job ref → job fps/VRAM ≡ same job w/ boot ref within noise, keep-warm reuse kept; ref feature in use → peak VRAM ≤ today's ref cfg +64 MiB.
- C22: zero-cost scope = boot `use_reference_image:false` (per-frame, boot time, VRAM unchanged); enabled+idle may add boot time & VRAM (≤ +64 MiB), ⊥ per-frame fps.
- C23: out of scope: runtime ref-res change (C17), >1 ref slot, travel across ref sizes, ctrl-channel ref ops (C20), MCP.
- C24: ref travel reuses prompt-travel machinery — `travel.progress` clock, `prompt_travel_full_execute_every`, `prompt_travel_refresh`; src/tgt VAE-encoded once at start, per step tensor blend only.
- C25: budget miss fallback = `reference_travel_refresh_every` (def 1; 2 → ref tokens refreshed every 2nd frame, blend held); ⊥ auto ref-res drop, ⊥ skip final full execute, ⊥ VRAM > +64 MiB.
- C26: swap storm: ≤1 ref change applied per rendered frame, newest wins (live queue drain + batch in-order); ⊥ server rate limit on /reference; `ref:set` broadcast + preview PNG coalesced ≤4/s, ending on latest version; docs: smooth change = /reference-travel.
- C27: ? ref-token cost on 4090 (0.058 ms/row = laptop-5090 figure) → measure in T32/T36 · ? latent-lerp visual quality (C19) · ? boot-time increase when ref enabled+idle (no cap set) → measure, then decide · ? today's ref-cfg VRAM baseline on 4090 → measure before T33. · resolved 2026-10-12 (R23/R24): ref-cfg VRAM baseline 20302 MB peak (static) / 20260 (idle); enabled-idle boot +15.3 s; static ref steady cost +2.1 ms/frame

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
- api: `/offer` ctrl `out-fps:<n>` — output frame-rate limit of this peer's own track, live; `0` = no limit (every published frame); whole number 0..240 → `ack:out-fps:<n>` | `err:out-fps`; never sent → 30; ⊥ broadcast
- api: `/healthz` `output_frames` — published output frames since boot (interpolated incl.)
- env: reuse `FLUXRT_STUN` / `FLUXRT_TURN_URL(+_USER/_PASS)` via `_rtc_config()`; `FLUXRT_STUN=none|off|""` → ⊥ STUN (`ice_config.stun_urls`); `start_show.ps1` sets `none` + `HF_HUB_OFFLINE=1` (show machine ⊥ internet)
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
- api: `POST /batch/jobs` + opt multipart `reference` (PNG/JPEG/WebP ≤10 MB) → applied in job reset before frame 1; absent → boot ref (current)
- api: `POST /reference` w/ active batch job → hard-swap that job's ref on next frame; live + batch both steered (mirror `/prompt`, run_webrtc.py:249-259); `--batch-only` w/o job → 503 (was 400); ref disabled at boot → 400 (V40)
- api: `POST /reference-travel?frames=<n|Ns>&mode=lerp|slerp` body=image → morph current ref → target over n rendered frames; live + running batch job; direct `/reference` mid-morph cancels
- ipc: child cmd `ref_travel(image, frames, mode)` (live: `command_queue`; batch: in-order `request_queue`, V41) → encode src & tgt latents once; per step `_cond_latent_cache[1]` = (current ref obj, lerp(src,tgt,t)) + `update_controller.reference_image_is_valid=False` (V42); full execute per prompt-travel stride/refresh (C24); t=1 → tgt set as ref
- config: `reference_travel_refresh_every: int`  // def 1; ref tokens refreshed every n-th morph frame, blend held between (C25)
- method: `StreamProcessor.start_reference_travel(...)`, `BatchJobManager.set_reference(rgb)` / `start_reference_travel(...)`
- route: `/comfy/edit` → template restored | 501 w/ msg (⊥ 500)

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
R9|ref load|ref = boot cfg `use_reference_image` + `reference_image_path` (cv2.imread, cwd-relative) → resize `reference_image_resolution` → PIL; unreadable → ref tokens off until POST /reference|model_inference_subprocess.py:594-610
R10|ref in model|ref → VAE latents cached `pipe._cond_latent_cache[idx]` by source → extra image tokens after input (`reference_list=[input, ref]`); `_set_reference_active` clears cache|model_inference_subprocess.py:790-804,957-963 · pipeline.py:654-660,927
R11|live swap|`POST /reference` → worker resize+set+`update_controller.reset_cache()` → hard cut, full recompute next frame, ⊥ blend; first activation ≈1s compile variant|run_webrtc.py:1246-1301 · model_inference_subprocess.py:757-769 · docs/perf-notes.md:146
R12|batch ref|batch proc cfg = deepcopy boot cfg → boot ref loaded per proc; keep-warm reset keeps `reference_image`, clears cond latent cache; `POST /batch/jobs` ⊥ ref field; manager steering = set_prompt + start_prompt_travel only|batch_render.py:415-440,476-489 · model_inference_subprocess.py:1195-1205
R13|batch-only ref API|`_reference_enabled()` needs live `sp` → `--batch-only` POST /reference → 400 "disabled"; ∄ way to change ref of running/parked batch proc except file swap + new proc (`batch_keep_warm=false`)|run_webrtc.py:1065-1066,1281-1290
R14|travel scope|prompt travel lerps/slerps `prompt_embeds` only; ref untouched|model_inference_subprocess.py `_advance_prompt_travel`
R15|comfy→ref|`/comfy/pull` installs newest ComfyUI output as ref (ok); `/comfy/edit` → 500, `scripts/comfy_workflows/qwen_edit_2509.api.json` ∉ checkout|run_webrtc.py:61-62,1316-1488
R16|sample path|`config_with_reference.json` → `local_samples/reference.png` ∉ repo → boot warn, ref off (rnatwoofour boot 2026-10-12)|configs/config_with_reference.json:19 · boot log
R17|ref prompt wording|static input, A-vs-B mean abs Δ 0-255: "everything in frame in the visual style of the reference image" 52.7 · "the person and the shelves rendered exactly like image 2, keep the layout of image 1" 36.0 · "restyle image 1 in the art style of image 2, same composition as image 1" 21.8 · "everything in frame repainted with the colors, materials and textures of image 2" 12.0; same ref twice → 0.0 ∴ wording sets ref strength ~4×|T31 2026-10-12 · live 576×320 steps 2 int8_linear tiny-vae 256² ref · RTX 4090 rnatwoofour · t31b.py
R18|ref set/swap cost|first set after boot ⊥ compile stall (max frame gap 104–131 ms; warm-up covers ref variant); hard swap → max gap 105–143 ms vs steady 85–87 ms = 1 slow frame, clean 1-frame cut|T31 run1+run2 · t31.py
R19|pixel cross-fade|repeated POST /reference: 4 steps/s → out fps 14→10 (−28%), proc 75→80 ms; 8–10 steps/s → out ≈8 fps (−40–45%), proc 75→118–125 ms (full recompute + VAE encode per step); frame Δ first-after-post ≈ fade mean (40.9/41.4 · 31.7/30.5 · 41.7/39.2) ∴ ⊥ visible per-step spikes over motion; looks gradual → misses C21 travel budget ∴ stopgap only|T31 run2 · steps.py
R20|ref clear stall|first `DELETE /reference` after boot → 1.6 s output gap once; later DELETE in same process ≤136 ms ∴ ref-removed variant ⊥ covered by warm-up (? exact variant)|T31 run1 vs run2, same server process
R21|VRAM probe|`/healthz` vram_mb swings ±400 MB under moving input (13.5 GB idle → 14.2–15.0 GB) w/ & w/o ref; ⊥ ref-attributable delta visible ∴ C21 +64 MiB check needs torch peak counters|T31 run1 health.csv
R22|latent ref travel|in-process, static input, A→B 9 pts: neighbor Δ evenness (max/mean) lerp 1.28 · slerp 1.42 · pixel blend 2.06 (back-loaded, last step 21.8 = 3× first); path/endpoints ≈1.85 all three ∴ ⊥ detour/ghost states; inject `_cond_latent_cache[1]` + `reference_image_is_valid=False` + full execute renders exact state, same state twice → 0.0 ∴ T35 design (V35/V42) works; lerp = default mode|T32 2026-10-12 · RTX 4090 · 576×320 steps 2 int8_linear tiny-vae 256² ref, compile off · t32.py
R23|ref baseline (main f284910)|perf_ab 300 fr moving clip, live preset (576×320 steps 2 int8_linear tiny-vae rife_cudagraphs interp 0, live warm-up): ref off 70.9 ms (14.11 fps) · enabled-idle 71.0 ms · static 256² ref 73.0 ms (+2.1 ms, +3%); boot (load+warm-up) off 31.5 s · idle 46.8 s (+15.3 s) · static 43.4 s; peak reserved (incl warm-up) off 20172 · idle 20260 (+88) · static 20302 MB (+130); steady serving ≈15.4 GB|T41 2026-10-12 · RTX 4090 · /tmp/t41/{ref_off,ref_idle,ref_static}/run.json
R24|ref event cost (main)|in-process, compiled+warmed, moving clip: steady w/ ref median 76.3 ms; hard swap frame 118.5–121 ms (1st 148.9) = +42–47 ms, next frame back ≈71 ms ∴ exactly 1 slow frame; prompt travel 48 fr mean 98–99 ms = 77–78% steady fps (∴ C21 ref-travel bar ≈78%); 1st ref clear after boot 1485 ms + 659 ms next frame (~2.1 s stall, R20 confirmed in-process), 2nd clear 105 ms; ref re-set 123 ms|T41 events.json · t41_events.py

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
V33: `/offer` output limit = viewer's choice — `out-fps:0` → `send_wait` 0 ∴ ∀ published frame sent (⊥ replaced by newer one while waiting); `out-fps:n` → sends ≥ 1/n s apart; peer that never sends it → 30 (old clients unchanged, V18); limit ⊥ pads a slower pipeline (V19 holds)
V34: per-job ref ⊥ leaks: next job w/o ref renders w/ boot ref (or none); reused proc (reset) vs fresh proc within noise floor (extends V29)
V35: ∀ ref change ⇒ next frame uses current ref state — set/job: fresh encode; travel: `_cond_latent_cache[1]` = (current ref obj, lerp(src,tgt,t)) — AND `reference_image_is_valid=False` ∴ ref tokens recomputed; ⊥ stale latent | stale ref K/V
V36: ref ops ⊥ change batch frame count/order (V22 holds); mid-job swap lands on next submitted frame
V37: ref travel t monotonic 0→1 over requested frames (Ns → same clock rule as prompt travel); new `/reference` | `/reference-travel` mid-morph → cancel, restart from current blended latent
V38: no ref op → output identical to pre-change (C18)
V39: `--batch-only` POST /reference w/o active job → 503 (like `/prompt`), ⊥ 500; w/ job → that job only
V40: boot `use_reference_image` false → ∀ ref op (`/reference`, `/reference-travel`, job `reference`) → 400 "reference disabled" at request ∴ ⊥ reaches worker (`reference_image_resolution` absent → KeyError, model_inference_subprocess.py:759)
V41: batch ref ops (job ref, mid-job swap, travel start) ride `request_queue` in order (`("ref", seq, …)`), ⊥ `command_queue` ∴ applied before exactly the next frame submitted after the call returns; worker logs applied frame index
V42: ref travel step ⇒ `reference_image_is_valid=False` every `reference_travel_refresh_every`-th frame + full execute on frame 1, every `prompt_travel_full_execute_every`-th frame & final (same stride/refresh as prompt travel) ∴ static regions reach target at t=1
V43: travel start → one `ref:set:v` broadcast, `GET /reference` = target from start; ⊥ per-step version bumps
V44: job `reference` > `MAX_REFERENCE_BYTES` → 413, undecodable → 400, both before job is queued
V45: ≤1 ref change applied per rendered frame, newest wins; `ref:set` broadcast + preview PNG ≤4/s, last emitted = latest version (C26)
V46: ref travel encodes src & tgt exactly once (VAE) per morph; ⊥ VAE work on morph steps (C24)
V47: first `DELETE /reference` after boot ⊥ compile stall — output gap ≤1 slow frame (warm-up covers ref set → clear → set)

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
T30|x|`output_rate.py` pure `parse_out_fps` + `send_wait`; `FluxRTTrack.recv` uses it; `/offer` ctrl `out-fps:`; `/healthz` `output_frames`; tests `test_output_rate.py`|V33,V19
T31|x|exp A (no code): live pixel cross-fade A→B, 8/16 steps via repeated POST /reference @100–250 ms — done 2026-10-12: gradual look, ⊥ per-step spikes; out fps −28% @4 steps/s, −40–45% @8–10 → misses C21 travel budget ∴ stopgap; ref wording ×4 (R17); clear stall once (R20)|R11,C19,R17,R18,R19,R20
T32|x|exp B: offline latent-lerp — encode 2 refs, fixed input, t∈{0,.25,.5,.75,1}; contact sheet vs exp A; neighbor LPIPS → go/no-go T35 → GO 2026-10-12: lerp most even (1.28 vs pixel 2.06), ⊥ ghosting; LPIPS ⊥ on box → pixel Δ used|R10,C19,R22
T33|.|batch per-job `reference` field: route → manager → child `("ref", seq, rgb or None)` before frames; reset restores boot ref; 413/400 before queue; disabled → 400; fake-proc tests|I.api,V34,V35,V40,V41,V44,C11
T34|.|`/reference` → active batch job under `--batch-only` (manager.set_reference, in-order IPC); live+batch both steered; 503 w/o job; test|V39,V40,V41,R13
T35|.|ref travel engine (live; T32 = GO, default mode lerp): child `ref_travel` — encode src/tgt once, per-step lerp into `_cond_latent_cache[1]`, `reference_image_is_valid=False`, prompt-travel stride/refresh, `reference_travel_refresh_every`, cancel rules; GPU-free unit tests on fake latents|V35,V37,V42,V46,C17,C19,C24,C25,R22
T36|.|ref travel A/B on 4090: mean fps over morph vs steady & vs prompt travel same preset; peak VRAM; quality sheet vs noise floor|C21,C14,T32
T37|x|`/comfy/edit`: restore template or 501 + msg; test|R15
T38|x|docs: `--set reference_image_path=<abs>` in perf-notes boot cmds + README; ship tiny sample or drop default path; strong ref wording "in the visual style of the reference image" (R17)|R16,R17
T39|.|`POST /reference-travel` route + `StreamProcessor.start_reference_travel` (live); 400 disabled; one `ref:set` bump at start, preview = target|V37,V40,V43,C20
T40|.|batch ref travel: `BatchJobManager.start_reference_travel` → in-order `request_queue`; fake-proc tests|V36,V41,V37
T41|~|perf budget harness (live cases done 2026-10-12 → R23/R24; batch-job-ref case waits for T33): `perf_ab` runs on 4090 for C21 cases (ref off, ref static, hard swap, travel, batch job ref) vs current main; record today's ref-cfg VRAM baseline & enabled-idle boot time; peak VRAM via torch `max_memory_reserved` in-process, ⊥ `/healthz` (R21)|C21,C22,C27,R21,R23,R24
T42|.|swap coalescing: ≤1 ref apply/frame (batch request_queue drains to newest ref), `ref:set` broadcast + preview PNG coalesce ≤4/s; test burst of 20 posts → 1 apply/frame, ≤4 broadcasts/s, last = latest|V45,C26
T43|x|warm-up covers ref set → clear → set (incl. text-active variants) ∴ first DELETE /reference after boot ⊥ stall; verify on 4090: first clear gap ≤1 slow frame|V47,R20

## §B BUGS
id|date|cause|fix
B1|2026-07-06|gap-yield `return`ed w/ pc alive → nobody drains track; aiortc decodes regardless → queue grows ~12MB/s if paused cam resumes (caught in review, pre-merge)|V9
B2|2026-07-08|webrtc sampled output tensor once per INPUT frame → interpolated frames dropped, out fps pinned ≈ input fps (~30) ≠ fps_interpolated ∴ scheduler `frame_counter` + `output_pump` publishes per scheduled frame|V20
B3|2026-07-08|Ctrl+C hang w/ live WHEP/WHIP ? uvicorn default graceful shutdown waits unbounded on open connections (keep-alive /healthz polls) before lifespan runs; in-process repro exits 0.1s ∴ bounded `timeout_graceful_shutdown=3` + upfront consume-task cancel — verify on GPU box|V21
B4|2026-10-12|`/comfy/edit` template `qwen_edit_2509.api.json` missing → every call 500|T37
B5|2026-10-12|first `DELETE /reference` after boot stalled output 1.6 s — ref-removed variant ⊥ warmed (T31)|V47
