# SPEC — FluxRT webrtc server

## §G GOAL
input: ∀ time ≤1 peer drives pipeline input; owner death/leave → oldest waiter takes over seamless (≤~9s worst case, next-frame when graceful); transient blip ⊥ force full renegotiation (client grace); no-waiter owner ⊥ evicted on gap. [done T1-T8]
egress: WHEP endpoint — standard players (GStreamer whepsrc, browser WHEP libs, studio-world) watch FluxRT output; per-session resolution/fps knobs for bandwidth (proposed). `/offer` + ownership behavior identical.
ingress: WHIP endpoint — standard publishers (OBS WHIP, GStreamer whipsink, browser) feed pipeline input as ordinary ownership claimants; realtime-client protocol & contention semantics untouched.

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

## §I INTERFACES
- fn: `owner_gap_should_release(gap_s: float, other_waiters: int) -> bool`  // pure, input_ownership.py; True iff gap_s ≥ OWNER_GAP_WITH_WAITER & other_waiters ≥ 1
- const: `OWNER_GAP_WITH_WAITER = 8.0`  // input_ownership.py
- method: `InputOwnership.num_other_waiters(pc) -> int`  // waiters excl. pc's own seq, under lock
- behavior: `_pump_owner_frames` gains gap supervision (recv wrapped in wait_for vs last-frame ts; claim time = initial ts) → policy fire → pump returns → existing `finally: release()` handoff path
- client: `engineSession.ts` — ice 'disconnected' → 3s grace timer → `scheduleRetry`; 'connected|completed' cancels; `pc.onconnectionstatechange === 'failed'` → `scheduleRetry`; `ch.onmessage` → `decodeCtrl` → per-clip inputRole via onStatus/store
- api: `/healthz` unchanged (`input_waiters`, `input_source` already exposed — e2e observability)
- api: `POST /whep` (Content-Type `application/sdp`, body=offer) → 201 + `Location: /whep/<uuid>` + answer SDP; 415 wrong ct; 400 empty/bad sdp
- api: `DELETE /whep/<uuid>` → 200 | 404; `PATCH /whep/<uuid>` → 405 (full ICE in answer, no trickle)
- api (proposed): `POST /whep?w=<px>&fps=<n>` — per-session output width (aspect kept, even-rounded, clamp [64, native]) & frame rate (clamp [1, 60], default 30)
- env: reuse `FLUXRT_STUN` / `FLUXRT_TURN_URL(+_USER/_PASS)` via `_rtc_config()`
- api: `POST /whip` (Content-Type `application/sdp`, offer w/ video) → 201 + `Location: /whip/<uuid>` + answer SDP; 415/400 same as WHEP; 503 when `sp is None` (like `/offer`)
- api: `DELETE /whip/<uuid>` → 200 (cancels consume task → ownership released via existing finally) | 404; `PATCH` → 405
- page: `GET /whip-client` → `scripts/whip_test_client.html` (publish test page); `GET /whep-client` → `scripts/whep_test_client.html` (playback test page); both mirror `/test` serving

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
V15 (proposed): `?w=` → resize INTER_AREA, aspect kept, even-rounded, clamp [64, native]; absent → native path identical (zero resize calls); `?fps=` clamp [1,60] default 30
V16: `POST /whip` valid SDP → 201 + Location + `application/sdp`; 415 wrong ct; 400 empty/bad; 503 no pipeline
V17: WHIP publisher = ordinary ownership claimant — same `consume_peer_input(track, pc, ownership, _frame_sink, notify=_input_notify)` path as `/offer` ∴ V1-V5,V9 apply unchanged; WHIP PC gets empty `_fluxrt_channels` → `send_to_pc` no-op; ⊥ datachannel wiring
V18: realtime-client surfaces byte-identical — `/offer` route, ctrl vocabulary, `/healthz` fields untouched; WHIP claim visible to client only as existing `input:peer` broadcast (same as 2nd browser sender)

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
T11|.|(proposed) `FluxRTTrack(fps, width=None)` — optional resize in recv; `/whep` parses `?w=`+`?fps=`|V15,R5,R6
T12|.|(proposed) test: `?w=256` viewer receives 256px frames & concurrent native viewer unchanged; absurd `?w=`/`?fps=` clamped|V15
T13|x|WHIP section in `run_webrtc.py`: `whip_sessions` registry, `POST /whip` (ownership-claimant track wiring, consume-task cancel on close), `DELETE`+404, `PATCH` 405|V16,V17,V18
T14|x|`scripts/whip_test_client.html` + `GET /whip-client` route (mirror `/test`)|I.page
T15|x|smoke test `scripts/test_whip.py`: shim + fake sp + `_frame_sink` collector — publish → ownership active & frames collected; 2nd publisher joins/leaves → 1st still owner; DELETE → ownership released, repeat 404|V16,V17

## §B BUGS
id|date|cause|fix
B1|2026-07-06|gap-yield `return`ed w/ pc alive → nobody drains track; aiortc decodes regardless → queue grows ~12MB/s if paused cam resumes (caught in review, pre-merge)|V9
