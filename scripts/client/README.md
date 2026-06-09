# FluxRT Client (React + TS)

Rewrite of the browser client (`scripts/webrtc_static/`) as a Vite + React 19 + TypeScript app with a composable media pipeline and on-device human sensing.

## Build & serve

```sh
cd scripts/client
yarn            # install
yarn vendor     # copy MediaPipe wasm + download .task models into public/ (once)
yarn build      # tsc + vite build -> dist/
```

`run_webrtc.py` automatically serves `dist/` at `/` when it exists (assets at `/assets`, MediaPipe runtime at `/mediapipe`). The legacy vanilla client stays available at `/legacy` (and at `/` when no build exists). No Node needed on the GPU box if `dist/` is committed/copied.

## Dev

```sh
yarn dev                          # http://localhost:5173, proxies API to :8765
FLUXRT_SERVER=http://gpu-box:8765 yarn dev   # point proxy elsewhere
yarn test                         # vitest (ctrl protocol codec, prompt features)
```

Camera works on localhost without TLS. WebRTC media is peer-to-peer (only signaling/control HTTP is proxied), so the dev machine must reach the server's LAN ICE candidates.

## Architecture

```
src/
  pipeline/            framework-agnostic media rail (no React)
    core/              types, AnalyzerBus, Compositor (shared per-frame logic)
    backends/          detect (capability probe)
                       streamsBackend  WebCodecs MSTP->worker->MSTG (Chrome/Edge)
                       canvasBackend   hidden <video> + canvas captureStream fallback
    workers/           pipeline.worker (compositing), vision.worker (MediaPipe)
    effects/           marker (pose-tracked dot+trail), drawLayer (brush/eraser)
    rail.ts            camera source -> effect chain -> output stream; strokes, taps
    visionClient.ts    typed wrapper around vision.worker
  vision/              ported from sense-human: engine, analyze, draw, types
  lib/                 ctrlProtocol (typed DataChannel codec), api (REST), features
  state/               zustand stores: session (WebRTC+ctrl), pipeline, prompt,
                       reference, sense; runtime.ts holds the Rail/vision singletons
  components/          input/ (camera, draw, marker), output/ (stage, prompt,
                       saved prompts, reference, comfy, lip), sense/ (panel, overlay)
```

Key invariants:

- **Effects run wherever compositing runs** (worker or main) — OffscreenCanvas-safe ops only, configured by name via postMessage.
- **Analyzers are taps, never in-chain**: the vision worker samples source frames (~15 Hz) and can never stall the 30 fps video rail. Pose results feed the marker effect through the rail bus; face+pose feed the Sense panel.
- **VideoFrame discipline** (streams backend): depth-1 valve, close-after-draw, open-frame counter logs leaks (leaks silently freeze the camera).
- **Preview element identity is preserved** across tab re-parenting — it backs `captureStream()` and comfy snapshots (`CanvasHost`).
- **Sense overlay is display-only**; nothing is baked into the stream sent to FLUX (unlike the hand marker, which is intentionally baked).

## Sense feature

MediaPipe FaceLandmarker (478 landmarks, 52 blendshapes) + PoseLandmarker in a worker → expression, attention, head pose, blink, posture, lean, hands raised, movement energy. Source selector: camera input or the AI output video. Heuristics in `src/vision/analyze.ts`.
