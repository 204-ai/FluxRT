"""
FluxRT WebRTC streaming server.

Streams the StreamProcessor output to any modern browser over WebRTC.
Includes an inline minimal client (HTML+JS) served at `/`.

Usage:
    python scripts/run_webrtc.py
    python scripts/run_webrtc.py --int8
    python scripts/run_webrtc.py --config configs/config_with_reference.json --port 8765

Then open `http://<linux-lan-ip>:8765/` on any LAN client.

Control:
    - Prompt / seed / steps updates: sent over a DataChannel.
    - Reference image upload: HTTP POST /reference with raw image bytes.
      Requires the active config to have `use_reference_image: true`
      (e.g. `--config configs/config_with_reference.json`).
"""

import argparse
import asyncio
import fractions
import io
import logging
import threading
import time
from typing import Optional

import av
import cv2
import httpx
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fluxrt.webrtc")


# ──────────────────────────────────────────────────────────────────────────────
# Globals — populated in main() so module import stays cheap.
# ──────────────────────────────────────────────────────────────────────────────
sp: Optional[StreamProcessor] = None
input_tensor = None
output_tensor = None
resolution = None

latest_rgb: Optional[np.ndarray] = None
latest_lock = threading.Lock()
producer_stop = threading.Event()

# Cached reference image as PNG bytes for GET /reference preview.
latest_reference_png: Optional[bytes] = None
reference_lock = threading.Lock()

pcs: set[RTCPeerConnection] = set()

# Open control DataChannels for cross-client broadcast (e.g. reference image sync).
ctrl_channels: set = set()
ref_version: int = 0

# Pipeline input ownership — first peer with sendrecv video wins; others ignored.
# When a peer owns the input, the local camera producer thread pauses its writes.
input_owner_pc: Optional[RTCPeerConnection] = None
input_owner_lock = threading.Lock()
peer_input_active = threading.Event()

MAX_REFERENCE_BYTES = 10 * 1024 * 1024  # 10 MB cap for uploaded reference images.

# Configured ComfyUI servers (populated in main() from --comfy-server flags).
# Maps display name → base URL (no trailing slash).
comfy_servers: dict[str, str] = {}

# Lip transfer (LivePortrait postprocessor) runtime state. The pipeline
# always loads the model when `lip_transfer.enable` is true in config, but
# the per-frame postprocess is gated by `set_lip_transfer(bool)`.
lip_active: bool = False


def broadcast_ctrl(msg: str) -> None:
    """Send a string message to every open control DataChannel."""
    for ch in list(ctrl_channels):
        try:
            ch.send(msg)
        except Exception as exc:
            log.debug("broadcast send failed, dropping channel: %s", exc)
            ctrl_channels.discard(ch)


# ──────────────────────────────────────────────────────────────────────────────
# Shared frame push — runs the BGR input through the pipeline and updates
# `latest_rgb` for any active video track to send out. Used by both the local
# camera producer thread and per-peer track consumers.
# ──────────────────────────────────────────────────────────────────────────────
def push_input_frame(frame_bgr: np.ndarray) -> None:
    global latest_rgb
    h, w = resolution["height"], resolution["width"]
    cropped = crop_maximal_rectangle(frame_bgr, h, w)
    input_tensor.copy_from(cropped)
    out_bgr = output_tensor.to_numpy()
    rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    with latest_lock:
        latest_rgb = rgb


# ──────────────────────────────────────────────────────────────────────────────
# Producer — local webcam frames. Pauses while a peer owns the input.
# ──────────────────────────────────────────────────────────────────────────────
def producer_loop(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Camera %d failed to open", camera_index)
        return

    log.info("Waiting for StreamProcessor to be ready...")
    while not sp.is_ready():
        if producer_stop.is_set():
            cap.release()
            return
        time.sleep(0.1)
    log.info("StreamProcessor ready, producing frames from local camera.")

    while not producer_stop.is_set():
        if peer_input_active.is_set():
            # A peer is driving input — skip local camera frames.
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            log.warning("Camera read failed, retrying...")
            time.sleep(0.05)
            continue

        push_input_frame(frame)

    cap.release()
    log.info("Producer stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Per-peer input consumer — pulls VideoFrames from a remote track and feeds
# them into the pipeline. First peer to deliver a track wins ownership;
# subsequent peers are ignored. Ownership is released on peer disconnect.
# ──────────────────────────────────────────────────────────────────────────────
async def consume_peer_input(track, pc: RTCPeerConnection) -> None:
    global input_owner_pc

    with input_owner_lock:
        if input_owner_pc is not None:
            log.info("Peer input already taken — ignoring extra track")
            return
        input_owner_pc = pc
        peer_input_active.set()
    log.info("Peer %x now drives input", id(pc))
    broadcast_ctrl("input:peer")

    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            except Exception as exc:
                log.warning("Peer track recv error: %s", exc)
                break

            bgr = frame.to_ndarray(format="bgr24")
            # Pipeline write/read is blocking — offload from event loop.
            await loop.run_in_executor(None, push_input_frame, bgr)
    finally:
        with input_owner_lock:
            if input_owner_pc is pc:
                input_owner_pc = None
                peer_input_active.clear()
                log.info("Peer %x released input", id(pc))
                broadcast_ctrl("input:server")


# ──────────────────────────────────────────────────────────────────────────────
# WebRTC video track — emits latest_rgb at a fixed frame rate.
# ──────────────────────────────────────────────────────────────────────────────
class FluxRTTrack(VideoStreamTrack):
    """Yields the latest FluxRT output frame as `av.VideoFrame`s at `fps` Hz."""

    kind = "video"

    def __init__(self, fps: int = 30):
        super().__init__()
        self.fps = fps
        self._t0 = time.time()
        self._n = 0
        h, w = resolution["height"], resolution["width"]
        self._blank = np.zeros((h, w, 3), dtype=np.uint8)

    async def recv(self) -> av.VideoFrame:
        # Pace ourselves; aiortc will pull frames as fast as recv() returns.
        self._n += 1
        target_t = self._t0 + self._n / self.fps
        delay = target_t - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            # Behind schedule — skip ahead so we don't accumulate debt.
            self._t0 = time.time() - self._n / self.fps

        with latest_lock:
            rgb = latest_rgb if latest_rgb is not None else self._blank

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self._n
        frame.time_base = fractions.Fraction(1, self.fps)
        return frame


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI signaling app.
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()


@app.post("/offer")
async def offer(request: Request):
    body = await request.json()
    offer_sdp = RTCSessionDescription(sdp=body["sdp"], type=body["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def _on_state():
        log.info("PC state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def _on_track(track):
        log.info("Inbound track from peer: kind=%s id=%s", track.kind, track.id)
        if track.kind == "video":
            asyncio.ensure_future(consume_peer_input(track, pc))

    @pc.on("datachannel")
    def _on_dc(channel):
        log.info("DataChannel opened: %s", channel.label)
        ctrl_channels.add(channel)

        # Send the current reference version to a fresh peer so it can
        # decide whether to refresh its preview.
        if latest_reference_png is not None:
            try:
                channel.send(f"ref:set:{ref_version}")
            except Exception:
                pass

        # Sync lip transfer state on connect.
        if _lip_enabled():
            try:
                channel.send("lip:on" if lip_active else "lip:off")
            except Exception:
                pass

        @channel.on("close")
        def _on_close():
            ctrl_channels.discard(channel)
            log.info("DataChannel closed: %s", channel.label)

        @channel.on("message")
        def _on_msg(msg):
            if not isinstance(msg, str):
                return
            if msg.startswith("prompt:"):
                new_prompt = msg[len("prompt:"):].strip()
                if new_prompt:
                    log.info("Prompt update: %r", new_prompt)
                    sp.set_prompt(new_prompt)
                    channel.send("ack:prompt")
            elif msg.startswith("seed:"):
                try:
                    seed = int(msg[len("seed:"):])
                    sp.set_seed(seed)
                    channel.send(f"ack:seed:{seed}")
                except ValueError:
                    channel.send("err:seed")
            elif msg.startswith("steps:"):
                try:
                    steps = int(msg[len("steps:"):])
                    sp.set_steps(steps)
                    channel.send(f"ack:steps:{steps}")
                except ValueError:
                    channel.send("err:steps")

    pc.addTrack(FluxRTTrack(fps=30))

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


@app.on_event("shutdown")
async def _shutdown():
    producer_stop.set()
    await asyncio.gather(*(pc.close() for pc in list(pcs)))
    pcs.clear()
    if sp is not None:
        sp.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Reference image upload — requires `use_reference_image: true` in config.
# Accepts raw image bytes (PNG / JPEG / WebP) as the request body.
# Browser posts with `Content-Type: application/octet-stream` (or any).
# ──────────────────────────────────────────────────────────────────────────────
def _reference_enabled() -> bool:
    return bool(sp and sp.config.get("use_reference_image", False))


def _lip_enabled() -> bool:
    """Whether lip transfer is configured + the LivePortrait model loaded."""
    if not sp:
        return False
    return bool(sp.config.get("lip_transfer", {}).get("enable", False))


@app.post("/lip-transfer")
async def post_lip_transfer(request: Request):
    """Toggle LivePortrait lip transfer on/off.
    Query: ?on=true|false (also accepts 1/0, yes/no)."""
    if not _lip_enabled():
        raise HTTPException(
            status_code=400,
            detail=(
                "Lip transfer not enabled in config. Restart with a config "
                "containing lip_transfer.enable = true and the LivePortrait "
                "directory present."
            ),
        )
    raw = request.query_params.get("on", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        on = True
    elif raw in ("0", "false", "no", "off"):
        on = False
    else:
        raise HTTPException(status_code=400, detail="?on= must be true/false")

    global lip_active
    lip_active = on
    sp.set_lip_transfer(on)
    log.info("Lip transfer: %s", "on" if on else "off")
    broadcast_ctrl("lip:on" if on else "lip:off")
    return JSONResponse({"ok": True, "lip_active": on})


def _apply_reference_bytes(body: bytes, source_label: str) -> dict:
    """Decode image bytes, set as reference, cache preview, bump version,
    broadcast. Returns the JSON-shaped result dict. Raises HTTPException
    on decode failure. Used by both /reference upload and /comfy/pull."""
    try:
        img = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decode error: {exc}")

    rgb = np.array(img)
    sp.set_reference_image(rgb)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    global latest_reference_png, ref_version
    with reference_lock:
        latest_reference_png = buf.getvalue()
        ref_version += 1
        version = ref_version

    log.info(
        "Reference image set from %s: %dx%d (%d bytes, v%d)",
        source_label, *rgb.shape[1::-1], len(body), version,
    )
    broadcast_ctrl(f"ref:set:{version}")

    return {
        "ok": True,
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "bytes": len(body),
        "version": version,
        "source": source_label,
    }


@app.post("/reference")
async def post_reference(request: Request):
    if not _reference_enabled():
        raise HTTPException(
            status_code=400,
            detail=(
                "Reference image conditioning is disabled. "
                "Restart with --config configs/config_with_reference.json"
            ),
        )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(body) > MAX_REFERENCE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (>{MAX_REFERENCE_BYTES // (1024 * 1024)} MB)",
        )

    return JSONResponse(_apply_reference_bytes(body, "upload"))


# ──────────────────────────────────────────────────────────────────────────────
# ComfyUI integration — pull latest output image from a configured server
# and use it as the reference image. See the SKILL.md in
# `~/Downloads/qwen-edit-test/` for the canonical comfy endpoints.
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/comfy/servers")
async def list_comfy_servers():
    return {
        "servers": [{"name": n, "url": u} for n, u in comfy_servers.items()],
    }


@app.post("/comfy/pull")
async def pull_comfy(request: Request):
    if not _reference_enabled():
        raise HTTPException(status_code=400, detail="Reference image disabled")

    name = request.query_params.get("server", "")
    if not name or name not in comfy_servers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown comfy server: {name!r}. "
            f"Known: {list(comfy_servers.keys())}",
        )
    base = comfy_servers[name].rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            hr = await client.get(f"{base}/history", params={"max_items": 20})
            hr.raise_for_status()
            history = hr.json()

            # `history` is dict insertion-ordered; newest entries last.
            chosen = None
            for prompt_id, entry in reversed(list(history.items())):
                outputs = entry.get("outputs", {}) or {}
                for node_id, out in outputs.items():
                    for image in out.get("images", []) or []:
                        if image.get("type") != "output":
                            continue
                        chosen = (prompt_id, node_id, image)
                        break
                    if chosen:
                        break
                if chosen:
                    break

            if not chosen:
                raise HTTPException(
                    status_code=404,
                    detail="No output images in recent comfy history",
                )

            prompt_id, node_id, image = chosen
            params = {
                "filename": image["filename"],
                "type": image["type"],
                "subfolder": image.get("subfolder", ""),
            }
            vr = await client.get(f"{base}/view", params=params)
            vr.raise_for_status()
            body = vr.content

    except httpx.HTTPError as exc:
        log.warning("Comfy pull HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Comfy server error: {exc}")

    label = f"comfy:{name}:{image['filename']}"
    result = _apply_reference_bytes(body, label)
    result["prompt_id"] = prompt_id
    result["node_id"] = node_id
    result["filename"] = image["filename"]
    return JSONResponse(result)


@app.delete("/reference")
async def delete_reference():
    if not _reference_enabled():
        raise HTTPException(status_code=400, detail="Reference image disabled")
    sp.set_reference_image(None)
    global latest_reference_png, ref_version
    with reference_lock:
        latest_reference_png = None
        ref_version += 1
        version = ref_version
    log.info("Reference image cleared (v%d)", version)
    broadcast_ctrl(f"ref:clear:{version}")
    return JSONResponse({"ok": True, "cleared": True, "version": version})


@app.get("/reference")
async def get_reference():
    with reference_lock:
        png = latest_reference_png
    if png is None:
        raise HTTPException(status_code=404, detail="No reference image set")
    return Response(content=png, media_type="image/png")


@app.get("/")
async def _index():
    return HTMLResponse(CLIENT_HTML)


@app.get("/healthz")
async def _health():
    return {
        "ready": bool(sp and sp.is_ready()),
        "peers": len(pcs),
        "resolution": resolution,
        "reference_enabled": _reference_enabled(),
        "reference_set": latest_reference_png is not None,
        "reference_version": ref_version,
        "input_source": "peer" if peer_input_active.is_set() else "server",
        "lip_enabled": _lip_enabled(),
        "lip_active": lip_active,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Minimal browser client — vanilla JS, no build step.
# ──────────────────────────────────────────────────────────────────────────────
CLIENT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FluxRT — WebRTC Viewer</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #eee; font: 14px/1.4 system-ui, sans-serif; }
  header { padding: 10px 14px; background: #1a1a1a; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 14px; margin: 0; font-weight: 600; }
  #status { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #333; }
  #status.live { background: #1f7a3a; }
  #status.err { background: #7a1f1f; }
  .stage { display: flex; justify-content: center; padding: 12px; background: #0a0a0a; gap: 8px; }
  .stage video { flex: 1 1 0; min-width: 0; max-width: 100%; background: #000; display: block; }
  .stage.split video { max-width: 50%; }
  .stage video#inv { display: none; }
  .stage.split video#inv { display: block; }
  .controls { padding: 10px 14px; display: flex; gap: 8px; flex-wrap: wrap; background: #1a1a1a; }
  input[type=text] { flex: 1 1 280px; padding: 8px 10px; background: #222; color: #eee; border: 1px solid #333; border-radius: 4px; }
  input[type=number] { width: 70px; padding: 8px 10px; background: #222; color: #eee; border: 1px solid #333; border-radius: 4px; }
  button { padding: 8px 14px; background: #2a6cd4; color: #fff; border: 0; border-radius: 4px; cursor: pointer; }
  button:hover { background: #3a7ce4; }
  button:disabled { background: #444; cursor: not-allowed; }
  .log { padding: 8px 14px; font-family: ui-monospace, monospace; font-size: 11px; color: #888; height: 6em; overflow-y: auto; background: #0a0a0a; }
  label { font-size: 12px; color: #aaa; display: flex; align-items: center; gap: 6px; }
  .ref {
    padding: 10px 14px; background: #1a1a1a; display: flex; align-items: center; gap: 10px;
    border-top: 1px solid #2a2a2a;
  }
  .ref.disabled { opacity: 0.5; pointer-events: none; }
  .ref .drop {
    flex: 1 1 auto; min-height: 64px; padding: 10px 14px;
    border: 2px dashed #444; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    color: #888; font-size: 12px; text-align: center; cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .ref .drop.over { border-color: #2a6cd4; background: #142035; color: #ccc; }
  .ref .preview { display: none; max-height: 80px; max-width: 120px; border-radius: 4px; border: 1px solid #333; }
  .ref .preview.shown { display: block; }
  .ref .meta { font-size: 11px; color: #888; min-width: 110px; }
</style>
</head>
<body>
  <header>
    <h1>FluxRT WebRTC</h1>
    <span id="status">idle</span>
  </header>

  <div class="stage" id="stage">
    <video id="inv" autoplay playsinline muted></video>
    <video id="v" autoplay playsinline muted></video>
  </div>

  <div class="controls">
    <button id="start">Start</button>
    <button id="stop" disabled>Stop</button>
    <input id="prompt" type="text" placeholder="Prompt — press Enter to apply">
    <label>seed <input id="seed" type="number" value="52"></label>
    <label>steps <input id="steps" type="number" value="2" min="1" max="8"></label>
  </div>

  <div class="controls" style="border-top:1px solid #2a2a2a;">
    <label><input id="useCam" type="checkbox"> Use my camera as input</label>
    <label><input id="showInput" type="checkbox" disabled> Show input preview (left)</label>
    <label><input id="flipInput" type="checkbox" disabled> Mirror input</label>
    <select id="camSelect" style="padding:6px 8px;background:#222;color:#eee;border:1px solid #333;border-radius:4px;flex:1 1 220px;" disabled>
      <option value="">— pick a camera —</option>
    </select>
    <span id="inputStatus" style="font-size:12px;color:#888;align-self:center;">input: server</span>
  </div>

  <div class="controls" style="border-top:1px solid #2a2a2a;">
    <label><input id="lipXfer" type="checkbox" disabled> Lip transfer</label>
    <span id="lipStatus" style="font-size:12px;color:#888;align-self:center;">lipsync: unavailable</span>
  </div>

  <div class="ref" id="comfyRow">
    <label style="font-size:12px;color:#aaa;">Comfy server:</label>
    <select id="comfySelect" style="padding:6px 8px;background:#222;color:#eee;border:1px solid #333;border-radius:4px;min-width:120px;">
      <option value="">(loading...)</option>
    </select>
    <button id="comfyPull">Pull latest output → reference</button>
    <span id="comfyStatus" style="font-size:11px;color:#888;"></span>
  </div>

  <div class="ref" id="refRow">
    <div class="drop" id="drop">Drop image here or click to choose a reference</div>
    <input type="file" id="file" accept="image/*" hidden>
    <img class="preview" id="preview" alt="reference preview">
    <div class="meta" id="refMeta">no reference</div>
    <button id="clearRef">Clear</button>
  </div>

  <pre class="log" id="log"></pre>

<script>
(() => {
  const v = document.getElementById('v');
  const status = document.getElementById('status');
  const startBtn = document.getElementById('start');
  const stopBtn = document.getElementById('stop');
  const promptIn = document.getElementById('prompt');
  const seedIn = document.getElementById('seed');
  const stepsIn = document.getElementById('steps');
  const logEl = document.getElementById('log');
  const useCam = document.getElementById('useCam');
  const showInput = document.getElementById('showInput');
  const flipInput = document.getElementById('flipInput');
  const camSelect = document.getElementById('camSelect');
  const inputStatus = document.getElementById('inputStatus');
  const stage = document.getElementById('stage');
  const inv = document.getElementById('inv');
  const lipXfer = document.getElementById('lipXfer');
  const lipStatus = document.getElementById('lipStatus');

  let pc = null, ch = null;
  // Camera pipeline: raw getUserMedia stream → hidden <video> → canvas (optional flip)
  // → captureStream. The canvas stream feeds both the WebRTC sender and the
  // local preview, so the preview always shows what the pipeline receives.
  let rawStream = null;       // from getUserMedia
  let canvasStream = null;    // from canvas.captureStream() — sent to peer
  let hiddenVideo = null;
  let drawCanvas = null;
  let drawCtx = null;
  let drawRAF = 0;
  let mirror = false;         // toggled by flipInput

  function applyInputPreview() {
    if (showInput.checked && canvasStream) {
      inv.srcObject = canvasStream;
      stage.classList.add('split');
    } else {
      inv.srcObject = null;
      stage.classList.remove('split');
    }
  }

  function applyFlip() {
    mirror = flipInput.checked;   // draw loop will pick it up next frame
  }

  async function buildCameraPipeline(constraints) {
    rawStream = await navigator.mediaDevices.getUserMedia(constraints);
    const [vt] = rawStream.getVideoTracks();
    const settings = vt.getSettings();
    const W = settings.width || 1280;
    const H = settings.height || 720;

    hiddenVideo = document.createElement('video');
    hiddenVideo.srcObject = rawStream;
    hiddenVideo.muted = true;
    hiddenVideo.playsInline = true;
    await hiddenVideo.play();

    drawCanvas = document.createElement('canvas');
    drawCanvas.width = W;
    drawCanvas.height = H;
    drawCtx = drawCanvas.getContext('2d');

    function drawFrame() {
      if (!hiddenVideo || !drawCtx) return;
      if (mirror) {
        drawCtx.save();
        drawCtx.scale(-1, 1);
        drawCtx.drawImage(hiddenVideo, -W, 0, W, H);
        drawCtx.restore();
      } else {
        drawCtx.drawImage(hiddenVideo, 0, 0, W, H);
      }
      drawRAF = requestAnimationFrame(drawFrame);
    }
    drawRAF = requestAnimationFrame(drawFrame);

    canvasStream = drawCanvas.captureStream(30);
    return canvasStream;
  }

  function tearDownCameraPipeline() {
    if (drawRAF) { cancelAnimationFrame(drawRAF); drawRAF = 0; }
    if (rawStream) {
      rawStream.getTracks().forEach(t => { try { t.stop(); } catch (_) {} });
      rawStream = null;
    }
    if (canvasStream) {
      canvasStream.getTracks().forEach(t => { try { t.stop(); } catch (_) {} });
      canvasStream = null;
    }
    if (hiddenVideo) {
      try { hiddenVideo.srcObject = null; } catch (_) {}
      hiddenVideo = null;
    }
    drawCanvas = null;
    drawCtx = null;
  }

  function logLine(s) {
    const t = new Date().toLocaleTimeString();
    logEl.textContent += `[${t}] ${s}\\n`;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setStatus(text, cls) {
    status.textContent = text;
    status.className = cls || '';
  }

  async function start() {
    startBtn.disabled = true;
    setStatus('connecting...', '');

    pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    pc.ontrack = (e) => {
      logLine('Track received');
      v.srcObject = e.streams[0];
    };

    pc.oniceconnectionstatechange = () => {
      logLine('ICE: ' + pc.iceConnectionState);
      if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
        setStatus('live', 'live');
        stopBtn.disabled = false;
      } else if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
        setStatus('disconnected', 'err');
      }
    };

    ch = pc.createDataChannel('ctrl');
    ch.onopen = () => logLine('Control channel open');
    ch.onmessage = (e) => onCtrlMessage(e.data);
    ch.onclose = () => logLine('Control channel closed');

    if (useCam.checked) {
      try {
        const constraints = {
          audio: false,
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30 },
            ...(camSelect.value ? { deviceId: { exact: camSelect.value } } : {}),
          },
        };
        mirror = flipInput.checked;
        const stream = await buildCameraPipeline(constraints);
        const [vt] = stream.getVideoTracks();
        const rawLabel = rawStream && rawStream.getVideoTracks()[0]
          ? rawStream.getVideoTracks()[0].label
          : 'camera';
        logLine('Local camera acquired: ' + rawLabel);
        pc.addTransceiver(vt, {
          direction: 'sendrecv',
          streams: [stream],
        });
        showInput.disabled = false;
        flipInput.disabled = false;
        applyInputPreview();
      } catch (e) {
        logLine('Camera access failed: ' + e.message);
        setStatus('camera blocked', 'err');
        startBtn.disabled = false;
        useCam.checked = false;
        tearDownCameraPipeline();
        pc.addTransceiver('video', { direction: 'recvonly' });
      }
    } else {
      pc.addTransceiver('video', { direction: 'recvonly' });
    }

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Wait for ICE gathering to complete (LAN: fast, simpler than trickle).
    await new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') return resolve();
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', check);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
    });

    const res = await fetch('/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });
    if (!res.ok) {
      setStatus('offer rejected', 'err');
      startBtn.disabled = false;
      return;
    }
    const answer = await res.json();
    await pc.setRemoteDescription(answer);
    logLine('SDP exchange complete');
  }

  function stop() {
    if (ch) { try { ch.close(); } catch (_) {} ch = null; }
    if (pc) { try { pc.close(); } catch (_) {} pc = null; }
    tearDownCameraPipeline();
    inv.srcObject = null;
    stage.classList.remove('split');
    showInput.disabled = true;
    flipInput.disabled = true;
    v.srcObject = null;
    setStatus('idle', '');
    startBtn.disabled = false;
    stopBtn.disabled = true;
    logLine('Stopped');
  }

  function sendCtrl(msg) {
    if (!ch || ch.readyState !== 'open') {
      logLine('Control channel not ready');
      return;
    }
    ch.send(msg);
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);

  promptIn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && promptIn.value.trim()) {
      sendCtrl('prompt:' + promptIn.value.trim());
    }
  });
  seedIn.addEventListener('change', () => sendCtrl('seed:' + seedIn.value));
  stepsIn.addEventListener('change', () => sendCtrl('steps:' + stepsIn.value));

  window.addEventListener('beforeunload', stop);

  // ── reference image upload ────────────────────────────────────────────────
  const refRow = document.getElementById('refRow');
  const drop = document.getElementById('drop');
  const fileIn = document.getElementById('file');
  const preview = document.getElementById('preview');
  const refMeta = document.getElementById('refMeta');
  const clearRefBtn = document.getElementById('clearRef');

  // Server's reference version we last observed/uploaded. Used to skip
  // refresh when the broadcast we received is for our own POST.
  let lastSeenRefVersion = 0;

  function refreshPreview(versionLabel) {
    preview.src = '/reference?t=' + Date.now();
    preview.classList.add('shown');
    refMeta.textContent = versionLabel
      ? `reference v${versionLabel}`
      : 'reference active';
  }

  function clearPreview() {
    preview.classList.remove('shown');
    preview.removeAttribute('src');
    refMeta.textContent = 'no reference';
  }

  function onCtrlMessage(msg) {
    if (typeof msg !== 'string') return;
    if (msg.startsWith('ref:set:')) {
      const v = parseInt(msg.slice('ref:set:'.length), 10);
      if (!isNaN(v) && v > lastSeenRefVersion) {
        lastSeenRefVersion = v;
        refreshPreview(v);
        logLine(`Reference updated by another client (v${v})`);
      }
    } else if (msg.startsWith('ref:clear')) {
      const v = parseInt(msg.split(':')[2] || '0', 10);
      if (!isNaN(v) && v > lastSeenRefVersion) {
        lastSeenRefVersion = v;
        clearPreview();
        logLine(`Reference cleared by another client (v${v})`);
      }
    } else if (msg === 'input:peer') {
      inputStatus.textContent = useCam.checked
        ? 'input: peer (you)'
        : 'input: peer (other client)';
      logLine('Pipeline input now from a peer');
    } else if (msg === 'input:server') {
      inputStatus.textContent = 'input: server';
      logLine('Pipeline input now from server camera');
    } else if (msg === 'lip:on' || msg === 'lip:off') {
      const on = (msg === 'lip:on');
      if (lipXfer.checked !== on) lipXfer.checked = on;
      lipStatus.textContent = on ? 'lipsync: ON' : 'lipsync: OFF';
      logLine('Lip transfer ' + (on ? 'enabled' : 'disabled'));
    } else {
      logLine('server: ' + msg);
    }
  }

  async function probeHealth() {
    try {
      const r = await fetch('/healthz');
      const j = await r.json();
      if (!j.reference_enabled) {
        refRow.classList.add('disabled');
        document.getElementById('comfyRow').classList.add('disabled');
        refMeta.textContent = 'disabled in config';
        drop.textContent = 'Reference disabled — start server with --config configs/config_with_reference.json';
      } else if (j.reference_set) {
        lastSeenRefVersion = j.reference_version || 0;
        refreshPreview(j.reference_version);
      }
      if (j.input_source === 'peer') {
        inputStatus.textContent = 'input: peer (other client)';
      } else {
        inputStatus.textContent = 'input: server';
      }
      if (j.lip_enabled) {
        lipXfer.disabled = false;
        lipXfer.checked = !!j.lip_active;
        lipStatus.textContent = j.lip_active ? 'lipsync: ON' : 'lipsync: OFF';
      } else {
        lipXfer.disabled = true;
        lipXfer.checked = false;
        lipStatus.textContent = 'lipsync: unavailable (add lip_transfer to config)';
      }
    } catch (_) {}
  }
  probeHealth();

  lipXfer.addEventListener('change', async () => {
    const on = lipXfer.checked;
    lipXfer.disabled = true;
    try {
      const r = await fetch('/lip-transfer?on=' + (on ? 'true' : 'false'), {
        method: 'POST',
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        logLine('Lip transfer toggle failed: ' + (err.detail || r.statusText));
        lipXfer.checked = !on;     // revert
        return;
      }
      const j = await r.json();
      lipStatus.textContent = j.lip_active ? 'lipsync: ON' : 'lipsync: OFF';
    } catch (e) {
      logLine('Lip transfer error: ' + e);
      lipXfer.checked = !on;
    } finally {
      lipXfer.disabled = false;
    }
  });

  // ── camera enumeration (requires getUserMedia permission to surface labels)
  async function refreshCameras() {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const cams = devs.filter(d => d.kind === 'videoinput');
      camSelect.innerHTML = '';
      if (cams.length === 0) {
        const opt = document.createElement('option');
        opt.value = ''; opt.textContent = '(no cameras detected — grant permission)';
        camSelect.appendChild(opt);
        return;
      }
      const def = document.createElement('option');
      def.value = ''; def.textContent = 'Default camera';
      camSelect.appendChild(def);
      cams.forEach((c, i) => {
        const opt = document.createElement('option');
        opt.value = c.deviceId;
        opt.textContent = c.label || `Camera ${i + 1}`;
        camSelect.appendChild(opt);
      });
    } catch (e) {
      logLine('Camera enumeration error: ' + e.message);
    }
  }

  showInput.addEventListener('change', applyInputPreview);
  flipInput.addEventListener('change', applyFlip);

  useCam.addEventListener('change', async () => {
    camSelect.disabled = !useCam.checked;
    if (!useCam.checked) {
      showInput.checked = false;
      showInput.disabled = true;
      flipInput.checked = false;
      flipInput.disabled = true;
      applyInputPreview();
      applyFlip();
    }
    if (useCam.checked) {
      // First call to enumerate after granting permission gives real labels.
      try {
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        tmp.getTracks().forEach(t => t.stop());
      } catch (e) {
        logLine('Camera permission denied: ' + e.message);
        useCam.checked = false;
        camSelect.disabled = true;
        return;
      }
      await refreshCameras();
    }
  });
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', () => {
      if (useCam.checked) refreshCameras();
    });
  }

  async function uploadReference(file) {
    if (!file || !file.type.startsWith('image/')) {
      logLine('Not an image file');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      logLine('File too large (>10 MB)');
      return;
    }
    refMeta.textContent = 'uploading...';
    try {
      const r = await fetch('/reference', {
        method: 'POST',
        headers: { 'Content-Type': file.type || 'application/octet-stream' },
        body: file,
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        logLine('Reference upload failed: ' + (err.detail || r.statusText));
        refMeta.textContent = 'upload failed';
        return;
      }
      const j = await r.json();
      logLine(`Reference set: ${j.size[0]}x${j.size[1]}, ${j.bytes} bytes (v${j.version})`);
      lastSeenRefVersion = j.version || lastSeenRefVersion;
      refMeta.textContent = `active ${j.size[0]}x${j.size[1]} (v${j.version})`;
      preview.src = URL.createObjectURL(file);
      preview.classList.add('shown');
    } catch (e) {
      logLine('Reference upload error: ' + e);
      refMeta.textContent = 'error';
    }
  }

  drop.addEventListener('click', () => fileIn.click());
  fileIn.addEventListener('change', () => {
    if (fileIn.files.length) uploadReference(fileIn.files[0]);
  });

  ['dragenter', 'dragover'].forEach(ev => {
    drop.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      drop.classList.add('over');
    });
  });
  ['dragleave', 'drop'].forEach(ev => {
    drop.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      drop.classList.remove('over');
    });
  });
  drop.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) {
      uploadReference(e.dataTransfer.files[0]);
    }
  });

  // Paste from clipboard (Cmd+V on macOS).
  window.addEventListener('paste', (e) => {
    if (!e.clipboardData) return;
    for (const item of e.clipboardData.items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) uploadReference(file);
        break;
      }
    }
  });

  // ── comfy server reference puller ─────────────────────────────────────────
  const comfySelect = document.getElementById('comfySelect');
  const comfyPullBtn = document.getElementById('comfyPull');
  const comfyStatus = document.getElementById('comfyStatus');

  async function loadComfyServers() {
    try {
      const r = await fetch('/comfy/servers');
      const j = await r.json();
      comfySelect.innerHTML = '';
      if (!j.servers || j.servers.length === 0) {
        const opt = document.createElement('option');
        opt.value = ''; opt.textContent = '(none configured)';
        comfySelect.appendChild(opt);
        comfyPullBtn.disabled = true;
        return;
      }
      j.servers.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.name;
        opt.textContent = `${s.name} (${s.url})`;
        comfySelect.appendChild(opt);
      });
    } catch (e) {
      logLine('Comfy server list error: ' + e);
    }
  }
  loadComfyServers();

  comfyPullBtn.addEventListener('click', async () => {
    const name = comfySelect.value;
    if (!name) return;
    comfyPullBtn.disabled = true;
    comfyStatus.textContent = `pulling from ${name}...`;
    try {
      const r = await fetch('/comfy/pull?server=' + encodeURIComponent(name), {
        method: 'POST',
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        comfyStatus.textContent = 'error: ' + (err.detail || r.statusText);
        logLine('Comfy pull failed: ' + (err.detail || r.statusText));
        return;
      }
      const j = await r.json();
      lastSeenRefVersion = j.version || lastSeenRefVersion;
      comfyStatus.textContent = `pulled ${j.filename} (v${j.version})`;
      refreshPreview(j.version);
      logLine(`Comfy pulled: ${j.filename} from ${name} (v${j.version})`);
    } catch (e) {
      comfyStatus.textContent = 'error: ' + e.message;
      logLine('Comfy pull error: ' + e);
    } finally {
      comfyPullBtn.disabled = false;
    }
  });

  clearRefBtn.addEventListener('click', async () => {
    try {
      const r = await fetch('/reference', { method: 'DELETE' });
      if (r.ok) {
        const j = await r.json().catch(() => ({}));
        if (j.version) lastSeenRefVersion = j.version;
        clearPreview();
        logLine('Reference cleared' + (j.version ? ` (v${j.version})` : ''));
      }
    } catch (e) {
      logLine('Clear error: ' + e);
    }
  });
})();
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Entry point.
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global sp, input_tensor, output_tensor, resolution

    parser = argparse.ArgumentParser(description="FluxRT WebRTC server")
    parser.add_argument("--config", default="configs/stream_processor_config.json")
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    parser.add_argument("--camera", type=int, default=0, help="cv2 camera index")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument(
        "--no-server-camera",
        action="store_true",
        help=(
            "Do not open the local OpenCV camera. Pipeline input then comes "
            "from the first WebRTC peer that sends a video track."
        ),
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to TLS cert (enables HTTPS, required by browsers for "
        "getUserMedia on non-localhost origins).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to TLS private key. Use together with --ssl-certfile.",
    )
    parser.add_argument(
        "--comfy-server",
        action="append",
        default=[],
        metavar="NAME=URL",
        help=(
            "Register a ComfyUI server the client can pull reference images "
            "from. Repeatable. Example: "
            "--comfy-server A=http://79.169.112.61:12040"
        ),
    )
    args = parser.parse_args()

    # Populate comfy_servers dict. Fall back to the two known production
    # servers (A=12040, B=12041) if none were passed on the CLI.
    raw_entries = args.comfy_server or [
        "A=http://79.169.112.61:12040",
        "B=http://79.169.112.61:12041",
    ]
    for entry in raw_entries:
        name, _, url = entry.partition("=")
        name = name.strip()
        url = url.strip().rstrip("/")
        if not name or not url:
            log.warning("Ignoring bad --comfy-server entry: %r", entry)
            continue
        comfy_servers[name] = url
    log.info("Configured comfy servers: %s", comfy_servers)

    log.info("Loading StreamProcessor from %s", args.config)
    sp = StreamProcessor(args.config)

    # Lip transfer only loads under int8 — bf16 + LivePortrait exceeds the
    # 4090's 24 GB. Mutate the in-memory config before sp.start() so the
    # inference subprocess never instantiates the postprocessor.
    # ModelInferenceSubprocess holds the same dict reference and the spawn
    # pickle happens during sp.start(), so this propagates correctly.
    lp_cfg = sp.config.get("lip_transfer")
    if lp_cfg and lp_cfg.get("enable"):
        if args.int8:
            log.info("Lip transfer: enabled (int8 mode)")
        else:
            log.warning(
                "Lip transfer disabled: bf16 mode + LivePortrait would exceed "
                "24 GB VRAM. Re-run with --int8 to enable lipsync."
            )
            lp_cfg["enable"] = False

    if args.int8:
        sp.enable_quantization()
    sp.start()

    if args.initial_prompt:
        sp.set_prompt(args.initial_prompt)

    input_tensor = sp.get_input_tensor()
    output_tensor = sp.get_output_tensor()
    resolution = sp.get_resolution()
    log.info("Resolution: %dx%d", resolution["width"], resolution["height"])

    if args.no_server_camera:
        log.info("--no-server-camera: skipping local camera; waiting for peer input.")
        # Mark the peer-input slot as needing a peer immediately so the
        # producer (if anything else ever started it) would yield.
        peer_input_active.set()
    else:
        producer_thread = threading.Thread(
            target=producer_loop, args=(args.camera,), daemon=True
        )
        producer_thread.start()

    use_tls = bool(args.ssl_certfile and args.ssl_keyfile)
    scheme = "https" if use_tls else "http"
    log.info("Open %s://<this-host>:%d/ in a LAN browser", scheme, args.port)
    if not use_tls:
        log.warning(
            "Running plain HTTP — browsers will block getUserMedia on non-"
            "localhost origins. Pass --ssl-certfile + --ssl-keyfile for HTTPS."
        )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
