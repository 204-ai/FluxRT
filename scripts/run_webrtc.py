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
import contextlib
import fractions
import io
import json
import logging
import os
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
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Directory holding the static browser client (index.html, app.js, app.css,
# input_processor.js). Served at /static, with / returning index.html.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webrtc_static")

# ComfyUI API-format workflow templates patched + queued by /comfy/edit.
WORKFLOWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfy_workflows")
QWEN_EDIT_TEMPLATE = os.path.join(WORKFLOWS_DIR, "qwen_edit_2509.api.json")

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

# Serializes the actual pipeline drive (shared-tensor write + read). Both the
# local producer thread and per-peer executor calls go through push_input_frame;
# without this they can race during the camera->peer ownership handoff and drive
# the pipeline from two threads at once.
pipeline_lock = threading.Lock()

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

# Strong refs to peer-input consumer tasks. asyncio only holds weak refs to
# tasks, so without this a consumer can be GC'd mid-run; if its `finally`
# never executes, peer_input_active stays set and the producer never resumes.
peer_input_tasks: set = set()

MAX_REFERENCE_BYTES = 10 * 1024 * 1024  # 10 MB cap for uploaded reference images.

# Configured ComfyUI servers (populated in main() from --comfy-server flags).
# Maps display name → base URL (no trailing slash).
comfy_servers: dict[str, str] = {}

# Lip transfer (LivePortrait postprocessor) runtime state. The pipeline
# always loads the model when `lip_transfer.enable` is true in config, but
# the per-frame postprocess is gated by `set_lip_transfer(bool)`.
lip_active: bool = False

# Mirror the pipeline state so /healthz and late-joining peers can see it.
# Updated from any path that changes the underlying value (API, DataChannel).
current_prompt: str = ""
current_seed: int = 0
current_steps: int = 0


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
    # Hold pipeline_lock across write+read so the producer thread and a peer's
    # executor call can't interleave shared-tensor access during handoff.
    with pipeline_lock:
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
_cleanup_done = threading.Event()


async def _graceful_cleanup() -> None:
    """Idempotent teardown: stop producer, close peers, stop the pipeline
    (which now joins/terminates its subprocesses with timeouts). Safe to call
    from the lifespan shutdown and from a signal handler."""
    if _cleanup_done.is_set():
        return
    _cleanup_done.set()

    log.info("Shutting down — stopping producer, peers, pipeline...")
    producer_stop.set()

    # Close all peer connections (best-effort, bounded).
    for pc in list(pcs):
        with contextlib.suppress(Exception):
            await pc.close()
    pcs.clear()

    # sp.stop() does blocking joins; run it off the event loop so a wedged
    # subprocess can't stall the loop while it escalates to terminate/kill.
    if sp is not None:
        loop = asyncio.get_running_loop()
        with contextlib.suppress(Exception):
            await loop.run_in_executor(None, sp.stop)
    log.info("Shutdown complete.")


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    await _graceful_cleanup()


app = FastAPI(lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
            task = asyncio.ensure_future(consume_peer_input(track, pc))
            peer_input_tasks.add(task)
            task.add_done_callback(peer_input_tasks.discard)

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

        # Send current pipeline parameter snapshot to a fresh peer.
        try:
            if current_prompt:
                channel.send(f"state:prompt:{current_prompt}")
            channel.send(f"state:seed:{current_seed}")
            channel.send(f"state:steps:{current_steps}")
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
            global current_prompt, current_seed, current_steps
            if msg.startswith("prompt:"):
                new_prompt = msg[len("prompt:"):].strip()
                if new_prompt:
                    log.info("Prompt update: %r", new_prompt)
                    current_prompt = new_prompt
                    sp.set_prompt(new_prompt)
                    channel.send("ack:prompt")
                    broadcast_ctrl(f"state:prompt:{new_prompt}")
            elif msg.startswith("seed:"):
                try:
                    seed = int(msg[len("seed:"):])
                    current_seed = seed
                    sp.set_seed(seed)
                    channel.send(f"ack:seed:{seed}")
                    broadcast_ctrl(f"state:seed:{seed}")
                except ValueError:
                    channel.send("err:seed")
            elif msg.startswith("steps:"):
                try:
                    steps = int(msg[len("steps:"):])
                    current_steps = steps
                    sp.set_steps(steps)
                    channel.send(f"ack:steps:{steps}")
                    broadcast_ctrl(f"state:steps:{steps}")
                except ValueError:
                    channel.send("err:steps")

    pc.addTrack(FluxRTTrack(fps=30))

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


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


@app.post("/prompt")
async def post_prompt(request: Request):
    """Set the generation prompt.
    Body: JSON {"prompt": "..."} OR raw text/plain.
    Query: ?prompt=... (URL-encoded) as a third option."""
    if not sp:
        raise HTTPException(status_code=503, detail="StreamProcessor not ready")

    prompt: str | None = None
    q = request.query_params.get("prompt")
    if q is not None:
        prompt = q
    else:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            try:
                payload = await request.json()
                prompt = payload.get("prompt") if isinstance(payload, dict) else None
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Bad JSON: {exc}")
        else:
            prompt = (await request.body()).decode("utf-8", errors="replace")

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    prompt = prompt.strip()
    global current_prompt
    current_prompt = prompt
    sp.set_prompt(prompt)
    log.info("Prompt set via API: %r", prompt)
    broadcast_ctrl(f"state:prompt:{prompt}")
    return JSONResponse({"ok": True, "prompt": prompt})


@app.post("/seed")
async def post_seed(request: Request):
    raw = request.query_params.get("value")
    if raw is None:
        try:
            body = await request.json()
            raw = body.get("value") if isinstance(body, dict) else None
        except Exception:
            raw = None
    if raw is None:
        raise HTTPException(status_code=400, detail="Missing ?value= (int)")
    try:
        seed = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Not an int: {raw!r}")

    global current_seed
    current_seed = seed
    sp.set_seed(seed)
    log.info("Seed set via API: %d", seed)
    broadcast_ctrl(f"state:seed:{seed}")
    return JSONResponse({"ok": True, "seed": seed})


@app.post("/steps")
async def post_steps(request: Request):
    raw = request.query_params.get("value")
    if raw is None:
        try:
            body = await request.json()
            raw = body.get("value") if isinstance(body, dict) else None
        except Exception:
            raw = None
    if raw is None:
        raise HTTPException(status_code=400, detail="Missing ?value= (int)")
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Not an int: {raw!r}")
    if steps < 1 or steps > 8:
        raise HTTPException(status_code=400, detail="steps must be 1..8")

    global current_steps
    current_steps = steps
    sp.set_steps(steps)
    log.info("Steps set via API: %d", steps)
    broadcast_ctrl(f"state:steps:{steps}")
    return JSONResponse({"ok": True, "steps": steps})


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


@app.post("/comfy/edit")
async def comfy_edit(request: Request):
    """Snap an image (raw bytes body), run it through the Qwen-Image-Edit
    2509 workflow on the selected comfy server, wait for the result, and
    install it as the reference image.
    Query: ?server=NAME&prompt=... (prompt falls back to the live prompt)."""
    if not _reference_enabled():
        raise HTTPException(status_code=400, detail="Reference image disabled")

    name = request.query_params.get("server", "")
    if not name or name not in comfy_servers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown comfy server: {name!r}. Known: {list(comfy_servers.keys())}",
        )
    base = comfy_servers[name].rstrip("/")

    prompt_text = request.query_params.get("prompt") or current_prompt or "enhance this person"

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")
    if len(body) > MAX_REFERENCE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        with open(QWEN_EDIT_TEMPLATE, "r") as f:
            wf = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow template error: {exc}")

    seed = int.from_bytes(os.urandom(4), "big")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # 1. upload the snapped frame to the comfy input/ folder
            files = {"image": ("fluxrt_snap.png", body, "image/png")}
            ur = await client.post(
                f"{base}/upload/image", files=files, data={"overwrite": "true"}
            )
            ur.raise_for_status()
            up = ur.json()
            uploaded_name = up["name"]

            # 2. patch the workflow: input image, prompt, seed
            wf["78"]["inputs"]["image"] = uploaded_name
            wf["111"]["inputs"]["prompt"] = prompt_text
            wf["3"]["inputs"]["seed"] = seed

            # 3. queue it
            pr = await client.post(f"{base}/prompt", json={"prompt": wf})
            pr.raise_for_status()
            prompt_id = pr.json()["prompt_id"]
            log.info("Qwen edit queued on %s: prompt_id=%s seed=%d", name, prompt_id, seed)

            # 4. poll history until the output image appears (~4-step, fast)
            out_image = None
            for _ in range(90):
                await asyncio.sleep(1.0)
                hr = await client.get(f"{base}/history/{prompt_id}")
                if hr.status_code != 200:
                    continue
                hist = hr.json()
                entry = hist.get(prompt_id)
                if not entry:
                    continue
                for _node, out in (entry.get("outputs", {}) or {}).items():
                    for img in out.get("images", []) or []:
                        if img.get("type") == "output":
                            out_image = img
                            break
                    if out_image:
                        break
                if out_image:
                    break

            if not out_image:
                raise HTTPException(status_code=504, detail="Qwen edit timed out")

            # 5. fetch the result
            vr = await client.get(
                f"{base}/view",
                params={
                    "filename": out_image["filename"],
                    "type": out_image["type"],
                    "subfolder": out_image.get("subfolder", ""),
                },
            )
            vr.raise_for_status()
            result_bytes = vr.content

    except httpx.HTTPError as exc:
        log.warning("Comfy edit HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Comfy server error: {exc}")

    # 6. install the edited image as the reference
    result = _apply_reference_bytes(result_bytes, f"qwen-edit:{name}:{out_image['filename']}")
    result["prompt_id"] = prompt_id
    result["filename"] = out_image["filename"]
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
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


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
        "prompt": current_prompt,
        "seed": current_seed,
        "steps": current_steps,
        **_perf_metrics(),
    }


def _perf_metrics() -> dict:
    """Pipeline FPS + VRAM snapshot for /healthz polling."""
    if not sp:
        return {
            "fps_pipeline": 0.0,
            "fps_interpolated": 0.0,
            "proc_time_ms": 0.0,
            "vram_mb": 0,
        }
    pt = sp.get_last_processing_time() or 0.0
    base = (1.0 / pt) if pt > 0 else 0.0
    exp = int(sp.config.get("interpolation_exp", 0))
    try:
        vram_mb = int(sp.get_reserved_memory())
    except Exception:
        vram_mb = 0
    return {
        "fps_pipeline": round(base, 2),
        "fps_interpolated": round(base * (2 ** exp), 2),
        "proc_time_ms": round(pt * 1000.0, 2),
        "vram_mb": vram_mb,
    }



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
        "--interp",
        type=int,
        default=None,
        metavar="EXP",
        help=(
            "Override interpolation_exp from config. RIFE emits 2^EXP output "
            "frames per generated frame (0=off, 1=2x, 2=4x, 3=8x). Boot-time "
            "only — sizes the shared frame buffer, cannot change at runtime."
        ),
    )
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

    config_path = args.config

    # --interp overrides interpolation_exp. StreamProcessor sizes the shared
    # frame buffer from this value in __init__, so it must be baked into the
    # config the constructor reads. Write a patched temp config and pass that.
    if args.interp is not None:
        if args.interp < 0 or args.interp > 4:
            parser.error("--interp must be in 0..4")
        import atexit
        import json as _json
        import tempfile

        with open(args.config) as f:
            cfg = _json.load(f)
        prev = cfg.get("interpolation_exp")
        cfg["interpolation_exp"] = args.interp
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="fluxrt_cfg_", delete=False
        )
        _json.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name
        # Don't leave the patched temp config behind in /tmp on exit.
        atexit.register(lambda p=config_path: os.path.exists(p) and os.unlink(p))
        log.info(
            "interpolation_exp override: %s -> %d (2^%d = %d output frames/frame)",
            prev, args.interp, args.interp, 2 ** args.interp,
        )

    log.info("Loading StreamProcessor from %s", config_path)
    sp = StreamProcessor(config_path)

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

    # Initialize parameter mirror from config + CLI overrides.
    global current_prompt, current_seed, current_steps
    current_prompt = args.initial_prompt or sp.config.get("default_prompt", "")
    current_seed = int(sp.config.get("default_seed", 0))
    current_steps = int(sp.config.get("default_steps", 2))

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
