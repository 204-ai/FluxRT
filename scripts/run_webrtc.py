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
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
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

MAX_REFERENCE_BYTES = 10 * 1024 * 1024  # 10 MB cap for uploaded reference images.


def broadcast_ctrl(msg: str) -> None:
    """Send a string message to every open control DataChannel."""
    for ch in list(ctrl_channels):
        try:
            ch.send(msg)
        except Exception as exc:
            log.debug("broadcast send failed, dropping channel: %s", exc)
            ctrl_channels.discard(ch)


# ──────────────────────────────────────────────────────────────────────────────
# Producer — webcam frames → pipeline → latest_rgb.
# ──────────────────────────────────────────────────────────────────────────────
def producer_loop(camera_index: int) -> None:
    global latest_rgb

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
    log.info("StreamProcessor ready, producing frames.")

    while not producer_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            log.warning("Camera read failed, retrying...")
            time.sleep(0.05)
            continue

        resized = crop_maximal_rectangle(frame, resolution["height"], resolution["width"])
        input_tensor.copy_from(resized)

        out_bgr = output_tensor.to_numpy()
        rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        with latest_lock:
            latest_rgb = rgb

    cap.release()
    log.info("Producer stopped.")


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

    try:
        img = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decode error: {exc}")

    rgb = np.array(img)

    # `set_reference_image` runs in the inference subprocess; it accepts
    # uint8 RGB and resizes to `reference_image_resolution` internally.
    sp.set_reference_image(rgb)

    # Cache a PNG preview for GET /reference.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    global latest_reference_png, ref_version
    with reference_lock:
        latest_reference_png = buf.getvalue()
        ref_version += 1
        version = ref_version

    log.info(
        "Reference image set: %dx%d (%d bytes uploaded, v%d)",
        *rgb.shape[1::-1], len(body), version,
    )
    broadcast_ctrl(f"ref:set:{version}")

    return JSONResponse(
        {
            "ok": True,
            "size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "bytes": len(body),
            "version": version,
        }
    )


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
  .stage { display: flex; justify-content: center; padding: 12px; background: #0a0a0a; }
  video { width: 100%; max-width: 1024px; background: #000; display: block; }
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

  <div class="stage">
    <video id="v" autoplay playsinline muted></video>
  </div>

  <div class="controls">
    <button id="start">Start</button>
    <button id="stop" disabled>Stop</button>
    <input id="prompt" type="text" placeholder="Prompt — press Enter to apply">
    <label>seed <input id="seed" type="number" value="52"></label>
    <label>steps <input id="steps" type="number" value="2" min="1" max="8"></label>
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

  let pc = null, ch = null;

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

    pc.addTransceiver('video', { direction: 'recvonly' });

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
        refMeta.textContent = 'disabled in config';
        drop.textContent = 'Reference disabled — start server with --config configs/config_with_reference.json';
      } else if (j.reference_set) {
        lastSeenRefVersion = j.reference_version || 0;
        refreshPreview(j.reference_version);
      }
    } catch (_) {}
  }
  probeHealth();

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
    args = parser.parse_args()

    log.info("Loading StreamProcessor from %s", args.config)
    sp = StreamProcessor(args.config)
    if args.int8:
        sp.enable_quantization()
    sp.start()

    if args.initial_prompt:
        sp.set_prompt(args.initial_prompt)

    input_tensor = sp.get_input_tensor()
    output_tensor = sp.get_output_tensor()
    resolution = sp.get_resolution()
    log.info("Resolution: %dx%d", resolution["width"], resolution["height"])

    producer_thread = threading.Thread(
        target=producer_loop, args=(args.camera,), daemon=True
    )
    producer_thread.start()

    log.info("Open http://<this-host>:%d/ in a LAN browser", args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
