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
    - Prompt updates: typed in the browser input, sent over a DataChannel.
    - Optional reference image upload could be added (not implemented here).
"""

import argparse
import asyncio
import fractions
import logging
import threading
import time
from typing import Optional

import av
import cv2
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

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

pcs: set[RTCPeerConnection] = set()


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


@app.get("/")
async def _index():
    return HTMLResponse(CLIENT_HTML)


@app.get("/healthz")
async def _health():
    return {
        "ready": bool(sp and sp.is_ready()),
        "peers": len(pcs),
        "resolution": resolution,
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
    ch.onmessage = (e) => logLine('server: ' + e.data);
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
