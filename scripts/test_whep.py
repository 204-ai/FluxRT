#!/usr/bin/env python3
"""WHEP endpoint smoke test (SPEC §T.10) — GPU-free.

Uses the torch-free `fluxrt` shim (same trick as tests/webrtc/conftest.py) to
import run_webrtc without StreamProcessor, seeds `latest_rgb` with a synthetic
frame, runs the FastAPI app in-process, and drives it with an aiortc client:

  V10: POST /whep (application/sdp) -> 201 + Location + application/sdp answer;
       wrong content-type -> 415; empty body -> 400
  V12: two concurrent viewers each receive frames at the native resolution
  V13: DELETE -> 200, repeat -> 404; client-side close -> session auto-cleaned

Usage: python scripts/test_whep.py   (needs aiortc/fastapi/uvicorn/numpy/cv2/httpx/PIL)
"""
import asyncio
import pathlib
import sys
import threading
import time
import types

import numpy as np
import requests
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription

# ── torch-free fluxrt shim (mirrors tests/webrtc/conftest.py) ────────────────
_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
_pkg = types.ModuleType("fluxrt")
_pkg.__path__ = [str(_REPO / "src" / "fluxrt")]
_pkg.StreamProcessor = type("StreamProcessor", (), {})  # never instantiated here
sys.modules["fluxrt"] = _pkg

import run_webrtc  # noqa: E402

PORT = 8978
BASE = f"http://127.0.0.1:{PORT}"
NATIVE = 512

passed = []


def check(name: str, cond: bool, detail: str = ""):
    if not cond:
        print(f"FAIL {name} {detail}")
        sys.exit(1)
    passed.append(name)
    print(f"ok   {name}")


def track_future(pc: RTCPeerConnection) -> asyncio.Future:
    got = asyncio.get_event_loop().create_future()

    @pc.on("track")
    def _on_track(track):
        if track.kind == "video" and not got.done():
            got.set_result(track)

    return got


async def whep_connect():
    pc = RTCPeerConnection()
    got = track_future(pc)
    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    r = requests.post(
        f"{BASE}/whep",
        data=pc.localDescription.sdp.encode(),
        headers={"Content-Type": "application/sdp"},
        timeout=10,
    )
    check("whep 201", r.status_code == 201, f"got {r.status_code}: {r.text[:200]}")
    check("whep Location", r.headers.get("Location", "").startswith("/whep/"), str(r.headers))
    check("whep answer content-type", "application/sdp" in r.headers.get("content-type", ""))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=r.text, type="answer"))
    return pc, got, r.headers["Location"]


async def recv_frames(got: asyncio.Future, n: int, timeout: float = 15.0):
    track = await asyncio.wait_for(got, timeout)
    frame = None
    for _ in range(n):
        frame = await asyncio.wait_for(track.recv(), timeout)
    return frame


async def wait_for(pred, timeout: float, what: str):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return
        await asyncio.sleep(0.5)
    print(f"FAIL timeout waiting for {what}")
    sys.exit(1)


async def main():
    r = requests.post(f"{BASE}/whep", json={"sdp": "x"}, timeout=5)
    check("whep 415 on json", r.status_code == 415, f"got {r.status_code}")
    r = requests.post(f"{BASE}/whep", data=b"", headers={"Content-Type": "application/sdp"}, timeout=5)
    check("whep 400 on empty", r.status_code == 400, f"got {r.status_code}")

    pc_a, got_a, loc_a = await whep_connect()
    pc_b, got_b, loc_b = await whep_connect()
    check("2 sessions registered", len(run_webrtc.whep_sessions) == 2)

    frame_a = await recv_frames(got_a, 5)
    check("viewer A frames @native", frame_a.width == NATIVE, f"{frame_a.width}x{frame_a.height}")
    frame_b = await recv_frames(got_b, 5)
    check("viewer B frames (concurrent)", frame_b.width == NATIVE)

    r = requests.delete(f"{BASE}{loc_a}", timeout=5)
    check("delete 200", r.status_code == 200, f"got {r.status_code}")
    r = requests.delete(f"{BASE}{loc_a}", timeout=5)
    check("delete repeat 404", r.status_code == 404, f"got {r.status_code}")
    check("registry after delete", len(run_webrtc.whep_sessions) == 1)
    await pc_a.close()

    await pc_b.close()
    await wait_for(lambda: len(run_webrtc.whep_sessions) == 0, 20, "auto-clean after client close")
    check("auto-clean on client close", True)

    print(f"\nPASS {len(passed)} checks")


if __name__ == "__main__":
    # Seed server globals normally set by main(): output resolution + a frame.
    run_webrtc.out_resolution = {"width": NATIVE, "height": NATIVE}
    rgb = np.zeros((NATIVE, NATIVE, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200  # non-black so a human eyeballing a viewer sees red
    with run_webrtc.latest_lock:
        run_webrtc.latest_rgb = rgb

    server = uvicorn.Server(uvicorn.Config(run_webrtc.app, host="127.0.0.1", port=PORT, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(40):
        try:
            requests.get(f"{BASE}/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.25)
    else:
        print("FAIL server never came up")
        sys.exit(1)
    try:
        asyncio.run(main())
    finally:
        server.should_exit = True
        t.join(timeout=10)
