#!/usr/bin/env python3
"""WHIP endpoint smoke test (SPEC §T.15) — GPU-free.

Same harness as scripts/test_whep.py: torch-free `fluxrt` shim, in-process
uvicorn, real aiortc clients. The pipeline is faked: `sp` is a bare object
(only truthiness is used by the 503 guard) and `_frame_sink` is replaced with
a counter, so ownership claims and frame flow are observable without a GPU.

  V16: POST /whip -> 201 + Location + application/sdp; 415 wrong ct;
       400 empty; 503 before the pipeline exists
  V17: publisher frames flow through consume_peer_input (ownership.is_active,
       sink counter grows); a second publisher joining+leaving does not steal
       ownership (counter keeps growing); DELETE cancels the consumer and
       releases ownership; repeat DELETE -> 404

Usage: python scripts/test_whip.py
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
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

# ── torch-free fluxrt shim (mirrors tests/webrtc/conftest.py) ────────────────
_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
_pkg = types.ModuleType("fluxrt")
_pkg.__path__ = [str(_REPO / "src" / "fluxrt")]
_pkg.StreamProcessor = type("StreamProcessor", (), {})
sys.modules["fluxrt"] = _pkg

import run_webrtc  # noqa: E402

PORT = 8980
BASE = f"http://127.0.0.1:{PORT}"

passed = []
sunk = {"n": 0}


def check(name: str, cond: bool, detail: str = ""):
    if not cond:
        print(f"FAIL {name} {detail}")
        sys.exit(1)
    passed.append(name)
    print(f"ok   {name}")


_RGB = np.zeros((512, 512, 3), dtype=np.uint8)


async def _counting_sink(frame) -> None:
    """Counts sunk frames AND publishes like push_input_frame does (latest_rgb +
    output_version bump) so the version-gated output tracks pace realistically."""
    sunk["n"] += 1
    with run_webrtc.latest_lock:
        run_webrtc.latest_rgb = _RGB
        run_webrtc.output_version += 1


class BlackTrack(VideoStreamTrack):
    async def recv(self):
        import av

        pts, time_base = await self.next_timestamp()
        frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format="rgb24")
        frame.pts, frame.time_base = pts, time_base
        return frame


async def whip_publish():
    pc = RTCPeerConnection()
    pc.addTrack(BlackTrack())
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    r = requests.post(
        f"{BASE}/whip",
        data=pc.localDescription.sdp.encode(),
        headers={"Content-Type": "application/sdp"},
        timeout=10,
    )
    check("whip 201", r.status_code == 201, f"got {r.status_code}: {r.text[:200]}")
    check("whip Location", r.headers.get("Location", "").startswith("/whip/"), str(r.headers))
    check("whip answer content-type", "application/sdp" in r.headers.get("content-type", ""))
    await pc.setRemoteDescription(RTCSessionDescription(sdp=r.text, type="answer"))
    return pc, r.headers["Location"]


async def wait_for(pred, timeout: float, what: str):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return
        await asyncio.sleep(0.5)
    print(f"FAIL timeout waiting for {what}")
    sys.exit(1)


async def wait_sunk_above(floor: int, timeout: float, what: str):
    await wait_for(lambda: sunk["n"] > floor, timeout, f"{what} (sunk stuck at {sunk['n']})")


async def main():
    # V16: 503 while no pipeline
    pc0 = RTCPeerConnection()
    pc0.addTrack(BlackTrack())
    await pc0.setLocalDescription(await pc0.createOffer())
    r = requests.post(f"{BASE}/whip", data=pc0.localDescription.sdp.encode(),
                      headers={"Content-Type": "application/sdp"}, timeout=5)
    check("whip 503 without pipeline", r.status_code == 503, f"got {r.status_code}")
    await pc0.close()

    run_webrtc.sp = object()  # 503 guard passes; the pipeline itself is never touched

    r = requests.post(f"{BASE}/whip", json={"sdp": "x"}, timeout=5)
    check("whip 415 on json", r.status_code == 415, f"got {r.status_code}")
    r = requests.post(f"{BASE}/whip", data=b"", headers={"Content-Type": "application/sdp"}, timeout=5)
    check("whip 400 on empty", r.status_code == 400, f"got {r.status_code}")

    # V17: publisher A claims ownership, frames hit the sink
    pc_a, loc_a = await whip_publish()
    await wait_sunk_above(0, 15, "publisher A ingress")
    check("V17 ownership active", run_webrtc.ownership.is_active())
    check("V17 frames reach sink", True)

    # V19: output fps == pipe fps — WHEP viewer capped high (?fps=60) so the
    # output_version gate dominates; received frames over 4s should match the
    # sink's processed-frame count over the same window.
    pcv = RTCPeerConnection()
    gotv = asyncio.get_event_loop().create_future()

    @pcv.on("track")
    def _on_vtrack(track):
        if track.kind == "video" and not gotv.done():
            gotv.set_result(track)

    pcv.addTransceiver("video", direction="recvonly")
    await pcv.setLocalDescription(await pcv.createOffer())
    r = requests.post(f"{BASE}/whep?fps=60", data=pcv.localDescription.sdp.encode(),
                      headers={"Content-Type": "application/sdp"}, timeout=10)
    check("whep viewer during whip publish", r.status_code == 201, f"got {r.status_code}")
    await pcv.setRemoteDescription(RTCSessionDescription(sdp=r.text, type="answer"))
    vtrack = await asyncio.wait_for(gotv, 15)
    await asyncio.wait_for(vtrack.recv(), 15)
    s0, n = sunk["n"], 0
    t_end = time.time() + 4
    while time.time() < t_end:
        await asyncio.wait_for(vtrack.recv(), 15)
        n += 1
    dsunk = sunk["n"] - s0
    check("V19 output fps == pipe fps", dsunk > 0 and 0.6 <= n / dsunk <= 1.4,
          f"recv {n} vs sunk {dsunk}")
    await pcv.close()

    # V17: publisher B joins (waiter) and leaves — A keeps owning, sink keeps growing
    pc_b, loc_b = await whip_publish()
    await asyncio.sleep(1)
    r = requests.delete(f"{BASE}{loc_b}", timeout=5)
    check("publisher B delete 200", r.status_code == 200, f"got {r.status_code}")
    await pc_b.close()
    mark = sunk["n"]
    await wait_sunk_above(mark, 10, "A still owner after B join/leave")
    check("V17 waiter join/leave doesn't steal ownership", True)

    # V17: DELETE releases ownership; repeat -> 404
    r = requests.delete(f"{BASE}{loc_a}", timeout=5)
    check("publisher A delete 200", r.status_code == 200, f"got {r.status_code}")
    r = requests.delete(f"{BASE}{loc_a}", timeout=5)
    check("delete repeat 404", r.status_code == 404, f"got {r.status_code}")
    await pc_a.close()
    await wait_for(lambda: not run_webrtc.ownership.is_active(), 15, "ownership release")
    check("V17 delete releases ownership", True)
    check("whip registry empty", len(run_webrtc.whip_sessions) == 0)

    print(f"\nPASS {len(passed)} checks")


if __name__ == "__main__":
    run_webrtc.out_resolution = {"width": 512, "height": 512}
    run_webrtc._frame_sink = _counting_sink

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
        run_webrtc.sp = None  # fake sp has no .stop(); let _graceful_cleanup skip it
        server.should_exit = True
        t.join(timeout=10)
