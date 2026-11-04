#!/usr/bin/env python3
"""WebRTC ingress / egress benchmark, GPU-free.

Measures the part of the live path that runs on the CPU around the model:
WebRTC input -> shared memory -> (model) -> output scheduler -> shared memory
-> output pump -> per-viewer encode -> WebRTC output.

  stages    per-frame cost of each conversion step, old vs new, in-process:
            egress (shared memory -> encoder input), VP8 settings (ms, Mbps,
            PSNR at equal bitrate), ingress (decoded frame -> shared memory),
            shared-memory tearing under a concurrent writer.

  loopback  end to end, with the real server code of a checkout (--repo):
            run_webrtc's /offer, ingest, output pump, tracks and aiortc
            encoders, the real StreamProcessor shared memory and output
            scheduler. Only the diffusion model is replaced: a stub process
            speaking the same shared-memory protocol, frame times varying
            like the spatial cache's. A client process publishes a camera
            whose frames carry a Gray-coded id and reads the ids back from
            the output: glass-to-glass latency, generated frames that reach
            the viewer, output fps, keyframes, bitrate; psutil measures the
            CPU of every server process over the same window.

  compare   tables from loopback result files.

    python scripts/perf_webrtc.py stages
    git worktree add /tmp/fluxrt-main main
    python scripts/perf_webrtc.py loopback --repo /tmp/fluxrt-main --out runs/old.json
    python scripts/perf_webrtc.py loopback --repo . --out runs/new.json
    python scripts/perf_webrtc.py compare runs/old.json runs/new.json

Needs aiortc, av, opencv, numpy, fastapi, uvicorn, psutil (scikit-image for
photographic test content; synthetic otherwise). Nothing else may load the
CPU while it runs: the client, server and stub share the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import types

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent


# ── torch-free import of a checkout's fluxrt (also in spawned children) ──────
def _install_repo(repo: str) -> None:
    """Import `repo`'s src/fluxrt and scripts/ without fluxrt's torch-importing
    package __init__s (the same shim as tests/webrtc/conftest.py), with the
    stub below standing in for the model process module."""
    root = pathlib.Path(repo).resolve()
    for p in (str(root / "scripts"), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "fluxrt" not in sys.modules:
        pkg = types.ModuleType("fluxrt")
        pkg.__path__ = [str(root / "src" / "fluxrt")]
        sys.modules["fluxrt"] = pkg
        sp_pkg = types.ModuleType("fluxrt.stream_processor")
        sp_pkg.__path__ = [str(root / "src" / "fluxrt" / "stream_processor")]
        sys.modules["fluxrt.stream_processor"] = sp_pkg
        stub = types.ModuleType("fluxrt.stream_processor.model_inference_subprocess")
        stub.ModelInferenceSubprocess = StubModelInference
        sys.modules[stub.__name__] = stub


# ── frame ids drawn into the picture ─────────────────────────────────────────
ID_BITS = 14  # client camera frame id
GEN_BITS = 14  # stub's generated-frame counter (12 bits) + in-between index (2 bits)
BAND_ID = (0.40, 0.56)  # vertical band of the client's id (normalized)
BAND_GEN = (0.64, 0.76)  # band of the stub's counter + phase (interpolated index)


def gray(n: int) -> int:
    return n ^ (n >> 1)


def ungray(g: int) -> int:
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def stamp(img: np.ndarray, band, value: int, bits: int) -> None:
    """Draw `bits` bits of `value` (Gray code: a blend of two consecutive
    values is ambiguous in one bit only) as black / white blocks, between a
    black and a white calibration block."""
    h, w = img.shape[:2]
    y0, y1 = int(band[0] * h), int(band[1] * h)
    cells = bits + 2
    x0, x1 = int(0.06 * w), int(0.94 * w)
    g = gray(value)
    for i in range(cells):
        a, b = x0 + (x1 - x0) * i // cells, x0 + (x1 - x0) * (i + 1) // cells
        if i == 0:
            v = 0
        elif i == cells - 1:
            v = 255
        else:
            v = 255 if (g >> (bits - i)) & 1 else 0
        img[y0:y1, a:b] = v


def read_stamp(luma: np.ndarray, band, bits: int):
    """The value of a stamp in a luma plane, or None when a block is neither
    clearly black nor clearly white (a blend, or codec damage)."""
    h, w = luma.shape[:2]
    y0, y1 = int(band[0] * h), int(band[1] * h)
    cells = bits + 2
    x0, x1 = int(0.06 * w), int(0.94 * w)
    means = []
    for i in range(cells):
        a, b = x0 + (x1 - x0) * i // cells, x0 + (x1 - x0) * (i + 1) // cells
        q, r = (b - a) // 4, (y1 - y0) // 4
        means.append(float(luma[y0 + r : y1 - r, a + q : b - q].mean()))
    lo, hi = means[0], means[-1]
    if hi - lo < 60:
        return None
    g = 0
    for m in means[1:-1]:
        t = (m - lo) / (hi - lo)
        if 0.3 < t < 0.7:
            return None
        g = (g << 1) | (t >= 0.5)
    return ungray(g)


# ── the stub model process ───────────────────────────────────────────────────
class StubModelInference:
    """Plays ModelInferenceSubprocess on the shared-memory side: reads the
    newest input frame, "generates" for a frame time (a sleep: the GPU's
    share, no CPU), upscales 2x when the flow upscaler is configured, blends
    RIFE-style in-betweens with the previous frame, stamps its frame counter
    and the in-between index, and hands the pack over the way
    sync_fps_and_send does (last_processing_time, frames_generated, batch,
    pack_is_ready). The picture itself passes through unchanged."""

    def __init__(self, config, input_name, batch_name, pack_is_ready, last_processing_time, *_, **__):
        from multiprocessing import Value

        self.config = config
        self.input_name, self.batch_name = input_name, batch_name
        self.pack_is_ready, self.last_processing_time = pack_is_ready, last_processing_time
        self.running = Value("b", False)
        self.frames_generated = Value("L", 0)
        self.memory_reserved = Value("i", 0)
        self.proc_ready = Value("b", False)
        self.process = None

    def start(self):
        from multiprocessing import Process

        self.running.value = True
        self.process = Process(target=self.process_main, daemon=True)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process is not None:
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
            self.process = None

    def exitcode(self):
        return None if self.process is None else self.process.exitcode

    def set_param(self, *a, **k):
        pass

    def start_prompt_travel(self, *a, **k):
        pass

    def process_main(self):
        import cv2

        from fluxrt.utils.shared_tensor import SharedTensor

        c = self.config
        h, w = c["resolution"]["height"], c["resolution"]["width"]
        up = 2 if c.get("enable_flow_upscaler") else 1
        n = 2 ** c.get("interpolation_exp", 1)
        inp = SharedTensor((h, w, 3), name=self.input_name)
        batch = SharedTensor((n, h * up, w * up, 3), name=self.batch_name)
        base = c.get("bench_frame_ms", 40) / 1000.0
        pattern = c.get("bench_frame_pattern", [1.0])
        prev, k = None, 0
        prev_time = time.time()
        frame = np.empty((h, w, 3), np.uint8)
        trace = open(c["bench_trace"], "a") if c.get("bench_trace") else None
        pack = np.empty((n, h * up, w * up, 3), np.uint8)
        self.proc_ready.value = True
        while self.running.value:
            # what condition_from_input does (seqlock read where the checkout has it)
            if hasattr(inp, "read_into"):
                inp.read_into(frame)
            else:
                np.copyto(frame, inp.array)
            t_read = time.time()
            time.sleep(base * pattern[k % len(pattern)])
            out = cv2.resize(frame, (w * up, h * up), interpolation=cv2.INTER_LINEAR) if up > 1 else frame
            last = prev if prev is not None else out
            for i in range(n):
                a = (i + 1) / n
                pack[i] = out if i == n - 1 else cv2.addWeighted(last, 1 - a, out, a, 0)
                stamp(pack[i], BAND_GEN, ((k % (1 << (GEN_BITS - 2))) << 2) | ((i + 1) % n), GEN_BITS)
            prev = out.copy()
            now = time.time()
            self.last_processing_time.value = now - prev_time
            prev_time = now
            self.frames_generated.value += 1
            batch.copy_from(pack)
            self.pack_is_ready.value = True
            if trace is not None:  # which camera frame this generated frame came from, and when
                fid = read_stamp(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), BAND_ID, ID_BITS)
                trace.write(json.dumps({"gen": k % (1 << (GEN_BITS - 2)), "id": fid, "read": t_read, "written": time.time()}) + "\n")
                trace.flush()
            k += 1


if os.environ.get("FLUXRT_BENCH_REPO"):  # spawned children of the bench server
    _install_repo(os.environ["FLUXRT_BENCH_REPO"])


# ── test content ─────────────────────────────────────────────────────────────
class Content:
    """Moving photographic content (a pan and slow zoom over a mosaic of
    scikit-image's sample photos), synthetic texture without scikit-image."""

    def __init__(self):
        import cv2

        try:
            import skimage.data as d

            tiles = [d.astronaut(), d.coffee(), d.rocket(), d.chelsea(), d.immunohistochemistry(), d.hubble_deep_field()[:512, :600]]
            rows = []
            for row in (tiles[:3], tiles[3:]):
                rows.append(np.hstack([cv2.resize(t, (int(t.shape[1] * 512 / t.shape[0]), 512), interpolation=cv2.INTER_AREA) for t in row]))
            width = min(r.shape[1] for r in rows)
            self.canvas = np.vstack([r[:, :width] for r in rows])[:, :, ::-1].copy()
            self.kind = "photos"
        except Exception:
            rng = np.random.default_rng(0)
            base = rng.integers(0, 256, (64, 112, 3), dtype=np.uint8)
            self.canvas = cv2.resize(cv2.GaussianBlur(cv2.resize(base, (1792, 1024), interpolation=cv2.INTER_CUBIC), (0, 0), 2), (1792, 1024))
            self.kind = "synthetic"

    def frame(self, h: int, w: int, t: float) -> np.ndarray:
        import cv2

        ch, cw = self.canvas.shape[:2]
        zoom = 0.75 + 0.1 * np.sin(t / 40)
        vh = int(ch * zoom * 0.8)
        vw = min(int(vh * w / h), cw)
        vh = int(vw * h / w)
        x = int((cw - vw) * (0.5 + 0.5 * np.sin(t / 25)))
        y = int((ch - vh) * (0.5 + 0.5 * np.cos(t / 33)))
        return cv2.resize(self.canvas[y : y + vh, x : x + vw], (w, h), interpolation=cv2.INTER_AREA)


# ── stages ───────────────────────────────────────────────────────────────────
def _bench(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255**2 / mse) if mse else float("inf")


def cmd_stages(args) -> None:
    import fractions

    import av
    import cv2

    _install_repo(str(_HERE.parent))
    from fluxrt.utils.crop_maximal_rectangle import crop_maximal_rectangle
    from fluxrt.utils.shared_tensor import SharedTensor
    from fluxrt.webrtc import frame_io, vp8_encoder

    content = Content()
    print(f"cv2 {cv2.__version__} ({cv2.getNumThreads()} threads), PyAV {av.__version__}, content: {content.kind}\n")

    if args.only in (None, 'egress'):
        print("## Egress: one output frame from shared memory to the encoder's input, per viewer\n")
        print("| output | viewers | old (copy, BGR->RGB, rgb24 frame, encoder's swscale) | new (seqlock copy, I420 once, yuv420p frame) | speed-up |")
        print("|---|---:|---:|---:|---:|")
        for h, w in ((320, 576), (640, 1152)):
            shm = SharedTensor((h, w, 3), create=True)
            shm.copy_from(content.frame(h, w, 3))
            conv = frame_io.I420Converter()
            for viewers in (1, 3):
                def old():
                    rgb = cv2.cvtColor(shm.array.copy(), cv2.COLOR_BGR2RGB)
                    for _ in range(viewers):
                        av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")

                def new():
                    out = frame_io.OutputFrame(1, shm.to_numpy())
                    out.i420(conv)
                    for _ in range(viewers):
                        out.video_frame()

                t_old, t_new = _bench(old), _bench(new)
                print(f"| {w}x{h} | {viewers} | {t_old:.2f} ms | {t_new:.2f} ms | {t_old / t_new:.1f}x |")
            shm.close_and_unlink()

    if args.only in (None, 'encoder'):
        # run_webrtc lifts aiortc's 1.5 Mbps VP8 ceiling at import; so does this
        from aiortc.codecs import vpx as _vpx

        _vpx.MAX_BITRATE = max(_vpx.MAX_BITRATE, 20_000_000)
        print("\n## VP8 encoder settings, photographic content with camera motion\n")
        print("| frame | bitrate | settings | ms / frame (median) | Mbps | PSNR |")
        print("|---|---:|---|---:|---:|---:|")
        stock = {"noise-sensitivity": "4"}
        for h, w, fps, rate in ((320, 576, 60, 8_000_000), (640, 1152, 24, 8_000_000), (640, 1152, 24, 3_000_000)):
            yuv = [frame_io.I420Converter()(content.frame(h, w, t)) for t in range(args.encoder_frames)]
            for label, opts in (("aiortc (denoiser 4)", stock), ("new default (denoiser 0)", {}), ("denoiser 0, cpu-used -10", {"cpu-used": "-10"})):
                cls = vp8_encoder.make_encoder_class({**vp8_encoder._DEFAULTS, **opts})
                enc = cls()
                enc.target_bitrate = rate
                dec = av.CodecContext.create("libvpx", "r")
                from aiortc.codecs.vpx import vp8_depayload

                times, size, psnrs = [], 0, []
                for i, src in enumerate(yuv):
                    f = av.VideoFrame.from_ndarray(src, format="yuv420p")
                    f.time_base = fractions.Fraction(1, 90000)
                    f.pts = int(i * 90000 / fps)
                    t0 = time.perf_counter()
                    payloads, _ = enc.encode(f)
                    times.append(time.perf_counter() - t0)
                    data = b"".join(vp8_depayload(p) for p in payloads)
                    size += len(data)
                    for out in dec.decode(av.Packet(data)):
                        if i >= 20:
                            psnrs.append(_psnr(out.to_ndarray()[:h], src[:h]))
                print(f"| {w}x{h} @ {fps} | {rate / 1e6:g} Mbps | {label} | {np.median(times[20:]) * 1e3:.2f} | {size * 8 * fps / len(yuv) / 1e6:.2f} | {np.mean(psnrs):.2f} dB |")

    if args.only in (None, 'ingress'):
        print("\n## Ingress: a decoded yuv420p frame into the model's input shared memory (576x320)\n")
        print("| input | old (to BGR, crop_maximal_rectangle, copy) | new (to BGR, crop + resize into a buffer, seqlock copy) | identical |")
        print("|---|---:|---:|---|")
        H, W = 320, 576
        ingress = frame_io.IngressConverter(H, W)
        shm = SharedTensor((H, W, 3), create=True)
        buf = np.empty((H, W, 3), np.uint8)
        for ih, iw in ((320, 576), (360, 640), (720, 1280), (1080, 1920)):
            vf = av.VideoFrame.from_ndarray(content.frame(ih, iw, 5), format="bgr24").reformat(format="yuv420p")

            def old():
                shm.copy_from(crop_maximal_rectangle(vf.to_ndarray(format="bgr24"), H, W))

            def new():
                ingress.write(vf, buf)
                shm.copy_from(buf)

            t_old, t_new = _bench(old, n=100), _bench(new, n=100)
            old()
            a = shm.to_numpy()
            new()
            print(f"| {iw}x{ih} | {t_old:.2f} ms | {t_new:.2f} ms | {np.array_equal(a, shm.to_numpy())} |")
        shm.close_and_unlink()

    if args.only in (None, 'shm'):
        print("\n## Shared memory: reads racing a writer in another process (1152x640 frames)\n")
        sys.path.insert(0, str(_HERE.parent / "tests"))
        import test_shared_tensor as tst

        print("| writer | reads | torn, unversioned (old) | torn, seqlock (new) |")
        print("|---|---:|---:|---:|")
        for gap, label in ((1 / 60, "60 Hz"), (0.001, "every 1 ms")):
            old = tst.torn_frames(False, gap, seconds=args.shm_seconds)
            new = tst.torn_frames(True, gap, seconds=args.shm_seconds)
            print(f"| {label} | {old['reads']} / {new['reads']} | {old['torn']} ({100 * old['torn'] / old['reads']:.2f}%) | {new['torn']} |")


# ── loopback: server ─────────────────────────────────────────────────────────
def cmd_server(args) -> None:
    import threading

    os.environ["FLUXRT_BENCH_REPO"] = args.repo  # for the spawned stub / scheduler
    _install_repo(args.repo)
    from fluxrt.stream_processor.stream_processor import StreamProcessor

    sys.modules["fluxrt"].StreamProcessor = StreamProcessor
    import uvicorn

    import run_webrtc

    with open(args.config) as f:
        config = json.load(f)
    sp = StreamProcessor(config)
    run_webrtc.sp = sp
    sp.start()
    run_webrtc.input_tensor = sp.get_input_tensor()
    run_webrtc.output_tensor = sp.get_output_tensor()
    run_webrtc.resolution = sp.get_resolution()
    run_webrtc.out_resolution = sp.get_out_resolution()
    if hasattr(run_webrtc, "publisher"):
        run_webrtc.publisher.i420 = bool(sp.config.get("egress_i420", True))
    run_webrtc._output_pump_on = True
    threading.Thread(target=run_webrtc.output_pump, daemon=True).start()
    run_webrtc.ownership.has_server_camera = False  # --no-server-camera
    with open(args.pids, "w") as f:  # so the driver can tell the children apart
        json.dump({"stub": sp.model_inference_subprocess.process.pid,
                   "scheduler": sp.output_scheduler_subprocess.process.pid}, f)
    uvicorn.run(run_webrtc.app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=3)


# ── loopback: client ─────────────────────────────────────────────────────────
async def _client(args) -> dict:
    import asyncio

    import av
    import httpx
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

    # the client stands in for a browser, whose receiver decodes a frame on
    # its last packet: the same for every checkout measured
    _install_repo(str(_HERE.parent))
    from fluxrt.webrtc import jitter

    jitter.install()
    content = Content()
    w, h = (int(v) for v in args.input_size.split("x"))
    frames = [content.frame(h, w, t) for t in range(90)]
    sent: dict[int, float] = {}

    import fractions

    fps = args.input_fps

    class Camera(VideoStreamTrack):
        def __init__(self):
            super().__init__()
            self.n = 0
            self.start = None

        async def recv(self):
            if self.start is None:
                self.start = time.time()
            wait = self.start + self.n / fps - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            pts, time_base = int(self.n * 90000 / fps), fractions.Fraction(1, 90000)
            img = frames[self.n % len(frames)].copy()
            fid = self.n % (1 << ID_BITS)
            stamp(img, BAND_ID, fid, ID_BITS)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts, frame.time_base = pts, time_base
            sent[fid] = time.time()
            self.n += 1
            return frame

    measuring = {"on": False}
    records = []

    async def read(track, viewer: int):
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            now = time.time()
            if not measuring["on"]:
                continue
            luma = frame.to_ndarray()[: frame.height]  # yuv420p: the Y plane
            fid = read_stamp(luma, BAND_ID, ID_BITS)
            gen = read_stamp(luma, BAND_GEN, GEN_BITS)
            records.append({
                "viewer": viewer, "t": now, "key": bool(frame.key_frame),
                "id": fid, "gen": None if gen is None else gen >> 2, "phase": None if gen is None else gen & 3,
                "latency": None if fid is None or fid not in sent else now - sent[fid],
            })

    pcs = []
    async with httpx.AsyncClient(base_url=args.url, timeout=20) as http:
        for _ in range(60):
            try:
                if (await http.get("/healthz")).json().get("ready"):
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        pc = RTCPeerConnection()
        pcs.append(pc)
        pc.addTrack(Camera())
        pc.on("track", lambda track: asyncio.ensure_future(read(track, 0)) if track.kind == "video" else None)
        ctrl = pc.createDataChannel("ctrl")
        # every published frame, as the web client asks with out-fps:0 (the
        # /offer track caps at 30 otherwise)
        ctrl.on("open", lambda: ctrl.send("out-fps:0"))
        await pc.setLocalDescription(await pc.createOffer())
        r = await http.post("/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
        r.raise_for_status()
        await pc.setRemoteDescription(RTCSessionDescription(**r.json()))
        for v in range(1, args.viewers):
            wpc = RTCPeerConnection()
            pcs.append(wpc)
            wpc.addTransceiver("video", direction="recvonly")
            wpc.on("track", lambda track, v=v: asyncio.ensure_future(read(track, v)))
            await wpc.setLocalDescription(await wpc.createOffer())
            r = await http.post("/whep?fps=60", content=wpc.localDescription.sdp.encode(), headers={"Content-Type": "application/sdp"})
            r.raise_for_status()
            await wpc.setRemoteDescription(RTCSessionDescription(sdp=r.text, type="answer"))
        await asyncio.sleep(args.warmup)
        before = (await http.get("/healthz")).json()
        measuring["on"] = True
        print("MEASURE_START", flush=True)
        t0 = time.time()
        await asyncio.sleep(args.seconds)
        measuring["on"] = False
        print("MEASURE_END", flush=True)
        t1 = time.time()
        after = (await http.get("/healthz")).json()
    for pc in pcs:
        await pc.close()
    return {"records": records, "sent": {str(k): v for k, v in sent.items()}, "t0": t0, "t1": t1,
            "healthz_before": before, "healthz_after": after, "content": content.kind}


def cmd_client(args) -> None:
    import asyncio

    result = asyncio.run(_client(args))
    with open(args.out, "w") as f:
        json.dump(result, f)


# ── loopback: driver ─────────────────────────────────────────────────────────
PRESETS = {
    # the show laptop's production config: 576x320 generated, flow upscaler
    # (1152x640 out), RIFE x2, ~12 generated fps
    "production": {"resolution": {"height": 320, "width": 576}, "enable_flow_upscaler": True,
                   "interpolation_exp": 1, "bench_frame_ms": 80},
    # the 4090's fast presets: 576x320 out, RIFE x2, ~30 generated fps
    "fast": {"resolution": {"height": 320, "width": 576}, "enable_flow_upscaler": False,
             "interpolation_exp": 1, "bench_frame_ms": 33},
}
# frame-time multipliers: the spatial cache makes consecutive frames vary 2x
PATTERN = [1.0, 0.45, 1.3, 0.6, 1.4, 0.5, 1.2, 0.55]


def _cpu_seconds(pids: dict) -> dict:
    import psutil

    out = {}
    for role, pid in pids.items():
        try:
            t = psutil.Process(pid).cpu_times()
            out[role] = t.user + t.system
        except psutil.Error:
            pass
    return out


def _unwrap(values, modulus: int):
    """Counters that wrap at `modulus`, in arrival order, made monotonic."""
    out, offset, last = [], 0, None
    for v in values:
        if last is not None and v + offset < last - modulus // 2:
            offset += modulus
        last = v + offset
        out.append(last)
    return out


def summarize(raw: dict) -> dict:
    recs = sorted((r for r in raw["records"] if r["viewer"] == 0), key=lambda r: r["t"])
    dur = raw["t1"] - raw["t0"]
    lat = np.array([r["latency"] for r in recs if r["phase"] == 0 and r["latency"] is not None]) * 1e3
    gens = _unwrap([r["gen"] for r in recs if r["gen"] is not None], 1 << (GEN_BITS - 2))
    shown = {g for g, r in zip(gens, [r for r in recs if r["gen"] is not None]) if r["phase"] == 0}
    span = (max(gens) - min(gens) + 1) if gens else 0
    cpu = raw["cpu"]
    # latency split (same clock in every process): camera frame sent -> the
    # model read it -> its pack was written -> the viewer received it
    parts = {"ingress": [], "model": [], "egress": []}
    sent = raw.get("sent", {})
    by_gen = {}
    for e in raw.get("trace", []):  # the newest entry per counter value
        by_gen[e["gen"]] = e
    for r in recs:
        e = by_gen.get(r["gen"]) if r["phase"] == 0 else None
        if e is None or e["id"] is None or e["id"] != r["id"] or str(e["id"]) not in sent:
            continue
        if not (raw["t0"] - 5 < e["written"] < r["t"]):
            continue
        parts["ingress"].append(e["read"] - sent[str(e["id"])])
        parts["model"].append(e["written"] - e["read"])
        parts["egress"].append(r["t"] - e["written"])
    split = {f"{k}_ms": float(np.median(v) * 1e3) if v else None for k, v in parts.items()}
    return {
        "label": raw["label"],
        "preset": raw["preset"], "input": raw["input_size"], "viewers": raw["viewers"],
        "out_fps": len(recs) / dur,
        "generated_fps": span / dur,
        "generated_shown_pct": 100.0 * len(shown) / span if span else 0.0,
        "latency_p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else None,
        "unreadable_pct": 100.0 * sum(r["gen"] is None for r in recs) / max(1, len(recs)),
        "keyframes": sum(r["key"] for r in raw["records"]),
        "server_cores": cpu.get("server", 0.0) / dur,
        "scheduler_cores": cpu.get("scheduler", 0.0) / dur,
        "stub_cores": cpu.get("stub", 0.0) / dur,
        "mbps_out": raw.get("mbps_out"),
        **split,
    }


def cmd_loopback(args) -> None:
    import socket
    import subprocess
    import tempfile

    preset = dict(PRESETS[args.preset])
    preset.setdefault("bench_frame_pattern", PATTERN)
    config = {"default_prompt": "x", "default_steps": 2, "default_seed": 0, "logging": False, **preset}
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="perf_webrtc_"))
    config["bench_trace"] = str(tmp / "trace.jsonl")
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            config[key] = json.loads(raw)
        except json.JSONDecodeError:
            config[key] = raw
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (tmp / "config.json").write_text(json.dumps(config))
    env = dict(os.environ, FLUXRT_STUN="none")
    me = str(pathlib.Path(__file__).resolve())
    server = subprocess.Popen([sys.executable, me, "_server", "--repo", args.repo, "--port", str(port),
                               "--config", str(tmp / "config.json"), "--pids", str(tmp / "pids.json")], env=env)
    try:
        client = subprocess.Popen(
            [sys.executable, me, "_client", "--url", f"http://127.0.0.1:{port}", "--seconds", str(args.seconds),
             "--warmup", str(args.warmup), "--input-size", args.input_size, "--input-fps", str(args.input_fps),
             "--viewers", str(args.viewers),
             "--out", str(tmp / "client.json")],
            stdout=subprocess.PIPE, text=True, env=env,
        )
        cpu0 = cpu1 = None
        for line in client.stdout:
            if line.startswith("MEASURE_START"):
                pids = {"server": server.pid, **json.loads((tmp / "pids.json").read_text())}
                cpu0 = _cpu_seconds(pids)
            elif line.startswith("MEASURE_END"):
                cpu1 = _cpu_seconds(pids)
                break
        client.wait(60)
        if cpu1 is None or client.returncode != 0:
            raise SystemExit(f"client failed (exit {client.returncode})")
    finally:
        server.terminate()
        try:
            server.wait(20)
        except subprocess.TimeoutExpired:
            server.kill()
    raw = json.loads((tmp / "client.json").read_text())
    trace = tmp / "trace.jsonl"
    raw["trace"] = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    net = (raw["healthz_after"] or {}).get("net") or {}
    raw.update({
        "label": args.label or pathlib.Path(args.repo).resolve().name, "repo": str(pathlib.Path(args.repo).resolve()),
        "preset": args.preset, "input_size": f"{args.input_size}@{args.input_fps:g}", "viewers": args.viewers, "config": config,
        "cpu": {k: cpu1[k] - cpu0[k] for k in cpu1 if k in cpu0},
        "mbps_out": net.get("out_mbps"),
    })
    raw["summary"] = summarize(raw)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(raw, f)
    print(json.dumps(raw["summary"], indent=2))


def cmd_compare(args) -> None:
    rows = [json.loads(pathlib.Path(p).read_text())["summary"] for p in args.runs]
    cols = [
        ("label", "run", "{}"), ("preset", "preset", "{}"), ("input", "input", "{}"), ("viewers", "viewers", "{}"),
        ("server_cores", "server CPU (cores)", "{:.3f}"), ("scheduler_cores", "scheduler CPU", "{:.3f}"),
        ("out_fps", "output fps", "{:.1f}"), ("generated_fps", "generated fps", "{:.1f}"),
        ("generated_shown_pct", "generated frames shown", "{:.1f}%"),
        ("latency_p50_ms", "latency p50 (ms)", "{:.1f}"), ("latency_p95_ms", "latency p95 (ms)", "{:.1f}"),
        ("ingress_ms", "of which ingress", "{:.1f}"), ("model_ms", "model (stub)", "{:.1f}"), ("egress_ms", "egress", "{:.1f}"),
        ("keyframes", "keyframes", "{}"), ("unreadable_pct", "unreadable", "{:.1f}%"),
    ]
    print("| " + " | ".join(c[1] for c in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join("—" if r.get(k) is None else f.format(r[k]) for k, _, f in cols) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stages")
    s.add_argument("--encoder-frames", type=int, default=150)
    s.add_argument("--shm-seconds", type=float, default=2.0)
    s.add_argument("--only", choices=["egress", "encoder", "ingress", "shm"], help="run one section")
    s = sub.add_parser("loopback")
    s.add_argument("--repo", default=str(_HERE.parent))
    s.add_argument("--label")
    s.add_argument("--preset", choices=sorted(PRESETS), default="production")
    s.add_argument("--input-size", default="640x360", help="client camera WxH (16:9: the band layout assumes little crop)")
    s.add_argument("--input-fps", type=float, default=30, help="client camera frame rate (the kiosk sends ~1.5x the generated fps)")
    s.add_argument("--viewers", type=int, default=1, help="viewers in total: the /offer peer plus WHEP viewers")
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--warmup", type=float, default=6)
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (JSON value)")
    s.add_argument("--out")
    s = sub.add_parser("compare")
    s.add_argument("runs", nargs="+")
    s = sub.add_parser("_server")
    s.add_argument("--repo", required=True)
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--pids", required=True)
    s = sub.add_parser("_client")
    s.add_argument("--url", required=True)
    s.add_argument("--seconds", type=float, required=True)
    s.add_argument("--warmup", type=float, required=True)
    s.add_argument("--input-size", required=True)
    s.add_argument("--input-fps", type=float, default=30)
    s.add_argument("--viewers", type=int, default=1)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"stages": cmd_stages, "loopback": cmd_loopback, "compare": cmd_compare, "_server": cmd_server, "_client": cmd_client}[args.cmd](args)


if __name__ == "__main__":
    main()
