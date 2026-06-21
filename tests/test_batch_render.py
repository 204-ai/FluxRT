"""GPU-free tests for the offline batch render: PyAV decode/encode round-trip and
the JobManager state machine against a STUB processor (no CUDA, no torch)."""

import os
import tempfile
import time

import av
import numpy as np
import pytest

from batch_render import BatchJobManager, Mp4Encoder, decode_video


# ── helpers ───────────────────────────────────────────────────────────────────
def make_mp4(path, n_frames=6, w=64, h=48, fps=10):
    """Write a synthetic CFR mp4 with `n_frames` distinct frames."""
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(n_frames):
        arr = np.full((h, w, 3), (i * 37) % 256, dtype=np.uint8)
        for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def make_mp4_bytes(**kw):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    try:
        make_mp4(path, **kw)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def count_frames(path):
    c = av.open(path)
    try:
        return sum(1 for _ in c.decode(c.streams.video[0]))
    finally:
        c.close()


class StubProcessor:
    """Stand-in for a batch StreamProcessor. Records lifecycle/config. By default
    echoes frames; `marker=True` emits a per-frame solid value (idx*8) so order is
    verifiable through the lossy codec; `out_scale` emits an upscaled output so the
    encoder's output-derived sizing is exercised."""

    def __init__(self, cfg, per_frame_sleep=0.0, fail=False, marker=False, out_scale=1):
        self.cfg = cfg
        self.per_frame_sleep = per_frame_sleep
        self.fail = fail
        self.marker = marker
        self.out_scale = out_scale
        self.started = self.stopped = False
        self.prompt = self.seed = self.steps = None
        self.n_frames = 0

    def start(self):
        self.started = True

    def is_ready(self):
        return True

    def worker_alive(self):
        return True

    def set_prompt(self, p):
        self.prompt = p

    def set_seed(self, s):
        self.seed = s

    def set_steps(self, s):
        self.steps = s

    def submit_frame(self, rgb):
        if self.per_frame_sleep:
            time.sleep(self.per_frame_sleep)
        if self.fail:
            raise RuntimeError("boom")
        idx = self.n_frames
        self.n_frames += 1
        h, w = rgb.shape[:2]
        if self.marker:
            return np.full((h * self.out_scale, w * self.out_scale, 3), (idx * 8) % 256, dtype=np.uint8)
        if self.out_scale != 1:
            return np.zeros((h * self.out_scale, w * self.out_scale, 3), dtype=np.uint8)
        return rgb.copy()  # echo: 1:1, preserves count

    def stop(self):
        self.stopped = True


def wait_until(pred, timeout=10.0, interval=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


# ── PyAV IO ───────────────────────────────────────────────────────────────────
def test_decode_then_encode_preserves_count_and_fps(tmp_path):
    src = str(tmp_path / "in.mp4")
    make_mp4(src, n_frames=8, fps=12)
    frames, fps = decode_video(src)
    assert len(frames) == 8
    assert frames[0].shape == (48, 64, 3) and frames[0].dtype == np.uint8
    assert round(fps) == 12

    out = str(tmp_path / "out.mp4")
    enc = Mp4Encoder(out, fps, frames[0].shape[1], frames[0].shape[0])
    for fr in frames:
        enc.write(fr)
    enc.close()
    enc.close()  # idempotent
    assert count_frames(out) == 8


# ── JobManager ────────────────────────────────────────────────────────────────
def _manager(stub_holder, base_config=None, **stub_kw):
    def make(cfg):
        stub = StubProcessor(cfg, **stub_kw)
        stub_holder.append(stub)
        return stub
    base = base_config if base_config is not None else {"resolution": {"height": 48, "width": 64}}
    return BatchJobManager(base_config=base, make_processor=make)


def _video_rate(path):
    c = av.open(path)
    try:
        return c.streams.video[0].average_rate
    finally:
        c.close()


def test_job_happy_path_renders_1to1(tmp_path):
    stubs = []
    mgr = _manager(stubs)
    job = mgr.submit(make_mp4_bytes(n_frames=6, fps=10), prompt="cat", seed=7, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state in ("done", "error"))
    j = mgr.get(job.id)
    assert j.state == "done", j.error
    assert j.frames_total == 6 and j.frames_done == 6
    assert count_frames(j.out_path) == 6           # exactly one output per input
    assert stubs[0].prompt == "cat" and stubs[0].seed == 7 and stubs[0].steps == 2
    assert stubs[0].started and stubs[0].stopped   # parked after the job


def test_second_submit_rejected_while_running():
    stubs = []
    mgr = _manager(stubs, per_frame_sleep=0.05)
    j1 = mgr.submit(make_mp4_bytes(n_frames=20), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(j1.id).state == "running")
    with pytest.raises(RuntimeError):
        mgr.submit(make_mp4_bytes(n_frames=5), prompt="b", seed=1, steps=2, fps=None)
    mgr.cancel(j1.id)
    assert wait_until(lambda: mgr.get(j1.id).state in ("canceled", "done"))


def test_cancel_midjob_sets_canceled_and_parks():
    stubs = []
    mgr = _manager(stubs, per_frame_sleep=0.05)
    job = mgr.submit(make_mp4_bytes(n_frames=40), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state == "running")
    mgr.cancel(job.id)
    assert wait_until(lambda: mgr.get(job.id).state == "canceled")
    assert wait_until(lambda: stubs[0].stopped)          # processor torn down
    assert mgr.active_job_id() is None                   # slot freed for the next job


def test_processor_error_sets_error_and_parks():
    stubs = []
    mgr = _manager(stubs, fail=True)
    job = mgr.submit(make_mp4_bytes(n_frames=4), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state == "error")
    assert "boom" in mgr.get(job.id).error
    assert wait_until(lambda: stubs[0].stopped)
    assert mgr.active_job_id() is None


def test_empty_video_errors_without_processor():
    stubs = []
    mgr = _manager(stubs)
    job = mgr.submit(b"not a video", prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state == "error")
    assert mgr.active_job_id() is None


def test_fps_override_and_source_fallback():
    stubs = []
    mgr = _manager(stubs)
    # explicit override wins over the 10 fps source
    j1 = mgr.submit(make_mp4_bytes(n_frames=5, fps=10), prompt="a", seed=1, steps=2, fps=24)
    assert wait_until(lambda: mgr.get(j1.id).state in ("done", "error"))
    assert mgr.get(j1.id).state == "done", mgr.get(j1.id).error
    assert round(float(_video_rate(mgr.get(j1.id).out_path))) == 24
    # None -> the source rate is used
    j2 = mgr.submit(make_mp4_bytes(n_frames=5, fps=10), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(j2.id).state == "done")
    assert round(float(_video_rate(mgr.get(j2.id).out_path))) == 10


def test_batch_config_overrides_and_isolates_base():
    base = {
        "resolution": {"height": 48, "width": 64},
        "interpolation_exp": 3,
        "logging": True,
        "lip_transfer": {"enable": True},
    }
    stubs = []
    mgr = _manager(stubs, base_config=base)
    job = mgr.submit(make_mp4_bytes(n_frames=3), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state == "done")
    cfg = stubs[0].cfg
    assert cfg["batch_mode"] is True
    assert cfg["interpolation_exp"] == 0
    assert cfg["logging"] is False
    assert cfg["lip_transfer"]["enable"] is False
    # the live base config (incl. its nested dict) is untouched
    assert base["interpolation_exp"] == 3
    assert base["logging"] is True
    assert base["lip_transfer"]["enable"] is True


def test_encoder_sized_from_output_not_input():
    stubs = []
    mgr = _manager(stubs, out_scale=2)  # 64x48 input -> 128x96 output
    job = mgr.submit(make_mp4_bytes(n_frames=4, w=64, h=48), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state == "done"), mgr.get(job.id).error
    c = av.open(mgr.get(job.id).out_path)
    try:
        st = c.streams.video[0]
        assert (st.codec_context.width, st.codec_context.height) == (128, 96)
    finally:
        c.close()


def test_output_order_preserved():
    stubs = []
    mgr = _manager(stubs, marker=True)  # frame k -> solid value k*8
    job = mgr.submit(make_mp4_bytes(n_frames=10), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state == "done")
    c = av.open(mgr.get(job.id).out_path)
    means = [float(f.to_ndarray(format="rgb24").mean()) for f in c.decode(c.streams.video[0])]
    c.close()
    assert len(means) == 10
    # strictly increasing means => frames came out in submission order (codec-noise tolerant)
    assert all(b > a for a, b in zip(means, means[1:])), means


def test_canceled_job_leaves_no_output_file():
    stubs = []
    mgr = _manager(stubs, per_frame_sleep=0.05)
    job = mgr.submit(make_mp4_bytes(n_frames=40), prompt="a", seed=1, steps=2, fps=None)
    assert wait_until(lambda: mgr.get(job.id).state == "running")
    mgr.cancel(job.id)
    assert wait_until(lambda: mgr.get(job.id).state == "canceled")
    j = mgr.get(job.id)
    assert not (j.out_path and os.path.exists(j.out_path))  # partial cleaned up


def test_preflight_rejection_blocks_submit():
    stubs = []
    mgr = _manager(stubs)

    def deny():
        raise RuntimeError("insufficient VRAM")

    mgr._preflight = deny
    with pytest.raises(RuntimeError, match="VRAM"):
        mgr.submit(make_mp4_bytes(n_frames=3), prompt="a", seed=1, steps=2, fps=None)
    assert mgr.active_job_id() is None  # rejected before claiming the slot
