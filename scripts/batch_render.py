"""Offline batch frame-for-frame render.

Runs a whole video through a DEDICATED StreamProcessor (batch_mode): exactly one
output frame per input frame, in order, with no drop / dup / fps-resampling /
interpolation (unlike the realtime live path). The render is temporally coherent
(each frame builds on the spatial cache of the ones before it, like the live
stream) and reproducible run-to-run for the same input+prompt+seed+steps — it is
NOT a per-frame-independent transform.

The dedicated instance shares no tensors / params / temporal caches with the live
WebRTC stream, so it cannot corrupt it (only GPU/VRAM is shared while a job runs).
The instance is created on submit and torn down on completion, so idle costs zero
GPU/VRAM. See docs/batch-render-spec.md.

The StreamProcessor is injected (`make_processor`) so the manager + video IO are
testable without CUDA (pass a stub that echoes frames).
"""

from __future__ import annotations

import copy
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, List, Optional

import av
import numpy as np


# ── Video IO (PyAV) ───────────────────────────────────────────────────────────
def decode_video(path: str) -> tuple[List[np.ndarray], Optional[Fraction]]:
    """Decode every frame in order. Returns (frames as uint8 RGB HxWx3, fps).
    fps is the exact rational rate (e.g. 30000/1001) so CFR timing is preserved."""
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        fps = stream.average_rate  # a Fraction, or None for unknown
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    finally:
        container.close()
    return frames, fps


class Mp4Encoder:
    """Streaming H.264 mp4 encoder — write() one RGB frame at a time (CFR), so the
    full output never has to be held in memory. close() is idempotent."""

    def __init__(self, path: str, fps, width: int, height: int):
        rate = fps if isinstance(fps, Fraction) else Fraction(fps or 25).limit_denominator(1000000)
        self._container = av.open(path, mode="w")
        self._stream = self._container.add_stream("libx264", rate=rate)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._closed = False

    def write(self, frame_rgb: np.ndarray) -> None:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_rgb), format="rgb24")
        for packet in self._stream.encode(vf):
            self._container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for packet in self._stream.encode():  # flush
            self._container.mux(packet)
        self._container.close()


# ── Job state ─────────────────────────────────────────────────────────────────
JOB_STATES = ("queued", "loading", "running", "encoding", "done", "error", "canceled")
_TERMINAL = ("done", "error", "canceled")


@dataclass
class BatchJob:
    id: str
    prompt: str
    seed: int
    steps: int
    fps: Optional[float]
    state: str = "queued"
    frames_total: int = 0
    frames_done: int = 0
    error: str = ""
    out_path: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    _cancel: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def status(self) -> dict:
        elapsed = time.time() - self.started_at
        eta = 0.0
        if self.frames_done and self.frames_total:
            per = elapsed / self.frames_done
            eta = max(0.0, per * (self.frames_total - self.frames_done))
        return {
            "id": self.id,
            "state": self.state,
            "frames_total": self.frames_total,
            "frames_done": self.frames_done,
            "fps": self.fps,
            "eta_s": round(eta, 1),
            "error": self.error,
        }


# make_processor(config_dict) -> object with:
#   start(), is_ready()->bool, set_prompt(str), set_seed(int), set_steps(int),
#   submit_frame(rgb)->rgb, stop(), and (optionally) worker_alive()->bool
ProcessorFactory = Callable[[dict], object]

# Bounded model-load wait: if the second model is not ready within this, the child
# almost certainly died (CUDA OOM) — fail the job instead of spinning forever.
_LOAD_TIMEOUT_S = 180.0


class BatchJobManager:
    """One job at a time (a second full model in VRAM). A worker thread spawns a
    dedicated batch StreamProcessor, renders frame-by-frame, streams the mp4, then
    tears the processor down. Finished jobs are retained (most-recent N) so their
    result can be downloaded; older ones are pruned and their files deleted."""

    def __init__(
        self,
        base_config: dict,
        make_processor: ProcessorFactory,
        out_dir: Optional[str] = None,
        preflight: Optional[Callable[[], None]] = None,
        max_retained: int = 20,
    ):
        self._base_config = base_config
        self._make = make_processor
        self._out_dir = out_dir or tempfile.gettempdir()
        self._preflight = preflight  # may raise RuntimeError (e.g. insufficient VRAM)
        self._max_retained = max_retained
        self._lock = threading.Lock()
        self._jobs: dict[str, BatchJob] = {}
        self._active: Optional[str] = None

    def submit(self, video_bytes: bytes, prompt: str, seed: int, steps: int, fps: Optional[float]) -> BatchJob:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("a batch render is already running")
            if self._preflight is not None:
                self._preflight()  # raises -> surfaced as 409 by the route
            self._prune_locked()
            job = BatchJob(id=uuid.uuid4().hex[:12], prompt=prompt, seed=seed, steps=steps, fps=fps)
            self._jobs[job.id] = job
            self._active = job.id
        job._thread = threading.Thread(target=self._run, args=(job, video_bytes), daemon=True)
        job._thread.start()
        return job

    def get(self, job_id: str) -> Optional[BatchJob]:
        return self._jobs.get(job_id)

    def active_job_id(self) -> Optional[str]:
        return self._active

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job._cancel.set()
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel the active job and give its worker a bounded chance to tear the
        batch processor down (so the 2nd model's subprocess + shared memory aren't
        orphaned at interpreter exit)."""
        active = self._active
        if not active:
            return
        self.cancel(active)
        job = self._jobs.get(active)
        thread = job._thread if job else None
        if thread is not None:
            thread.join(timeout)

    def _prune_locked(self) -> None:
        finished = [j for j in self._jobs.values() if j.state in _TERMINAL]
        if len(finished) <= self._max_retained:
            return
        finished.sort(key=lambda j: j.started_at)
        for j in finished[: len(finished) - self._max_retained]:
            if j.out_path and os.path.exists(j.out_path):
                try:
                    os.unlink(j.out_path)
                except OSError:
                    pass
            self._jobs.pop(j.id, None)

    def _batch_config(self) -> dict:
        # deepcopy so nested dicts (resolution, lip_transfer, …) are NOT shared with
        # the live sp.config — the batch instance must not mutate the live byte path.
        cfg = copy.deepcopy(self._base_config)
        cfg["batch_mode"] = True
        cfg["interpolation_exp"] = 0  # exactly 1 output per input (no RIFE tween)
        cfg["logging"] = False
        # A second model + LivePortrait would OOM; batch never lip-syncs.
        lp = cfg.get("lip_transfer")
        if isinstance(lp, dict):
            lp["enable"] = False
        return cfg

    def _run(self, job: BatchJob, video_bytes: bytes) -> None:
        proc = None
        encoder = None
        in_path = None
        out_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                in_path = f.name
            frames, src_fps = decode_video(in_path)
            if not frames:
                raise ValueError("no frames decoded from input video")
            job.frames_total = len(frames)
            out_fps = job.fps or src_fps or 25.0

            job.state = "loading"
            proc = self._make(self._batch_config())
            proc.start()
            # Wait for the model to load, but fail fast if the child dies (CUDA OOM)
            # or never becomes ready — otherwise this loop (and the single job slot)
            # would hang forever.
            alive = getattr(proc, "worker_alive", None)
            load_deadline = time.time() + _LOAD_TIMEOUT_S
            while not proc.is_ready():
                if job._cancel.is_set():
                    raise _Canceled()
                if alive is not None and not alive():
                    raise RuntimeError("batch inference subprocess died during model load (likely CUDA OOM)")
                if time.time() >= load_deadline:
                    raise RuntimeError("batch processor failed to become ready within the load timeout")
                time.sleep(0.1)
            # Params applied once, before the first frame (drained on the first render).
            proc.set_prompt(job.prompt)
            proc.set_seed(job.seed)
            proc.set_steps(job.steps)

            job.state = "running"
            out_path = os.path.join(self._out_dir, f"fluxrt-render-{job.id}.mp4")
            for i, fr in enumerate(frames):
                if job._cancel.is_set():
                    raise _Canceled()
                out = proc.submit_frame(fr)
                if encoder is None:  # size the encoder from the first output (may be upscaled)
                    encoder = Mp4Encoder(out_path, out_fps, out.shape[1], out.shape[0])
                encoder.write(out)
                job.frames_done = i + 1

            job.state = "encoding"
            encoder.close()
            encoder = None
            job.out_path = out_path
            job.state = "done"
        except _Canceled:
            job.state = "canceled"
        except Exception as exc:  # noqa: BLE001 — surface any failure as job state
            job.error = str(exc)
            job.state = "error"
        finally:
            if encoder is not None:
                try:
                    encoder.close()
                except Exception:
                    pass
            if proc is not None:
                try:
                    proc.stop()  # park: frees the 2nd model's VRAM/GPU when idle
                except Exception:
                    pass
            # A canceled/failed job keeps no file; only a completed render is retained.
            if job.state in ("canceled", "error") and out_path and os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
            if in_path and os.path.exists(in_path):
                try:
                    os.unlink(in_path)
                except OSError:
                    pass
            with self._lock:
                self._active = None


class _Canceled(Exception):
    pass
