"""Offline batch frame-for-frame render.

Runs a whole video through a DEDICATED StreamProcessor (batch_mode): exactly one
output frame per input frame, in order, with no drop / dup / fps-resampling /
interpolation (unlike the realtime live path). The render is temporally coherent
(each frame builds on the spatial cache of the ones before it, like the live
stream) and reproducible run-to-run for the same input+prompt+seed+steps — it is
NOT a per-frame-independent transform.

The dedicated instance shares no tensors / params / temporal caches with the live
WebRTC stream, so it cannot corrupt it (only GPU/VRAM is shared while a job runs).
By default the instance is created on submit and torn down on completion, so idle
costs zero GPU/VRAM; with keep-warm (default under --batch-only) it stays loaded
between jobs and is reset() instead. See docs/batch-render-spec.md.

Decode (+ crop to model resolution) and encode each run on their own thread
behind a small bounded queue, so the GPU render never waits on x264 or on PyAV.

The StreamProcessor is injected (`make_processor`) so the manager + video IO are
testable without CUDA (pass a stub that echoes frames).
"""

from __future__ import annotations

import copy
import functools
import io
import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Iterator, List, Optional

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
        frames = list(iter_video_frames(container, stream))
    finally:
        container.close()
    return frames, fps


def iter_video_frames(container, stream) -> Iterator[np.ndarray]:
    """Decode one frame at a time (uint8 RGB HxWx3), in order."""
    for f in container.decode(stream):
        yield f.to_ndarray(format="rgb24")


@functools.lru_cache(maxsize=None)
def _codec_works(name: str) -> bool:
    """Can this PyAV/FFmpeg build (and machine) actually encode with `name`? An
    nvenc encoder can be compiled in and still fail to open without a driver, so
    encode a tiny clip to memory rather than only looking the codec up."""
    try:
        container = av.open(io.BytesIO(), mode="w", format="mp4")
        try:
            stream = container.add_stream(name, rate=25)
            stream.width, stream.height, stream.pix_fmt = 256, 256, "yuv420p"
            frame = av.VideoFrame.from_ndarray(np.zeros((256, 256, 3), np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        return True
    except Exception:
        return False


def resolve_encoder(name: Optional[str]) -> str:
    """The `batch_encoder` config value, or libx264 when it can't be used here."""
    name = name or "libx264"
    if name == "libx264" or _codec_works(name):
        return name
    print(f"batch encoder {name} unavailable; using libx264")
    return "libx264"


class Mp4Encoder:
    """Streaming H.264 mp4 encoder — write() one RGB frame at a time (CFR), so the
    full output never has to be held in memory. close() is idempotent."""

    def __init__(self, path: str, fps, width: int, height: int, codec: str = "libx264"):
        rate = fps if isinstance(fps, Fraction) else Fraction(fps or 25).limit_denominator(1000000)
        self._container = av.open(path, mode="w")
        self._stream = self._container.add_stream(codec, rate=rate)
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


# ── Overlapped IO ─────────────────────────────────────────────────────────────
# Frames buffered between each IO thread and the render loop. Small: a 1440p RGB
# frame is 11 MB, and the render is the slow stage — the queues only have to
# cover jitter so the GPU never waits on decode or encode.
_QUEUE_DEPTH = 4
_END = object()

# FLUXRT_PROFILE=1: per-stage timings, averaged and printed every _PROFILE_EVERY
# rendered frames (the inference child prints its own render / download split).
_PROFILE = os.environ.get("FLUXRT_PROFILE", "") == "1"
_PROFILE_EVERY = 24


class _Spans:
    """Mean ms per stage since the last print. Stages are added from the decode,
    encode and job threads; a lost update under a race only skews one sample."""

    def __init__(self):
        self._sums: dict[str, list] = {}

    def add(self, name: str, seconds: float) -> None:
        acc = self._sums.setdefault(name, [0.0, 0])
        acc[0] += seconds
        acc[1] += 1

    def maybe_print(self, frames_done: int) -> None:
        if frames_done % _PROFILE_EVERY:
            return
        sums, self._sums = self._sums, {}
        parts = [f"{k} {1000 * s / n:.1f}" for k, (s, n) in sums.items() if n]
        print(f"batch profile @ frame {frames_done} (ms/frame): " + ", ".join(parts))


class _Decoder:
    """Decodes the input on its own thread and prepares each frame (crop to the
    model resolution) ahead of the render, a bounded number of frames at a time."""

    def __init__(self, path: str, prepare: Callable[[np.ndarray], np.ndarray], spans: Optional[_Spans]):
        self._container = av.open(path)  # raises here for a non-video upload
        try:
            stream = self._container.streams.video[0]
        except Exception:
            self._container.close()
            raise
        self.fps = stream.average_rate  # a Fraction, or None for unknown
        self.total = int(stream.frames or 0)  # 0 = not in the header
        self.count = 0
        self._stream = stream
        self._prepare = prepare
        self._spans = spans
        self._q: queue.Queue = queue.Queue(_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="batch-decode", daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _loop(self) -> None:
        try:
            frames = iter_video_frames(self._container, self._stream)
            while True:
                t0 = time.perf_counter()
                frame = next(frames, None)
                if frame is None:
                    break
                t1 = time.perf_counter()
                frame = self._prepare(frame)
                if self._spans is not None:
                    self._spans.add("decode", t1 - t0)
                    self._spans.add("crop", time.perf_counter() - t1)
                self.count += 1
                if not self._put(frame):
                    return
            self._put(_END)
        except Exception as exc:  # surfaced to the job by get()
            self._put(exc)

    def get(self, cancel: threading.Event) -> Optional[np.ndarray]:
        """The next prepared frame, or None at the end of the video. Raises the
        decoder's exception, or _Canceled when the job is canceled while waiting."""
        while True:
            if cancel.is_set():
                raise _Canceled()
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _END:
                return None
            if isinstance(item, Exception):
                raise item
            return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self._container.close()


class _EncoderThread:
    """Mp4Encoder.write on its own thread behind a bounded queue, so x264 encodes
    frame n while the GPU renders frame n+1. Frames are written in put() order.
    An encoder error is raised from the next put() or from close()."""

    def __init__(self, make_encoder: Callable[[np.ndarray], Mp4Encoder], spans: Optional[_Spans]):
        self._make = make_encoder  # sized from the first frame (the output may be upscaled)
        self._enc: Optional[Mp4Encoder] = None
        self._spans = spans
        self._error: Optional[BaseException] = None
        self._q: queue.Queue = queue.Queue(_QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._loop, name="batch-encode", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is _END:
                break
            if self._error is not None:
                continue  # keep draining so put() never blocks on a dead encoder
            try:
                t0 = time.perf_counter()
                if self._enc is None:
                    self._enc = self._make(item)
                self._enc.write(item)
                if self._spans is not None:
                    self._spans.add("encode", time.perf_counter() - t0)
            except BaseException as exc:
                self._error = exc
        if self._enc is not None:
            try:
                self._enc.close()  # flush; also releases the file on abort
            except BaseException as exc:
                if self._error is None:
                    self._error = exc

    def put(self, frame: np.ndarray) -> None:
        if self._error is not None:
            raise self._error
        self._q.put(frame)

    def close(self) -> None:
        """Encode everything queued, flush and close; raises an encoder error."""
        self._q.put(_END)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def abort(self) -> None:
        """Drop what's queued and close the file (canceled / failed job)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._error = self._error or _Canceled()
        self._q.put(_END)
        self._thread.join()


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
    interp: int = 0  # RIFE interpolation exponent: 0 = 1:1, k>0 = 2**k frames per input
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
            "interp": self.interp,
            "eta_s": round(eta, 1),
            "error": self.error,
        }


# make_processor(config_dict) -> object with:
#   start(), is_ready()->bool, set_prompt(str), set_seed(int), set_steps(int),
#   submit_frame(rgb)->rgb, stop(), and optionally worker_alive()->bool,
#   worker_exitcode()->int|None and reset() (required for keep-warm reuse)
ProcessorFactory = Callable[[dict], object]

# Bounded model-load wait: if the second model is not ready within this, the child
# almost certainly hung. A child that dies (CUDA OOM) is caught at once by
# worker_alive(); the bound is generous because loading includes the compile
# warm-up when compile_models is on.
_LOAD_TIMEOUT_S = 600.0


@dataclass
class _Warm:
    proc: object
    interp: int
    timer: Optional[threading.Timer] = None


class BatchJobManager:
    """One job at a time (a second full model in VRAM). A worker thread spawns a
    dedicated batch StreamProcessor, renders frame-by-frame, streams the mp4, then
    tears the processor down — or, with keep_warm, parks it for the next job with
    the same interp (reset between jobs). Finished jobs are retained (most-recent N)
    so their result can be downloaded; older ones are pruned and their files deleted."""

    def __init__(
        self,
        base_config: dict,
        make_processor: ProcessorFactory,
        out_dir: Optional[str] = None,
        preflight: Optional[Callable[[], None]] = None,
        max_retained: int = 20,
        keep_warm: Optional[bool] = None,
    ):
        self._base_config = base_config
        self._make = make_processor
        self._out_dir = out_dir or tempfile.gettempdir()
        self._preflight = preflight  # may raise RuntimeError (e.g. insufficient VRAM)
        self._max_retained = max_retained
        # Keep the processor loaded between jobs (None = the config's batch_keep_warm).
        self._keep_warm = bool(base_config.get("batch_keep_warm", False) if keep_warm is None else keep_warm)
        self._idle_s = float(base_config.get("batch_keep_warm_idle_s", 0) or 0)  # 0 = never evict
        self._encoder = resolve_encoder(base_config.get("batch_encoder"))
        self._lock = threading.Lock()
        self._jobs: dict[str, BatchJob] = {}
        self._active: Optional[str] = None
        self._active_proc = None  # the running batch StreamProcessor (for live prompt steering)
        self._warm: Optional[_Warm] = None  # parked processor between jobs (keep_warm)
        self._closing = False
        self._latest_frame = None  # most recent rendered frame (RGB), for the live preview

    def submit(self, video_bytes: bytes, prompt: str, seed: int, steps: int, fps: Optional[float], interp: Optional[int] = None) -> BatchJob:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("a batch render is already running")
            # A parked processor already holds its VRAM (and is torn down before a
            # different one loads), so there is nothing new to fit.
            if self._preflight is not None and self._warm is None:
                self._preflight()  # raises -> surfaced as 409 by the route
            self._prune_locked()
            # interp=None inherits the server's interpolation_exp (the --interp flag /
            # config); an explicit value (e.g. from the UI) overrides it.
            interp_v = self._base_config.get("interpolation_exp", 0) if interp is None else interp
            job = BatchJob(id=uuid.uuid4().hex[:12], prompt=prompt, seed=seed, steps=steps, fps=fps, interp=max(0, min(4, int(interp_v))))
            self._jobs[job.id] = job
            self._active = job.id
        job._thread = threading.Thread(target=self._run, args=(job, video_bytes), daemon=True)
        job._thread.start()
        return job

    def get(self, job_id: str) -> Optional[BatchJob]:
        return self._jobs.get(job_id)

    def active_job_id(self) -> Optional[str]:
        return self._active

    def latest_jpeg(self) -> Optional[bytes]:
        """JPEG of the most recently rendered frame (for the live preview), or None."""
        frame = self._latest_frame
        if frame is None:
            return None
        import cv2  # lazy: keeps the GPU-free test import torch/cv2-free

        ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1])  # RGB -> BGR for cv2
        return buf.tobytes() if ok else None

    def set_prompt(self, text: str) -> bool:
        """Live-steer the running render's prompt (hard cut). No-op if no job runs.
        Applied on the next rendered frame (the worker drains the command queue)."""
        p = self._active_proc
        if p is None:
            return False
        try:
            p.set_prompt(text)
            return True
        except Exception:
            return False

    def start_prompt_travel(self, text: str, frames: int = 48, mode: str = "slerp") -> bool:
        """Live-steer with a slerp/lerp morph over the next `frames` rendered frames."""
        p = self._active_proc
        if p is None:
            return False
        try:
            p.start_prompt_travel(text, frames=frames, mode=mode)
            return True
        except Exception:
            return False

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job._cancel.set()
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel the active job and give its worker a bounded chance to tear the
        batch processor down, then tear down a parked one (so the 2nd model's
        subprocess + shared memory aren't orphaned at interpreter exit)."""
        self._closing = True  # a job finishing now tears its processor down
        active = self._active
        if active:
            self.cancel(active)
            job = self._jobs.get(active)
            thread = job._thread if job else None
            if thread is not None:
                thread.join(timeout)
        with self._lock:
            warm, self._warm = self._warm, None
        if warm is not None:
            self._park_cancel(warm)
            self._teardown(warm.proc)

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

    def _batch_config(self, interp: int) -> dict:
        # deepcopy so nested dicts (resolution, lip_transfer, …) are NOT shared with
        # the live sp.config — the batch instance must not mutate the live byte path.
        cfg = copy.deepcopy(self._base_config)
        cfg["batch_mode"] = True
        # 0 = exactly one output per input (1:1); k>0 = RIFE-interpolate 2**k frames
        # per input (the encoder fps is scaled by the same factor).
        cfg["interpolation_exp"] = max(0, min(4, int(interp)))
        cfg["logging"] = False
        # A second model + LivePortrait would OOM; batch never lip-syncs.
        lp = cfg.get("lip_transfer")
        if isinstance(lp, dict):
            lp["enable"] = False
        return cfg

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop + resize to the model resolution here in the parent (on the decode
        thread), so the child gets a model-size frame instead of a full 1080p/4K
        one pickled through its queue. Same call as the child's crop, which stays
        as a no-op guard."""
        res = self._base_config.get("resolution")
        if not res:
            return frame
        h, w = int(res["height"]), int(res["width"])
        if frame.shape[:2] == (h, w):
            return frame
        from fluxrt.utils.crop_maximal_rectangle import crop_maximal_rectangle

        return crop_maximal_rectangle(
            frame, h, w, area_downscale=bool(self._base_config.get("area_downscale", True))
        )

    def _run(self, job: BatchJob, video_bytes: bytes) -> None:
        proc = None
        decoder = None
        encoder = None
        in_path = None
        out_path = None
        final = "error"
        spans = _Spans() if _PROFILE else None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                in_path = f.name
            # Decoding starts now and runs ahead while the model loads; the first
            # frame is awaited before loading so an empty video fails without one.
            decoder = _Decoder(in_path, self._prepare_frame, spans)
            frame = decoder.get(job._cancel)
            if frame is None:
                raise ValueError("no frames decoded from input video")
            job.frames_total = decoder.total or decoder.count
            # Interpolation emits 2**interp frames per input, so scale the encoded fps
            # by the same factor to keep the output the same duration (just smoother).
            factor = 2 ** job.interp
            out_fps = (job.fps or decoder.fps or 25.0) * factor

            job.state = "loading"
            self._latest_frame = None  # drop the previous render's last frame
            proc = self._acquire_processor(job)
            # Params applied once, before the first frame (drained on the first render).
            proc.set_prompt(job.prompt)
            proc.set_seed(job.seed)
            proc.set_steps(job.steps)

            job.state = "running"
            out_path = os.path.join(self._out_dir, f"fluxrt-render-{job.id}.mp4")
            codec = self._encoder
            # Sized from the first output (may be upscaled).
            encoder = _EncoderThread(
                lambda out: Mp4Encoder(out_path, out_fps, out.shape[1], out.shape[0], codec=codec), spans
            )
            done = 0
            while frame is not None:
                if job._cancel.is_set():
                    raise _Canceled()
                t0 = time.perf_counter()
                # submit_frame returns a LIST (one frame, or 2**interp interpolated).
                outs = proc.submit_frame(frame)
                if spans is not None:
                    spans.add("submit", time.perf_counter() - t0)
                for out in outs:
                    encoder.put(out)
                    self._latest_frame = out  # live preview tracks the latest rendered frame
                done += 1
                job.frames_done = done
                job.frames_total = max(decoder.total, decoder.count)
                if spans is not None:
                    spans.maybe_print(done)
                frame = decoder.get(job._cancel)
            job.frames_total = done  # the header count can be off; this is what was rendered

            job.state = "encoding"
            enc, encoder = encoder, None
            enc.close()
            job.out_path = out_path
            final = "done"
        except _Canceled:
            final = "canceled"
        except Exception as exc:  # noqa: BLE001 — surface any failure as job state
            job.error = str(exc)
            final = "error"
        finally:
            if encoder is not None:
                encoder.abort()
            if decoder is not None:
                try:
                    decoder.close()
                except Exception:
                    pass
            self._active_proc = None
            if proc is not None:
                # A failed processor is never reused; a canceled one is clean between
                # frames (submit_frame is synchronous) and gets reset() like any other.
                self._release_processor(proc, job.interp, reusable=final != "error")
            # A canceled/failed job keeps no file; only a completed render is retained.
            if final != "done" and out_path and os.path.exists(out_path):
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
            # Terminal state last: a client that sees it can submit the next job.
            job.state = final

    # ── processor lifecycle (keep-warm) ───────────────────────────────────────
    def _acquire_processor(self, job: BatchJob):
        """The parked processor if it was built for this job's interp and resets
        cleanly; otherwise a fresh one, loaded and ready."""
        with self._lock:
            warm, self._warm = self._warm, None
        if warm is not None:
            self._park_cancel(warm)
            if warm.interp == job.interp:
                try:
                    warm.proc.reset()
                    self._active_proc = warm.proc
                    return warm.proc
                except Exception as exc:  # noqa: BLE001 — a suspect processor is rebuilt
                    print(f"batch processor reset failed ({exc}); reloading")
            self._teardown(warm.proc)

        proc = self._make(self._batch_config(job.interp))
        self._active_proc = proc  # exposed for live prompt steering
        try:
            proc.start()
            self._wait_ready(proc, job)
        except BaseException:
            self._active_proc = None
            self._teardown(proc)
            raise
        return proc

    def _wait_ready(self, proc, job: BatchJob) -> None:
        # Wait for the model to load, but fail fast if the child dies (CUDA OOM)
        # or never becomes ready — otherwise this loop (and the single job slot)
        # would hang forever.
        alive = getattr(proc, "worker_alive", None)
        load_deadline = time.time() + _LOAD_TIMEOUT_S
        while not proc.is_ready():
            if job._cancel.is_set():
                raise _Canceled()
            if alive is not None and not alive():
                raise RuntimeError(
                    "batch inference subprocess died during model load (likely CUDA OOM)"
                    + _exitcode_suffix(proc)
                )
            if time.time() >= load_deadline:
                raise RuntimeError("batch processor failed to become ready within the load timeout")
            time.sleep(0.1)

    def _release_processor(self, proc, interp: int, reusable: bool) -> None:
        if not (reusable and self._keep_warm and not self._closing and hasattr(proc, "reset")):
            self._teardown(proc)
            return
        warm = _Warm(proc, interp)
        if self._idle_s > 0:
            warm.timer = threading.Timer(self._idle_s, self._evict_idle, args=(warm,))
            warm.timer.daemon = True
        with self._lock:
            self._warm = warm
        if warm.timer is not None:
            warm.timer.start()

    def _evict_idle(self, warm: _Warm) -> None:
        with self._lock:
            if self._warm is not warm:
                return  # already picked up by a job, or replaced
            self._warm = None
        self._teardown(warm.proc)

    @staticmethod
    def _park_cancel(warm: _Warm) -> None:
        if warm.timer is not None:
            warm.timer.cancel()

    @staticmethod
    def _teardown(proc) -> None:
        try:
            proc.stop()  # park: frees the 2nd model's VRAM/GPU when idle
        except Exception:
            pass


def _exitcode_suffix(proc) -> str:
    code = getattr(proc, "worker_exitcode", None)
    code = code() if code is not None else None
    return f" (exitcode={code})" if code is not None else ""


class _Canceled(Exception):
    pass
