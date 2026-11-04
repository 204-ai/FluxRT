"""Frames between WebRTC and the pipeline's shared memory (torch-free).

Ingress: an inbound frame (aiortc's decoded av.VideoFrame, the local
transport's RGBX frame, or a BGR camera / Spout image) becomes the pipeline
input: BGR at the model size, centre-cropped to its aspect ratio
(fluxrt.utils.crop_maximal_rectangle, bit-identical), written straight into
shared memory.

Egress: each published output frame is converted to I420 once, in the output
pump's thread, and every viewer's track hands the encoder a yuv420p frame.
aiortc's encoders convert anything else themselves, per viewer and per
frame, and PyAV builds a new swscale context for every such call:
2.5 ms at 576x320 and 5.4 ms at 1152x640 per frame per viewer, on the encoder
thread. The conversion here takes 0.5 / 1.0 ms once per frame.

Colour: BT.601, limited range, the convention VP8 and H.264 receivers assume
(swscale's default, what aiortc used). Chroma is the average of each 2x2
block. OpenCV's own BGR2YUV_I420 takes chroma from one pixel of the block
(colour fringes up to ~90 levels on edges); swscale filters it, to within 9
levels of the box average; this is within 1 level.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

import cv2
import numpy as np

try:  # present on the server; only OutputFrame.video_frame needs it
    import av
except Exception:  # pragma: no cover - exercised only without PyAV
    av = None


# ── crop geometry ─────────────────────────────────────────────────────────────
def crop_box(in_h: int, in_w: int, out_h: int, out_w: int) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the largest centred rectangle with the output's aspect
    ratio: the crop of crop_maximal_rectangle, same rounding."""
    aspect = out_w / out_h
    if aspect > in_w / in_h:
        crop_h = max(1, min(in_h, int(round(in_w / aspect))))
        crop_w = in_w
    else:
        crop_w = max(1, min(in_w, int(round(in_h * aspect))))
        crop_h = in_h
    return (in_w - crop_w) // 2, (in_h - crop_h) // 2, crop_w, crop_h


# ── I420 ──────────────────────────────────────────────────────────────────────
def i420_planes(i420: np.ndarray, h: int, w: int):
    """Y (h, w), U and V (h/2, w/2) views of a contiguous (h*3/2, w) I420 buffer."""
    flat = i420.reshape(-1)
    q = h * w // 4
    return (
        flat[: h * w].reshape(h, w),
        flat[h * w : h * w + q].reshape(h // 2, w // 2),
        flat[h * w + q :].reshape(h // 2, w // 2),
    )


class I420Converter:
    """BGR -> I420 with box-filtered chroma. Keeps its scratch buffers, so a
    stream of same-size frames allocates only the output. Not thread-safe:
    one per thread."""

    def __init__(self):
        self._scratch = {}

    def __call__(self, bgr: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = bgr.shape[:2]
        if h % 2 or w % 2:
            raise ValueError(f"I420 needs even dimensions, got {w}x{h}")
        if out is None:
            out = np.empty((h * 3 // 2, w), np.uint8)
        key = (h, w)
        if key not in self._scratch:
            self._scratch[key] = (
                np.empty((h // 2, w // 2, 3), np.uint8),
                np.empty((h, w, 3), np.uint8),
                np.empty((h * 3 // 2, w), np.uint8),
            )
        half, blocks, tmp = self._scratch[key]
        # luma: OpenCV's fixed-point BT.601 limited range (its chroma is
        # point-sampled, so it is replaced below)
        cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420, dst=out)
        # chroma of each 2x2 block's mean: an exact 2x area downscale, then
        # nearest upscale so every pixel of a block has the block's mean, so
        # whichever pixel the conversion samples gives the box average
        cv2.resize(bgr, (w // 2, h // 2), dst=half, interpolation=cv2.INTER_AREA)
        cv2.resize(half, (w, h), dst=blocks, interpolation=cv2.INTER_NEAREST)
        cv2.cvtColor(blocks, cv2.COLOR_BGR2YUV_I420, dst=tmp)
        out[h:] = tmp[h:]
        return out


def scale_i420(i420: np.ndarray, h: int, w: int, out_h: int, out_w: int) -> np.ndarray:
    """Resize an I420 frame plane by plane (area filter): a per-viewer
    downscale (WHEP ?w=) without going back to RGB."""
    out = np.empty((out_h * 3 // 2, out_w), np.uint8)
    for src, dst in zip(i420_planes(i420, h, w), i420_planes(out, out_h, out_w)):
        cv2.resize(src, (dst.shape[1], dst.shape[0]), dst=dst, interpolation=cv2.INTER_AREA)
    return out


# ── egress: published output frames ───────────────────────────────────────────
class OutputFrame:
    """One published output frame. `bgr` is owned by this object and never
    written again; the other forms are derived on first use and cached
    (computing one twice from two threads is harmless)."""

    __slots__ = ("version", "bgr", "_rgb", "_i420", "_scaled")

    def __init__(self, version: int, bgr: np.ndarray):
        self.version = version
        self.bgr = bgr
        self._rgb = None
        self._i420 = None
        self._scaled = {}

    @property
    def height(self) -> int:
        return self.bgr.shape[0]

    @property
    def width(self) -> int:
        return self.bgr.shape[1]

    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            self._rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        return self._rgb

    def i420(self, converter: Optional[I420Converter] = None) -> np.ndarray:
        if self._i420 is None:
            self._i420 = (converter or I420Converter())(self.bgr)
        return self._i420

    def i420_at(self, width: int, height: int) -> np.ndarray:
        """I420 at another (even) size, cached per size: several viewers at
        the same WHEP ?w= share one resize."""
        if (width, height) == (self.width, self.height):
            return self.i420()
        key = (width, height)
        scaled = self._scaled.get(key)
        if scaled is None:
            scaled = scale_i420(self.i420(), self.height, self.width, height, width)
            self._scaled[key] = scaled
        return scaled

    def video_frame(self, size: Optional[tuple[int, int]] = None) -> "av.VideoFrame":
        """A new yuv420p av.VideoFrame (each viewer's encoder sets its own
        pts and may force a keyframe on it, so frames are never shared)."""
        w, h = size or (self.width, self.height)
        return av.VideoFrame.from_ndarray(self.i420_at(w, h), format="yuv420p")


class FramePublisher:
    """The newest output frame and its version, for every consumer: WebRTC
    tracks (asyncio), the local transport, the Spout sender (threads).

    Consumers wait for a version newer than the one they have instead of
    polling: publish() wakes them. With `i420`, publish() converts the frame
    to I420 in the publishing thread (the output pump), before any track
    asks for it."""

    def __init__(self, i420: bool = True):
        self.i420 = i420
        # also prepare RGB in publish() (the local transport, Spout): it
        # would otherwise be converted on first use, on the event loop
        self.want_rgb = False
        self._converter = I420Converter()
        self._cond = threading.Condition()
        self._latest: Optional[OutputFrame] = None
        self._version = 0
        self._waiters: set = set()  # (loop, future) of tracks waiting for a frame

    @property
    def version(self) -> int:
        return self._version

    def latest(self) -> Optional[OutputFrame]:
        return self._latest

    def publish(self, bgr: np.ndarray) -> OutputFrame:
        """Publish an output frame. `bgr` must not be written afterwards
        (pass a fresh copy, e.g. SharedTensor.to_numpy())."""
        frame = OutputFrame(0, bgr)
        i420 = self.i420 and bgr.shape[0] % 2 == 0 and bgr.shape[1] % 2 == 0
        if i420:
            frame.i420(self._converter)
        if self.want_rgb or not i420:  # rgb24 tracks, local transport, Spout
            frame.rgb()
        with self._cond:
            frame.version = self._version + 1
            self._latest = frame
            self._version = frame.version
            waiters, self._waiters = self._waiters, set()
            self._cond.notify_all()
        for loop, future in waiters:
            try:
                loop.call_soon_threadsafe(_wake, future)
            except RuntimeError:  # that loop is closed
                pass
        return frame

    async def next_frame(self, after: int, timeout: float) -> Optional[OutputFrame]:
        """The newest frame once its version is above `after`, or None after
        `timeout` seconds."""
        frame = self._latest
        if frame is not None and frame.version > after:
            return frame
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        entry = (loop, future)
        with self._cond:
            frame = self._latest
            if frame is not None and frame.version > after:
                return frame
            self._waiters.add(entry)
        try:
            await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            with self._cond:
                self._waiters.discard(entry)
        frame = self._latest
        return frame if frame is not None and frame.version > after else None

    def wait_frame(self, after: int, timeout: float) -> Optional[OutputFrame]:
        """Blocking twin of next_frame, for consumer threads."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._latest is None or self._latest.version <= after:
                left = deadline - time.monotonic()
                if left <= 0 or not self._cond.wait(left):
                    if self._latest is None or self._latest.version <= after:
                        return None
            return self._latest


def _wake(future) -> None:
    if not future.done():
        future.set_result(None)


# ── ingress: inbound frames -> the pipeline input ────────────────────────────
class IngressConverter:
    """Writes an inbound frame into the pipeline input `dst` ((h, w, 3) BGR,
    usually a SharedTensor's array inside SharedTensor.writing()), with
    crop_maximal_rectangle's crop and filter, bit-identical to it.

    What it saves over "to BGR, crop_maximal_rectangle, copy into shared
    memory": the resize writes straight into dst (no intermediate frame, no
    copy), a frame already at the model size is converted straight into dst,
    and the local transport's RGBX frames are cropped and resized before the
    channel swap (4-channel resize, one pass less).

    Not done, measured (docs/perf-notes.md): scaling a larger yuv420p frame
    in swscale or on the Y/U/V planes is 1.5-3x faster but 2-10 dB further
    from the ideal resize at 720p / 1080p (aliasing, chroma resolution); the
    full-resolution BGR conversion then OpenCV's area filter stays."""

    def __init__(self, height: int, width: int, area_downscale: bool = True):
        self.height = height
        self.width = width
        self.area_downscale = area_downscale

    def _filter(self, crop_w: int, crop_h: int) -> int:
        shrinking = crop_w > self.width or crop_h > self.height
        return cv2.INTER_AREA if shrinking and self.area_downscale else cv2.INTER_LINEAR

    def write(self, frame, dst: np.ndarray) -> None:
        if isinstance(frame, np.ndarray):
            self._write_bgr(frame, dst)
        elif hasattr(frame, "rgbx"):  # local transport
            self._write_rgbx(frame.rgbx, dst)
        else:  # av.VideoFrame from aiortc's decoder
            self._write_bgr(frame.to_ndarray(format="bgr24"), dst)

    def _write_bgr(self, bgr: np.ndarray, dst: np.ndarray) -> None:
        x, y, w, h = crop_box(bgr.shape[0], bgr.shape[1], self.height, self.width)
        crop = bgr[y : y + h, x : x + w]
        if (w, h) == (self.width, self.height):
            np.copyto(dst, crop)
        else:
            cv2.resize(crop, (self.width, self.height), dst=dst, interpolation=self._filter(w, h))

    def _write_rgbx(self, rgbx: np.ndarray, dst: np.ndarray) -> None:
        x, y, w, h = crop_box(rgbx.shape[0], rgbx.shape[1], self.height, self.width)
        crop = rgbx[y : y + h, x : x + w]
        if (w, h) != (self.width, self.height):
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self._filter(w, h))
        cv2.cvtColor(crop, cv2.COLOR_RGBA2BGR, dst=dst)
