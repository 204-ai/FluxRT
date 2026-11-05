"""H.264 for the output stream on the GPU's encoder, NVENC (torch-free).

The pip PyAV wheels are built with NVENC (`h264_nvenc`; the NVIDIA driver
has to be >= 570), so the hardware encoder needs no other package. Per
output frame and viewer it takes an encode that libvpx does in 10-14 ms at
1152x640 on the CPU (scripts/perf_webrtc.py stages) off the CPU entirely;
NVIDIA quotes 800-900 frames per second at 1080p for the fastest presets.

This encoder replaces aiortc's H264Encoder for video/h264 senders. It keeps
everything of aiortc's but the codec: the RTP packetization (STAP-A / FU-A),
pre-encoded packets (`pack`), the bitrate setter that REMB drives and the
rebuild when the target moves by over 10 %. What it changes:

  - h264_nvenc with the low-latency settings: tune ull, CBR, no B-frames,
    no lookahead, no output delay, profile baseline (what aiortc's SDP
    offers, 42e01f).
  - forced-idr: a keyframe request (aiortc sets pict_type I) becomes an
    IDR with SPS/PPS; without it NVENC writes a plain I frame, which a
    decoder that lost the stream cannot start from, and the viewer keeps
    asking (a PLI loop).
  - the frames' own 90 kHz timestamps (aiortc rebases them to 1/30 s).
  - a GOP of 3000 frames, like the VP8 encoder: keyframes come on request.

If NVENC cannot be opened (no NVIDIA GPU, an older driver, a PyAV build
without it), the first failure is logged and every H.264 encoder of the
process uses aiortc's libx264 path instead.

Environment (read once at install):
  FLUXRT_NVENC_PRESET  p1 (fastest) .. p7 (best quality per bit), default p4
"""

from __future__ import annotations

import fractions
import logging
import os
import threading

log = logging.getLogger("fluxrt.webrtc")

NVENC = "h264_nvenc"
# Frame rate NVENC's rate control budgets for. CBR gives each frame
# bitrate / frame rate; the output runs at 12-60 frames a second depending
# on the preset, so at fewer frames the stream simply stays under the target.
RC_FRAME_RATE = 30
GOP = 3000


def nvenc_options(preset: str = "p4") -> dict:
    return {
        "preset": preset,
        "tune": "ull",
        "rc": "cbr",
        "zerolatency": "1",
        "delay": "0",
        "rc-lookahead": "0",
        "forced-idr": "1",
        "profile": "baseline",
    }


def options_from_env(env=None) -> dict:
    env = os.environ if env is None else env
    preset = env.get("FLUXRT_NVENC_PRESET", "") or "p4"
    if preset not in {f"p{i}" for i in range(1, 8)}:
        raise ValueError("FLUXRT_NVENC_PRESET must be p1..p7")
    return nvenc_options(preset)


class _State:
    """Process-wide: whether the hardware encoder has failed to open (then
    every encoder falls back), and which encoder the last open used."""

    def __init__(self):
        self.lock = threading.Lock()
        self.failed = False
        self.reason = None
        self.in_use = None


state = _State()


def make_encoder_class(options: dict, codec_name: str = NVENC):
    """An aiortc H264Encoder subclass that encodes with `codec_name`, or None
    if this aiortc's encoder does not look like the one this was written
    against (1.x). codec_name is a parameter so the tests can run the same
    code on libx264."""
    import av
    from aiortc.codecs import h264

    base = getattr(h264, "H264Encoder", None)
    needed = ("_split_bitstream", "_packetize", "_encode_frame", "encode", "pack")
    if base is None or not all(hasattr(base, n) for n in needed):
        return None

    class HardwareH264Encoder(base):
        """aiortc's H264Encoder with the codec opened on `codec_name`."""

        hardware = codec_name

        def __init__(self):
            super().__init__()
            self.rebuilds = -1
            self._hardware = False  # this instance's codec is codec_name's

        def _open(self, frame):
            codec = av.CodecContext.create(codec_name, "w")
            codec.width = frame.width
            codec.height = frame.height
            codec.bit_rate = self.target_bitrate
            codec.pix_fmt = "yuv420p"
            codec.framerate = fractions.Fraction(RC_FRAME_RATE, 1)
            codec.time_base = fractions.Fraction(1, 90000)
            codec.gop_size = GOP
            codec.max_b_frames = 0
            codec.options = dict(options)
            return codec

        def _encode_frame(self, frame, force_keyframe):
            if state.failed and not self._hardware:
                yield from super()._encode_frame(frame, force_keyframe)
                return
            if self.codec and (
                frame.width != self.codec.width
                or frame.height != self.codec.height
                # aiortc's rule: adjust the bitrate only when it moves by over 10 %
                or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
            ):
                self.codec = None

            frame.pict_type = (
                av.video.frame.PictureType.I if force_keyframe else av.video.frame.PictureType.NONE
            )
            first = self.codec is None
            try:
                if first:
                    self.codec = self._open(frame)
                data = b"".join(bytes(p) for p in self.codec.encode(frame))
            except Exception as error:  # the codec opens on the first frame
                if not first:
                    raise
                self.codec = None
                with state.lock:
                    if not state.failed:
                        state.failed = True
                        state.reason = f"{type(error).__name__}: {error}"
                        log.warning("H.264: %s unavailable (%s); using aiortc's libx264", codec_name, state.reason)
                self._hardware = False
                yield from super()._encode_frame(frame, force_keyframe)
                return
            if first:
                self._hardware = True
                self.rebuilds += 1
                state.in_use = codec_name
            if data:
                yield from self._split_bitstream(data)

    HardwareH264Encoder.options = dict(options)
    return HardwareH264Encoder


def nvenc_available(codec_name: str = NVENC) -> bool:
    """Whether the hardware encoder opens and encodes here (one small frame;
    tens of milliseconds the first time, cached after)."""
    with state.lock:
        if state.failed:
            return False
        if state.in_use == codec_name:
            return True
    import fractions as _fractions

    import av
    import numpy as np

    try:
        codec = av.CodecContext.create(codec_name, "w")
        codec.width, codec.height = 256, 144
        codec.pix_fmt = "yuv420p"
        codec.bit_rate = 1_000_000
        codec.framerate = _fractions.Fraction(30, 1)
        codec.time_base = _fractions.Fraction(1, 90000)
        codec.options = nvenc_options()
        frame = av.VideoFrame.from_ndarray(np.zeros((216, 256), np.uint8), format="yuv420p")
        frame.pts, frame.time_base = 0, _fractions.Fraction(1, 90000)
        list(codec.encode(frame))
        list(codec.encode(None))
    except Exception as error:
        with state.lock:
            state.failed = True
            state.reason = f"{type(error).__name__}: {error}"
        return False
    with state.lock:
        state.in_use = codec_name
    return True


def status() -> dict:
    """For /healthz: which encoder H.264 senders use, and why not NVENC."""
    with state.lock:
        return {
            "h264_encoder": "libx264" if state.failed else (state.in_use or "not opened yet"),
            "nvenc_error": state.reason,
        }


_installed = None


def install(env=None, codec_name: str = NVENC):
    """Route aiortc's video/h264 senders to the hardware encoder (falling
    back to libx264 per process). Returns the encoder class, or None when
    aiortc's encoder is left in place. Installs on top of whatever
    get_encoder currently is (e.g. the VP8 encoder of vp8_encoder.install)."""
    global _installed
    try:
        options = options_from_env(env)
    except ValueError as error:
        log.warning("H.264: %s; keeping aiortc's encoder", error)
        return None
    cls = make_encoder_class(options, codec_name)
    if cls is None:
        log.warning("H.264: unexpected aiortc encoder; keeping aiortc's")
        return None
    from aiortc import rtcrtpsender

    current = rtcrtpsender.get_encoder
    base = getattr(current, "_fluxrt_h264_base", current)

    def get_encoder(codec):
        if codec.mimeType.lower() == "video/h264":
            return cls()
        return base(codec)

    get_encoder._fluxrt_h264_base = base
    rtcrtpsender.get_encoder = get_encoder
    _installed = cls
    log.info("H.264 encoder: %s %s (libx264 if it cannot open)", codec_name, options)
    return cls
