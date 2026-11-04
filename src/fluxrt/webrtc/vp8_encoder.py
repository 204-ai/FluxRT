"""libvpx settings for the VP8 output stream (torch-free).

aiortc's Vp8Encoder hard-codes options chosen for camera video. One of them
costs time and quality on FluxRT's output: `noise-sensitivity: 4`, libvpx's
temporal denoiser. It filters sensor noise out of the source before encoding;
a generated frame has none, so it only smears detail. Measured on photographic
content with camera motion at equal bitrate (scripts/perf_webrtc.py encoder):
1152x640 / 8 Mbps 17.2 -> 15.4 ms per frame and 44.4 -> 45.6 dB PSNR;
576x320 / 8 Mbps 6.3 -> 5.6 ms and 41.5 -> 43.2 dB.

Environment (read once at install):
  FLUXRT_VP8_DENOISE   libvpx noise-sensitivity, 0 (default here) .. 6; aiortc used 4
  FLUXRT_VP8_CPU_USED  libvpx cpu-used, -16 .. 16 (default -6, aiortc's). Higher
                       magnitude is faster at lower quality per bit: -10 measured
                       ~20% faster, -1.3 to -1.6 dB at equal bitrate
  FLUXRT_VP8_THREADS   encoder threads; 0 / unset = aiortc's choice by frame size

Everything else (rate control, keyframe interval, packetization) is aiortc's,
and this encoder replaces aiortc's only for video/vp8 senders created after
install().
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("fluxrt.webrtc")

_DEFAULTS = {"noise-sensitivity": "0"}


def options_from_env(env=None) -> tuple[dict, int]:
    """(libvpx option overrides, thread count or 0) from the environment.
    ValueError on an out-of-range value."""
    env = os.environ if env is None else env
    options = dict(_DEFAULTS)
    if env.get("FLUXRT_VP8_DENOISE", "") != "":
        denoise = int(env["FLUXRT_VP8_DENOISE"])
        if not 0 <= denoise <= 6:
            raise ValueError("FLUXRT_VP8_DENOISE must be 0..6")
        options["noise-sensitivity"] = str(denoise)
    if env.get("FLUXRT_VP8_CPU_USED", "") != "":
        cpu_used = int(env["FLUXRT_VP8_CPU_USED"])
        if not -16 <= cpu_used <= 16:
            raise ValueError("FLUXRT_VP8_CPU_USED must be -16..16")
        options["cpu-used"] = str(cpu_used)
    threads = int(env.get("FLUXRT_VP8_THREADS", "0") or 0)
    if threads < 0:
        raise ValueError("FLUXRT_VP8_THREADS must be >= 0")
    return options, threads


def make_encoder_class(options: dict, threads: int = 0):
    """A Vp8Encoder subclass that opens libvpx with `options` merged over
    aiortc's, or None if this aiortc's encoder does not look like the one
    this was written against (1.x)."""
    import multiprocessing

    import av
    from aiortc.codecs import vpx

    base = getattr(vpx, "Vp8Encoder", None)
    needed = ("number_of_threads", "convert_timebase", "VIDEO_TIME_BASE")
    if base is None or not all(hasattr(vpx, n) for n in needed) or not hasattr(base, "_packetize"):
        return None

    class TunedVp8Encoder(base):
        """aiortc's Vp8Encoder.encode with FluxRT's libvpx options."""

        def encode(self, frame, force_keyframe: bool = False):
            assert isinstance(frame, av.VideoFrame)
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")

            if self.codec and (
                frame.width != self.codec.width
                or frame.height != self.codec.height
                # aiortc adjusts the bitrate only when it moves by over 10%
                # (a new codec: libvpx ignores bit_rate changes once open)
                or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
            ):
                self.codec = None

            if force_keyframe:
                frame.pict_type = av.video.frame.PictureType.I

            if self.codec is None:
                bitrate = self.target_bitrate
                self.codec = av.CodecContext.create("libvpx", "w")
                self.codec.width = frame.width
                self.codec.height = frame.height
                self.codec.bit_rate = bitrate
                self.codec.pix_fmt = "yuv420p"
                self.codec.gop_size = 3000  # kf_max_dist
                self.codec.qmin = 2  # rc_min_quantizer
                self.codec.qmax = 56  # rc_max_quantizer
                self.codec.options = {
                    # aiortc's set (rc_buf_sz = 1000 ms, CBR via minrate = maxrate)
                    "bufsize": str(bitrate),
                    "cpu-used": "-6",
                    "deadline": "realtime",
                    "lag-in-frames": "0",
                    "minrate": str(bitrate),
                    "maxrate": str(bitrate),
                    "noise-sensitivity": "4",
                    "overshoot-pct": "15",
                    "partitions": "0",
                    "static-thresh": "1",
                    "undershoot-pct": "100",
                    **options,
                }
                self.codec.thread_count = threads or vpx.number_of_threads(
                    frame.width * frame.height, multiprocessing.cpu_count()
                )
                self.rebuilds = getattr(self, "rebuilds", -1) + 1

            data = b"".join(bytes(packet) for packet in self.codec.encode(frame))
            payloads = self._packetize(data, self.picture_id)
            timestamp = vpx.convert_timebase(frame.pts, frame.time_base, vpx.VIDEO_TIME_BASE)
            self.picture_id = (self.picture_id + 1) % (1 << 15)
            return payloads, timestamp

    TunedVp8Encoder.options = dict(options)
    TunedVp8Encoder.threads = threads
    return TunedVp8Encoder


_installed = None


def install(env=None):
    """Route aiortc's video/vp8 senders to the tuned encoder. Returns the
    encoder class, or None when aiortc's encoder is left in place."""
    global _installed
    try:
        options, threads = options_from_env(env)
    except ValueError as error:
        log.warning("VP8: %s; keeping aiortc's encoder settings", error)
        return None
    cls = make_encoder_class(options, threads)
    if cls is None:
        log.warning("VP8: unexpected aiortc encoder; keeping aiortc's settings")
        return None
    from aiortc import rtcrtpsender

    original = getattr(rtcrtpsender.get_encoder, "_fluxrt_original", rtcrtpsender.get_encoder)

    def get_encoder(codec):
        if codec.mimeType.lower() == "video/vp8":
            return cls()
        return original(codec)

    get_encoder._fluxrt_original = original
    rtcrtpsender.get_encoder = get_encoder
    _installed = cls
    log.info("VP8 encoder: libvpx %s, threads %s", options, threads or "auto")
    return cls
