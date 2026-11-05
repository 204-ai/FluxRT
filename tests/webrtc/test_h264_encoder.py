"""The NVENC H.264 encoder (fluxrt.webrtc.h264_encoder). There is no NVIDIA
GPU in CI: the encoder's own logic runs on libx264 through the same class,
the NVENC options are checked against h264_nvenc's option list in the
bundled libavcodec, and the fallback is what a machine without NVENC does."""

import ctypes
import fractions
import glob
import os

import av
import numpy as np
import pytest

from fluxrt.webrtc import h264_encoder


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(h264_encoder, "state", h264_encoder._State())


def codec_options(name: str) -> set:
    """Option names of an encoder's private class in PyAV's libavcodec."""
    libdir = os.path.join(os.path.dirname(av.__file__), "..", "av.libs")
    avc_path = glob.glob(os.path.join(libdir, "libavcodec*"))
    avu_path = glob.glob(os.path.join(libdir, "libavutil*"))
    if not avc_path or not avu_path:
        pytest.skip("PyAV's bundled libraries not found")
    avc, avu = ctypes.CDLL(avc_path[0]), ctypes.CDLL(avu_path[0])
    avc.avcodec_get_class.restype = ctypes.c_void_p
    avu.av_opt_child_class_iterate.restype = ctypes.c_void_p
    avu.av_opt_child_class_iterate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    avu.av_opt_next.restype = ctypes.c_void_p
    avu.av_opt_next.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    parent, it = avc.avcodec_get_class(), ctypes.c_void_p(None)
    while True:
        cls = avu.av_opt_child_class_iterate(parent, ctypes.byref(it))
        if not cls:
            pytest.skip(f"{name} not in this libavcodec")
        if ctypes.cast(cls, ctypes.POINTER(ctypes.c_char_p))[0] == name.encode():
            break
    holder, prev, names = ctypes.c_void_p(cls), None, set()
    while True:
        prev = avu.av_opt_next(ctypes.byref(holder), prev)
        if not prev:
            return names
        names.add(ctypes.cast(prev, ctypes.POINTER(ctypes.c_char_p))[0].decode())


def test_the_wheel_has_nvenc_and_every_option_exists():
    av.codec.Codec("h264_nvenc", "w")  # raises if this PyAV build has no NVENC
    names = codec_options("h264_nvenc")
    options = h264_encoder.nvenc_options()
    assert set(options) <= names, set(options) - names
    # the values are named constants of those options
    assert {"p4", "ull", "cbr", "baseline"} <= names


@pytest.mark.parametrize("env,preset", [({}, "p4"), ({"FLUXRT_NVENC_PRESET": "p1"}, "p1")])
def test_preset_from_the_environment(env, preset):
    assert h264_encoder.options_from_env(env)["preset"] == preset


def test_a_bad_preset_is_rejected():
    with pytest.raises(ValueError):
        h264_encoder.options_from_env({"FLUXRT_NVENC_PRESET": "fast"})


def frames(n, w=576, h=320, step=3000):
    """yuv420p frames with 90 kHz pts, as FluxRTTrack hands them over."""
    y, x = np.mgrid[0:h, 0:w]
    for i in range(n):
        rgb = np.stack([(x + 4 * i) % 256, (y + 3 * i) % 256, (x + y) % 256], -1).astype(np.uint8)
        f = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        f.time_base = fractions.Fraction(1, 90000)
        f.pts = i * step
        yield f


def decode_all(encoder, frame_iter, keyframe_at=()):
    from aiortc.codecs.h264 import h264_depayload

    dec = av.CodecContext.create("h264", "r")
    decoded, timestamps, keys = 0, [], 0
    for i, f in enumerate(frame_iter):
        pts = f.pts
        payloads, ts = encoder.encode(f, force_keyframe=i in keyframe_at)
        timestamps.append((pts, ts))
        # one access unit per frame: depayload every RTP payload and decode
        au = b"".join(h264_depayload(p) for p in payloads)
        for out in dec.decode(av.Packet(au)):
            decoded += 1
            keys += out.key_frame
    return decoded, timestamps, keys


def test_encodes_a_decodable_stream_with_the_frames_own_timestamps():
    cls = h264_encoder.make_encoder_class({"tune": "zerolatency", "preset": "ultrafast"}, codec_name="libx264")
    enc = cls()
    decoded, timestamps, keys = decode_all(enc, frames(12))
    assert decoded == 12
    assert all(abs(ts - pts) <= 1 for pts, ts in timestamps)  # 90 kHz in, 90 kHz out
    assert enc.rebuilds == 0 and enc._hardware
    assert h264_encoder.state.in_use == "libx264"


def test_a_keyframe_request_produces_a_keyframe():
    cls = h264_encoder.make_encoder_class({"tune": "zerolatency", "preset": "ultrafast"}, codec_name="libx264")
    decoded, _, keys = decode_all(cls(), frames(10), keyframe_at={6})
    assert decoded == 10 and keys == 2  # the first frame and the requested one


def test_a_bitrate_change_over_ten_percent_rebuilds():
    cls = h264_encoder.make_encoder_class({"tune": "zerolatency", "preset": "ultrafast"}, codec_name="libx264")
    enc = cls()
    it = frames(3)
    enc.encode(next(it))
    enc.target_bitrate = int(enc.target_bitrate * 1.05)
    enc.encode(next(it))
    assert enc.rebuilds == 0
    enc.target_bitrate = int(enc.target_bitrate * 1.5)
    enc.encode(next(it))
    assert enc.rebuilds == 1


def test_without_the_hardware_encoder_it_falls_back_to_libx264():
    """What a machine without NVENC (this one) does: the first open fails,
    the frame and every later one go through aiortc's libx264 path."""
    cls = h264_encoder.make_encoder_class(h264_encoder.nvenc_options(), codec_name="h264_nvenc")
    try:
        av.CodecContext.create("h264_nvenc", "w")
    except Exception:
        pytest.skip("no h264_nvenc in this build")
    if h264_encoder.nvenc_available():
        pytest.skip("this machine has a working NVENC")
    h264_encoder.state.failed = False  # the probe set it; let the encoder find out itself
    enc = cls()
    decoded, _, _ = decode_all(enc, frames(5))
    assert decoded == 5
    assert h264_encoder.state.failed and not enc._hardware
    assert h264_encoder.status()["h264_encoder"] == "libx264"
    # a second encoder does not try again
    decoded, _, _ = decode_all(cls(), frames(3))
    assert decoded == 3


def test_install_routes_h264_senders_and_keeps_the_vp8_route(monkeypatch):
    from aiortc import rtcrtpsender
    from aiortc.codecs import get_capabilities

    from fluxrt.webrtc import vp8_encoder

    monkeypatch.setattr(rtcrtpsender, "get_encoder", rtcrtpsender.get_encoder)
    vp8 = vp8_encoder.install({})
    h264 = h264_encoder.install({})
    codecs = {c.mimeType.lower(): c for c in get_capabilities("video").codecs}
    assert isinstance(rtcrtpsender.get_encoder(codecs["video/h264"]), h264)
    assert isinstance(rtcrtpsender.get_encoder(codecs["video/vp8"]), vp8)
    base = rtcrtpsender.get_encoder._fluxrt_h264_base
    h264_encoder.install({})  # again: wraps the same base, not itself
    assert rtcrtpsender.get_encoder._fluxrt_h264_base is base
