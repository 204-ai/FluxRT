"""The tuned VP8 encoder (fluxrt.webrtc.vp8_encoder): settings from the
environment, a real libvpx encode that a decoder reads back, and the hook
into aiortc's sender."""

import fractions

import av
import numpy as np
import pytest

from fluxrt.webrtc import vp8_encoder


def test_defaults_turn_the_denoiser_off_and_keep_the_rest():
    options, threads = vp8_encoder.options_from_env({})
    assert options == {"noise-sensitivity": "0"} and threads == 0


def test_environment_overrides():
    options, threads = vp8_encoder.options_from_env(
        {"FLUXRT_VP8_DENOISE": "4", "FLUXRT_VP8_CPU_USED": "-10", "FLUXRT_VP8_THREADS": "3"}
    )
    assert options == {"noise-sensitivity": "4", "cpu-used": "-10"} and threads == 3


@pytest.mark.parametrize("env", [{"FLUXRT_VP8_DENOISE": "9"}, {"FLUXRT_VP8_CPU_USED": "-17"}, {"FLUXRT_VP8_THREADS": "-1"}])
def test_out_of_range_values_are_rejected(env):
    with pytest.raises(ValueError):
        vp8_encoder.options_from_env(env)


def frames(n, h=320, w=576):
    y, x = np.mgrid[0:h, 0:w]
    for i in range(n):
        rgb = np.stack([(x + 3 * i) % 256, (y + 2 * i) % 256, (x + y) % 256], axis=-1).astype(np.uint8)
        f = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        f.time_base = fractions.Fraction(1, 90000)
        f.pts = i * 3000
        yield f


def test_encoded_frames_decode():
    from aiortc.codecs.vpx import vp8_depayload

    cls = vp8_encoder.make_encoder_class({"noise-sensitivity": "0"})
    enc = cls()
    dec = av.CodecContext.create("libvpx", "r")
    decoded = 0
    for frame in frames(10):
        pts = frame.pts  # PyAV rescales the frame's pts to the codec's time base in encode()
        payloads, timestamp = enc.encode(frame)
        assert payloads and abs(timestamp - pts) <= 1
        data = b"".join(vp8_depayload(p) for p in payloads)
        decoded += len(dec.decode(av.Packet(data)))
    assert decoded == 10
    assert enc.rebuilds == 0


def test_the_options_reach_libvpx():
    """Same frames, denoiser on and off: different bitstreams (PyAV empties
    codec.options once libvpx has taken them, so they cannot be read back)."""
    streams = []
    for denoise in ("0", "4"):
        enc = vp8_encoder.make_encoder_class({"noise-sensitivity": denoise})()
        streams.append(b"".join(b"".join(enc.encode(f)[0]) for f in frames(8)))
    assert streams[0] != streams[1]


def test_a_bitrate_change_over_ten_percent_rebuilds_like_aiortc():
    enc = vp8_encoder.make_encoder_class({})()
    it = frames(3)
    enc.encode(next(it))
    enc.target_bitrate = int(enc.target_bitrate * 1.05)
    enc.encode(next(it))
    assert enc.rebuilds == 0
    enc.target_bitrate = int(enc.target_bitrate * 1.5)
    enc.encode(next(it))
    assert enc.rebuilds == 1


def test_install_routes_vp8_senders_only(monkeypatch):
    from aiortc import rtcrtpsender
    from aiortc.codecs import get_capabilities

    monkeypatch.setattr(rtcrtpsender, "get_encoder", rtcrtpsender.get_encoder)
    cls = vp8_encoder.install({})
    assert cls is not None
    codecs = {c.mimeType.lower(): c for c in get_capabilities("video").codecs}
    assert isinstance(rtcrtpsender.get_encoder(codecs["video/vp8"]), cls)
    assert not isinstance(rtcrtpsender.get_encoder(codecs["video/h264"]), cls)
    # installing again wraps aiortc's function, not the previous wrapper
    original = rtcrtpsender.get_encoder._fluxrt_original
    vp8_encoder.install({})
    assert rtcrtpsender.get_encoder._fluxrt_original is original


def test_a_bad_environment_keeps_aiortc_s_encoder(monkeypatch):
    from aiortc import rtcrtpsender

    before = rtcrtpsender.get_encoder
    assert vp8_encoder.install({"FLUXRT_VP8_DENOISE": "x"}) is None
    assert rtcrtpsender.get_encoder is before
