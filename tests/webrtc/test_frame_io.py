"""Frame conversions between WebRTC and the pipeline (fluxrt.webrtc.frame_io):
bit-identity with the old paths where claimed, colour accuracy of the I420
egress, and the publisher's wake-ups. No GPU, no network."""

import asyncio
import threading
import time

import av
import cv2
import numpy as np
import pytest

from fluxrt.utils.crop_maximal_rectangle import crop_maximal_rectangle
from fluxrt.webrtc import frame_io
from fluxrt.webrtc.local_transport import RawFrame

H, W = 320, 576  # the model input


def picture(h, w, seed=0):
    """Deterministic test image with smooth areas, fine texture and hard
    colour edges (where chroma subsampling shows)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([
        127 + 100 * np.cos(x / 23.0) * np.sin(y / 17.0),
        127 + 90 * np.sin((x + y) / 11.0),
        127 + 110 * np.cos(np.hypot(x - w / 2, y - h / 2) / 9.0),
    ], axis=-1)
    img += rng.normal(0, 12, img.shape)
    for _ in range(12):  # flat colour rectangles: hard chroma edges
        x0, y0 = rng.integers(0, w - 8), rng.integers(0, h - 8)
        img[y0 : y0 + rng.integers(4, h // 3), x0 : x0 + rng.integers(4, w // 3)] = rng.integers(0, 256, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def reference_i420(bgr):
    """Float BT.601 limited range, chroma averaged over each 2x2 block."""
    b, g, r = (bgr[..., i].astype(np.float64) for i in range(3))
    k = 224 / 255
    y = 16 + 219 / 255 * (0.299 * r + 0.587 * g + 0.114 * b)
    cb = 128 + k * (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 128 + k * (0.5 * r - 0.418688 * g - 0.081312 * b)
    h, w = y.shape
    box = lambda c: c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return y, box(cb), box(cr)


# ── crop geometry ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("size", [(360, 640), (480, 640), (720, 1280), (1080, 1920), (800, 1920), (320, 576), (100, 100)])
def test_crop_box_is_crop_maximal_rectangle_s_crop(size):
    ih, iw = size
    img = np.arange(ih * iw, dtype=np.uint32).reshape(ih, iw)
    x, y, w, h = frame_io.crop_box(ih, iw, H, W)
    # crop_maximal_rectangle's crop, recovered from its centring
    aspect = W / H
    if aspect > iw / ih:
        ch, cw = max(1, min(ih, int(round(iw / aspect)))), iw
    else:
        cw, ch = max(1, min(iw, int(round(ih * aspect)))), ih
    assert (x, y, w, h) == ((iw - cw) // 2, (ih - ch) // 2, cw, ch)
    assert img[y : y + h, x : x + w].shape == (h, w)


# ── egress: I420 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("size", [(320, 576), (640, 1152)])
def test_i420_matches_bt601_with_box_averaged_chroma(size):
    bgr = picture(*size)
    h, w = size
    out = frame_io.I420Converter()(bgr)
    y, u, v = frame_io.i420_planes(out, h, w)
    ry, ru, rv = reference_i420(bgr)
    assert np.abs(y - ry).max() <= 1
    assert np.abs(u - ru).max() <= 1 and np.abs(v - rv).max() <= 1
    # OpenCV's own conversion point-samples chroma: what this replaces
    _, cu, _ = frame_io.i420_planes(cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420), h, w)
    assert np.abs(cu - ru).max() > 20


def test_i420_is_what_the_encoder_would_otherwise_make():
    """Same picture as aiortc's per-frame swscale conversion, to rounding
    (swscale filters chroma, so it differs by a few levels on edges)."""
    bgr = picture(H, W, seed=3)
    ours = frame_io.I420Converter()(bgr)
    theirs = av.VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), format="rgb24").reformat(format="yuv420p").to_ndarray()
    assert psnr(ours[:H], theirs[:H]) > 50  # luma
    assert psnr(ours[H:], theirs[H:]) > 40  # chroma


def test_i420_converter_reuses_its_scratch_and_rejects_odd_sizes():
    conv = frame_io.I420Converter()
    a = conv(picture(64, 96))
    b = conv(picture(64, 96, seed=1))
    assert a is not b and not np.array_equal(a, b)
    with pytest.raises(ValueError):
        conv(np.zeros((63, 96, 3), np.uint8))


def test_scale_i420_keeps_planes_consistent():
    bgr = picture(H, W)
    i420 = frame_io.I420Converter()(bgr)
    small = frame_io.scale_i420(i420, H, W, 160, 288)
    direct = frame_io.I420Converter()(cv2.resize(bgr, (288, 160), interpolation=cv2.INTER_AREA))
    assert small.shape == (240, 288)
    assert psnr(small, direct) > 38


def test_output_frame_derives_and_caches_its_forms():
    bgr = picture(H, W)
    frame = frame_io.OutputFrame(1, bgr)
    assert np.array_equal(frame.rgb(), bgr[:, :, ::-1])
    assert frame.i420() is frame.i420()
    assert frame.i420_at(288, 160) is frame.i420_at(288, 160)
    vf = frame.video_frame()
    assert (vf.format.name, vf.width, vf.height) == ("yuv420p", W, H)
    assert frame.video_frame() is not vf  # each viewer gets its own frame object
    assert frame.video_frame((288, 160)).width == 288


# ── egress: publisher ─────────────────────────────────────────────────────────
def test_publish_converts_eagerly_and_bumps_the_version():
    pub = frame_io.FramePublisher(i420=True)
    assert pub.latest() is None and pub.version == 0
    f = pub.publish(picture(H, W))
    assert f.version == 1 and pub.version == 1 and f._i420 is not None
    plain = frame_io.FramePublisher(i420=False).publish(picture(H, W))
    assert plain._i420 is None and plain._rgb is not None
    both = frame_io.FramePublisher(i420=True)
    both.want_rgb = True
    f = both.publish(picture(H, W))
    assert f._i420 is not None and f._rgb is not None


async def test_a_waiting_track_wakes_on_publish():
    pub = frame_io.FramePublisher()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    waiter = asyncio.ensure_future(pub.next_frame(0, timeout=2.0))
    await asyncio.sleep(0.01)
    threading.Thread(target=pub.publish, args=(picture(H, W),)).start()  # the pump thread
    frame = await waiter
    assert frame is not None and frame.version == 1
    assert loop.time() - t0 < 0.5
    # already newer: no wait at all
    assert (await pub.next_frame(0, timeout=0.0)).version == 1


async def test_next_frame_times_out_with_none():
    pub = frame_io.FramePublisher()
    pub.publish(picture(H, W))
    assert await pub.next_frame(1, timeout=0.02) is None
    assert not pub._waiters  # cleaned up


def test_wait_frame_blocks_a_thread_until_publish():
    pub = frame_io.FramePublisher(i420=False)
    threading.Timer(0.02, pub.publish, args=(np.zeros((4, 4, 3), np.uint8),)).start()
    t0 = time.monotonic()
    frame = pub.wait_frame(0, timeout=2.0)
    assert frame is not None and time.monotonic() - t0 < 1.0
    assert pub.wait_frame(1, timeout=0.01) is None


# ── ingress ───────────────────────────────────────────────────────────────────
def old_path_bgr(bgr):
    return crop_maximal_rectangle(bgr, H, W)


@pytest.mark.parametrize("size", [(320, 576), (360, 640), (480, 640), (720, 1280), (180, 320)])
def test_bgr_ingress_is_bit_identical_and_writes_in_place(size):
    bgr = picture(*size)
    dst = np.zeros((H, W, 3), np.uint8)
    frame_io.IngressConverter(H, W).write(bgr, dst)
    assert np.array_equal(dst, old_path_bgr(bgr))


@pytest.mark.parametrize("size", [(320, 576), (360, 640), (720, 1280)])
def test_rgbx_ingress_is_bit_identical(size):
    bgr = picture(*size)
    rgbx = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    old = old_path_bgr(RawFrame(rgbx).to_ndarray(format="bgr24"))
    dst = np.zeros((H, W, 3), np.uint8)
    frame_io.IngressConverter(H, W).write(RawFrame(rgbx), dst)
    assert np.array_equal(dst, old)


def yuv_frame(bgr):
    return av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p")


@pytest.mark.parametrize("size", [(320, 576), (360, 640), (480, 640), (720, 1280), (1080, 1920)])
def test_video_frame_ingress_is_bit_identical(size):
    vf = yuv_frame(picture(*size))
    dst = np.zeros((H, W, 3), np.uint8)
    frame_io.IngressConverter(H, W).write(vf, dst)
    assert np.array_equal(dst, old_path_bgr(vf.to_ndarray(format="bgr24")))


def test_area_downscale_off_uses_linear_like_crop_maximal_rectangle():
    bgr = picture(720, 1280)
    dst = np.zeros((H, W, 3), np.uint8)
    frame_io.IngressConverter(H, W, area_downscale=False).write(bgr, dst)
    assert np.array_equal(dst, crop_maximal_rectangle(bgr, H, W, area_downscale=False))


def test_writes_into_a_shared_tensor_in_place():
    from fluxrt.utils.shared_tensor import SharedTensor

    t = SharedTensor((H, W, 3), create=True)
    try:
        bgr = picture(720, 1280)
        with t.writing() as dst:
            frame_io.IngressConverter(H, W).write(bgr, dst)
        assert t.sequence() == 2
        assert np.array_equal(t.to_numpy(), old_path_bgr(bgr))
    finally:
        t.close_and_unlink()
