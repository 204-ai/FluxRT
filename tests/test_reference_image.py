"""Boot reference loader (src/fluxrt/utils/reference_image.py) — T38 / R16."""

import cv2
import numpy as np

from fluxrt.utils.reference_image import load_reference_image


def test_empty_path_is_off_with_how_to():
    image, msg = load_reference_image("", 256, 256)
    assert image is None
    assert "no reference_image_path" in msg
    assert "--set reference_image_path=" in msg and "POST /reference" in msg


def test_missing_path_names_it_and_never_calls_opencv(tmp_path, capfd):
    path = str(tmp_path / "local_samples" / "reference.png")
    image, msg = load_reference_image(path, 256, 256)
    assert image is None
    assert path in msg and "not found" in msg
    assert "--set reference_image_path=" in msg
    # the old path printed OpenCV's own "can't open/read file" warning
    assert "can't open/read file" not in capfd.readouterr().err


def test_undecodable_file_is_off(tmp_path):
    path = tmp_path / "ref.png"
    path.write_bytes(b"not an image")
    image, msg = load_reference_image(str(path), 256, 256)
    assert image is None
    assert "could not decode" in msg


def test_valid_image_is_rgb_and_resized(tmp_path):
    bgr = np.zeros((40, 60, 3), np.uint8)
    bgr[:, :] = (255, 0, 0)  # pure blue in OpenCV's BGR order
    path = str(tmp_path / "ref.png")
    assert cv2.imwrite(path, bgr)
    image, msg = load_reference_image(path, 32, 16)
    assert msg is None
    assert image.shape == (16, 32, 3) and image.dtype == np.uint8
    assert tuple(image[0, 0]) == (0, 0, 255)  # RGB: blue lands in the last channel


def test_matches_previous_inline_load(tmp_path):
    # same pixels as the code it replaced: imread -> BGR2RGB -> resize (default interp)
    rng = np.random.default_rng(0)
    path = str(tmp_path / "ref.png")
    cv2.imwrite(path, rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))
    old = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (24, 20))
    new, _ = load_reference_image(path, 24, 20)
    assert np.array_equal(old, new)
