"""Load the boot reference image (`reference_image_path`) for the inference worker.

Only a readable image counts. An empty, missing or undecodable path leaves the
reference off (its tokens stay out of the sequence until POST /reference) and
returns one line saying which case it was and how to set one — instead of
OpenCV's "can't open/read file" warning, which named neither the config key nor
the fix. A missing path is checked before cv2.imread, so OpenCV never warns.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

HOW_TO_SET = "set one with POST /reference, or at boot with --set reference_image_path=/abs/path.png"


def load_reference_image(path: str, width: int, height: int) -> tuple[np.ndarray | None, str | None]:
    """Return (RGB uint8 image resized to width x height, None), or (None, message)."""
    off = f"reference tokens stay off until one is set ({HOW_TO_SET})"
    if not path:
        return None, f"use_reference_image: no reference_image_path — {off}."
    if not os.path.isfile(path):
        return None, (
            f"use_reference_image: reference_image_path {path!r} not found "
            f"(relative paths resolve from {os.getcwd()}) — {off}."
        )
    image = cv2.imread(path)
    if image is None:
        return None, f"use_reference_image: could not decode {path!r} as an image — {off}."
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (width, height)), None
