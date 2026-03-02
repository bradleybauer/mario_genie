from __future__ import annotations

import numpy as np


TARGET_HW = (256, 256)


def pad_to_square_256(frame: np.ndarray) -> np.ndarray:
    """
    Pad NES frame (typically 240x256x3) to 256x256x3.

    Assumes RGB uint8 input with shape (H, W, C).
    Pads vertically with zeros (black bars) centered by default.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}")

    h, w, _ = frame.shape
    target_h, target_w = TARGET_HW

    if h > target_h or w > target_w:
        raise ValueError(f"Input frame {frame.shape} exceeds target {TARGET_HW}")

    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = np.pad(
        frame,
        ((top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return padded.astype(np.uint8, copy=False)
