from __future__ import annotations

from PIL import Image
import numpy as np
from mario_world_model_phase1.config import IMAGE_SIZE

TARGET_HW = (IMAGE_SIZE, IMAGE_SIZE)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Pad NES frame (typically 240x256x3) to 256x256x3, then optional resize to IMAGE_SIZE.

    Assumes RGB uint8 input with shape (H, W, C).
    Pads vertically with zeros (black bars) centered by default.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}")

    h, w, _ = frame.shape
    # Pad to 256x256 first
    pad_h = max(0, 256 - h)
    pad_w = max(0, 256 - w)
    
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
    
    if IMAGE_SIZE != 256:
        # Resize using PIL
        img = Image.fromarray(padded.astype(np.uint8, copy=False))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        padded = np.array(img)

    return padded.astype(np.uint8, copy=False)
