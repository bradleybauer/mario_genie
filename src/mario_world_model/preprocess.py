from __future__ import annotations

from PIL import Image
import numpy as np
from mario_world_model.config import IMAGE_SIZE

# Pre-compute padding for the standard NES frame size (240×256).
# Avoids recalculating and reallocating np.pad every frame.
_NES_H, _NES_W = 240, 256
_PAD_H = max(0, 256 - _NES_H)
_PAD_TOP = _PAD_H // 2
_PAD_BOTTOM = _PAD_H - _PAD_TOP
# Width padding is 0 for 256-wide frames, so no horizontal pad needed.

# Reusable output buffer for the standard case (256×256×3)
_PADDED_BUF: np.ndarray | None = None


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Pad NES frame (typically 240x256x3) to 256x256x3, then optional resize to IMAGE_SIZE.

    Assumes RGB uint8 input with shape (H, W, C).
    Pads vertically with zeros (black bars) centered by default.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {frame.shape}")

    h, w, _ = frame.shape

    # Fast path for standard NES frames (240×256) — skip np.pad overhead
    if h == _NES_H and w == _NES_W:
        global _PADDED_BUF
        if _PADDED_BUF is None:
            _PADDED_BUF = np.zeros((256, 256, 3), dtype=np.uint8)
        _PADDED_BUF[_PAD_TOP:_PAD_TOP + _NES_H, :, :] = frame
        padded = _PADDED_BUF
    else:
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
        img = Image.fromarray(padded.astype(np.uint8, copy=False))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        padded = np.array(img)

    return padded.astype(np.uint8, copy=False)
