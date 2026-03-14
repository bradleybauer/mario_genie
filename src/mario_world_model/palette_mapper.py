"""Dynamic NES palette tracker.

Maintains a growing mapping from RGB colours → uint8 palette indices.
New colours observed at runtime are assigned the next free index and the
palette file is re-saved automatically.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np


class PaletteMapper:
    """Maps RGB pixels to palette indices, growing the palette on the fly.

    Parameters
    ----------
    palette_path : str | Path
        JSON file that stores ``[[r, g, b], ...]`` (uint8 values).
        Created if it does not exist; updated whenever new colours appear.
    """

    # Dense LUT: maps packed RGB (r<<16|g<<8|b) → uint8 index.
    # Max packed value for uint8 RGB is 0xFFFFFF (~16M), but NES palettes
    # have ≤64 unique colours so we use a sparse→dense approach:
    # a fixed-size numpy array covering the full 24-bit RGB space would be
    # 16 MB — acceptable.  Rebuilt only when new colours appear.
    _LUT_SIZE = (1 << 24)  # 16,777,216

    def __init__(self, palette_path: str | Path):
        self._path = Path(palette_path)
        self._lock = threading.Lock()

        # rgb_to_idx: packed int (r<<16 | g<<8 | b) → uint8 index
        self._rgb_to_idx: dict[int, int] = {}
        # palette: list of [r, g, b] for serialisation
        self._palette: list[list[int]] = []
        # Dense vectorised LUT (rebuilt when palette changes)
        self._lut: np.ndarray | None = None
        # Counter for frames since last new colour — enables fast path
        self._frames_since_new: int = 0

        if self._path.exists():
            with open(self._path, "r") as f:
                stored = json.load(f)
            for rgb in stored:
                self._register_color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            self._rebuild_lut()
            print(f"[palette] Loaded {len(self._palette)} colours from {self._path}")
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── public API ───────────────────────────────────────────────────

    @property
    def num_colors(self) -> int:
        return len(self._palette)

    def _rebuild_lut(self) -> None:
        """Rebuild the dense numpy LUT from the current palette dict."""
        lut = np.zeros(self._LUT_SIZE, dtype=np.uint8)
        for packed, idx in self._rgb_to_idx.items():
            lut[packed] = idx
        self._lut = lut

    def map_frame(self, frame_hwc: np.ndarray) -> np.ndarray:
        """Convert an RGB ``(H, W, 3)`` uint8 frame to ``(H, W)`` uint8 indices."""
        h, w, _ = frame_hwc.shape
        flat = frame_hwc.reshape(-1, 3)

        packed = (
            flat[:, 0].astype(np.int32) << 16
            | flat[:, 1].astype(np.int32) << 8
            | flat[:, 2].astype(np.int32)
        )

        if self._lut is not None and self._frames_since_new > 60:
            # Fast path: palette is settled, skip np.unique entirely
            return self._lut[packed].reshape(h, w)

        # Check for new colours (rare after warmup)
        if self._lut is not None:
            unique_packed = np.unique(packed)
            new_colors = []
            with self._lock:
                for p in unique_packed:
                    if int(p) not in self._rgb_to_idx:
                        pi = int(p)
                        new_colors.append(((pi >> 16) & 0xFF, (pi >> 8) & 0xFF, pi & 0xFF))

                if new_colors:
                    self._frames_since_new = 0
                    for r, g, b in new_colors:
                        self._register_color(r, g, b)
                    self._save()
                    self._rebuild_lut()
                    print(
                        f"[palette] +{len(new_colors)} new colours → "
                        f"{len(self._palette)} total"
                    )
                else:
                    self._frames_since_new += 1
        else:
            # First frame — register all colours and build LUT
            unique_packed = np.unique(packed)
            with self._lock:
                for p in unique_packed:
                    pi = int(p)
                    if pi not in self._rgb_to_idx:
                        self._register_color((pi >> 16) & 0xFF, (pi >> 8) & 0xFF, pi & 0xFF)
                self._save()
                self._rebuild_lut()
                print(f"[palette] Initial scan → {len(self._palette)} colours")

        # Vectorised lookup via dense LUT — no Python loop over pixels
        return self._lut[packed].reshape(h, w)

    # ── internals ────────────────────────────────────────────────────

    def _register_color(self, r: int, g: int, b: int) -> int:
        key = (r << 16) | (g << 8) | b
        if key in self._rgb_to_idx:
            return self._rgb_to_idx[key]
        idx = len(self._palette)
        if idx > 255:
            raise RuntimeError(
                f"Palette overflow: more than 256 unique colours observed. "
                f"Check that nearest-neighbour downsampling is active."
            )
        self._rgb_to_idx[key] = idx
        self._palette.append([r, g, b])
        return idx

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._palette, f)
