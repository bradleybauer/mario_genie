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

    def __init__(self, palette_path: str | Path):
        self._path = Path(palette_path)
        self._lock = threading.Lock()

        # rgb_to_idx: packed int (r<<16 | g<<8 | b) → uint8 index
        self._rgb_to_idx: dict[int, int] = {}
        # palette: list of [r, g, b] for serialisation
        self._palette: list[list[int]] = []

        if self._path.exists():
            with open(self._path, "r") as f:
                stored = json.load(f)
            for rgb in stored:
                self._register_color(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            print(f"[palette] Loaded {len(self._palette)} colours from {self._path}")
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── public API ───────────────────────────────────────────────────

    @property
    def num_colors(self) -> int:
        return len(self._palette)

    def map_frame(self, frame_hwc: np.ndarray) -> np.ndarray:
        """Convert an RGB ``(H, W, 3)`` uint8 frame to ``(H, W)`` uint8 indices."""
        h, w, _ = frame_hwc.shape
        flat = frame_hwc.reshape(-1, 3)

        packed = (
            flat[:, 0].astype(np.int32) << 16
            | flat[:, 1].astype(np.int32) << 8
            | flat[:, 2].astype(np.int32)
        )
        unique_packed = np.unique(packed)

        # Fast path: check for new colours
        new_colors = []
        with self._lock:
            for p in unique_packed:
                if int(p) not in self._rgb_to_idx:
                    r = (int(p) >> 16) & 0xFF
                    g = (int(p) >> 8) & 0xFF
                    b = int(p) & 0xFF
                    new_colors.append((r, g, b, int(p)))

            if new_colors:
                for r, g, b, p in new_colors:
                    self._register_color(r, g, b)
                self._save()
                print(
                    f"[palette] +{len(new_colors)} new colours → "
                    f"{len(self._palette)} total"
                )

            # Vectorised lookup via a numpy array indexed by packed RGB
            # Build a lookup table only for the packed values in this frame
            out = np.empty(len(packed), dtype=np.uint8)
            for i, p in enumerate(packed):
                out[i] = self._rgb_to_idx[int(p)]

        return out.reshape(h, w)

    def get_palette_rgb(self) -> np.ndarray:
        """Return the current palette as ``(K, 3)`` uint8 array."""
        with self._lock:
            return np.array(self._palette, dtype=np.uint8)

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
