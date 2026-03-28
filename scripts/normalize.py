#!/usr/bin/env python3
"""Normalize raw Mesen2 recordings into compact training data.

Reads data/raw/ and produces data/normalized/:
  - Per-recording .npz files with palette-indexed frames, reduced actions,
    and reduced RAM columns.
  - Shared JSON mapping files (palette.json, actions.json, ram_addresses.json).

Usage:
    python scripts/normalize.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "normalized"
PALETTE_FILE = RAW_DIR / "smb1_palette.pal"

CROP_H = 224
CROP_W = 224
NES_W = 256
NES_H = 240


def load_palette(path: Path) -> np.ndarray:
    """Load .pal file as (N, 3) uint8 RGB array."""
    return np.fromfile(str(path), dtype=np.uint8).reshape(-1, 3)


def build_rgb_lut(palette: np.ndarray) -> np.ndarray:
    """Build (256, 256, 256) → palette_index lookup table.

    For duplicate RGB entries, the first palette index wins.
    Unmapped RGB values get 255 (sentinel).
    """
    lut = np.full((256, 256, 256), 255, dtype=np.uint8)
    for i, (r, g, b) in enumerate(palette):
        if lut[r, g, b] == 255:
            lut[r, g, b] = i
    return lut


def decode_avi(avi_path: Path, expected_frames: int) -> np.ndarray:
    """Decode AVI → (N, H, W, 3) uint8 RGB via ffmpeg."""
    cmd = [
        "ffmpeg", "-i", str(avi_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "error", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error on {avi_path.name}: {proc.stderr.decode()}")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    return raw.reshape(expected_frames, NES_H, NES_W, 3)


def center_crop(frames: np.ndarray) -> np.ndarray:
    """Center crop (N, 240, 256, 3) → (N, 224, 224, 3)."""
    y0 = (NES_H - CROP_H) // 2
    x0 = (NES_W - CROP_W) // 2
    return frames[:, y0:y0 + CROP_H, x0:x0 + CROP_W]


def rgb_to_palette_indices(
    frames_rgb: np.ndarray,
    lut: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """Map (N, H, W, 3) RGB → (N, H, W) palette indices.

    Uses the precomputed exact-match LUT. Any pixel that doesn't match
    exactly (shouldn't happen with lossless video) falls back to
    nearest-neighbor and updates the LUT for future lookups.
    """
    indices = lut[frames_rgb[..., 0], frames_rgb[..., 1], frames_rgb[..., 2]]

    unmatched = indices == 255
    if unmatched.any():
        um_pixels = frames_rgb[unmatched]
        unique_um = np.unique(um_pixels, axis=0)
        print(f"    Warning: {len(unique_um)} unmatched color(s), using nearest neighbor")
        pal16 = palette.astype(np.int16)
        for color in unique_um:
            dists = np.sum((pal16 - color.astype(np.int16)) ** 2, axis=1)
            lut[color[0], color[1], color[2]] = np.argmin(dists)
        indices = lut[frames_rgb[..., 0], frames_rgb[..., 1], frames_rgb[..., 2]]

    return indices


def scan_raw_recordings() -> list[Path]:
    """Find recording base paths in RAW_DIR (keyed by .ram.npy presence)."""
    return [
        Path(str(f).removesuffix(".ram.npy"))
        for f in sorted(RAW_DIR.glob("*.ram.npy"))
    ]


def validate_recording_lengths(base: Path, frame_numbers: np.ndarray, actions: np.ndarray, ram: np.ndarray) -> None:
    """Ensure all per-frame arrays agree on the number of frames."""
    expected = len(frame_numbers)
    actual = {
        "frames": expected,
        "actions": len(actions),
        "ram": len(ram),
    }
    if len(set(actual.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in actual.items())
        raise ValueError(f"Mismatched frame counts for {base.name}: {details}")


def main():
    palette = load_palette(PALETTE_FILE)
    lut = build_rgb_lut(palette)
    bases = scan_raw_recordings()

    if not bases:
        print("No recordings found in", RAW_DIR, file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(bases)} recording(s)\n")

    # ── Load all recordings into memory ──────────────────────────────
    loaded = []
    for base in bases:
        frame_numbers = np.load(str(base) + ".frames.npy")
        actions = np.load(str(base) + ".input.npy")
        ram = np.load(str(base) + ".ram.npy")
        validate_recording_lengths(base, frame_numbers, actions, ram)

        n = len(frame_numbers)
        avi = Path(str(base) + ".avi")

        print(f"  Loading {base.name} ({n} frames) ...")
        rgb = decode_avi(avi, n)
        cropped = center_crop(rgb)
        pal_idx = rgb_to_palette_indices(cropped, lut, palette)

        loaded.append((base.name, pal_idx, actions, ram))

    # ── Compute reduced mappings across the full dataset ─────────────

    # Palette: which indices actually appear after crop?
    used_pal = set()
    for _, pal_idx, _, _ in loaded:
        used_pal.update(np.unique(pal_idx).tolist())
    used_pal_sorted = sorted(used_pal)
    pal_remap = np.zeros(len(palette), dtype=np.uint8)
    for new, old in enumerate(used_pal_sorted):
        pal_remap[old] = new

    # Actions: which button combinations actually appear?
    used_act = set()
    for _, _, actions, _ in loaded:
        used_act.update(np.unique(actions).tolist())
    used_act_sorted = sorted(used_act)
    act_remap = np.zeros(256, dtype=np.uint8)
    for new, old in enumerate(used_act_sorted):
        act_remap[old] = new

    # RAM: which columns are NOT constant across every frame of every recording?
    all_ram = np.concatenate([ram for _, _, _, ram in loaded])
    constant_mask = all_ram.min(axis=0) == all_ram.max(axis=0)
    kept_cols = np.where(~constant_mask)[0]

    print(f"\nPalette: {len(used_pal_sorted)} / {len(palette)} colors used")
    print(f"Actions: {len(used_act_sorted)} unique values")
    print(f"RAM:     {len(kept_cols)} / {all_ram.shape[1]} non-constant addresses")

    # ── Write JSON mapping files ─────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    palette_info = {
        "num_colors": len(used_pal_sorted),
        "reduced_to_original_index": used_pal_sorted,
        "colors_rgb": [palette[i].tolist() for i in used_pal_sorted],
    }
    action_info = {
        "num_actions": len(used_act_sorted),
        "reduced_to_original_value": used_act_sorted,
    }
    ram_info = {
        "num_addresses": int(len(kept_cols)),
        "kept_addresses": kept_cols.tolist(),
    }

    for name, obj in [
        ("palette.json", palette_info),
        ("actions.json", action_info),
        ("ram_addresses.json", ram_info),
    ]:
        path = OUT_DIR / name
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"  Wrote {path}")

    # ── Convert and save each recording ──────────────────────────────
    for rec_name, pal_idx, actions, ram in loaded:
        frames_out = pal_remap[pal_idx]
        actions_out = act_remap[actions]
        ram_out = ram[:, kept_cols]

        out_path = OUT_DIR / (rec_name + ".npz")
        np.savez_compressed(
            str(out_path),
            frames=frames_out,
            actions=actions_out,
            ram=ram_out,
        )
        print(f"  Saved {out_path.name}  "
              f"frames={frames_out.shape} actions={actions_out.shape} ram={ram_out.shape}")

    print(f"\nDone. Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
