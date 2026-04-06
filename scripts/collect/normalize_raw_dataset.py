#!/usr/bin/env python3
"""Convert raw Mesen2 recordings into compact training data.

Reads data/raw/ and produces data/normalized/:
  - Per-recording .npz files with palette-indexed frames, reduced actions,
        reduced RAM columns, frame-aligned audio, and audio metadata.
  - Shared JSON mapping files (palette.json, palette_distribution.json,
      actions.json, ram_addresses.json).

Usage:
    python scripts/normalize.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "normalized"
PALETTE_FILE = RAW_DIR / "smb1_palette.pal"

# Suffixes associated with each recording base path
RAW_SUFFIXES = [".frames.npy", ".input.npy", ".ram.npy", ".avi", ".mss", ".wram.npy"]

CROP_H = 224
CROP_W = 224
NES_W = 256
NES_H = 240

# Audio normalization parameters (derived from dataset-wide spectrum analysis)
AUDIO_SAMPLE_RATE = 24000  # Retains 98.5% of dataset spectral energy
AUDIO_DTYPE = np.int16


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
    frame_bytes = NES_H * NES_W * 3
    actual_frames = len(raw) // frame_bytes
    if actual_frames < expected_frames:
        raise ValueError(
            f"AVI has {actual_frames} frames but expected {expected_frames} in {avi_path.name}"
        )
    if actual_frames != expected_frames:
        print(f"    Note: AVI has {actual_frames} frames, trimming to {expected_frames}")
    raw = raw[: expected_frames * frame_bytes]
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


def decode_audio(avi_path: Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Decode full audio track from AVI → (S,) int16 mono at *sample_rate* via ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to extract audio from AVI files")
    cmd = [
        "ffmpeg", "-i", str(avi_path),
        "-vn", "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-v", "error", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio error on {avi_path.name}: {proc.stderr.decode()}"
        )
    return np.frombuffer(proc.stdout, dtype=np.int16)


def frame_aligned_audio(
    audio: np.ndarray,
    n_frames: int,
    fps: float,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice mono audio into (N, samples_per_frame) aligned to video frames.

    Uses cumulative rounding so drift never exceeds one sample over the
    full recording, even though fps is not an exact divisor of sample_rate.
    Each frame's chunk is truncated or zero-padded to a fixed window width
    (the ceiling of samples_per_frame) so the output array is rectangular.
    Returns both the padded audio chunks and the valid sample count for each
    frame before zero padding.
    """
    samples_per_frame = sample_rate / fps
    window_size = int(np.ceil(samples_per_frame))
    boundaries = np.round(np.arange(n_frames + 1) * samples_per_frame).astype(np.int64)

    out = np.zeros((n_frames, window_size), dtype=audio.dtype)
    lengths = np.zeros((n_frames,), dtype=np.uint16)
    for i in range(n_frames):
        start = int(boundaries[i])
        end = min(int(boundaries[i + 1]), len(audio))
        length = min(end - start, window_size)
        if length > 0:
            out[i, :length] = audio[start : start + length]
            lengths[i] = length
    return out, lengths


def probe_fps(avi_path: Path) -> float:
    """Read the video frame rate from an AVI file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(avi_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed on {avi_path.name}")
    num, den = result.stdout.strip().split("/")
    return int(num) / int(den)


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

    # ── Remove empty recordings (0 frames) ───────────────────────────
    valid_bases = []
    for base in bases:
        frame_numbers = np.load(str(base) + ".frames.npy")
        if len(frame_numbers) == 0:
            print(f"  Deleting empty recording: {base.name}")
            for suffix in RAW_SUFFIXES:
                p = Path(str(base) + suffix)
                if p.exists():
                    p.unlink()
            # Also delete periodic save-state directory if present
            states_dir = Path(str(base) + "_states")
            if states_dir.is_dir():

                shutil.rmtree(states_dir)
        else:
            valid_bases.append(base)
    bases = valid_bases

    if not bases:
        print("No valid recordings remaining.", file=sys.stderr)
        sys.exit(1)

    # ── Pass 1: collect unique palette/action/RAM stats (no frames kept) ──
    print("Pass 1: scanning metadata and frames for mappings ...")
    used_pal = set()
    palette_counts_full = np.zeros(len(palette), dtype=np.int64)
    used_act = set()
    ram_min = None
    ram_max = None

    for base in bases:
        frame_numbers = np.load(str(base) + ".frames.npy")
        actions = np.load(str(base) + ".input.npy")
        ram = np.load(str(base) + ".ram.npy")
        validate_recording_lengths(base, frame_numbers, actions, ram)

        n = len(frame_numbers)
        avi = Path(str(base) + ".avi")

        print(f"  Scanning {base.name} ({n} frames) ...")
        rgb = decode_avi(avi, n)
        cropped = center_crop(rgb)
        pal_idx = rgb_to_palette_indices(cropped, lut, palette)

        palette_counts_full += np.bincount(
            pal_idx.ravel(),
            minlength=len(palette),
        )[:len(palette)]
        used_pal.update(np.unique(pal_idx).tolist())
        used_act.update(np.unique(actions).tolist())

        if ram_min is None:
            ram_min = ram.min(axis=0)
            ram_max = ram.max(axis=0)
        else:
            ram_min = np.minimum(ram_min, ram.min(axis=0))
            ram_max = np.maximum(ram_max, ram.max(axis=0))

        del rgb, cropped, pal_idx  # free memory

    # Build remapping tables
    used_pal_sorted = sorted(used_pal)
    pal_remap = np.zeros(len(palette), dtype=np.uint8)
    for new, old in enumerate(used_pal_sorted):
        pal_remap[old] = new

    reduced_palette_counts = palette_counts_full[used_pal_sorted]
    total_palette_pixels = int(reduced_palette_counts.sum())
    if total_palette_pixels > 0:
        reduced_palette_probs = reduced_palette_counts.astype(np.float64) / total_palette_pixels
    else:
        reduced_palette_probs = np.zeros_like(reduced_palette_counts, dtype=np.float64)

    used_act_sorted = sorted(used_act)
    act_remap = np.zeros(256, dtype=np.uint8)
    for new, old in enumerate(used_act_sorted):
        act_remap[old] = new

    constant_mask = ram_min == ram_max
    # Exclude Stack ($0100–$01FF) and OAM ($0200–$02FF) — noise / redundant with image
    stack_oam_mask = np.zeros(ram_min.shape[0], dtype=bool)
    stack_oam_mask[0x100:0x300] = True
    kept_cols = np.where(~constant_mask & ~stack_oam_mask)[0]

    print(f"\nPalette: {len(used_pal_sorted)} / {len(palette)} colors used")
    print(f"Actions: {len(used_act_sorted)} unique values")
    print(f"RAM:     {len(kept_cols)} / {ram_min.shape[0]} non-constant addresses (Stack+OAM excluded)")

    # ── Write JSON mapping files ─────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    palette_info = {
        "num_colors": len(used_pal_sorted),
        "reduced_to_original_index": used_pal_sorted,
        "colors_rgb": [palette[i].tolist() for i in used_pal_sorted],
    }
    palette_distribution_info = {
        "num_colors": len(used_pal_sorted),
        "total_pixels": total_palette_pixels,
        "counts": [int(v) for v in reduced_palette_counts.tolist()],
        "probabilities": [float(v) for v in reduced_palette_probs.tolist()],
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
        ("palette_distribution.json", palette_distribution_info),
        ("actions.json", action_info),
        ("ram_addresses.json", ram_info),
    ]:
        path = OUT_DIR / name
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"  Wrote {path}")

    # ── Pass 2: re-decode, remap, and save each recording ────────────
    print("\nPass 2: converting and saving ...")
    for base in bases:
        frame_numbers = np.load(str(base) + ".frames.npy")
        actions = np.load(str(base) + ".input.npy")
        ram = np.load(str(base) + ".ram.npy")

        n = len(frame_numbers)
        avi = Path(str(base) + ".avi")

        print(f"  Processing {base.name} ({n} frames) ...")
        rgb = decode_avi(avi, n)
        cropped = center_crop(rgb)
        pal_idx = rgb_to_palette_indices(cropped, lut, palette)
        del rgb, cropped

        frames_out = pal_remap[pal_idx]
        del pal_idx
        actions_out = act_remap[actions]
        ram_out = ram[:, kept_cols]

        # Extract frame-aligned mono audio at the target sample rate
        fps = probe_fps(avi)
        raw_audio = decode_audio(avi)
        audio_out, audio_lengths = frame_aligned_audio(raw_audio, n, fps)
        del raw_audio

        out_path = OUT_DIR / (base.name + ".npz")
        np.savez_compressed(
            str(out_path),
            frames=frames_out,
            actions=actions_out,
            ram=ram_out,
            audio=audio_out,
            audio_lengths=audio_lengths,
            audio_sample_rate=np.int32(AUDIO_SAMPLE_RATE),
            video_fps=np.float32(fps),
        )
        print(f"  Saved {out_path.name}  "
              f"frames={frames_out.shape} actions={actions_out.shape} "
              f"ram={ram_out.shape} audio={audio_out.shape}")
        del frames_out, actions_out, ram_out, audio_out, audio_lengths

    print(f"\nDone. Output in {OUT_DIR}")


if __name__ == "__main__":
    main()
