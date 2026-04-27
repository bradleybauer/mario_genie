#!/usr/bin/env python3
"""Convert raw Mesen2 recordings into compact training data.

Reads data/raw/ and produces data/normalized/:
  - Per-recording .npz files with palette-indexed frames, reduced actions,
        reduced RAM columns, frame-aligned audio, and audio metadata.
  - Shared JSON mapping files (palette.json, palette_distribution.json,
      actions.json, ram_addresses.json).

Usage:
    python scripts/collect/normalize_raw_dataset.py
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.smb1_memory_map import SMB1_RAM_LABELS  # noqa: E402

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

CONSOLE = Console()

_PASS1_PALETTE: np.ndarray | None = None
_PASS1_LUT: np.ndarray | None = None
_PASS1_TMPDIR: Path | None = None

_PASS2_PAL_REMAP: np.ndarray | None = None
_PASS2_ACT_REMAP: np.ndarray | None = None
_PASS2_KEPT_COLS: np.ndarray | None = None
_PASS2_OUT_DIR: Path | None = None


def build_progress() -> Progress:
    """Build a Rich progress bar matching existing repo scripts."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.fields[rate]:>6.1f} rec/s"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        refresh_per_second=8,
    )


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


def decode_avi_and_audio(
    avi_path: Path,
    expected_frames: int,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode video, audio, and fps from AVI.

    Returns (rgb_frames, audio_samples, fps).
    """
    # Quick metadata read for fps
    fps = probe_fps(avi_path)

    video_cmd = [
        "ffmpeg", "-threads", "1", "-i", str(avi_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "error", "-",
    ]
    audio_cmd = [
        "ffmpeg", "-threads", "1", "-i", str(avi_path),
        "-vn", "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-v", "error", "-",
    ]
    video_result = subprocess.run(video_cmd, capture_output=True)
    if video_result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg video error on {avi_path.name}: {video_result.stderr.decode()}"
        )

    audio_result = subprocess.run(audio_cmd, capture_output=True)
    if audio_result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio error on {avi_path.name}: {audio_result.stderr.decode()}"
        )

    # Parse video
    raw = np.frombuffer(video_result.stdout, dtype=np.uint8)
    frame_bytes = NES_H * NES_W * 3
    actual_frames = len(raw) // frame_bytes
    if actual_frames < expected_frames:
        raise ValueError(
            f"AVI has {actual_frames} frames but expected {expected_frames} in {avi_path.name}"
        )
    if actual_frames != expected_frames:
        print(f"    Note: AVI has {actual_frames} frames, trimming to {expected_frames}")
    raw = raw[: expected_frames * frame_bytes]
    rgb = raw.reshape(expected_frames, NES_H, NES_W, 3)

    # Parse audio
    audio = np.frombuffer(audio_result.stdout, dtype=AUDIO_DTYPE)

    return rgb, audio, fps


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


def init_pass1_worker(palette: np.ndarray, tmpdir_str: str) -> None:
    """Initialize per-process state for the decode/statistics pass."""
    global _PASS1_PALETTE, _PASS1_LUT, _PASS1_TMPDIR
    _PASS1_PALETTE = palette
    _PASS1_LUT = build_rgb_lut(palette)
    _PASS1_TMPDIR = Path(tmpdir_str)


def process_recording_pass1(task: tuple[int, str]) -> dict[str, Any]:
    """Decode one recording, cache intermediate arrays, and return summary stats."""
    if _PASS1_PALETTE is None or _PASS1_LUT is None or _PASS1_TMPDIR is None:
        raise RuntimeError("Pass 1 worker state was not initialized")

    index, base_str = task
    base = Path(base_str)
    frame_numbers = np.load(str(base) + ".frames.npy")
    actions = np.load(str(base) + ".input.npy")
    ram = np.load(str(base) + ".ram.npy")
    validate_recording_lengths(base, frame_numbers, actions, ram)

    frame_count = len(frame_numbers)
    avi = Path(str(base) + ".avi")
    rgb, raw_audio, fps = decode_avi_and_audio(avi, frame_count)
    cropped = center_crop(rgb)
    del rgb
    pal_idx = rgb_to_palette_indices(cropped, _PASS1_LUT, _PASS1_PALETTE)
    del cropped

    palette_counts = np.bincount(pal_idx.ravel(), minlength=len(_PASS1_PALETTE))[: len(_PASS1_PALETTE)]
    ram_min = ram.min(axis=0)
    ram_max = ram.max(axis=0)
    ram_unique_per_col = [np.unique(ram[:, c]).tolist() for c in range(ram.shape[1])]
    audio_out, audio_lengths = frame_aligned_audio(raw_audio, frame_count, fps)
    del raw_audio

    pal_path = _PASS1_TMPDIR / f"{index}_pal.npy"
    audio_path = _PASS1_TMPDIR / f"{index}_audio.npy"
    audio_len_path = _PASS1_TMPDIR / f"{index}_audio_len.npy"
    np.save(str(pal_path), pal_idx)
    np.save(str(audio_path), audio_out)
    np.save(str(audio_len_path), audio_lengths)

    return {
        "index": index,
        "name": base.name,
        "frame_count": frame_count,
        "fps": float(fps),
        "palette_counts": palette_counts,
        "used_palette": np.unique(pal_idx),
        "used_actions": np.unique(actions),
        "ram_min": ram_min,
        "ram_max": ram_max,
        "ram_unique_per_col": ram_unique_per_col,
        "pal_idx_path": str(pal_path),
        "audio_path": str(audio_path),
        "audio_lengths_path": str(audio_len_path),
    }


def init_pass2_worker(
    pal_remap: np.ndarray,
    act_remap: np.ndarray,
    kept_cols: np.ndarray,
    out_dir_str: str,
) -> None:
    """Initialize per-process state for the remap/save pass."""
    global _PASS2_PAL_REMAP, _PASS2_ACT_REMAP, _PASS2_KEPT_COLS, _PASS2_OUT_DIR
    _PASS2_PAL_REMAP = pal_remap
    _PASS2_ACT_REMAP = act_remap
    _PASS2_KEPT_COLS = kept_cols
    _PASS2_OUT_DIR = Path(out_dir_str)


def process_recording_pass2(task: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
    """Load cached arrays, remap them, and write a normalized .npz file."""
    if _PASS2_PAL_REMAP is None or _PASS2_ACT_REMAP is None or _PASS2_KEPT_COLS is None or _PASS2_OUT_DIR is None:
        raise RuntimeError("Pass 2 worker state was not initialized")

    index, base_str, cache_entry = task
    base = Path(base_str)
    actions = np.load(str(base) + ".input.npy")
    ram = np.load(str(base) + ".ram.npy")

    pal_path = Path(cache_entry["pal_idx_path"])
    audio_path = Path(cache_entry["audio_path"])
    audio_lengths_path = Path(cache_entry["audio_lengths_path"])

    pal_idx = np.load(str(pal_path))
    audio_out = np.load(str(audio_path))
    audio_lengths = np.load(str(audio_lengths_path))

    frames_out = _PASS2_PAL_REMAP[pal_idx].astype(np.uint8, copy=False)
    actions_out = _PASS2_ACT_REMAP[actions]
    ram_out = ram[:, _PASS2_KEPT_COLS]

    out_path = _PASS2_OUT_DIR / (base.name + ".npz")
    np.savez_compressed(
        str(out_path),
        frames=frames_out,
        actions=actions_out,
        ram=ram_out,
        audio=audio_out,
        audio_lengths=audio_lengths,
        audio_sample_rate=np.int32(AUDIO_SAMPLE_RATE),
        video_fps=np.float32(cache_entry["fps"]),
    )

    pal_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)
    audio_lengths_path.unlink(missing_ok=True)

    return {
        "index": index,
        "name": base.name,
        "frames_shape": tuple(int(v) for v in frames_out.shape),
        "actions_shape": tuple(int(v) for v in actions_out.shape),
        "ram_shape": tuple(int(v) for v in ram_out.shape),
        "audio_shape": tuple(int(v) for v in audio_out.shape),
    }


def run_tasks(
    tasks: list[Any],
    *,
    description: str,
    fn,
    initializer=None,
    initargs: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    """Run tasks sequentially with a Rich progress bar."""
    if not tasks:
        return []

    results: list[dict[str, Any] | None] = [None] * len(tasks)
    start_time = time.time()
    completed = 0

    with build_progress() as progress:
        task_id = progress.add_task(description, total=len(tasks), rate=0.0)

        if initializer is not None:
            initializer(*initargs)
        for task in tasks:
            result = fn(task)
            results[result["index"]] = result
            completed += 1
            elapsed = max(time.time() - start_time, 1e-9)
            progress.update(task_id, advance=1, rate=completed / elapsed)

    return [result for result in results if result is not None]


def deduplicate_vary_together(
    bases: list[Path], kept_cols: np.ndarray
) -> np.ndarray:
    """Collapse kept RAM addresses that share identical per-frame traces.

    Streams each recording's .ram.npy and updates one hasher per kept column so
    memory stays bounded. For every vary-together group, the lowest-indexed
    address is retained as the representative and the others are dropped from
    ``kept_cols``. Returns the reduced kept_cols array.
    """
    if kept_cols.size < 2:
        return kept_cols

    n_cols = int(kept_cols.size)
    hashers = [hashlib.blake2b(digest_size=16) for _ in range(n_cols)]
    for base in bases:
        ram = np.load(str(base) + ".ram.npy", mmap_mode="r")
        kept = np.ascontiguousarray(ram[:, kept_cols])
        for i in range(n_cols):
            hashers[i].update(np.ascontiguousarray(kept[:, i]).tobytes())

    buckets: dict[bytes, list[int]] = {}
    for i, h in enumerate(hashers):
        buckets.setdefault(h.digest(), []).append(i)

    groups = [group for group in buckets.values() if len(group) >= 2]
    groups.sort(key=lambda g: (-len(g), g[0]))

    grouped_cols = sum(len(g) for g in groups)
    dropped_cols = sum(len(g) - 1 for g in groups)
    CONSOLE.print(
        f"RAM:     {len(groups)} vary-together group(s) covering {grouped_cols} / {n_cols} kept addresses; "
        f"dropping {dropped_cols} redundant address(es)"
    )

    if not groups:
        return kept_cols

    shown = groups
    for gi, group in enumerate(shown, start=1):
        addrs = [int(kept_cols[i]) for i in group]
        addr_strs = []
        for a in addrs:
            label = SMB1_RAM_LABELS.get(a, ("", ""))[0]
            addr_strs.append(f"${a:04X}" + (f"({label})" if label else ""))
        kept_addr = int(kept_cols[group[0]])
        CONSOLE.print(
            f"  group {gi} (size={len(group)}, keep ${kept_addr:04X}): {', '.join(addr_strs)}"
        )
    if len(groups) > len(shown):
        CONSOLE.print(f"  ... {len(groups) - len(shown)} more group(s) hidden")

    drop_local = {i for group in groups for i in group[1:]}
    keep_local = np.array(
        [i for i in range(n_cols) if i not in drop_local], dtype=kept_cols.dtype
    )
    return kept_cols[keep_local]


def filter_empty_recordings(bases: list[Path]) -> list[Path]:
    """Delete empty recordings in place and return the remaining base paths."""
    valid_bases: list[Path] = []
    deleted_count = 0
    for base in bases:
        frame_numbers = np.load(str(base) + ".frames.npy")
        if len(frame_numbers) == 0:
            deleted_count += 1
            for suffix in RAW_SUFFIXES:
                path = Path(str(base) + suffix)
                if path.exists():
                    path.unlink()
            states_dir = Path(str(base) + "_states")
            if states_dir.is_dir():
                shutil.rmtree(states_dir)
            continue
        valid_bases.append(base)

    if deleted_count:
        CONSOLE.print(f"Removed {deleted_count} empty recording(s).")
    return valid_bases


def main():
    palette = load_palette(PALETTE_FILE)
    bases = scan_raw_recordings()

    if not bases:
        print("No recordings found in", RAW_DIR, file=sys.stderr)
        sys.exit(1)

    CONSOLE.print(f"Found {len(bases)} recording(s) in {RAW_DIR}")
    bases = filter_empty_recordings(bases)

    if not bases:
        print("No valid recordings remaining.", file=sys.stderr)
        sys.exit(1)

    CONSOLE.print("Using sequential processing")

    # ── Pass 1: decode each AVI once, collect stats, cache to temp ──
    tmpdir = Path(tempfile.mkdtemp(prefix="normalize_"))
    CONSOLE.print("Pass 1: decoding, cropping, palette indexing, and audio alignment ...")

    used_pal: set[int] = set()
    palette_counts_full = np.zeros(len(palette), dtype=np.int64)
    used_act: set[int] = set()
    ram_min: np.ndarray | None = None
    ram_max: np.ndarray | None = None
    ram_unique_sets: list[set[int]] | None = None
    cache_meta: list[dict[str, Any]] = [dict() for _ in bases]

    try:
        pass1_results = run_tasks(
            [(i, str(base)) for i, base in enumerate(bases)],
            description="Pass 1: decoding recordings",
            fn=process_recording_pass1,
            initializer=init_pass1_worker,
            initargs=(palette, str(tmpdir)),
        )

        for result in pass1_results:
            palette_counts_full += np.asarray(result["palette_counts"], dtype=np.int64)
            used_pal.update(int(v) for v in np.asarray(result["used_palette"]).tolist())
            used_act.update(int(v) for v in np.asarray(result["used_actions"]).tolist())
            file_ram_min = np.asarray(result["ram_min"])
            file_ram_max = np.asarray(result["ram_max"])
            file_unique = result["ram_unique_per_col"]
            if ram_min is None:
                ram_min = file_ram_min
                ram_max = file_ram_max
                ram_unique_sets = [set(vals) for vals in file_unique]
            else:
                ram_min = np.minimum(ram_min, file_ram_min)
                ram_max = np.maximum(ram_max, file_ram_max)
                for col, vals in enumerate(file_unique):
                    ram_unique_sets[col].update(vals)
            cache_meta[result["index"]] = {
                "pal_idx_path": result["pal_idx_path"],
                "audio_path": result["audio_path"],
                "audio_lengths_path": result["audio_lengths_path"],
                "fps": result["fps"],
            }

        if ram_min is None or ram_max is None or ram_unique_sets is None:
            raise RuntimeError("No recording statistics were collected")

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

        CONSOLE.print(f"Palette: {len(used_pal_sorted)} / {len(palette)} colors used")
        CONSOLE.print(f"Actions: {len(used_act_sorted)} unique values")
        CONSOLE.print(
            f"RAM:     {len(kept_cols)} / {ram_min.shape[0]} non-constant addresses (Stack+OAM excluded)"
        )
        kept_cols = deduplicate_vary_together(bases, kept_cols)
        CONSOLE.print(f"RAM:     {len(kept_cols)} address(es) kept after vary-together dedup")

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
        values_per_address = [sorted(ram_unique_sets[c]) for c in kept_cols]
        ram_info = {
            "num_addresses": int(len(kept_cols)),
            "kept_addresses": kept_cols.tolist(),
            "values_per_address": values_per_address,
        }

        for name, obj in [
            ("palette.json", palette_info),
            ("palette_distribution.json", palette_distribution_info),
            ("actions.json", action_info),
            ("ram_addresses.json", ram_info),
        ]:
            path = OUT_DIR / name
            with path.open("w") as handle:
                json.dump(obj, handle, indent=2)
            CONSOLE.print(f"Wrote {path}")

        # ── Pass 2: load cached data, remap, and save (no ffmpeg) ────────
        CONSOLE.print("Pass 2: remapping arrays and writing compressed .npz files ...")
        pass2_results = run_tasks(
            [(i, str(base), cache_meta[i]) for i, base in enumerate(bases)],
            description="Pass 2: saving normalized files",
            fn=process_recording_pass2,
            initializer=init_pass2_worker,
            initargs=(pal_remap, act_remap, kept_cols, str(OUT_DIR)),
        )

        total_frames = sum(int(result["frames_shape"][0]) for result in pass2_results)
        CONSOLE.print(f"Done. Wrote {len(pass2_results)} normalized file(s), {total_frames:,} frames total.")
        CONSOLE.print(f"Output in {OUT_DIR}")

        # ── Write dataset index for fast loading ─────────────────────────
        from src.data.dataset_index import build_normalized_index, write_index

        ds_index = build_normalized_index(OUT_DIR)
        idx_path = write_index(OUT_DIR, ds_index)
        CONSOLE.print(f"Wrote {idx_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
