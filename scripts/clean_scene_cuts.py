#!/usr/bin/env python3
"""
Remove sequences with unnatural scene cuts from dataset chunks.

Rewrites each .npz chunk in-place, keeping only sequences that contain no
unnatural transitions (random level jumps or mid-level respawns).  The
corresponding .meta.json files are regenerated from the filtered data.

Empty chunks (all sequences removed) are deleted.

Usage:
    python scripts/clean_scene_cuts.py data/nes
    python scripts/clean_scene_cuts.py data/nes --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.dataset_paths import find_chunk_files
from mario_world_model.storage import (
    ChunkMeta,
    _as_action_shape_bt,
    _as_frame_shape_btchw,
    _compute_chunk_summaries,
)

# ---------------------------------------------------------------------------
# Natural level progression (duplicated from scan_scene_cuts.py)
# ---------------------------------------------------------------------------

NATURAL_SUCCESSORS: dict[tuple[int, int], set[tuple[int, int]]] = {}

for w in range(1, 9):
    for s in range(1, 5):
        succs = set()
        if s < 4:
            succs.add((w, s + 1))
        elif w < 8:
            succs.add((w + 1, 1))
        succs.add((w, s))
        NATURAL_SUCCESSORS[(w, s)] = succs

for dest in [(2, 1), (3, 1), (4, 1)]:
    NATURAL_SUCCESSORS[(1, 2)].add(dest)
for dest in [(5, 1), (6, 1), (7, 1), (8, 1)]:
    NATURAL_SUCCESSORS[(4, 2)].add(dest)

MID_LEVEL_X_THRESHOLD = 80


def _is_sequence_clean(
    world_t: np.ndarray,
    stage_t: np.ndarray,
    dones_t: np.ndarray,
    x_pos_t: np.ndarray,
) -> bool:
    """Return True if a single sequence has no unnatural scene cuts."""
    seq_len = len(world_t)
    for t in range(seq_len - 1):
        w_b, s_b = int(world_t[t]), int(stage_t[t])
        w_a, s_a = int(world_t[t + 1]), int(stage_t[t + 1])
        d_b = bool(dones_t[t])
        x_a = int(x_pos_t[t + 1])

        if not d_b and (w_b, s_b) == (w_a, s_a):
            continue  # No boundary, same level — fine

        from_level = (w_b, s_b)
        to_level = (w_a, s_a)

        if from_level == to_level:
            if d_b and x_a > MID_LEVEL_X_THRESHOLD:
                return False  # Mid-level respawn
            continue

        # Level changed
        allowed = NATURAL_SUCCESSORS.get(from_level, set())
        if to_level not in allowed:
            return False  # Unnatural level jump

    return True


def process_chunk(
    filepath: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Filter a single chunk file.  Returns stats about what was done."""
    npz_path = Path(filepath)
    meta_path = npz_path.with_name(npz_path.stem + ".meta.json")

    npz = np.load(filepath)
    available = set(npz.files)

    required = {"frames", "actions", "dones", "world", "stage", "x_pos"}
    if not required.issubset(available):
        missing = required - available
        return {"skipped": True, "reason": f"missing arrays: {missing}"}

    world = npz["world"]
    stage = npz["stage"]
    dones = np.asarray(npz["dones"], dtype=bool)
    x_pos = npz["x_pos"]
    num_seqs = world.shape[0]

    # Find clean sequence indices
    clean_mask = np.array([
        _is_sequence_clean(world[i], stage[i], dones[i], x_pos[i])
        for i in range(num_seqs)
    ])
    num_clean = int(clean_mask.sum())
    num_dirty = num_seqs - num_clean

    result = {
        "skipped": False,
        "total": num_seqs,
        "clean": num_clean,
        "dirty": num_dirty,
        "deleted_chunk": False,
    }

    if num_dirty == 0:
        return result  # Nothing to do

    if dry_run:
        return result

    if num_clean == 0:
        # Remove entire chunk + meta
        npz_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        result["deleted_chunk"] = True
        return result

    # Build filtered arrays
    all_keys = list(npz.files)
    filtered = {}
    for key in all_keys:
        filtered[key] = npz[key][clean_mask]

    # Rewrite the npz (compressed)
    np.savez_compressed(npz_path, **filtered)

    # Regenerate meta.json
    action_summ, exact_prog_summ = _compute_chunk_summaries(filtered)
    frames_shape = filtered["frames"].shape
    actions_shape = filtered["actions"].shape

    # Parse chunk index from filename
    chunk_index = int(npz_path.stem.split("_")[1])

    meta = ChunkMeta(
        chunk_index=chunk_index,
        num_sequences=int(frames_shape[0]),
        sequence_length=int(frames_shape[1]),
        frame_shape_btchw=_as_frame_shape_btchw(tuple(int(x) for x in frames_shape)),
        action_shape_bt=_as_action_shape_bt(tuple(int(x) for x in actions_shape)),
        total_frames=int(frames_shape[0] * frames_shape[1]),
        total_actions=int(np.prod(actions_shape)),
        action_summary=action_summ,
        exact_progression_summary=exact_prog_summ,
    )
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Remove sequences with unnatural scene cuts from dataset chunks"
    )
    parser.add_argument("data_dirs", nargs="+", type=Path, help="Data directories to clean")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be removed without modifying files",
    )
    args = parser.parse_args()

    all_chunks: list[str] = []
    for d in args.data_dirs:
        all_chunks.extend(find_chunk_files(d))

    if not all_chunks:
        print("No .npz chunk files found.")
        sys.exit(1)

    action = "Would remove" if args.dry_run else "Removing"
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Scanning {len(all_chunks)} chunk(s)...\n")

    total_seqs = 0
    total_clean = 0
    total_dirty = 0
    chunks_modified = 0
    chunks_deleted = 0
    chunks_skipped = 0

    for filepath in all_chunks:
        result = process_chunk(filepath, dry_run=args.dry_run)

        if result.get("skipped"):
            chunks_skipped += 1
            print(f"  SKIP {Path(filepath).name}: {result['reason']}")
            continue

        total_seqs += result["total"]
        total_clean += result["clean"]
        total_dirty += result["dirty"]

        if result["dirty"] > 0:
            name = Path(filepath).name
            if result.get("deleted_chunk"):
                chunks_deleted += 1
                print(f"  {name}: DELETED (all {result['total']} sequences were dirty)")
            else:
                chunks_modified += 1
                print(f"  {name}: {action} {result['dirty']}/{result['total']} sequences "
                      f"({result['clean']} kept)")

    print(f"\n{'='*60}")
    print(f"{'DRY RUN ' if args.dry_run else ''}SUMMARY")
    print(f"{'='*60}")
    print(f"Chunks scanned:    {len(all_chunks)}")
    if chunks_skipped:
        print(f"Chunks skipped:    {chunks_skipped}")
    print(f"Chunks modified:   {chunks_modified}")
    print(f"Chunks deleted:    {chunks_deleted}")
    print(f"Chunks unchanged:  {len(all_chunks) - chunks_skipped - chunks_modified - chunks_deleted}")
    print(f"")
    print(f"Total sequences:   {total_seqs}")
    print(f"Dirty (removed):   {total_dirty}")
    print(f"Clean (kept):      {total_clean}")
    if total_seqs > 0:
        pct = 100.0 * total_clean / total_seqs
        print(f"Data retained:     {pct:.1f}%")


if __name__ == "__main__":
    main()
