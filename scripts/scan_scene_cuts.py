#!/usr/bin/env python3
"""
Scan dataset chunks for "unnatural" scene cuts.

An unnatural scene cut is any frame-to-frame transition that could NOT happen
during normal gameplay:

  1. Level jump:  (world, stage) changes to something other than the natural
     successor.  Natural successors after flag_get: W-1→W-2, W-2→W-3, W-3→W-4,
     W-4→(W+1)-1.  Death always respawns the *same* level.
  2. Mid-level respawn:  Same level continues after a done, but x_pos jumps to
     a mid-level position (evidence of replay fast-forward leaking into data,
     or random-level collection restarting at a different point).

Usage:
    python scripts/scan_scene_cuts.py data/nes
    python scripts/scan_scene_cuts.py data/nes --verbose
    python scripts/scan_scene_cuts.py data/nes --cross-sequence
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.dataset_paths import find_chunk_files

# ---------------------------------------------------------------------------
# Natural level progression in Super Mario Bros
# ---------------------------------------------------------------------------
# After grabbing the flag (flag_get), the game advances to the next stage.
# World X-4 → (X+1)-1.  World 8-4 loops or ends, but we treat it as terminal.
# Warp zones can skip levels, but we can't distinguish warp from random-reset
# in the data, so we list warp destinations as natural too.

NATURAL_SUCCESSORS: dict[tuple[int, int], set[tuple[int, int]]] = {}

for w in range(1, 9):
    for s in range(1, 5):
        succs = set()
        # Normal progression
        if s < 4:
            succs.add((w, s + 1))
        elif w < 8:
            succs.add((w + 1, 1))
        # Same level (death respawn)
        succs.add((w, s))
        NATURAL_SUCCESSORS[(w, s)] = succs

# Warp zone destinations (accessed via hidden pipes)
# 1-2 has warp to 2-1, 3-1, 4-1
for dest in [(2, 1), (3, 1), (4, 1)]:
    NATURAL_SUCCESSORS[(1, 2)].add(dest)
# 4-2 has warp to 5-1, 6-1, 7-1, 8-1
for dest in [(5, 1), (6, 1), (7, 1), (8, 1)]:
    NATURAL_SUCCESSORS[(4, 2)].add(dest)

# Mid-level x_pos threshold: if after a done the new x_pos is above this,
# it's suspicious (normal respawn starts near x=0..~40).
MID_LEVEL_X_THRESHOLD = 80


def scan_chunk(
    filepath: str,
    *,
    verbose: bool = False,
    cross_sequence: bool = False,
) -> dict:
    """Scan one .npz chunk for unnatural scene cuts.

    Returns a dict with counts and details.
    """
    npz = np.load(filepath, mmap_mode="r")

    required = {"world", "stage", "dones", "x_pos"}
    available = set(npz.files)
    if not required.issubset(available):
        missing = required - available
        return {"skipped": True, "reason": f"missing arrays: {missing}"}

    world = np.asarray(npz["world"])   # (N, T)
    stage = np.asarray(npz["stage"])   # (N, T)
    dones = np.asarray(npz["dones"], dtype=bool)
    x_pos = np.asarray(npz["x_pos"])
    flag_get = np.asarray(npz["flag_get"]) if "flag_get" in available else None

    num_seqs, seq_len = world.shape

    stats = {
        "total_transitions": 0,
        "level_jumps": 0,
        "mid_level_respawns": 0,
        "death_same_level": 0,
        "flag_next_level": 0,
        "details": [],
    }

    # Per-sequence tracking by cut type
    dirty_seqs: set[int] = set()
    seqs_with_death_same: set[int] = set()
    seqs_with_flag_next: set[int] = set()
    seqs_with_level_jump: set[int] = set()
    seqs_with_mid_respawn: set[int] = set()

    # Counter for (from_level, to_level) jumps
    jump_counter: Counter[tuple[tuple[int, int], tuple[int, int]]] = Counter()
    mid_respawn_counter: Counter[tuple[int, int]] = Counter()

    def _check_transition(
        w_before: int, s_before: int, w_after: int, s_after: int,
        x_after: int, done_before: bool, fg_before: int,
        loc: str,
        seq_idx: int = -1,
    ):
        stats["total_transitions"] += 1
        from_level = (int(w_before), int(s_before))
        to_level = (int(w_after), int(s_after))

        if from_level == to_level:
            # Same level transition across a done boundary
            if done_before and x_after > MID_LEVEL_X_THRESHOLD:
                stats["mid_level_respawns"] += 1
                mid_respawn_counter[to_level] += 1
                if seq_idx >= 0:
                    dirty_seqs.add(seq_idx)
                    seqs_with_mid_respawn.add(seq_idx)
                if verbose:
                    stats["details"].append(
                        f"  {loc}: mid-level respawn {w_after}-{s_after} x={x_after}"
                    )
            else:
                stats["death_same_level"] += 1
                if seq_idx >= 0:
                    seqs_with_death_same.add(seq_idx)
            return

        # Level changed — check if it's a natural successor
        allowed = NATURAL_SUCCESSORS.get(from_level, set())
        if to_level in allowed:
            stats["flag_next_level"] += 1
            if seq_idx >= 0:
                seqs_with_flag_next.add(seq_idx)
            return

        # Unnatural level jump
        stats["level_jumps"] += 1
        jump_counter[(from_level, to_level)] += 1
        if seq_idx >= 0:
            dirty_seqs.add(seq_idx)
            seqs_with_level_jump.add(seq_idx)
        if verbose:
            extra = f" (flag)" if fg_before else " (death/trunc)"
            stats["details"].append(
                f"  {loc}: {w_before}-{s_before} → {w_after}-{s_after}{extra}"
            )

    # --- Within-sequence transitions ---
    for seq_idx in range(num_seqs):
        for t in range(seq_len - 1):
            w_b, s_b = int(world[seq_idx, t]), int(stage[seq_idx, t])
            w_a, s_a = int(world[seq_idx, t + 1]), int(stage[seq_idx, t + 1])
            x_a = int(x_pos[seq_idx, t + 1])
            d_b = bool(dones[seq_idx, t])
            fg_b = int(flag_get[seq_idx, t]) if flag_get is not None else 0

            # Only check transitions where the done flag fires or level changes
            if d_b or (w_b, s_b) != (w_a, s_a):
                _check_transition(w_b, s_b, w_a, s_a, x_a, d_b, fg_b,
                                  f"seq {seq_idx} t={t}→{t+1}",
                                  seq_idx=seq_idx)

    # --- Cross-sequence transitions (last frame of seq N → first frame of seq N+1) ---
    if cross_sequence and num_seqs > 1:
        for seq_idx in range(num_seqs - 1):
            w_b = int(world[seq_idx, -1])
            s_b = int(stage[seq_idx, -1])
            w_a = int(world[seq_idx + 1, 0])
            s_a = int(stage[seq_idx + 1, 0])
            x_a = int(x_pos[seq_idx + 1, 0])
            d_b = bool(dones[seq_idx, -1])
            fg_b = int(flag_get[seq_idx, -1]) if flag_get is not None else 0

            if d_b or (w_b, s_b) != (w_a, s_a):
                _check_transition(w_b, s_b, w_a, s_a, x_a, d_b, fg_b,
                                  f"seq {seq_idx}→{seq_idx+1} (cross)")

    stats["jump_counter"] = jump_counter
    stats["mid_respawn_counter"] = mid_respawn_counter
    stats["total_sequences"] = num_seqs
    stats["dirty_sequences"] = len(dirty_seqs)
    stats["clean_sequences"] = num_seqs - len(dirty_seqs)
    stats["seqs_death_same"] = len(seqs_with_death_same)
    stats["seqs_flag_next"] = len(seqs_with_flag_next)
    stats["seqs_level_jump"] = len(seqs_with_level_jump)
    stats["seqs_mid_respawn"] = len(seqs_with_mid_respawn)
    stats["seq_len"] = seq_len
    return stats


def main():
    parser = argparse.ArgumentParser(description="Scan dataset for unnatural scene cuts")
    parser.add_argument("data_dirs", nargs="+", type=Path, help="Data directories to scan")
    parser.add_argument("--verbose", action="store_true", help="Print every unnatural cut")
    parser.add_argument(
        "--cross-sequence", action="store_true",
        help="Also check transitions between consecutive sequences within a chunk "
             "(these don't affect training but indicate collection patterns)",
    )
    args = parser.parse_args()

    all_chunks: list[str] = []
    for d in args.data_dirs:
        all_chunks.extend(find_chunk_files(d))

    if not all_chunks:
        print("No .npz chunk files found.")
        sys.exit(1)

    print(f"Scanning {len(all_chunks)} chunk(s)...\n")

    total_transitions = 0
    total_level_jumps = 0
    total_mid_respawns = 0
    total_death_same = 0
    total_flag_next = 0
    total_sequences = 0
    total_dirty = 0
    total_clean = 0
    total_seqs_death_same = 0
    total_seqs_flag_next = 0
    total_seqs_level_jump = 0
    total_seqs_mid_respawn = 0
    seq_len = 0
    skipped = 0

    global_jump_counter: Counter[tuple[tuple[int, int], tuple[int, int]]] = Counter()
    global_mid_counter: Counter[tuple[int, int]] = Counter()

    for filepath in all_chunks:
        result = scan_chunk(filepath, verbose=args.verbose, cross_sequence=args.cross_sequence)
        if result.get("skipped"):
            skipped += 1
            print(f"  SKIP {filepath}: {result['reason']}")
            continue

        total_transitions += result["total_transitions"]
        total_level_jumps += result["level_jumps"]
        total_mid_respawns += result["mid_level_respawns"]
        total_death_same += result["death_same_level"]
        total_flag_next += result["flag_next_level"]
        total_sequences += result["total_sequences"]
        total_dirty += result["dirty_sequences"]
        total_clean += result["clean_sequences"]
        total_seqs_death_same += result["seqs_death_same"]
        total_seqs_flag_next += result["seqs_flag_next"]
        total_seqs_level_jump += result["seqs_level_jump"]
        total_seqs_mid_respawn += result["seqs_mid_respawn"]
        seq_len = result["seq_len"]
        global_jump_counter.update(result["jump_counter"])
        global_mid_counter.update(result["mid_respawn_counter"])

        chunk_unnatural = result["level_jumps"] + result["mid_level_respawns"]
        if chunk_unnatural > 0:
            name = Path(filepath).name
            print(f"  {name}: {chunk_unnatural} unnatural cuts "
                  f"({result['level_jumps']} level jumps, "
                  f"{result['mid_level_respawns']} mid-level respawns)")
            for line in result.get("details", []):
                print(line)

    # --- Summary ---
    total_unnatural = total_level_jumps + total_mid_respawns
    total_natural = total_death_same + total_flag_next
    total_frames = total_sequences * seq_len
    clean_frames = total_clean * seq_len
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Chunks scanned:       {len(all_chunks) - skipped}")
    if skipped:
        print(f"Chunks skipped:       {skipped}")
    print(f"")
    print(f"Total sequences:      {total_sequences}")
    print(f"Sequence length:      {seq_len}")
    print(f"Total frames:         {total_frames}")
    print(f"")
    print(f"--- Transition counts (frame-to-frame) ---")
    print(f"Total transitions:    {total_transitions}")
    print(f"Natural:              {total_natural}")
    print(f"  Death → same level: {total_death_same}")
    print(f"  Flag → next level:  {total_flag_next}")
    print(f"Unnatural:            {total_unnatural}")
    print(f"  Level jumps:        {total_level_jumps}")
    print(f"  Mid-level respawns: {total_mid_respawns}")
    if total_transitions > 0:
        pct = 100.0 * total_unnatural / total_transitions
        print(f"Unnatural rate:       {pct:.2f}%")
    print(f"")
    print(f"--- Sequence counts (videos) ---")
    print(f"Seqs with death → same level:   {total_seqs_death_same}")
    print(f"Seqs with flag → next level:    {total_seqs_flag_next}")
    print(f"Seqs with level jump:           {total_seqs_level_jump}")
    print(f"Seqs with mid-level respawn:    {total_seqs_mid_respawn}")
    print(f"")
    print(f"Sequences with unnatural cuts:  {total_dirty}")
    print(f"Clean sequences:                {total_clean}")
    if total_sequences > 0:
        retain_pct = 100.0 * total_clean / total_sequences
        print(f"Data retained if clean only:    {retain_pct:.1f}%  ({clean_frames} / {total_frames} frames)")

    if global_jump_counter:
        print(f"\nTop level jumps (from → to):")
        for (fr, to), count in global_jump_counter.most_common(20):
            print(f"  {fr[0]}-{fr[1]} → {to[0]}-{to[1]}:  {count}")

    if global_mid_counter:
        print(f"\nMid-level respawns by level:")
        for level, count in global_mid_counter.most_common(20):
            print(f"  {level[0]}-{level[1]}:  {count}")


if __name__ == "__main__":
    main()
