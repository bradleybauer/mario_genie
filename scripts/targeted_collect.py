#!/usr/bin/env python3
"""Targeted data collection driver for SMB1.

Iterates a loop:
  1. Load all existing recordings and compute coverage over (world, stage, x_bin).
  2. Gather all available save states and tag each with its (world, stage, x_bin).
  3. Score save states by inverse-frequency weighting — save states in
     under-represented bins are preferred.
  4. Launch Mesen with the chosen save state and auto-start data recording.
  5. Wait for the user to close the emulator, then repeat.

The balancing dimensions default to (world, stage, x_bin) but can be extended
by adding extra RAM addresses to BALANCE_DIMS below.

Usage
-----
  conda activate mario
  python scripts/targeted_collect.py

  # Override bin width (default 128 pixels)
  python scripts/targeted_collect.py --bin-width 64

  # Dry-run: just print what would be launched
  python scripts/targeted_collect.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MESEN_BIN = PROJECT_ROOT / "mesen" / "bin" / "linux-x64" / "Release" / "Mesen"
ROM_PATH = PROJECT_ROOT / "nes" / "Super Mario Bros. (Japan, USA).nes"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LUA_SCRIPT = PROJECT_ROOT / "scripts" / "collector_autostart.lua"

# ---------------------------------------------------------------------------
# RAM byte indices (into the 2048-byte NES internal RAM snapshot)
# ---------------------------------------------------------------------------

RAM_WORLD = 0x075F       # world number (0-indexed)
RAM_STAGE = 0x075C       # stage number (0-indexed)
RAM_X_PAGE = 0x006D      # player horizontal page in level
RAM_X_SCREEN = 0x0086    # player X position on screen (0-255)

# Optional extra dimensions you might want to balance over in the future.
# Each entry is (ram_index, label, optional_bin_width_or_None).
# None means treat as categorical (no binning).
EXTRA_BALANCE_DIMS: list[tuple[int, str, int | None]] = [
    # Uncomment to also balance over powerup state:
    # (0x0756, "powerup", None),
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SaveStateInfo(NamedTuple):
    path: Path
    recording_name: str
    frame_number: int | None
    world: int
    stage: int
    x_bin: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_x_position(ram_row: np.ndarray) -> int:
    """Compute absolute level-x from a single RAM snapshot row."""
    return int(ram_row[RAM_X_PAGE]) * 256 + int(ram_row[RAM_X_SCREEN])


def bin_value(value: int, bin_width: int) -> int:
    return (value // bin_width) * bin_width


def extract_bin_key(ram_row: np.ndarray, bin_width: int) -> tuple:
    """Return the balance-dimension key for one RAM snapshot."""
    world = int(ram_row[RAM_WORLD])
    stage = int(ram_row[RAM_STAGE])
    x_pos = compute_x_position(ram_row)
    x_b = bin_value(x_pos, bin_width)
    key = [world, stage, x_b]
    for ram_idx, _label, extra_bw in EXTRA_BALANCE_DIMS:
        val = int(ram_row[ram_idx])
        if extra_bw is not None:
            val = bin_value(val, extra_bw)
        key.append(val)
    return tuple(key)


def dim_labels() -> list[str]:
    labels = ["world", "stage", "x_bin"]
    for _, label, _ in EXTRA_BALANCE_DIMS:
        labels.append(label)
    return labels


# ---------------------------------------------------------------------------
# Step 1: compute current dataset coverage
# ---------------------------------------------------------------------------

def compute_coverage(bin_width: int) -> Counter:
    """Count frames per (world, stage, x_bin, ...) across all recordings."""
    from mario_world_model.npy_db import load_recordings

    counts: Counter = Counter()
    recordings = load_recordings(data_dir=RAW_DATA_DIR, verbose=False)
    total_frames = 0
    for rec in recordings:
        for i in range(len(rec.frames)):
            key = extract_bin_key(rec.ram[i], bin_width)
            counts[key] += 1
            total_frames += 1

    print(f"\n  Recordings loaded: {len(recordings)}")
    print(f"  Total frames:      {total_frames:,}")
    print(f"  Unique bins:       {len(counts)}")
    return counts


# ---------------------------------------------------------------------------
# Step 2: gather all available save states and tag them
# ---------------------------------------------------------------------------

def gather_save_states(bin_width: int) -> list[SaveStateInfo]:
    """Find every .mss file in the raw data directory and read its
    associated RAM snapshot to determine (world, stage, x_bin)."""
    import re
    from mario_world_model.npy_db import load_recordings

    recordings = load_recordings(data_dir=RAW_DATA_DIR, verbose=False)
    states: list[SaveStateInfo] = []

    for rec in recordings:
        # Initial save state
        if rec.mss_path and rec.mss_path.exists():
            # Use the first frame's RAM to tag this state
            key = extract_bin_key(rec.ram[0], bin_width)
            states.append(SaveStateInfo(
                path=rec.mss_path,
                recording_name=rec.name,
                frame_number=int(rec.frames[0]),
                world=key[0],
                stage=key[1],
                x_bin=key[2],
            ))

        # Periodic save states
        for frame_num, state_path in rec.save_states:
            # Find the closest frame index in the recording
            idx = np.searchsorted(rec.frames, frame_num)
            if idx >= len(rec.frames):
                idx = len(rec.frames) - 1
            key = extract_bin_key(rec.ram[idx], bin_width)
            states.append(SaveStateInfo(
                path=state_path,
                recording_name=rec.name,
                frame_number=frame_num,
                world=key[0],
                stage=key[1],
                x_bin=key[2],
            ))

    return states


# ---------------------------------------------------------------------------
# Step 3: pick the best save state via inverse-frequency weighting
# ---------------------------------------------------------------------------

def pick_best_state(
    coverage: Counter,
    states: list[SaveStateInfo],
) -> SaveStateInfo:
    """Score each save state by inverse frequency of its bin.

    States in bins with zero coverage get the highest priority.
    Among equally-weighted states, pick one at random to add variety.
    """
    labels = dim_labels()

    def state_key(s: SaveStateInfo) -> tuple:
        key = [s.world, s.stage, s.x_bin]
        # If EXTRA_BALANCE_DIMS were used they would already be in the
        # SaveStateInfo fields — for now just the core 3.
        return tuple(key)

    # Compute inverse-frequency weight for each state
    weights = np.empty(len(states), dtype=np.float64)
    for i, s in enumerate(states):
        count = coverage[state_key(s)]
        if count == 0:
            weights[i] = float("inf")
        else:
            weights[i] = 1.0 / count

    # Separate infinite (unseen) from finite
    inf_mask = np.isinf(weights)
    if inf_mask.any():
        # Pick randomly among completely unseen bins
        candidates = np.where(inf_mask)[0]
    else:
        # Pick from the top-10% least-covered states
        threshold = np.percentile(weights, 90)
        candidates = np.where(weights >= threshold)[0]

    chosen_idx = int(np.random.choice(candidates))
    chosen = states[chosen_idx]

    s_key = state_key(chosen)
    current_count = coverage[s_key]
    print(f"\n  Selected save state:")
    print(f"    File:        {chosen.path.name}")
    print(f"    Recording:   {chosen.recording_name}")
    print(f"    Frame:       {chosen.frame_number}")
    for lbl, val in zip(labels, s_key):
        print(f"    {lbl:12s}:  {val}")
    print(f"    Bin count:   {current_count} frames in dataset")

    return chosen


# ---------------------------------------------------------------------------
# Step 4: launch Mesen
# ---------------------------------------------------------------------------

def launch_mesen(state: SaveStateInfo, dry_run: bool = False) -> None:
    """Launch Mesen with the ROM, save state, and auto-record Lua script."""

    now = datetime.now().strftime("%m_%d_%Y %I:%M:%S %p")
    rom_name = ROM_PATH.stem
    output_base = str(RAW_DATA_DIR / f"{rom_name} {now}")

    # Patch the Lua script with the output path (restore after launch)
    lua_text = LUA_SCRIPT.read_text()
    placeholder = "__OUTPUT_BASE_PLACEHOLDER__"
    if placeholder not in lua_text:
        print("ERROR: Lua script missing placeholder marker.", file=sys.stderr)
        sys.exit(1)
    patched_lua = lua_text.replace(placeholder, output_base)
    LUA_SCRIPT.write_text(patched_lua)

    cmd = [
        str(MESEN_BIN),
        str(ROM_PATH),
        str(state.path),
        str(LUA_SCRIPT),
    ]

    print(f"\n  Output base: {output_base}")
    print(f"  Command:     {' '.join(cmd[:1])} \\")
    for c in cmd[1:]:
        print(f"               {c} \\")

    if dry_run:
        # Restore placeholder immediately
        LUA_SCRIPT.write_text(lua_text)
        print("\n  [DRY RUN] Skipping launch.")
        return

    print("\n  Launching Mesen... (close the emulator window when done)\n")
    try:
        subprocess.run(cmd)
    finally:
        # Always restore the placeholder so the script is clean for next run
        LUA_SCRIPT.write_text(lua_text)
    print("\n  Mesen closed.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Targeted data collection driver for SMB1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bin-width", type=int, default=128,
        help="X-position bin width in pixels (default: 128).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be launched without starting Mesen.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single iteration instead of looping.",
    )
    args = parser.parse_args()

    if not MESEN_BIN.exists():
        print(f"ERROR: Mesen binary not found: {MESEN_BIN}", file=sys.stderr)
        sys.exit(1)
    if not ROM_PATH.exists():
        print(f"ERROR: ROM not found: {ROM_PATH}", file=sys.stderr)
        sys.exit(1)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}")
        print(f"{'='*60}")

        # 1. Compute coverage
        print("\n[1/3] Computing dataset coverage...")
        coverage = compute_coverage(args.bin_width)

        # 2. Gather save states
        print("\n[2/3] Scanning available save states...")
        states = gather_save_states(args.bin_width)
        if not states:
            print("  No save states found! Play at least one recording first.")
            sys.exit(1)
        print(f"  Found {len(states)} save states")

        # 3. Pick best and launch
        print("\n[3/3] Selecting best save state...")
        chosen = pick_best_state(coverage, states)
        launch_mesen(chosen, dry_run=args.dry_run)

        if args.once or args.dry_run:
            break

        print("\n  Press Ctrl+C to stop the collection loop.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCollection loop stopped by user.")
