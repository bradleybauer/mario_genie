#!/usr/bin/env python3
"""Find RAM/WRAM addresses that are constant across all recordings.

Scans all recordings in data/ and reports addresses
whose value never changes across any frame in any recording.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from src.data.npy_db import load_recordings
from src.data.smb1_memory_map import SMB1_RAM_LABELS


def find_constant_addresses():
    recordings = load_recordings()
    if not recordings:
        print("No recordings found in data/", file=sys.stderr)
        sys.exit(1)

    ram = np.concatenate([r.ram for r in recordings])
    has_wram = any(r.wram is not None for r in recordings)
    wram = np.concatenate([r.wram for r in recordings if r.wram is not None]) if has_wram else None
    total_frames = ram.shape[0]

    print(f"\n{'='*60}")
    print(f"Total files: {len(recordings)}  Total frames: {total_frames:,}")
    print(f"{'='*60}")

    # RAM: constant iff min == max across all frames
    ram_min = ram.min(axis=0)
    ram_max = ram.max(axis=0)
    ram_constant = ram_min == ram_max
    ram_const_idxs = np.where(ram_constant)[0]

    print(f"\nConstant RAM addresses: {len(ram_const_idxs)} / {ram.shape[1]}")
    if len(ram_const_idxs) > 0:
        print(f"{'Address':>10}  {'Index':>6}  {'Value':>5}  {'Hex':>5}  Label")
        print("-" * 60)
        for idx in ram_const_idxs:
            val = ram_min[idx]
            label = ""
            if idx in SMB1_RAM_LABELS:
                label = SMB1_RAM_LABELS[idx][0]
            print(f"  ${idx:04X}      {idx:5d}   {val:4d}   0x{val:02X}  {label}")

    # WRAM
    if wram is not None:
        wram_min = wram.min(axis=0)
        wram_max = wram.max(axis=0)
        wram_constant = wram_min == wram_max
        wram_const_idxs = np.where(wram_constant)[0]
        print(f"\nConstant WRAM addresses: {len(wram_const_idxs)} / {wram.shape[1]}")
        if len(wram_const_idxs) > 0:
            print(f"{'Address':>10}  {'Index':>6}  {'Value':>5}  {'Hex':>5}")
            print("-" * 35)
            for idx in wram_const_idxs:
                val = wram_min[idx]
                nes_addr = 0x6000 + idx
                print(f"  ${nes_addr:04X}      {idx:5d}   {val:4d}   0x{val:02X}")
    else:
        print("\nNo WRAM data found in any file.")

    # Summary
    ram_zero = np.sum(ram_constant & (ram_min == 0))
    ram_nonzero = np.sum(ram_constant & (ram_min != 0))
    print(f"\nRAM: {ram_zero} always-zero, {ram_nonzero} constant non-zero")

    # Summary: addresses that are always zero (common for unused)
    ram_zero = np.sum(ram_constant & (ram_min == 0))
    ram_nonzero = np.sum(ram_constant & (ram_min != 0))
    print(f"\nRAM: {ram_zero} always-zero, {ram_nonzero} constant non-zero")


if __name__ == "__main__":
    find_constant_addresses()
