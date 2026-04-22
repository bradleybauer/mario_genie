#!/usr/bin/env python3
"""Find sets of RAM/WRAM addresses that vary together across recordings.

Groups addresses whose byte trajectory is identical over every loaded frame.
This is useful for spotting redundant memory locations or mirrored state.

By default, constant-only groups are omitted because they are already covered by
find_constant_addresses.py. Use --include-constant to include them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.npy_db import load_recordings
from src.data.smb1_memory_map import SMB1_RAM_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum number of addresses required to report a group (default: 2).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Maximum number of groups to print per memory space (default: 50).",
    )
    parser.add_argument(
        "--include-constant",
        action="store_true",
        help="Include groups whose shared trajectory is a single constant value.",
    )
    return parser.parse_args()


def _column_digest(column: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(column)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()


def find_identical_groups(matrix: np.ndarray) -> list[list[int]]:
    digest_buckets: dict[bytes, list[list[int]]] = {}

    for idx in range(matrix.shape[1]):
        digest = _column_digest(matrix[:, idx])
        candidate_groups = digest_buckets.setdefault(digest, [])

        for group in candidate_groups:
            if np.array_equal(matrix[:, idx], matrix[:, group[0]]):
                group.append(idx)
                break
        else:
            candidate_groups.append([idx])

    groups: list[list[int]] = []
    for bucket_groups in digest_buckets.values():
        groups.extend(bucket_groups)
    return groups


def format_ram_entry(index: int) -> str:
    label = SMB1_RAM_LABELS.get(index, ("", ""))[0]
    suffix = f"  {label}" if label else ""
    return f"  ${index:04X}      {index:5d}{suffix}"


def format_wram_entry(index: int) -> str:
    nes_addr = 0x6000 + index
    return f"  ${nes_addr:04X}      {index:5d}"


def report_groups(
    *,
    name: str,
    matrix: np.ndarray | None,
    min_group_size: int,
    top: int,
    include_constant: bool,
    formatter,
) -> None:
    if matrix is None:
        print(f"\nNo {name} data found.")
        return

    groups = find_identical_groups(matrix)
    filtered: list[tuple[list[int], int]] = []
    hidden_constant_groups = 0

    for group in groups:
        if len(group) < min_group_size:
            continue
        unique_values = int(np.unique(matrix[:, group[0]]).size)
        if unique_values == 1 and not include_constant:
            hidden_constant_groups += 1
            continue
        filtered.append((group, unique_values))

    filtered.sort(key=lambda item: (-len(item[0]), item[1], item[0][0]))

    grouped_addresses = sum(len(group) for group, _ in filtered)
    print(f"\n{name} groups with identical trajectories: {len(filtered)}")
    print(f"{name} addresses participating: {grouped_addresses} / {matrix.shape[1]}")
    if hidden_constant_groups:
        print(f"Skipped {hidden_constant_groups} constant-only group(s); rerun with --include-constant to show them.")

    if not filtered:
        print("No groups matched the current filters.")
        return

    groups_to_show = filtered[:top]
    for group_index, (group, unique_values) in enumerate(groups_to_show, start=1):
        preview = np.unique(matrix[:, group[0]])[:8]
        preview_str = ", ".join(str(int(v)) for v in preview)
        if unique_values > len(preview):
            preview_str += ", ..."
        print(
            f"\nGroup {group_index}: size={len(group)}  unique_values={unique_values}"
            f"  sample_values=[{preview_str}]"
        )
        for idx in group:
            print(formatter(idx))

    remaining = len(filtered) - len(groups_to_show)
    if remaining > 0:
        print(f"\n... {remaining} additional group(s) hidden by --top={top}.")


def main() -> None:
    args = parse_args()
    if args.min_group_size < 2:
        print("--min-group-size must be at least 2", file=sys.stderr)
        sys.exit(2)
    if args.top < 1:
        print("--top must be at least 1", file=sys.stderr)
        sys.exit(2)

    recordings = load_recordings(verbose=False)
    if not recordings:
        print("No recordings found in data/", file=sys.stderr)
        sys.exit(1)

    ram = np.concatenate([recording.ram for recording in recordings], axis=0)

    wram_recordings = [recording for recording in recordings if recording.wram is not None]
    wram = (
        np.concatenate([recording.wram for recording in wram_recordings], axis=0)
        if wram_recordings
        else None
    )

    print(f"\n{'=' * 72}")
    print(f"Total files: {len(recordings)}  Total RAM frames: {ram.shape[0]:,}")
    if wram is not None:
        print(f"WRAM files: {len(wram_recordings)}  Total WRAM frames: {wram.shape[0]:,}")
    print(f"{'=' * 72}")

    report_groups(
        name="RAM",
        matrix=ram,
        min_group_size=args.min_group_size,
        top=args.top,
        include_constant=args.include_constant,
        formatter=format_ram_entry,
    )
    report_groups(
        name="WRAM",
        matrix=wram,
        min_group_size=args.min_group_size,
        top=args.top,
        include_constant=args.include_constant,
        formatter=format_wram_entry,
    )


if __name__ == "__main__":
    main()