#!/usr/bin/env python3
"""Analyse per-address value cardinality in the normalized RAM dataset.

Scans all normalized .npz files, collects the set of unique values each
RAM address takes on, and prints a histogram of cardinalities (how many
addresses have 2 unique values, how many have 3, etc.).

Optionally updates data/normalized/ram_addresses.json with a
"values_per_address" list so that downstream training code can build a
categorical prediction head per address.

Note: normalize_raw_dataset.py already writes values_per_address during
normalization.  This script is useful for inspection or for re-generating
the field from existing normalized data without re-running the full pipeline.

Usage:
    # Print distribution only
    python scripts/eval/ram_value_distribution.py

    # Also update ram_addresses.json
    python scripts/eval/ram_value_distribution.py --update-json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized"
RAM_ADDRESSES_JSON = NORMALIZED_DIR / "ram_addresses.json"

CONSOLE = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--update-json",
        action="store_true",
        help="Write values_per_address into ram_addresses.json.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show the top-N highest cardinality addresses (default: 20).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    npz_files = sorted(NORMALIZED_DIR.glob("*.npz"))
    if not npz_files:
        print("No .npz files found in", NORMALIZED_DIR, file=sys.stderr)
        sys.exit(1)

    with RAM_ADDRESSES_JSON.open() as f:
        ram_info = json.load(f)
    kept_addresses: list[int] = ram_info["kept_addresses"]
    n_addresses = len(kept_addresses)

    CONSOLE.print(f"Scanning {len(npz_files)} normalized recording(s), {n_addresses} RAM addresses ...")

    # For each address column, collect the set of unique byte values
    unique_sets: list[set[int]] = [set() for _ in range(n_addresses)]

    for npz_path in npz_files:
        data = np.load(str(npz_path), mmap_mode="r")
        ram = np.asarray(data["ram"])  # (N_frames, n_addresses) uint8
        for col in range(n_addresses):
            unique_sets[col].update(np.unique(ram[:, col]).tolist())

    # Compute cardinalities
    cardinalities = np.array([len(s) for s in unique_sets])

    # ── Histogram of cardinalities ──
    counts_by_card = np.bincount(cardinalities)
    nonzero = np.nonzero(counts_by_card)[0]

    table = Table(title="RAM Address Cardinality Distribution")
    table.add_column("Unique Values", justify="right", style="cyan")
    table.add_column("# Addresses", justify="right", style="green")
    table.add_column("Cumulative %", justify="right")
    cumulative = 0
    for card in nonzero:
        count = int(counts_by_card[card])
        cumulative += count
        pct = 100.0 * cumulative / n_addresses
        table.add_row(str(card), str(count), f"{pct:.1f}%")
    CONSOLE.print(table)

    CONSOLE.print(f"\nTotal addresses: {n_addresses}")
    CONSOLE.print(f"Min cardinality: {int(cardinalities.min())}")
    CONSOLE.print(f"Max cardinality: {int(cardinalities.max())}")
    CONSOLE.print(f"Mean cardinality: {cardinalities.mean():.1f}")
    CONSOLE.print(f"Median cardinality: {int(np.median(cardinalities))}")

    # ── Top-N highest cardinality addresses ──
    top_n = min(args.top, n_addresses)
    top_indices = np.argsort(cardinalities)[::-1][:top_n]

    top_table = Table(title=f"Top {top_n} Highest Cardinality Addresses")
    top_table.add_column("NES Address", justify="right", style="cyan")
    top_table.add_column("Column Index", justify="right")
    top_table.add_column("Unique Values", justify="right", style="green")
    for idx in top_indices:
        nes_addr = kept_addresses[idx]
        top_table.add_row(f"${nes_addr:04X}", str(idx), str(cardinalities[idx]))
    CONSOLE.print(top_table)

    # ── Optionally update JSON ──
    if args.update_json:
        values_per_address = [sorted(s) for s in unique_sets]
        ram_info["values_per_address"] = values_per_address
        with RAM_ADDRESSES_JSON.open("w") as f:
            json.dump(ram_info, f, indent=2)
        CONSOLE.print(f"\nUpdated {RAM_ADDRESSES_JSON} with values_per_address.")
    else:
        CONSOLE.print(f"\nRun with --update-json to write values_per_address into {RAM_ADDRESSES_JSON.name}")


if __name__ == "__main__":
    main()
