#!/usr/bin/env python3
"""Generate dataset_index.json for existing normalized and/or latent data.

Usage:
    python scripts/collect/build_dataset_index.py                      # both
    python scripts/collect/build_dataset_index.py --normalized-only
    python scripts/collect/build_dataset_index.py --latent-only
    python scripts/collect/build_dataset_index.py --data-dir data/normalized
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_index import (
    build_latent_index,
    build_normalized_index,
    write_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset_index.json for fast dataset loading.")
    parser.add_argument("--data-dir", type=str, default=None, help="Explicit directory to index (auto-detects type)")
    parser.add_argument("--normalized-only", action="store_true")
    parser.add_argument("--latent-only", action="store_true")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            print(f"Directory not found: {data_dir}", file=sys.stderr)
            sys.exit(1)
        npz_files = sorted(data_dir.glob("*.npz"))
        if not npz_files:
            print(f"No .npz files in {data_dir}", file=sys.stderr)
            sys.exit(1)
        # Auto-detect: if first file has 'latents' key, treat as latent
        import numpy as np
        with np.load(npz_files[0], mmap_mode="r") as f:
            is_latent = "latents" in f.files
        if is_latent:
            index = build_latent_index(data_dir)
        else:
            index = build_normalized_index(data_dir)
        path = write_index(data_dir, index)
        print(f"Wrote {path} ({len(index['files'])} files)")
        return

    normalized_dir = PROJECT_ROOT / "data" / "normalized"
    latent_dir = PROJECT_ROOT / "data" / "latents"

    if not args.latent_only and normalized_dir.is_dir() and list(normalized_dir.glob("*.npz")):
        index = build_normalized_index(normalized_dir)
        path = write_index(normalized_dir, index)
        print(f"Wrote {path} ({len(index['files'])} files)")

    if not args.normalized_only and latent_dir.is_dir() and list(latent_dir.glob("*.npz")):
        index = build_latent_index(latent_dir)
        path = write_index(latent_dir, index)
        print(f"Wrote {path} ({len(index['files'])} files)")


if __name__ == "__main__":
    main()
