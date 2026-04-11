#!/usr/bin/env python3
"""Validate latent_stats.json schema and consistency.

This validator enforces component-wise latent normalization stats:
- latent_stats_version == 2
- normalization_scheme == component_chw_shared_time
- component_{mean,std,std_clamped} with shape (C, H, W)

Optionally verifies all latent .npz files in a directory match the declared
latent shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REQUIRED_SCHEME = "component_chw_shared_time"
REQUIRED_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate latent_stats.json schema and latent shapes.")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/latents/latent_stats.json"),
        help="Path to latent_stats.json (default: data/latents/latent_stats.json)",
    )
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=None,
        help="Optional directory containing latent .npz files to shape-check. Defaults to stats parent.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _validate_stats_schema(stats: dict) -> tuple[list[str], tuple[int, int, int] | None]:
    errors: list[str] = []

    version = stats.get("latent_stats_version")
    if version is None:
        errors.append("missing key: latent_stats_version")
    else:
        try:
            version_int = int(version)
            if version_int != REQUIRED_VERSION:
                errors.append(
                    f"latent_stats_version must be {REQUIRED_VERSION}, got {version_int}"
                )
        except (TypeError, ValueError):
            errors.append(f"latent_stats_version must be an integer, got {version!r}")

    scheme = stats.get("normalization_scheme")
    if scheme != REQUIRED_SCHEME:
        errors.append(
            f"normalization_scheme must be '{REQUIRED_SCHEME}', got {scheme!r}"
        )

    latent_shape = stats.get("latent_shape")
    shape_tuple: tuple[int, int, int] | None = None
    if not isinstance(latent_shape, dict):
        errors.append("missing or invalid key: latent_shape")
    else:
        channels = latent_shape.get("channels")
        height = latent_shape.get("height")
        width = latent_shape.get("width")
        try:
            shape_tuple = (int(channels), int(height), int(width))
            if any(v <= 0 for v in shape_tuple):
                errors.append(f"latent_shape must be positive, got {shape_tuple}")
        except (TypeError, ValueError):
            errors.append(f"invalid latent_shape values: {latent_shape!r}")

    eps = stats.get("std_epsilon", 1e-6)
    try:
        eps = max(float(eps), 1e-12)
    except (TypeError, ValueError):
        errors.append(f"std_epsilon must be numeric, got {stats.get('std_epsilon')!r}")
        eps = 1e-6

    for key in ("component_mean", "component_std", "component_std_clamped"):
        if key not in stats:
            errors.append(f"missing key: {key}")

    if shape_tuple is None:
        return errors, None

    channels, height, width = shape_tuple
    expected_shape = (channels, height, width)

    for key in ("component_mean", "component_std", "component_std_clamped"):
        if key not in stats:
            continue
        arr = np.asarray(stats[key], dtype=np.float32)
        if arr.shape != expected_shape:
            errors.append(f"{key} shape mismatch: expected {expected_shape}, got {arr.shape}")
            continue
        if not np.isfinite(arr).all():
            errors.append(f"{key} contains non-finite values")

    if "component_std_clamped" in stats:
        std_clamped = np.asarray(stats["component_std_clamped"], dtype=np.float32)
        if std_clamped.shape == expected_shape and float(std_clamped.min()) < eps:
            errors.append(
                f"component_std_clamped contains values below std_epsilon ({eps})"
            )

    count = stats.get("count_per_component")
    try:
        if int(count) <= 0:
            errors.append(f"count_per_component must be > 0, got {count!r}")
    except (TypeError, ValueError):
        errors.append(f"count_per_component must be an integer, got {count!r}")

    if "global_stats" not in stats or not isinstance(stats["global_stats"], dict):
        errors.append("missing or invalid key: global_stats")

    return errors, shape_tuple


def _validate_latent_files(latent_dir: Path, expected_shape: tuple[int, int, int]) -> list[str]:
    errors: list[str] = []
    c_exp, h_exp, w_exp = expected_shape

    npz_files = sorted(latent_dir.glob("*.npz"))
    if not npz_files:
        errors.append(f"no latent .npz files found in {latent_dir}")
        return errors

    for path in npz_files:
        with np.load(path, mmap_mode="r") as data:
            if "latents" not in data.files:
                errors.append(f"{path.name}: missing latents array")
                continue
            latents = data["latents"]
            if latents.ndim != 4:
                errors.append(
                    f"{path.name}: invalid latents ndim {latents.ndim}; expected 4 (C,T,H,W)"
                )
                continue
            c, _t, h, w = (int(v) for v in latents.shape)
            if (c, h, w) != (c_exp, h_exp, w_exp):
                errors.append(
                    f"{path.name}: latent shape mismatch (C,H,W)=({c},{h},{w}), "
                    f"expected ({c_exp},{h_exp},{w_exp})"
                )

    return errors


def main() -> int:
    args = parse_args()
    stats_path = args.stats_path.resolve()
    if not stats_path.is_file():
        print(f"Error: stats file not found: {stats_path}", file=sys.stderr)
        return 1

    stats = _load_json(stats_path)
    schema_errors, shape_tuple = _validate_stats_schema(stats)

    latent_dir = args.latent_dir.resolve() if args.latent_dir is not None else stats_path.parent
    file_errors: list[str] = []
    if shape_tuple is not None:
        file_errors = _validate_latent_files(latent_dir, shape_tuple)

    errors = schema_errors + file_errors
    if errors:
        print("Latent stats validation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 2

    channels, height, width = shape_tuple
    print(
        "Latent stats validation passed: "
        f"version={int(stats['latent_stats_version'])}, "
        f"scheme={stats['normalization_scheme']}, "
        f"shape=(C={channels}, H={height}, W={width}), "
        f"files={int(stats.get('num_files', 0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
