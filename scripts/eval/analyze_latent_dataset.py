#!/usr/bin/env python3
"""Analyze latent dataset structure and dynamics.

This script streams over latent .npz recordings and reports:
  - global latent statistics
  - per-channel activity and temporal motion
  - spatial energy heatmaps over the latent grid
  - lagged similarity of pooled latent states
  - action-conditioned latent motion
  - recording-level outliers by energy and motion

Usage:
    python scripts/analyze_latent_dataset.py
    python scripts/analyze_latent_dataset.py --data-dir dataset/latents --output-dir results/latent_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_style import apply_plot_style
from src.models.latent_utils import load_json
apply_plot_style()
console = Console()

_NES_BUTTONS = [
    (0, "Up"),
    (1, "Down"),
    (2, "Left"),
    (3, "Right"),
    (4, "Start"),
    (5, "Select"),
    (6, "B"),
    (7, "A"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a directory of latent .npz recordings.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Latent dataset directory. Defaults to data/latents.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for summary.json and plots.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=2048,
        help="Frames to process at once from each recording.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit analysis to the first N recordings (0 = all).",
    )
    parser.add_argument(
        "--max-frames-per-file",
        type=int,
        default=0,
        help="Limit analysis to the first N frames of each recording (0 = all).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=240,
        help="Maximum frame lag for pooled latent cosine similarity.",
    )
    parser.add_argument(
        "--lag-step",
        type=int,
        default=20,
        help="Stride between evaluated frame lags for pooled latent cosine similarity.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top channels/actions/files to print.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not save matplotlib plots even when --output-dir is set.",
    )
    args = parser.parse_args()

    if args.chunk_frames <= 0:
        parser.error("--chunk-frames must be > 0")
    if args.max_files < 0:
        parser.error("--max-files must be >= 0")
    if args.max_frames_per_file < 0:
        parser.error("--max-frames-per-file must be >= 0")
    if args.max_lag < 0:
        parser.error("--max-lag must be >= 0")
    if args.lag_step <= 0:
        parser.error("--lag-step must be > 0")
    if args.top_k <= 0:
        parser.error("--top-k must be > 0")
    return args


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        path = Path(args.data_dir).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Latent dataset directory not found: {path}")
        return path

    candidate = PROJECT_ROOT / "data" / "latents"
    if candidate.is_dir():
        return candidate.resolve()

    raise FileNotFoundError(
        "Could not infer latent dataset directory. Checked data/latents."
    )


def action_label(original_value: int) -> str:
    parts = [name for bit, name in _NES_BUTTONS if original_value & (1 << bit)]
    return "+".join(parts) if parts else "None"


def format_bytes(byte_count: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(byte_count)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_action_capacity(state: dict[str, np.ndarray], min_size: int) -> None:
    current = len(state["frame_count"])
    if min_size <= current:
        return

    new_size = max(min_size, max(1, current * 2))
    for key, array in list(state.items()):
        expanded = np.zeros(new_size, dtype=array.dtype)
        expanded[:current] = array
        state[key] = expanded


def update_action_frame_stats(
    state: dict[str, np.ndarray],
    action_ids: np.ndarray,
    values: np.ndarray,
) -> None:
    if action_ids.size == 0:
        return
    ensure_action_capacity(state, int(action_ids.max()) + 1)
    state["frame_count"] += np.bincount(action_ids, minlength=len(state["frame_count"]))
    state["frame_sum"] += np.bincount(action_ids, weights=values, minlength=len(state["frame_sum"]))
    weights = np.square(values.astype(np.float64, copy=False))
    state["frame_sum_sq"] += np.bincount(
        action_ids,
        weights=weights,
        minlength=len(state["frame_sum_sq"]),
    )


def update_action_delta_stats(
    state: dict[str, np.ndarray],
    action_ids: np.ndarray,
    values: np.ndarray,
) -> None:
    if action_ids.size == 0:
        return
    ensure_action_capacity(state, int(action_ids.max()) + 1)
    state["delta_count"] += np.bincount(action_ids, minlength=len(state["delta_count"]))
    state["delta_sum"] += np.bincount(action_ids, weights=values, minlength=len(state["delta_sum"]))
    weights = np.square(values.astype(np.float64, copy=False))
    state["delta_sum_sq"] += np.bincount(
        action_ids,
        weights=weights,
        minlength=len(state["delta_sum_sq"]),
    )


def build_action_row(index: int, stats: dict[str, Any], action_values: list[int] | None) -> dict[str, Any]:
    frame_count = int(stats["frame_count"][index])
    delta_count = int(stats["delta_count"][index])
    frame_mean = float(stats["frame_sum"][index] / frame_count) if frame_count else 0.0
    frame_var = float(stats["frame_sum_sq"][index] / frame_count - frame_mean * frame_mean) if frame_count else 0.0
    delta_mean = float(stats["delta_sum"][index] / delta_count) if delta_count else 0.0
    delta_var = float(stats["delta_sum_sq"][index] / delta_count - delta_mean * delta_mean) if delta_count else 0.0
    original_value = action_values[index] if action_values is not None and index < len(action_values) else index
    return {
        "index": index,
        "original_value": int(original_value),
        "label": action_label(int(original_value)),
        "frame_count": frame_count,
        "frame_rms_mean": frame_mean,
        "frame_rms_std": math.sqrt(max(frame_var, 0.0)),
        "delta_count": delta_count,
        "delta_rms_mean": delta_mean,
        "delta_rms_std": math.sqrt(max(delta_var, 0.0)),
    }


def build_lag_values(max_lag: int, lag_step: int) -> list[int]:
    """Return the evaluated lag values for pooled-state similarity."""
    if max_lag <= 0:
        return []
    return list(range(lag_step, max_lag + 1, lag_step))


def save_plots(output_dir: Path, summary: dict[str, Any], *, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_rows = summary["channel_rows"]
    action_rows = summary["action_rows"]
    lag_rows = summary["lag_rows"]
    spatial_rms = np.asarray(summary["spatial_rms"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    channels = np.arange(len(channel_rows))
    channel_std = [row["std"] for row in channel_rows]
    channel_delta = [row["delta_rms"] for row in channel_rows]
    ax.bar(channels - 0.2, channel_std, width=0.4, label="std")
    ax.bar(channels + 0.2, channel_delta, width=0.4, label="delta_rms")
    ax.set_title("Per-channel latent activity")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Magnitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "channel_activity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(spatial_rms, cmap="magma")
    ax.set_title("Spatial latent RMS")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_rms.png", dpi=160)
    plt.close(fig)

    if lag_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        lags = [row["lag"] for row in lag_rows]
        cosine = [row["mean_cosine"] for row in lag_rows]
        ax.plot(lags, cosine, marker="o")
        ax.set_title("Lagged cosine similarity of pooled latent states")
        ax.set_xlabel("Frame lag")
        ax.set_ylabel("Mean cosine similarity")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "lagged_similarity.png", dpi=160)
        plt.close(fig)

    if action_rows:
        top_actions = sorted(action_rows, key=lambda row: row["delta_rms_mean"], reverse=True)[:top_k]
        labels = [f"{row['index']}:{row['label']}" for row in top_actions]
        values = [row["delta_rms_mean"] for row in top_actions]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.8))
        ax.bar(np.arange(len(labels)), values)
        ax.set_title("Actions with largest latent motion")
        ax.set_ylabel("Mean delta RMS")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / "action_delta_rms.png", dpi=160)
        plt.close(fig)


def analyze_dataset(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = resolve_data_dir(args)
    lag_values = build_lag_values(args.max_lag, args.lag_step)
    npz_files = sorted(path for path in data_dir.iterdir() if path.suffix == ".npz")
    if args.max_files > 0:
        npz_files = npz_files[: args.max_files]
    if not npz_files:
        raise FileNotFoundError(f"No latent .npz files found in {data_dir}")

    latent_config = load_json(data_dir / "latent_config.json") if (data_dir / "latent_config.json").is_file() else None
    actions_info = load_json(data_dir / "actions.json") if (data_dir / "actions.json").is_file() else None
    action_values = None
    if actions_info is not None and actions_info.get("reduced_to_original_value") is not None:
        action_values = [int(v) for v in actions_info["reduced_to_original_value"]]

    first_file = npz_files[0]
    with np.load(first_file, mmap_mode="r") as probe:
        if "latents" not in probe.files:
            raise ValueError(f"{first_file.name} is missing latents")
        latent_shape = tuple(int(dim) for dim in probe["latents"].shape)
        if len(latent_shape) != 4:
            raise ValueError(f"Invalid latent shape in {first_file.name}: {latent_shape}")
        channels, _, height, width = latent_shape
        has_actions = "actions" in probe.files

    action_stats = {
        "frame_count": np.zeros(max(len(action_values) if action_values is not None else 1, 1), dtype=np.int64),
        "frame_sum": np.zeros(max(len(action_values) if action_values is not None else 1, 1), dtype=np.float64),
        "frame_sum_sq": np.zeros(max(len(action_values) if action_values is not None else 1, 1), dtype=np.float64),
        "delta_count": np.zeros(max(len(action_values) if action_values is not None else 1, 1), dtype=np.int64),
        "delta_sum": np.zeros(max(len(action_values) if action_values is not None else 1, 1), dtype=np.float64),
        "delta_sum_sq": np.zeros(max(len(action_values) if action_values is not None else 1, 1), dtype=np.float64),
    }

    total_frames = 0
    total_values = 0
    total_deltas = 0
    global_sum = 0.0
    global_sum_sq = 0.0
    global_abs_sum = 0.0
    global_min = float("inf")
    global_max = float("-inf")
    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_sum_sq = np.zeros(channels, dtype=np.float64)
    channel_abs_sum = np.zeros(channels, dtype=np.float64)
    channel_delta_sum_sq = np.zeros(channels, dtype=np.float64)
    spatial_sum_sq = np.zeros((height, width), dtype=np.float64)
    pooled_sum = np.zeros(channels, dtype=np.float64)
    pooled_xtx = np.zeros((channels, channels), dtype=np.float64)
    pooled_count = 0
    lag_cosine_sum = np.zeros(len(lag_values), dtype=np.float64)
    lag_cosine_count = np.zeros(len(lag_values), dtype=np.int64)
    recording_rows: list[dict[str, Any]] = []

    use_live_progress = sys.stdout.isatty()
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        disable=not use_live_progress,
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing latents", total=len(npz_files))

        for file_path in npz_files:
            with np.load(file_path, mmap_mode="r") as npz:
                latents = npz["latents"]
                if latents.ndim != 4:
                    raise ValueError(
                        f"{file_path.name} has invalid latents shape {tuple(latents.shape)}; expected (C, T, H, W)"
                    )
                if int(latents.shape[0]) != channels or int(latents.shape[2]) != height or int(latents.shape[3]) != width:
                    raise ValueError(f"{file_path.name} has inconsistent latent shape {tuple(latents.shape)}")
                if has_actions and "actions" not in npz.files:
                    raise ValueError(f"{file_path.name} is missing actions even though earlier files include them")

                actions = np.asarray(npz["actions"]) if has_actions and "actions" in npz.files else None
                frame_limit = int(latents.shape[1])
                if args.max_frames_per_file > 0:
                    frame_limit = min(frame_limit, args.max_frames_per_file)
                if actions is not None and int(actions.shape[0]) < frame_limit:
                    raise ValueError(
                        f"{file_path.name} has {frame_limit} latent frames but only {actions.shape[0]} actions"
                    )

                file_value_count = 0
                file_abs_sum = 0.0
                file_frame_rms_sum = 0.0
                file_delta_rms_sum = 0.0
                file_delta_count = 0
                tail_pooled = np.empty((0, channels), dtype=np.float32)
                prev_frame: np.ndarray | None = None

                for start in range(0, frame_limit, args.chunk_frames):
                    end = min(start + args.chunk_frames, frame_limit)
                    chunk = np.asarray(latents[:, start:end], dtype=np.float32)
                    chunk_actions = np.asarray(actions[start:end], dtype=np.int64) if actions is not None else None

                    chunk_sq = np.square(chunk)
                    chunk_abs = np.abs(chunk)
                    chunk_pooled = chunk.mean(axis=(2, 3), dtype=np.float32).transpose(1, 0)
                    frame_rms = np.sqrt(chunk_sq.mean(axis=(0, 2, 3), dtype=np.float32)).astype(np.float32, copy=False)

                    total_frames += int(chunk.shape[1])
                    pooled_count += int(chunk_pooled.shape[0])
                    value_count = int(chunk.size)
                    total_values += value_count
                    file_value_count += value_count

                    global_sum += float(chunk.sum(dtype=np.float64))
                    global_sum_sq += float(chunk_sq.sum(dtype=np.float64))
                    global_abs_sum += float(chunk_abs.sum(dtype=np.float64))
                    global_min = min(global_min, float(chunk.min()))
                    global_max = max(global_max, float(chunk.max()))

                    channel_sum += chunk.sum(axis=(1, 2, 3), dtype=np.float64)
                    channel_sum_sq += chunk_sq.sum(axis=(1, 2, 3), dtype=np.float64)
                    channel_abs_sum += chunk_abs.sum(axis=(1, 2, 3), dtype=np.float64)
                    spatial_sum_sq += chunk_sq.sum(axis=(0, 1), dtype=np.float64)
                    pooled_sum += chunk_pooled.sum(axis=0, dtype=np.float64)
                    pooled_xtx += chunk_pooled.T.astype(np.float64) @ chunk_pooled.astype(np.float64)

                    file_abs_sum += float(chunk_abs.sum(dtype=np.float64))
                    file_frame_rms_sum += float(frame_rms.sum(dtype=np.float64))

                    if chunk_actions is not None:
                        update_action_frame_stats(action_stats, chunk_actions, frame_rms.astype(np.float64))

                    if lag_values:
                        combined = np.concatenate((tail_pooled, chunk_pooled), axis=0)
                        tail_only = int(tail_pooled.shape[0])
                        for lag_index, lag in enumerate(lag_values):
                            if combined.shape[0] <= lag:
                                break
                            lhs = combined[:-lag]
                            rhs = combined[lag:]
                            drop = max(tail_only - lag, 0)
                            if drop > 0:
                                lhs = lhs[drop:]
                                rhs = rhs[drop:]
                            if lhs.shape[0] == 0:
                                continue
                            numer = np.sum(lhs * rhs, axis=1, dtype=np.float64)
                            denom = np.linalg.norm(lhs, axis=1) * np.linalg.norm(rhs, axis=1)
                            valid = denom > 1e-8
                            if not np.any(valid):
                                continue
                            lag_cosine_sum[lag_index] += float(np.sum(numer[valid] / denom[valid], dtype=np.float64))
                            lag_cosine_count[lag_index] += int(np.count_nonzero(valid))
                        tail_pooled = combined[-lag_values[-1] :].copy()

                    if prev_frame is not None:
                        delta0 = chunk[:, :1] - prev_frame[:, None, :, :]
                        delta0_sq = np.square(delta0)
                        delta0_rms = np.sqrt(delta0_sq.mean(axis=(0, 2, 3), dtype=np.float32)).astype(np.float32, copy=False)
                        channel_delta_sum_sq += delta0_sq.sum(axis=(1, 2, 3), dtype=np.float64)
                        total_deltas += 1
                        file_delta_rms_sum += float(delta0_rms.sum(dtype=np.float64))
                        file_delta_count += 1
                        if chunk_actions is not None:
                            update_action_delta_stats(action_stats, chunk_actions[:1], delta0_rms.astype(np.float64))

                    if chunk.shape[1] > 1:
                        delta = chunk[:, 1:] - chunk[:, :-1]
                        delta_sq = np.square(delta)
                        delta_rms = np.sqrt(delta_sq.mean(axis=(0, 2, 3), dtype=np.float32)).astype(np.float32, copy=False)
                        channel_delta_sum_sq += delta_sq.sum(axis=(1, 2, 3), dtype=np.float64)
                        total_deltas += int(delta.shape[1])
                        file_delta_rms_sum += float(delta_rms.sum(dtype=np.float64))
                        file_delta_count += int(delta.shape[1])
                        if chunk_actions is not None:
                            update_action_delta_stats(action_stats, chunk_actions[1:], delta_rms.astype(np.float64))

                    prev_frame = chunk[:, -1].copy()

                recording_rows.append(
                    {
                        "file": file_path.name,
                        "frames": frame_limit,
                        "mean_abs": file_abs_sum / max(file_value_count, 1),
                        "frame_rms_mean": file_frame_rms_sum / max(frame_limit, 1),
                        "delta_rms_mean": file_delta_rms_sum / max(file_delta_count, 1),
                    }
                )

            progress.update(task, advance=1)

    if total_values == 0:
        raise RuntimeError("No latent values were analyzed.")

    channel_value_count = total_frames * height * width
    channel_mean = channel_sum / max(channel_value_count, 1)
    channel_var = channel_sum_sq / max(channel_value_count, 1) - np.square(channel_mean)
    channel_std = np.sqrt(np.maximum(channel_var, 0.0))
    channel_abs_mean = channel_abs_sum / max(channel_value_count, 1)
    channel_delta_rms = np.sqrt(channel_delta_sum_sq / max(total_deltas * height * width, 1))

    global_mean = global_sum / total_values
    global_var = global_sum_sq / total_values - global_mean * global_mean
    global_std = math.sqrt(max(global_var, 0.0))
    global_rms = math.sqrt(max(global_sum_sq / total_values, 0.0))
    global_abs_mean = global_abs_sum / total_values
    spatial_rms = np.sqrt(spatial_sum_sq / max(total_frames * channels, 1))

    pooled_mean = pooled_sum / max(pooled_count, 1)
    pooled_cov = pooled_xtx / max(pooled_count, 1) - np.outer(pooled_mean, pooled_mean)
    pooled_eigvals = np.linalg.eigvalsh(pooled_cov)
    pooled_eigvals = np.clip(np.sort(pooled_eigvals)[::-1], a_min=0.0, a_max=None)
    eigval_sum = float(pooled_eigvals.sum())
    if eigval_sum > 0.0:
        eigval_probs = pooled_eigvals / eigval_sum
        effective_rank = float(math.exp(-np.sum(eigval_probs * np.log(np.maximum(eigval_probs, 1e-12)))))
    else:
        effective_rank = 0.0

    lag_rows = []
    for lag_index, lag in enumerate(lag_values):
        count = int(lag_cosine_count[lag_index])
        lag_rows.append(
            {
                "lag": lag,
                "pair_count": count,
                "mean_cosine": float(lag_cosine_sum[lag_index] / count) if count else 0.0,
            }
        )

    channel_rows = []
    for channel_idx in range(channels):
        channel_rows.append(
            {
                "channel": channel_idx,
                "mean": float(channel_mean[channel_idx]),
                "std": float(channel_std[channel_idx]),
                "abs_mean": float(channel_abs_mean[channel_idx]),
                "delta_rms": float(channel_delta_rms[channel_idx]),
            }
        )

    action_rows = []
    for action_idx in range(len(action_stats["frame_count"])):
        if int(action_stats["frame_count"][action_idx]) == 0 and int(action_stats["delta_count"][action_idx]) == 0:
            continue
        action_rows.append(build_action_row(action_idx, action_stats, action_values))

    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "num_files": len(npz_files),
        "total_frames": total_frames,
        "latent_shape": {
            "channels": channels,
            "height": height,
            "width": width,
        },
        "source_latent_dtype": "float16",
        "approx_dataset_bytes": int(sum(path.stat().st_size for path in npz_files)),
        "latent_config": latent_config,
        "global_stats": {
            "mean": global_mean,
            "std": global_std,
            "rms": global_rms,
            "abs_mean": global_abs_mean,
            "min": global_min,
            "max": global_max,
        },
        "pooled_state": {
            "effective_rank": effective_rank,
            "eigenvalues": [float(v) for v in pooled_eigvals[: min(8, len(pooled_eigvals))]],
        },
        "channel_rows": channel_rows,
        "lag_rows": lag_rows,
        "action_rows": action_rows,
        "recording_rows": recording_rows,
        "spatial_rms": spatial_rms,
    }
    return summary


def print_summary(summary: dict[str, Any], *, top_k: int) -> None:
    dataset_table = Table(title="Latent Dataset Overview")
    dataset_table.add_column("Field")
    dataset_table.add_column("Value", justify="right")
    dataset_table.add_row("Data dir", summary["data_dir"])
    dataset_table.add_row("Recordings", f"{summary['num_files']:,}")
    dataset_table.add_row("Frames", f"{summary['total_frames']:,}")
    dataset_table.add_row(
        "Latent shape",
        (
            f"{summary['latent_shape']['channels']} x {summary['latent_shape']['height']} x "
            f"{summary['latent_shape']['width']}"
        ),
    )
    dataset_table.add_row("Compressed size", format_bytes(int(summary["approx_dataset_bytes"])))
    console.print(dataset_table)

    global_stats = summary["global_stats"]
    global_table = Table(title="Global Latent Statistics")
    global_table.add_column("Metric")
    global_table.add_column("Value", justify="right")
    global_table.add_row("Mean", f"{global_stats['mean']:.5f}")
    global_table.add_row("Std", f"{global_stats['std']:.5f}")
    global_table.add_row("RMS", f"{global_stats['rms']:.5f}")
    global_table.add_row("Mean |x|", f"{global_stats['abs_mean']:.5f}")
    global_table.add_row("Min", f"{global_stats['min']:.5f}")
    global_table.add_row("Max", f"{global_stats['max']:.5f}")
    global_table.add_row("Pooled effective rank", f"{summary['pooled_state']['effective_rank']:.2f}")
    console.print(global_table)

    channel_rows = summary["channel_rows"]
    top_motion = sorted(channel_rows, key=lambda row: row["delta_rms"], reverse=True)[:top_k]
    channel_table = Table(title=f"Top {min(top_k, len(top_motion))} Moving Channels")
    channel_table.add_column("Channel", justify="right")
    channel_table.add_column("Std", justify="right")
    channel_table.add_column("Mean |x|", justify="right")
    channel_table.add_column("Delta RMS", justify="right")
    for row in top_motion:
        channel_table.add_row(
            str(row["channel"]),
            f"{row['std']:.5f}",
            f"{row['abs_mean']:.5f}",
            f"{row['delta_rms']:.5f}",
        )
    console.print(channel_table)

    lag_rows = summary["lag_rows"]
    if lag_rows:
        lag_table = Table(title="Lagged Pooled-State Similarity")
        lag_table.add_column("Lag", justify="right")
        lag_table.add_column("Mean Cosine", justify="right")
        lag_table.add_column("Pairs", justify="right")
        for row in lag_rows[: min(top_k, len(lag_rows))]:
            lag_table.add_row(str(row["lag"]), f"{row['mean_cosine']:.5f}", f"{row['pair_count']:,}")
        console.print(lag_table)

    action_rows = [row for row in summary["action_rows"] if row["delta_count"] > 0]
    if action_rows:
        action_rows = sorted(action_rows, key=lambda row: row["delta_rms_mean"], reverse=True)
        action_table = Table(title=f"Top {min(top_k, len(action_rows))} Actions By Latent Motion")
        action_table.add_column("Idx", justify="right")
        action_table.add_column("Buttons")
        action_table.add_column("Frames", justify="right")
        action_table.add_column("Frame RMS", justify="right")
        action_table.add_column("Delta RMS", justify="right")
        for row in action_rows[:top_k]:
            action_table.add_row(
                str(row["index"]),
                row["label"],
                f"{row['frame_count']:,}",
                f"{row['frame_rms_mean']:.5f}",
                f"{row['delta_rms_mean']:.5f}",
            )
        console.print(action_table)

    recording_rows = summary["recording_rows"]
    if recording_rows:
        recording_rows = sorted(recording_rows, key=lambda row: row["delta_rms_mean"], reverse=True)
        recording_table = Table(title=f"Top {min(top_k, len(recording_rows))} Most Dynamic Recordings")
        recording_table.add_column("Recording")
        recording_table.add_column("Frames", justify="right")
        recording_table.add_column("Mean |x|", justify="right")
        recording_table.add_column("Delta RMS", justify="right")
        for row in recording_rows[:top_k]:
            recording_table.add_row(
                row["file"],
                f"{row['frames']:,}",
                f"{row['mean_abs']:.5f}",
                f"{row['delta_rms_mean']:.5f}",
            )
        console.print(recording_table)


def main() -> None:
    args = parse_args()
    summary = analyze_dataset(args)
    print_summary(summary, top_k=args.top_k)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "summary.json").open("w") as handle:
            json.dump(to_serializable(summary), handle, indent=2)
        console.print(f"[green]Wrote summary:[/green] {output_dir / 'summary.json'}")
        if not args.skip_plots:
            save_plots(output_dir, summary, top_k=args.top_k)
            console.print(f"[green]Wrote plots to:[/green] {output_dir}")


if __name__ == "__main__":
    main()