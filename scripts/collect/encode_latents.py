#!/usr/bin/env python3
"""Pre-encode normalized frames into VAE latents for faster DiT training.

Reads data/normalized/*.npz, encodes each recording's frames through a frozen
video VAE, and writes per-recording .npz files to data/latents/ containing
latents (float16) and actions.

Usage:
    python scripts/encode_latents.py --video-vae-checkpoint checkpoints/my_vae/video_vae_best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.ltx_video_vae import LTXVideoVAE
from path_utils import serialize_project_path

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-encode normalized frames into VAE latents.")
    parser.add_argument("--data-dir", type=str, default="data/normalized",
                        help="Directory containing normalized .npz files.")
    parser.add_argument("--output-dir", type=str, default="data/latents",
                        help="Directory to write latent .npz files.")
    parser.add_argument("--video-vae-checkpoint", type=str, default=None,
                        help="Path to the video VAE checkpoint (.pt).")
    parser.add_argument("--video-vae-config", type=str, default=None,
                        help="Path to the video VAE config.json (inferred from checkpoint dir if omitted).")
    parser.add_argument("--batch-frames", type=int, default=64,
                        help="Number of frames to encode at once (temporal batch size).")
    parser.add_argument("--onehot-dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--autocast", action="store_true",
                        help="Use torch.autocast for encoding.")
    parser.add_argument("--autocast-dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode files that already exist in output-dir.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip encoding and recompute latent_stats.json from existing latent .npz files in output-dir.",
    )
    args = parser.parse_args()
    if not args.stats_only and not args.video_vae_checkpoint:
        parser.error("--video-vae-checkpoint is required unless --stats-only is set")
    return args


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def infer_vae_config_path(args: argparse.Namespace) -> Path:
    if args.video_vae_config is not None:
        return Path(args.video_vae_config).resolve()
    if args.video_vae_checkpoint is None:
        raise FileNotFoundError("--video-vae-checkpoint is required to infer the VAE config path")
    sibling = Path(args.video_vae_checkpoint).resolve().parent / "config.json"
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        "Could not infer VAE config from checkpoint directory; pass --video-vae-config explicitly."
    )


def load_video_vae(
    checkpoint_path: Path,
    config_path: Path,
    num_colors: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    cfg = load_json(config_path)
    patch_size = int(cfg.get("patch_size", 4))
    base_channels = int(cfg.get("base_channels", 64))
    latent_channels = int(cfg.get("latent_channels", 64))
    vae_num_colors = int(cfg.get("num_colors", num_colors))

    if vae_num_colors != num_colors:
        raise ValueError(
            f"VAE config expects num_colors={vae_num_colors} but palette has {num_colors}."
        )

    vae = LTXVideoVAE(
        num_colors=vae_num_colors,
        patch_size=patch_size,
        base_channels=base_channels,
        latent_channels=latent_channels,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    summary = {
        "patch_size": patch_size,
        "base_channels": base_channels,
        "latent_channels": latent_channels,
        "num_colors": vae_num_colors,
    }
    return vae, summary


def frames_to_one_hot(
    frames: torch.Tensor,
    num_colors: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert (B, T, H, W) palette indices to (B, num_colors, T, H, W) one-hot."""
    if frames.dtype != torch.long:
        frames = frames.long()
    out = torch.zeros(
        frames.shape[0], num_colors, frames.shape[1], frames.shape[2], frames.shape[3],
        dtype=dtype, device=frames.device,
    )
    out.scatter_(1, frames.unsqueeze(1), 1)
    return out


def encode_file(
    vae: torch.nn.Module,
    frames_np: np.ndarray,
    *,
    num_colors: int,
    batch_frames: int,
    onehot_dtype: torch.dtype,
    device: torch.device,
    autocast_ctx,
) -> np.ndarray:
    """Encode all frames from one recording, returning latents as float16 numpy."""
    total_frames = frames_np.shape[0]
    latent_chunks: list[np.ndarray] = []

    for start in range(0, total_frames, batch_frames):
        end = min(start + batch_frames, total_frames)
        chunk = torch.from_numpy(frames_np[start:end]).to(device)
        # VAE expects (B, C, T, H, W) — use batch=1 with T=chunk_len
        one_hot = frames_to_one_hot(chunk.unsqueeze(0), num_colors, onehot_dtype)
        with torch.inference_mode(), autocast_ctx:
            mean, _ = vae.encode(one_hot)
        # mean shape: (1, latent_ch, T, H', W')
        latent_chunks.append(mean[0].cpu().to(torch.float16).numpy())

    # Concatenate along temporal axis (dim 1 in (C, T, H', W'))
    return np.concatenate(latent_chunks, axis=1)


def compute_latent_channel_stats(
    latent_files: list[Path],
    *,
    chunk_frames: int = 2048,
) -> dict[str, Any]:
    """Compute per-channel latent statistics over encoded latent files."""
    if not latent_files:
        raise RuntimeError("No latent files were provided for statistics computation.")

    channel_sum: np.ndarray | None = None
    channel_sum_sq: np.ndarray | None = None
    channels = 0
    height = 0
    width = 0
    count_per_channel = 0
    total_values = 0
    global_sum = 0.0
    global_sum_sq = 0.0
    global_abs_sum = 0.0
    global_min = float("inf")
    global_max = float("-inf")

    for npz_path in latent_files:
        with np.load(npz_path, mmap_mode="r") as data:
            if "latents" not in data.files:
                raise ValueError(f"{npz_path.name} is missing latents array")

            latents = data["latents"]
            if latents.ndim != 4:
                raise ValueError(
                    f"{npz_path.name} has invalid latents shape {tuple(latents.shape)}; expected (C, T, H, W)"
                )

            file_channels, file_frames, file_height, file_width = (int(v) for v in latents.shape)
            if channel_sum is None:
                channels = file_channels
                height = file_height
                width = file_width
                channel_sum = np.zeros(channels, dtype=np.float64)
                channel_sum_sq = np.zeros(channels, dtype=np.float64)
            elif (file_channels, file_height, file_width) != (channels, height, width):
                raise ValueError(
                    f"{npz_path.name} has inconsistent latent shape {tuple(latents.shape)}; "
                    f"expected ({channels}, T, {height}, {width})"
                )

            for start in range(0, file_frames, chunk_frames):
                end = min(start + chunk_frames, file_frames)
                chunk = np.asarray(latents[:, start:end], dtype=np.float32)
                chunk64 = chunk.astype(np.float64, copy=False)
                chunk_sq = np.square(chunk64)
                chunk_abs = np.abs(chunk64)

                assert channel_sum is not None
                assert channel_sum_sq is not None
                channel_sum += chunk64.sum(axis=(1, 2, 3), dtype=np.float64)
                channel_sum_sq += chunk_sq.sum(axis=(1, 2, 3), dtype=np.float64)

                count_per_channel += int((end - start) * height * width)
                total_values += int(chunk.size)
                global_sum += float(chunk64.sum(dtype=np.float64))
                global_sum_sq += float(chunk_sq.sum(dtype=np.float64))
                global_abs_sum += float(chunk_abs.sum(dtype=np.float64))
                global_min = min(global_min, float(chunk.min()))
                global_max = max(global_max, float(chunk.max()))

    if channel_sum is None or channel_sum_sq is None or count_per_channel <= 0 or total_values <= 0:
        raise RuntimeError("Unable to compute latent statistics from provided files.")

    channel_mean = channel_sum / float(count_per_channel)
    channel_var = channel_sum_sq / float(count_per_channel) - np.square(channel_mean)
    channel_std = np.sqrt(np.maximum(channel_var, 0.0))

    std_epsilon = 1e-6
    channel_std_clamped = np.maximum(channel_std, std_epsilon)

    global_mean = global_sum / float(total_values)
    global_var = global_sum_sq / float(total_values) - global_mean * global_mean
    global_std = float(np.sqrt(max(global_var, 0.0)))
    global_rms = float(np.sqrt(max(global_sum_sq / float(total_values), 0.0)))

    return {
        "num_files": len(latent_files),
        "latent_shape": {
            "channels": channels,
            "height": height,
            "width": width,
        },
        "count_per_channel": int(count_per_channel),
        "std_epsilon": std_epsilon,
        "channel_mean": [float(v) for v in channel_mean],
        "channel_std": [float(v) for v in channel_std],
        "channel_std_clamped": [float(v) for v in channel_std_clamped],
        "global_stats": {
            "mean": float(global_mean),
            "std": global_std,
            "rms": global_rms,
            "abs_mean": float(global_abs_sum / float(total_values)),
            "min": float(global_min),
            "max": float(global_max),
        },
    }


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_config_path = output_dir / "latent_config.json"
    latent_meta = load_json(latent_config_path) if latent_config_path.is_file() else {}

    if args.stats_only:
        latent_npz_files = sorted(p for p in output_dir.iterdir() if p.suffix == ".npz")
        if not latent_npz_files:
            raise RuntimeError(f"No latent .npz files found in {output_dir}.")

        console.print(f"[stats] Computing latent channel stats over {len(latent_npz_files)} files...")
        stats_start = time.time()
        latent_stats = compute_latent_channel_stats(latent_npz_files)
        stats_path = output_dir / "latent_stats.json"
        with stats_path.open("w") as f:
            json.dump(latent_stats, f, indent=2)
        latent_meta["latent_stats_file"] = stats_path.name
        latent_meta["latent_stats_path"] = str(stats_path.resolve())
        with latent_config_path.open("w") as f:
            json.dump(latent_meta, f, indent=2)
        stats_elapsed = time.time() - stats_start
        console.print(f"[stats] Wrote {stats_path} in {stats_elapsed:.1f}s")
        console.print(f"Output: {output_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    onehot_dtype = dtype_map[args.onehot_dtype]
    autocast_dtype = dtype_map[args.autocast_dtype]

    autocast_enabled = args.autocast and device.type == "cuda"
    if autocast_enabled and autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        console.print("[autocast] bfloat16 unsupported; falling back to float16.")
        autocast_dtype = torch.float16
    if not autocast_enabled and onehot_dtype != torch.float32:
        console.print("[encode] Non-autocast encoding requires float32 inputs; overriding --onehot-dtype to float32.")
        onehot_dtype = torch.float32

    def make_autocast_ctx():
        if autocast_enabled:
            return torch.autocast(device_type="cuda", dtype=autocast_dtype)
        return nullcontext()

    data_dir = Path(args.data_dir)

    # Load palette to get num_colors
    palette_path = data_dir / "palette.json"
    if not palette_path.is_file():
        raise FileNotFoundError(f"Missing {palette_path}")
    palette_info = load_json(palette_path)
    num_colors = len(palette_info["colors_rgb"])
    console.print(f"[palette] {num_colors} colors")

    # Load VAE
    vae_checkpoint = Path(args.video_vae_checkpoint).resolve()
    vae_config = infer_vae_config_path(args)
    vae, vae_summary = load_video_vae(vae_checkpoint, vae_config, num_colors, device)
    console.print(f"[vae] latent_channels={vae_summary['latent_channels']}")

    # Copy metadata files
    for meta_name in ["palette.json", "palette_distribution.json", "actions.json", "ram_addresses.json"]:
        src = data_dir / meta_name
        if src.is_file():
            shutil.copy2(src, output_dir / meta_name)

    # Save full VAE config alongside latents for exact reproducibility
    vae_config_full = load_json(vae_config)
    # Copy the actual config.json used by the VAE
    shutil.copy2(vae_config, output_dir / "video_vae_config.json")
    latent_meta = {
        "source_data_dir": serialize_project_path(data_dir, project_root=PROJECT_ROOT),
        "video_vae_checkpoint": serialize_project_path(vae_checkpoint, project_root=PROJECT_ROOT),
        "video_vae_config_path": serialize_project_path(vae_config, project_root=PROJECT_ROOT),
        "video_vae_config": vae_config_full,
        "num_colors": num_colors,
    }
    with latent_config_path.open("w") as f:
        json.dump(latent_meta, f, indent=2)

    # Collect normalized .npz files
    npz_files = sorted(p for p in data_dir.iterdir() if p.suffix == ".npz")
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    console.print(f"Found {len(npz_files)} recordings to encode")

    # Skip already-encoded files unless --overwrite
    if not args.overwrite:
        to_encode = []
        skipped = 0
        for p in npz_files:
            if (output_dir / p.name).is_file():
                skipped += 1
            else:
                to_encode.append(p)
        if skipped:
            console.print(f"Skipping {skipped} already-encoded files (use --overwrite to re-encode)")
        npz_files = to_encode

    total_frames_encoded = 0
    start_time = time.time()

    if npz_files:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            refresh_per_second=2,
        ) as progress:
            task = progress.add_task("Encoding", total=len(npz_files), status="")

            for npz_path in npz_files:
                with np.load(npz_path) as data:
                    frames = np.asarray(data["frames"])
                    actions = np.asarray(data["actions"])

                num_frames = frames.shape[0]
                progress.update(task, status=f"{npz_path.name} ({num_frames} frames)")

                latents = encode_file(
                    vae,
                    frames,
                    num_colors=num_colors,
                    batch_frames=args.batch_frames,
                    onehot_dtype=onehot_dtype,
                    device=device,
                    autocast_ctx=make_autocast_ctx(),
                )

                out_path = output_dir / npz_path.name
                np.savez_compressed(
                    out_path,
                    latents=latents,
                    actions=actions,
                )

                total_frames_encoded += num_frames
                elapsed = max(time.time() - start_time, 1e-6)
                fps = total_frames_encoded / elapsed
                progress.update(task, advance=1, status=f"{fps:.0f} frames/s total")

        elapsed = time.time() - start_time
        console.print(
            f"Done. Encoded {total_frames_encoded:,} frames from {len(npz_files)} files "
            f"in {elapsed:.1f}s ({total_frames_encoded / max(elapsed, 1e-6):.0f} frames/s)"
        )
    else:
        console.print("No new files to encode; recomputing latent stats from existing outputs.")

    latent_npz_files = sorted(p for p in output_dir.iterdir() if p.suffix == ".npz")
    if not latent_npz_files:
        raise RuntimeError(f"No latent .npz files found in {output_dir}.")

    console.print(f"[stats] Computing latent channel stats over {len(latent_npz_files)} files...")
    stats_start = time.time()
    latent_stats = compute_latent_channel_stats(latent_npz_files)
    stats_path = output_dir / "latent_stats.json"
    with stats_path.open("w") as f:
        json.dump(latent_stats, f, indent=2)
    stats_elapsed = time.time() - stats_start
    console.print(f"[stats] Wrote {stats_path} in {stats_elapsed:.1f}s")

    latent_meta["latent_stats_file"] = stats_path.name
    latent_meta["latent_stats_path"] = serialize_project_path(stats_path, project_root=PROJECT_ROOT)
    with latent_config_path.open("w") as f:
        json.dump(latent_meta, f, indent=2)

    console.print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
