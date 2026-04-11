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
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.video_vae import VideoVAE
from src.data.video_frames import SUPPORTED_FRAME_SIZES, resize_palette_frames
from src.path_utils import serialize_project_path

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-encode normalized frames into VAE latents.")
    parser.add_argument("--data-dir", type=str, default="data/normalized",
                        help="Directory containing normalized .npz files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write latent .npz files (defaults to data/latents or data/latents_<frame-size>).")
    parser.add_argument("--video-vae-checkpoint", type=str, default=None,
                        help="Path to the video VAE checkpoint (.pt).")
    parser.add_argument("--video-vae-config", type=str, default=None,
                        help="Path to the video VAE config.json (inferred from checkpoint dir if omitted).")
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        choices=SUPPORTED_FRAME_SIZES,
        help="Frame size to encode after runtime resizing.",
    )
    parser.add_argument("--batch-frames", type=int, default=64,
                        help="Number of frames to encode at once (temporal batch size).")
    parser.add_argument("--min-batch-frames", type=int, default=1,
                        help="Minimum frames per chunk when auto-shrinking after CUDA OOM.")
    parser.add_argument("--onehot-dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--autocast", action="store_true",
                        help="Use torch.autocast for encoding.")
    parser.add_argument("--autocast-dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--no-auto-batch-shrink", action="store_true",
                        help="Disable automatic batch-frames reduction on CUDA OOM.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode files that already exist in output-dir.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip encoding and recompute latent_stats.json from existing latent .npz files in output-dir.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = "data/latents" if args.frame_size == 224 else f"data/latents_{args.frame_size}"
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
    model_cfg = dict(cfg.get("model", {}))
    data_cfg = dict(cfg.get("data", {}))
    base_channels = int(model_cfg.get("base_channels", cfg.get("base_channels", 64)))
    latent_channels = int(model_cfg.get("latent_channels", cfg.get("latent_channels", 64)))
    temporal_downsample = int(model_cfg.get("temporal_downsample", cfg.get("temporal_downsample", 0)))
    global_bottleneck_attn = bool(
        model_cfg.get("global_bottleneck_attn", cfg.get("global_bottleneck_attn", False))
    )
    global_bottleneck_attn_heads = int(
        model_cfg.get("global_bottleneck_attn_heads", cfg.get("global_bottleneck_attn_heads", 8))
    )
    onehot_conv = bool(model_cfg.get("onehot_conv", cfg.get("onehot_conv", False)))
    vae_num_colors = int(data_cfg.get("num_colors", cfg.get("num_colors", num_colors)))

    if vae_num_colors != num_colors:
        raise ValueError(
            f"VAE config expects num_colors={vae_num_colors} but palette has {num_colors}."
        )

    vae = VideoVAE(
        num_colors=vae_num_colors,
        base_channels=base_channels,
        latent_channels=latent_channels,
        temporal_downsample=temporal_downsample,
        onehot_conv=onehot_conv,
        global_bottleneck_attn=global_bottleneck_attn,
        global_bottleneck_attn_heads=global_bottleneck_attn_heads,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    summary = {
        "base_channels": base_channels,
        "latent_channels": latent_channels,
        "temporal_downsample": temporal_downsample,
        "global_bottleneck_attn": global_bottleneck_attn,
        "global_bottleneck_attn_heads": global_bottleneck_attn_heads,
        "onehot_conv": onehot_conv,
        "num_colors": vae_num_colors,
    }
    return vae, summary


def downsample_actions(actions_np: np.ndarray, *, temporal_downsample: int) -> np.ndarray:
    if temporal_downsample == 0:
        return actions_np
    if temporal_downsample != 1:
        raise ValueError("temporal_downsample must be 0 or 1")
    if actions_np.shape[0] == 0:
        return actions_np
    indices = np.arange(0, actions_np.shape[0], 2, dtype=np.int64) + 1
    indices = np.clip(indices, 0, actions_np.shape[0] - 1)
    return actions_np[indices]


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
    min_batch_frames: int,
    onehot_dtype: torch.dtype,
    device: torch.device,
    autocast_ctx,
    auto_batch_shrink: bool,
    use_onehot_conv: bool = False,
    require_even_batch_frames: bool = False,
) -> np.ndarray:
    """Encode all frames from one recording, returning latents as float16 numpy."""
    total_frames = frames_np.shape[0]
    latent_chunks: list[np.ndarray] = []

    if batch_frames < 1:
        raise ValueError("batch_frames must be >= 1")
    if min_batch_frames < 1:
        raise ValueError("min_batch_frames must be >= 1")
    if require_even_batch_frames and (batch_frames % 2 != 0 or min_batch_frames % 2 != 0):
        raise ValueError("batch_frames and min_batch_frames must be even when temporal downsampling is enabled")

    effective_batch_frames = max(batch_frames, min_batch_frames)
    start = 0
    while start < total_frames:
        current_batch_frames = min(effective_batch_frames, total_frames - start)
        if require_even_batch_frames and current_batch_frames > 1 and current_batch_frames % 2 != 0:
            current_batch_frames -= 1
        end = min(start + current_batch_frames, total_frames)

        chunk = None
        one_hot = None
        mean = None

        try:
            chunk = torch.from_numpy(frames_np[start:end]).to(device)
            # VAE expects (B, C, T, H, W) — use batch=1 with T=chunk_len
            if use_onehot_conv:
                one_hot = chunk.unsqueeze(0).to(torch.uint8)
            else:
                one_hot = frames_to_one_hot(chunk.unsqueeze(0), num_colors, onehot_dtype)
            with torch.inference_mode(), autocast_ctx:
                mean, _ = vae.encode(one_hot)
            # mean shape: (1, latent_ch, T, H', W')
            latent_chunks.append(mean[0].cpu().to(torch.float16).numpy())
            start = end
        except torch.OutOfMemoryError as exc:
            if device.type != "cuda" or not auto_batch_shrink:
                raise

            if current_batch_frames <= min_batch_frames:
                raise RuntimeError(
                    f"CUDA OOM even at minimum batch-frames ({min_batch_frames}). "
                    "Try lowering --frame-size, enabling --autocast, or using CPU."
                ) from exc

            next_batch_frames = max(min_batch_frames, current_batch_frames // 2)
            if require_even_batch_frames and next_batch_frames % 2 != 0:
                next_batch_frames -= 1
            if next_batch_frames >= current_batch_frames:
                next_batch_frames = current_batch_frames - 1
                if require_even_batch_frames and next_batch_frames % 2 != 0:
                    next_batch_frames -= 1
            next_batch_frames = max(min_batch_frames, next_batch_frames)

            console.print(
                f"[yellow][oom][/yellow] CUDA OOM at batch-frames={current_batch_frames}; "
                f"retrying with batch-frames={next_batch_frames}"
            )
            effective_batch_frames = next_batch_frames
            torch.cuda.empty_cache()
        finally:
            del chunk, one_hot, mean

    # Concatenate along temporal axis (dim 1 in (C, T, H', W'))
    return np.concatenate(latent_chunks, axis=1)


def compute_latent_component_stats(
    latent_files: list[Path],
    *,
    chunk_frames: int = 2048,
) -> dict[str, Any]:
    """Compute per-component latent statistics over encoded latent files.

    Components are indexed by (C, H, W) and aggregated over time/file axes.
    """
    if not latent_files:
        raise RuntimeError("No latent files were provided for statistics computation.")

    component_sum: np.ndarray | None = None
    component_sum_sq: np.ndarray | None = None
    channels = 0
    height = 0
    width = 0
    count_per_component = 0
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
            if component_sum is None:
                channels = file_channels
                height = file_height
                width = file_width
                component_sum = np.zeros((channels, height, width), dtype=np.float64)
                component_sum_sq = np.zeros((channels, height, width), dtype=np.float64)
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

                assert component_sum is not None
                assert component_sum_sq is not None
                component_sum += chunk64.sum(axis=1, dtype=np.float64)
                component_sum_sq += chunk_sq.sum(axis=1, dtype=np.float64)

                count_per_component += int(end - start)
                total_values += int(chunk.size)
                global_sum += float(chunk64.sum(dtype=np.float64))
                global_sum_sq += float(chunk_sq.sum(dtype=np.float64))
                global_abs_sum += float(chunk_abs.sum(dtype=np.float64))
                global_min = min(global_min, float(chunk.min()))
                global_max = max(global_max, float(chunk.max()))

    if component_sum is None or component_sum_sq is None or count_per_component <= 0 or total_values <= 0:
        raise RuntimeError("Unable to compute latent statistics from provided files.")

    component_mean = component_sum / float(count_per_component)
    component_var = component_sum_sq / float(count_per_component) - np.square(component_mean)
    component_std = np.sqrt(np.maximum(component_var, 0.0))

    std_epsilon = 1e-6
    component_std_clamped = np.maximum(component_std, std_epsilon)

    global_mean = global_sum / float(total_values)
    global_var = global_sum_sq / float(total_values) - global_mean * global_mean
    global_std = float(np.sqrt(max(global_var, 0.0)))
    global_rms = float(np.sqrt(max(global_sum_sq / float(total_values), 0.0)))

    return {
        "latent_stats_version": 2,
        "normalization_scheme": "component_chw_shared_time",
        "num_files": len(latent_files),
        "latent_shape": {
            "channels": channels,
            "height": height,
            "width": width,
        },
        "count_per_component": int(count_per_component),
        "std_epsilon": std_epsilon,
        "component_mean": component_mean.tolist(),
        "component_std": component_std.tolist(),
        "component_std_clamped": component_std_clamped.tolist(),
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

        console.print(f"[stats] Computing latent component stats over {len(latent_npz_files)} files...")
        stats_start = time.time()
        latent_stats = compute_latent_component_stats(latent_npz_files)
        stats_path = output_dir / "latent_stats.json"
        with stats_path.open("w") as f:
            json.dump(latent_stats, f, indent=2)
        latent_meta["latent_stats_file"] = stats_path.name
        latent_meta["latent_stats_path"] = serialize_project_path(stats_path, project_root=PROJECT_ROOT)
        latent_meta["latent_stats_version"] = int(latent_stats["latent_stats_version"])
        latent_meta["latent_normalization_scheme"] = str(latent_stats["normalization_scheme"])
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
    if vae_summary["onehot_conv"]:
        if args.onehot_dtype != "float32":
            console.print("[encode] onehot_conv enabled; --onehot-dtype is ignored.")
    elif not autocast_enabled and onehot_dtype != torch.float32:
        console.print("[encode] Non-autocast encoding requires float32 inputs; overriding --onehot-dtype to float32.")
        onehot_dtype = torch.float32

    console.print(
        f"[vae] latent_channels={vae_summary['latent_channels']} "
        f"temporal_downsample={vae_summary['temporal_downsample']} "
        f"onehot_conv={vae_summary['onehot_conv']}"
    )
    if vae_summary["temporal_downsample"] > 0 and args.batch_frames % 2 != 0:
        raise ValueError("--batch-frames must be even when the VAE uses temporal downsampling")
    if vae_summary["temporal_downsample"] > 0 and args.min_batch_frames % 2 != 0:
        raise ValueError("--min-batch-frames must be even when the VAE uses temporal downsampling")

    if device.type == "cuda":
        console.print(
            "[encode] CUDA settings: "
            f"batch_frames={args.batch_frames}, min_batch_frames={args.min_batch_frames}, "
            f"autocast={autocast_enabled}, auto_batch_shrink={not args.no_auto_batch_shrink}"
        )

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
        "temporal_downsample": int(vae_summary["temporal_downsample"]),
        "latent_temporal_stride": int(2 ** vae_summary["temporal_downsample"]),
        "action_downsample": "last_of_pair" if vae_summary["temporal_downsample"] > 0 else "none",
        "latent_stats_version": 2,
        "latent_normalization_scheme": "component_chw_shared_time",
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
    source_frame_height: int | None = None
    source_frame_width: int | None = None

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

                file_height = int(frames.shape[1])
                file_width = int(frames.shape[2])
                if source_frame_height is None:
                    source_frame_height = file_height
                    source_frame_width = file_width
                elif (file_height, file_width) != (source_frame_height, source_frame_width):
                    raise ValueError(
                        f"{npz_path.name} has frame shape {(file_height, file_width)}; expected "
                        f"{(source_frame_height, source_frame_width)}"
                    )

                frames = resize_palette_frames(
                    frames,
                    target_height=args.frame_size,
                    target_width=args.frame_size,
                )

                num_frames = frames.shape[0]
                progress.update(task, status=f"{npz_path.name} ({num_frames} frames)")

                latents = encode_file(
                    vae,
                    frames,
                    num_colors=num_colors,
                    batch_frames=args.batch_frames,
                    min_batch_frames=args.min_batch_frames,
                    onehot_dtype=onehot_dtype,
                    device=device,
                    autocast_ctx=make_autocast_ctx(),
                    auto_batch_shrink=not args.no_auto_batch_shrink,
                    use_onehot_conv=bool(vae_summary["onehot_conv"]),
                    require_even_batch_frames=vae_summary["temporal_downsample"] > 0,
                )
                latent_actions = downsample_actions(
                    actions,
                    temporal_downsample=vae_summary["temporal_downsample"],
                )

                out_path = output_dir / npz_path.name
                np.savez_compressed(
                    out_path,
                    latents=latents,
                    actions=latent_actions,
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

    console.print(f"[stats] Computing latent component stats over {len(latent_npz_files)} files...")
    stats_start = time.time()
    latent_stats = compute_latent_component_stats(latent_npz_files)
    stats_path = output_dir / "latent_stats.json"
    with stats_path.open("w") as f:
        json.dump(latent_stats, f, indent=2)
    stats_elapsed = time.time() - stats_start
    console.print(f"[stats] Wrote {stats_path} in {stats_elapsed:.1f}s")

    latent_meta["latent_stats_file"] = stats_path.name
    latent_meta["latent_stats_path"] = serialize_project_path(stats_path, project_root=PROJECT_ROOT)
    latent_meta["latent_stats_version"] = int(latent_stats["latent_stats_version"])
    latent_meta["latent_normalization_scheme"] = str(latent_stats["normalization_scheme"])
    latent_meta["frame_height"] = int(args.frame_size)
    latent_meta["frame_width"] = int(args.frame_size)
    latent_meta["source_frame_height"] = int(source_frame_height or latent_meta.get("source_frame_height", args.frame_size))
    latent_meta["source_frame_width"] = int(source_frame_width or latent_meta.get("source_frame_width", args.frame_size))
    latent_meta["latent_channels"] = int(latent_stats["latent_shape"]["channels"])
    latent_meta["latent_height"] = int(latent_stats["latent_shape"]["height"])
    latent_meta["latent_width"] = int(latent_stats["latent_shape"]["width"])
    with latent_config_path.open("w") as f:
        json.dump(latent_meta, f, indent=2)

    # ── Write dataset index for fast loading ─────────────────────────
    from src.data.dataset_index import build_latent_index, write_index

    ds_index = build_latent_index(output_dir)
    idx_path = write_index(output_dir, ds_index)
    console.print(f"Wrote {idx_path}")

    console.print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
