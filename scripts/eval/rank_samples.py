#!/usr/bin/env python3
"""Rank normalized video clips by reconstruction quality.

This script currently supports checkpoints produced by
`scripts/train_video_vae.py` (continuous video VAE).

It evaluates reconstruction quality for each sample and writes:
- a JSON ranking report (best to worst), and
- optional best/worst preview PNGs.

Usage:
    python scripts/rank_samples.py checkpoints/video_vae_run/video_vae_latest.pt
    python scripts/rank_samples.py checkpoints/video_vae_run/video_vae_latest.pt --sample-fraction 0.1 --top-k 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.video_vae import VideoVAE
from src.data.normalized_dataset import NormalizedSequenceDataset, load_palette_tensor
from src.training.palette_video_vae_training import (
    frames_to_one_hot,
    save_video_preview,
    split_context_targets,
)


@dataclass
class SampleResult:
    dataset_idx: int
    loss: float
    pixel_accuracy: float
    file: str
    t_start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank dataset samples by reconstruction quality")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to a video_vae_*.pt checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: read from config.json next to checkpoint)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: sample_rankings.json next to checkpoint)",
    )
    parser.add_argument(
        "--save-images",
        type=int,
        default=10,
        help="Save previews for N best and N worst samples (0 to disable)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Print top-K best and worst samples",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Evaluate a random fraction of the dataset (0 < f <= 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--onehot-dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Override one-hot dtype (default: config onehot_dtype or float32)",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if not (0.0 < args.sample_fraction <= 1.0):
        parser.error("--sample-fraction must be in (0, 1]")
    if args.save_images < 0:
        parser.error("--save-images must be >= 0")
    if args.top_k <= 0:
        parser.error("--top-k must be > 0")

    return args


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[dtype_name]


def _load_run_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path.resolve().parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {config_path}")
    with config_path.open() as handle:
        return json.load(handle)


def _build_model_from_config(config: dict, num_colors: int, device: torch.device) -> VideoVAE:
    model_name = config.get("model_name", "")
    model_cfg = dict(config.get("model", {}))
    normalized_model_name = model_name
    if model_name.endswith("video_vae") and model_name != "video_vae" and "_" in model_name:
        normalized_model_name = model_name.split("_", maxsplit=1)[-1]

    if model_name and normalized_model_name != "video_vae":
        raise ValueError(
            "Unsupported model_name in config: "
            f"{model_name!r}. This script currently supports only 'video_vae'."
        )

    model = VideoVAE(
        num_colors=num_colors,
        base_channels=int(model_cfg.get("base_channels", config.get("base_channels", 64))),
        latent_channels=int(model_cfg.get("latent_channels", config.get("latent_channels", 64))),
        temporal_downsample=int(model_cfg.get("temporal_downsample", config.get("temporal_downsample", 0))),
        dropout=float(model_cfg.get("dropout", config.get("dropout", 0.0))),
        onehot_conv=bool(model_cfg.get("onehot_conv", config.get("onehot_conv", False))),
        global_bottleneck_attn=bool(
            model_cfg.get("global_bottleneck_attn", config.get("global_bottleneck_attn", False))
        ),
        global_bottleneck_attn_heads=int(
            model_cfg.get("global_bottleneck_attn_heads", config.get("global_bottleneck_attn_heads", 8))
        ),
    ).to(device)
    model.eval()
    return model


def _extract_model_state(raw_checkpoint: object) -> tuple[dict[str, torch.Tensor], int | str]:
    if isinstance(raw_checkpoint, dict) and "model" in raw_checkpoint:
        step = raw_checkpoint.get("step", "?")
        return raw_checkpoint["model"], step
    if isinstance(raw_checkpoint, dict):
        return raw_checkpoint, "?"
    raise TypeError("Checkpoint format is not a state-dict dictionary")


def _build_eval_indices(total: int, fraction: float, seed: int) -> list[int] | None:
    if fraction >= 1.0:
        return None
    count = max(1, int(total * fraction))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=generator)[:count].tolist()


def _score_batch(
    *,
    model: VideoVAE,
    frames: torch.Tensor,
    num_colors: int,
    context_frames: int,
    onehot_dtype: torch.dtype,
    onehot_buffer: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = frames_to_one_hot(frames, num_colors=num_colors, dtype=onehot_dtype, out=onehot_buffer)
    outputs = model(inputs, sample_posterior=False)
    logits, targets = split_context_targets(outputs.logits, frames, context_frames)

    per_elem_loss = F.cross_entropy(logits.float(), targets, reduction="none")
    sample_loss = per_elem_loss.mean(dim=(1, 2, 3))

    predictions = logits.argmax(dim=1)
    sample_acc = (predictions == targets).float().mean(dim=(1, 2, 3))

    return sample_loss, sample_acc, inputs


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _load_run_config(checkpoint_path)
    data_dir = Path(args.data_dir or config.get("data_dir", "data/normalized")).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    palette = load_palette_tensor(data_dir).to(device)
    num_colors = int(palette.shape[0])

    model = _build_model_from_config(config, num_colors=num_colors, device=device)
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict, step = _extract_model_state(raw_checkpoint)
    model.load_state_dict(state_dict)
    del raw_checkpoint, state_dict

    data_cfg = dict(config.get("data", {}))
    training_cfg = dict(config.get("training", {}))
    clip_frames = int(data_cfg.get("clip_frames", config.get("clip_frames", 16)))
    context_frames = int(data_cfg.get("context_frames", config.get("context_frames", 0)))
    total_clip_frames = clip_frames + context_frames

    onehot_dtype_name = args.onehot_dtype or str(training_cfg.get("onehot_dtype", config.get("onehot_dtype", "float32")))
    onehot_dtype = _resolve_dtype(onehot_dtype_name)
    frame_size = int(data_cfg.get("frame_height", 224))

    dataset = NormalizedSequenceDataset(
        data_dir=data_dir,
        clip_frames=total_clip_frames,
        frame_size=frame_size,
        num_workers=args.num_workers,
    )
    eval_indices = _build_eval_indices(len(dataset), args.sample_fraction, args.seed)
    eval_dataset = Subset(dataset, eval_indices) if eval_indices is not None else dataset

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Step: {step}")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)} clips")
    if eval_indices is None:
        print("Scoring all clips")
    else:
        print(f"Scoring {len(eval_indices)} clips ({args.sample_fraction:.1%} sample)")

    results: list[SampleResult] = []
    onehot_buffer: torch.Tensor | None = None
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring"):
            frames = batch["frames"].to(device, non_blocking=True).long()
            file_idx = batch["file_idx"].tolist()
            start_idx = batch["start_idx"].tolist()

            sample_loss, sample_acc, onehot_buffer = _score_batch(
                model=model,
                frames=frames,
                num_colors=num_colors,
                context_frames=context_frames,
                onehot_dtype=onehot_dtype,
                onehot_buffer=onehot_buffer,
            )

            batch_size = frames.shape[0]
            for i in range(batch_size):
                dataset_idx = sample_offset + i if eval_indices is None else eval_indices[sample_offset + i]
                data_file = str(dataset.data_files[int(file_idx[i])])
                results.append(
                    SampleResult(
                        dataset_idx=int(dataset_idx),
                        loss=float(sample_loss[i].item()),
                        pixel_accuracy=float(sample_acc[i].item()),
                        file=data_file,
                        t_start=int(start_idx[i]),
                    )
                )
            sample_offset += batch_size

    if not results:
        raise RuntimeError("No samples were scored")

    results.sort(key=lambda item: item.loss)

    losses = np.array([item.loss for item in results], dtype=np.float64)
    accuracies = np.array([item.pixel_accuracy for item in results], dtype=np.float64)

    print("\n" + "=" * 72)
    print(f"Scored clips: {len(results)}")
    print(
        "Loss  - "
        f"mean: {losses.mean():.6f}  median: {np.median(losses):.6f}  std: {losses.std():.6f}"
    )
    print(
        "Acc   - "
        f"mean: {accuracies.mean():.4f}  median: {np.median(accuracies):.4f}"
    )
    print("=" * 72)

    k = min(args.top_k, len(results))
    print(f"\nTop {k} best (lowest loss):")
    for item in results[:k]:
        print(
            f"  loss={item.loss:.6f}  acc={item.pixel_accuracy:.4f}  "
            f"file={os.path.basename(item.file)}  t={item.t_start}"
        )

    print(f"\nTop {k} worst (highest loss):")
    for item in results[-k:][::-1]:
        print(
            f"  loss={item.loss:.6f}  acc={item.pixel_accuracy:.4f}  "
            f"file={os.path.basename(item.file)}  t={item.t_start}"
        )

    output_path = Path(args.output) if args.output else checkpoint_path.resolve().parent / "sample_rankings.json"
    output_payload = {
        "checkpoint": str(checkpoint_path),
        "step": step,
        "num_scored": len(results),
        "model_name": "video_vae",
        "clip_frames": clip_frames,
        "context_frames": context_frames,
        "total_clip_frames": total_clip_frames,
        "data_dir": str(data_dir),
        "mean_loss": float(losses.mean()),
        "median_loss": float(np.median(losses)),
        "std_loss": float(losses.std()),
        "mean_pixel_accuracy": float(accuracies.mean()),
        "samples": [asdict(item) for item in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output_payload, handle, indent=2)
    print(f"\nSaved ranking JSON to {output_path}")

    if args.save_images <= 0:
        return

    preview_dir = checkpoint_path.resolve().parent / "sample_analysis"
    preview_dir.mkdir(parents=True, exist_ok=True)
    n_save = min(args.save_images, len(results))

    def save_preview(item: SampleResult, label: str) -> None:
        sample = dataset[item.dataset_idx]
        frames = sample["frames"].unsqueeze(0).to(device)
        inputs = frames_to_one_hot(frames.long(), num_colors=num_colors, dtype=onehot_dtype)
        with torch.no_grad():
            outputs = model(inputs, sample_posterior=False)
        logits, targets = split_context_targets(outputs.logits, frames.long(), context_frames)

        filename = (
            f"{label}_loss{item.loss:.4f}_acc{item.pixel_accuracy:.4f}_"
            f"t{item.t_start}.png"
        )
        save_video_preview(
            preview_dir / filename,
            targets.detach().cpu(),
            logits.detach().cpu(),
            palette.detach().cpu(),
            max_frames=targets.shape[1],
        )

    print(f"Saving {n_save} best and {n_save} worst previews to {preview_dir}")
    for idx, item in enumerate(tqdm(results[:n_save], desc="Best previews")):
        save_preview(item, f"best_{idx:03d}")

    for idx, item in enumerate(tqdm(results[-n_save:][::-1], desc="Worst previews")):
        save_preview(item, f"worst_{idx:03d}")


if __name__ == "__main__":
    main()
