#!/usr/bin/env python3
"""Train the Video VAE (symmetric 7x7-latent architecture)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from rich.console import Console
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.gan_discriminator import build_palette_discriminator, count_trainable_parameters
from src.data.video_frames import SUPPORTED_FRAME_SIZES
from src.training.gan_training import LeCAMEMA, hinge_discriminator_loss, hinge_generator_loss, set_requires_grad
from src.training.losses import focal_cross_entropy, softened_inverse_frequency_weights, spatial_weight_map, temporal_change_weight
from src.models.video_vae import VideoVAE
from src.models.auxiliary_heads import (
    NextFramePredictor,
    RAMAlignmentHead,
    load_frozen_ram_vae,
    temporal_smoothness_loss,
)
from src.models.ram_vae import RAMVAE
from src.data.normalized_dataset import NormalizedSequenceDataset, load_palette_tensor
from src.training.palette_video_vae_training import (
    apply_palette_index_augmentation,
    evaluate_video_vae,
    frames_to_one_hot,
    save_video_preview,
)
from src.system_info import collect_system_info, print_system_info
from src.training.mlflow_utils import MLflowRun, parse_mlflow_tags
from src.training.trainer_common import (
    add_mlflow_args,
    add_resume_scheduler_args,
    build_trainer_config,
    configure_resume_scheduler,
    configure_cuda_runtime,
    format_bytes,
    gpu_stats,
    is_periodic_event_due,
    make_output_dir,
    preview_path,
    seed_everything,
    should_log_step,
    validate_resume_scheduler_args,
)
from src.training.training_utils import (
    ThroughputTracker,
    advance_progress,
    build_eval_loader,
    build_progress,
    build_replacement_train_loader,
    create_accelerator_runtime,
    get_model_state_dict,
    infinite_batches,
    load_model_state_dict,
    save_json,
    save_metrics_json,
    split_train_eval_dataset,
)


console = Console()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Video VAE (symmetric 7x7-latent architecture).")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        choices=SUPPORTED_FRAME_SIZES,
        help="Output frame size used for training and eval previews.",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Linear warmup from 0 to --lr over this many steps (default: 2000)")
    parser.add_argument("--kl-weight", type=float, default=5e-4, help="Weight for KL divergence loss term.")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for eval only (defaults to --batch-size).",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <= 0 to disable clipping.",
    )
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.005,
        help="Dropout probability used in model residual blocks.",
    )
    parser.add_argument(
        "--temporal-downsample",
        type=int,
        default=1,
        choices=[0, 1],
        help="Number of temporal downsamples in the VAE bottleneck (0 or 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF16 matmul precision.",
    )
    parser.add_argument(
        "--onehot-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype for one-hot input tensors (reduces memory vs float32 when using float16/bfloat16).",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode passed to Accelerator.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the Video VAE with torch.compile.",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.3,
                        help="Focal loss gamma (0 = standard cross-entropy, 2 = typical focal)")
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable class-weighted reconstruction loss using distribution JSON from --data-dir.",
    )
    parser.add_argument(
        "--class-weights-file",
        type=str,
        default="palette_distribution.json",
        help="Distribution JSON filename in --data-dir containing class counts/probabilities.",
    )
    parser.add_argument(
        "--class-weight-soften",
        type=float,
        default=0.1,
        help="Softening exponent for inverse-frequency weights (0=uniform, 1=full inverse-frequency).",
    )
    parser.add_argument(
        "--class-weight-radius",
        type=float,
        default=3.0,
        help="Spatial LogSumExp pooling radius for per-pixel class weights (0=disabled).",
    )
    parser.add_argument(
        "--class-weight-hardness",
        type=float,
        default=20.0,
        help="LogSumExp hardness β for spatial class-weight pooling (higher=closer to local max).",
    )
    parser.add_argument(
        "--class-weight-temporal-ema",
        type=float,
        default=0.9,
        help="Causal max-decay persistence for spatial weight map (0=disabled, 0.9=strong carry-over).",
    )
    parser.add_argument(
        "--temporal-change-boost",
        type=float,
        default=.1,
        help="Additive weight boost for pixels that changed from the previous frame (0=disabled).",
    )
    parser.add_argument(
        "--palette-aug-sample-prob",
        type=float,
        default=0.5,
        help="Per-sample probability of applying any palette augmentation at all.",
    )
    parser.add_argument(
        "--palette-aug-prob",
        type=float,
        default=0.4,
        help="Per-pixel probability of replacing an encoder input palette index once a sample is selected for augmentation.",
    )
    parser.add_argument(
        "--palette-aug-file",
        type=str,
        default="palette_distribution.json",
        help="Distribution JSON filename in --data-dir used to sample palette replacements for input corruption.",
    )
    parser.add_argument(
        "--use-gan",
        action="store_true",
        help="Enable adversarial training with a compact 3D discriminator.",
    )
    parser.add_argument(
        "--gan-weight",
        type=float,
        default=0.1,
        help="Generator adversarial loss weight.",
    )
    parser.add_argument(
        "--gan-lr",
        type=float,
        default=2e-4,
        help="Discriminator learning rate.",
    )
    parser.add_argument(
        "--gan-start-step",
        type=int,
        default=100000,
        help="Global step to start GAN updates.",
    )
    parser.add_argument(
        "--gan-target-size",
        type=str,
        default="~5m",
        choices=["~10m", "~5m"],
        help="Discriminator size preset.",
    )
    parser.add_argument(
        "--use-lecam",
        action="store_true",
        help="Enable LeCAM regularization for discriminator stabilization.",
    )
    parser.add_argument(
        "--lecam-weight",
        type=float,
        default=0.1,
        help="LeCAM regularization weight added to discriminator loss.",
    )
    parser.add_argument(
        "--lecam-decay",
        type=float,
        default=0.999,
        help="EMA decay for LeCAM running real/fake logits.",
    )
    # --- Auxiliary losses ---
    parser.add_argument(
        "--next-frame-weight",
        type=float,
        default=0.1,
        help="Weight for next-frame latent prediction loss (0=disabled).",
    )
    parser.add_argument(
        "--temporal-smooth-weight",
        type=float,
        default=0.01,
        help="Weight for temporal smoothness loss (0=disabled).",
    )
    parser.add_argument(
        "--ram-align-weight",
        type=float,
        default=0,
        help="Weight for RAM alignment loss (0=disabled).",
    )
    parser.add_argument(
        "--ram-vae-checkpoint",
        type=str,
        default=None,
        help="Path to frozen RAM VAE checkpoint (required when --ram-align-weight > 0).",
    )
    parser.add_argument("--num-actions", type=int, default=42,
                        help="Number of discrete actions for next-frame predictor embedding.")
    parser.add_argument("--action-embed-dim", type=int, default=16,
                        help="Action embedding dimension for next-frame predictor.")
    parser.add_argument("--next-frame-hidden-dim", type=int, default=128,
                        help="Hidden dim for next-frame predictor MLP.")

    parser.add_argument("--resume-from", type=str, default=None)
    add_resume_scheduler_args(parser, default_tail_final_lr_scale=0.25)
    add_mlflow_args(parser)
    args = parser.parse_args(argv)
    try:
        args.mlflow_tags = parse_mlflow_tags(args.mlflow_tag)
    except ValueError as exc:
        parser.error(str(exc))
    if args.clip_frames <= 0:
        parser.error("--clip-frames must be > 0")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    if args.max_grad_norm < 0:
        parser.error("--max-grad-norm must be >= 0")
    if not (0.0 <= args.palette_aug_sample_prob <= 1.0):
        parser.error("--palette-aug-sample-prob must be in [0, 1]")
    if not (0.0 <= args.palette_aug_prob <= 1.0):
        parser.error("--palette-aug-prob must be in [0, 1]")
    if not (0.0 <= args.dropout < 1.0):
        parser.error("--dropout must be in [0, 1)")
    if args.ram_align_weight > 0 and args.ram_vae_checkpoint is None:
        parser.error("--ram-vae-checkpoint is required when --ram-align-weight > 0")
    validate_resume_scheduler_args(parser, args)
    return args


def load_class_weights(
    data_dir: str,
    *,
    num_classes: int,
    filename: str,
    soften: float,
    device: torch.device,
) -> torch.Tensor:
    dist_path = Path(data_dir) / filename
    if not dist_path.is_file():
        raise FileNotFoundError(
            f"Class weight distribution file not found: {dist_path}. "
            "Run scripts/normalize.py first or disable --use-class-weights."
        )

    with dist_path.open() as handle:
        dist = json.load(handle)

    counts_data = dist.get("counts")
    probs_data = dist.get("probabilities")
    if counts_data is not None:
        counts = torch.tensor(counts_data, dtype=torch.float32)
    elif probs_data is not None:
        counts = torch.tensor(probs_data, dtype=torch.float32)
    else:
        raise ValueError(
            f"{dist_path} must contain either 'counts' or 'probabilities'"
        )

    if counts.ndim != 1 or counts.numel() != num_classes:
        raise ValueError(
            f"{dist_path} has {counts.numel()} classes but model expects {num_classes}"
        )

    weights = softened_inverse_frequency_weights(counts, soften=soften)
    return weights.to(device)


def load_palette_probabilities(
    data_dir: str,
    *,
    num_classes: int,
    filename: str,
    device: torch.device,
) -> torch.Tensor:
    dist_path = Path(data_dir) / filename
    if not dist_path.is_file():
        raise FileNotFoundError(
            f"Palette augmentation distribution file not found: {dist_path}. "
            "Run scripts/normalize.py first or disable --palette-aug-prob."
        )

    with dist_path.open() as handle:
        dist = json.load(handle)

    probs_data = dist.get("probabilities")
    counts_data = dist.get("counts")
    if probs_data is not None:
        probs = torch.tensor(probs_data, dtype=torch.float32)
    elif counts_data is not None:
        probs = torch.tensor(counts_data, dtype=torch.float32)
    else:
        raise ValueError(
            f"{dist_path} must contain either 'counts' or 'probabilities'"
        )

    if probs.ndim != 1 or probs.numel() != num_classes:
        raise ValueError(
            f"{dist_path} has {probs.numel()} classes but model expects {num_classes}"
        )

    probs = probs / probs.sum().clamp_min(1e-12)
    return probs.to(device)


def save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scheduler_metadata: dict[str, int | float | str],
    step: int,
    best_eval: float,
    metrics: list[dict[str, float]],
    accelerator: Accelerator | None = None,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    lecam_ema: LeCAMEMA | None = None,
    next_frame_predictor: torch.nn.Module | None = None,
    ram_align_head: torch.nn.Module | None = None,
) -> None:
    model_state = get_model_state_dict(model, accelerator=accelerator)
    state = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scheduler_metadata": scheduler_metadata,
        "step": step,
        "best_eval": best_eval,
        "metrics": metrics,
    }
    if discriminator is not None and discriminator_optimizer is not None:
        state["discriminator"] = get_model_state_dict(discriminator, accelerator=accelerator)
        state["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        if lecam_ema is not None:
            state["lecam_ema"] = lecam_ema.state_dict()
    if next_frame_predictor is not None:
        state["next_frame_predictor"] = next_frame_predictor.state_dict()
    if ram_align_head is not None:
        state["ram_align_head"] = ram_align_head.state_dict()
    torch.save(state, path)


def _optimizer_group_sizes(state: object) -> list[int]:
    if not isinstance(state, dict):
        return []
    param_groups = state.get("param_groups")
    if not isinstance(param_groups, list):
        return []
    sizes: list[int] = []
    for group in param_groups:
        if not isinstance(group, dict):
            sizes.append(0)
            continue
        params = group.get("params")
        sizes.append(len(params) if isinstance(params, list) else 0)
    return sizes


def _load_resume_training_config(resume_from: str) -> tuple[dict[str, object] | None, Path | None]:
    cfg_path = Path(resume_from).resolve().parent / "config.json"
    if not cfg_path.is_file():
        return None, None
    try:
        with cfg_path.open() as handle:
            cfg = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None, cfg_path
    training_cfg = cfg.get("training")
    if isinstance(training_cfg, dict):
        return training_cfg, cfg_path
    return None, cfg_path


def _format_resume_optimizer_mismatch(
    *,
    error: Exception,
    resume_from: str,
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, object],
    args: argparse.Namespace,
    next_frame_predictor: torch.nn.Module | None,
    ram_align_head: torch.nn.Module | None,
) -> str:
    current_sizes = _optimizer_group_sizes(optimizer.state_dict())
    checkpoint_sizes = _optimizer_group_sizes(checkpoint.get("optimizer"))
    lines: list[str] = [
        "[resume] Failed to load optimizer state.",
        f"Reason: {error}",
        f"Checkpoint: {resume_from}",
        f"Current optimizer param_groups={len(current_sizes)} sizes={current_sizes}",
        f"Checkpoint optimizer param_groups={len(checkpoint_sizes)} sizes={checkpoint_sizes}",
        "This usually means the trainable parameter set changed since the checkpoint was created.",
    ]

    current_has_nfp = next_frame_predictor is not None
    current_has_ram = ram_align_head is not None
    checkpoint_has_nfp = "next_frame_predictor" in checkpoint
    checkpoint_has_ram = "ram_align_head" in checkpoint
    checkpoint_has_gan = "discriminator_optimizer" in checkpoint
    lines.append(
        "Current features: "
        f"use_gan={bool(args.use_gan)}, "
        f"next_frame_head={current_has_nfp}, "
        f"ram_align_head={current_has_ram}, "
        f"base_channels={int(args.base_channels)}, "
        f"latent_channels={int(args.latent_channels)}, "
        f"temporal_downsample={int(args.temporal_downsample)}"
    )
    lines.append(
        "Checkpoint state keys: "
        f"has_discriminator_optimizer={checkpoint_has_gan}, "
        f"has_next_frame_predictor={checkpoint_has_nfp}, "
        f"has_ram_align_head={checkpoint_has_ram}"
    )

    training_cfg, cfg_path = _load_resume_training_config(resume_from)
    if training_cfg is not None:
        tracked_keys = [
            "use_gan",
            "next_frame_weight",
            "ram_align_weight",
            "base_channels",
            "latent_channels",
            "temporal_downsample",
            "num_actions",
            "action_embed_dim",
            "next_frame_hidden_dim",
        ]
        mismatch_notes: list[str] = []
        for key in tracked_keys:
            if not hasattr(args, key):
                continue
            current_value = getattr(args, key)
            checkpoint_value = training_cfg.get(key)
            if checkpoint_value is None:
                continue
            if current_value != checkpoint_value:
                mismatch_notes.append(f"{key}: current={current_value} checkpoint={checkpoint_value}")
        if mismatch_notes:
            lines.append(f"Config mismatches from {cfg_path}:")
            lines.extend(mismatch_notes)
        else:
            lines.append(f"Config file checked: {cfg_path} (no tracked mismatches found).")
    elif cfg_path is not None:
        lines.append(f"Config file could not be parsed for comparison: {cfg_path}")

    lines.append("If you changed architecture/aux flags intentionally, start a new run without --resume-from.")
    lines.append("If you intended to resume exactly, reuse the original flags from the checkpoint config.")
    return "\n".join(lines)


def _format_resume_discriminator_optimizer_mismatch(
    *,
    error: Exception,
    resume_from: str,
    discriminator_optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, object],
    args: argparse.Namespace,
) -> str:
    current_sizes = _optimizer_group_sizes(discriminator_optimizer.state_dict())
    checkpoint_sizes = _optimizer_group_sizes(checkpoint.get("discriminator_optimizer"))
    lines: list[str] = [
        "[resume] Failed to load discriminator optimizer state.",
        f"Reason: {error}",
        f"Checkpoint: {resume_from}",
        f"Current discriminator optimizer param_groups={len(current_sizes)} sizes={current_sizes}",
        f"Checkpoint discriminator optimizer param_groups={len(checkpoint_sizes)} sizes={checkpoint_sizes}",
        "This usually means discriminator architecture or GAN settings changed.",
        f"Current GAN flags: use_gan={bool(args.use_gan)}, gan_target_size={args.gan_target_size}, use_lecam={bool(args.use_lecam)}",
        "Use the same GAN flags as the original run, or restart without --resume-from.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.use_lecam and not args.use_gan:
        raise SystemExit("--use-lecam requires --use-gan")
    if not (0.0 < args.lecam_decay < 1.0):
        raise SystemExit("--lecam-decay must be in (0, 1)")
    seed_everything(args.seed)

    output_dir = make_output_dir(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        run_name=args.run_name,
        default_prefix="video_vae",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = create_accelerator_runtime(
        output_dir=output_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator = runtime.accelerator
    device = runtime.device
    is_main_process = runtime.is_main_process

    if is_main_process:
        console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        if args.tf16:
            configure_cuda_runtime(matmul_precision="medium")
            if is_main_process:
                console.print("[tf16] TF16 matmul precision enabled")
        else:
            configure_cuda_runtime(matmul_precision="high")
            if is_main_process:
                console.print("[tf32] TF32 matmul precision enabled")

    onehot_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    onehot_dtype = onehot_dtype_map[args.onehot_dtype]
    if args.mixed_precision != "no" and is_main_process:
        console.print(f"[mixed-precision] Enabled ({args.mixed_precision})")

    system_info = collect_system_info()
    if is_main_process:
        print_system_info(system_info)

    eval_batch_size = args.eval_batch_size or args.batch_size
    use_aux_actions = args.next_frame_weight > 0
    use_aux_ram = args.ram_align_weight > 0

    def build_data_for_frame_size(
        frame_size: int,
        *,
        stage_label: str,
    ) -> tuple[
        NormalizedSequenceDataset,
        torch.utils.data.Dataset,
        torch.utils.data.Dataset | None,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader | None,
        int,
        int,
        int,
        int,
    ]:
        if is_main_process:
            console.print(
                f"[dataset:{stage_label}] Building sequence index at {frame_size}x{frame_size} "
                "(can take a few minutes on large runs)..."
            )
        with accelerator.main_process_first():
            stage_dataset = NormalizedSequenceDataset(
                data_dir=args.data_dir,
                clip_frames=args.clip_frames,
                frame_size=frame_size,
                include_actions=use_aux_actions,
                include_ram=use_aux_ram,
                num_workers=args.num_workers,
                system_info=system_info,
            )
        if is_main_process:
            console.print(f"[dataset:{stage_label}] Index build complete.")
        if len(stage_dataset) == 0:
            raise RuntimeError("No training samples were found.")

        stage_frame_height = stage_dataset.frame_height
        stage_frame_width = stage_dataset.frame_width
        stage_source_frame_height = stage_dataset.source_frame_height
        stage_source_frame_width = stage_dataset.source_frame_width
        if stage_frame_height is None or stage_frame_width is None:
            raise RuntimeError("Could not determine training frame shape from dataset")
        if is_main_process:
            console.print(
                f"[dataset:{stage_label}] Found {len(stage_dataset)} sequence segments of {args.clip_frames} frames."
            )
            console.print(
                f"[dataset:{stage_label}] Dataset: {stage_dataset.num_files} files, {len(stage_dataset):,} samples, "
                f"{format_bytes(stage_dataset.dataset_bytes)} on disk."
            )

        stage_train_dataset, stage_eval_dataset = split_train_eval_dataset(
            stage_dataset,
            eval_samples=args.eval_samples,
            seed=args.seed,
        )
        if stage_eval_dataset is not None and is_main_process:
            console.print(
                f"[dataset:{stage_label}] Eval split: {len(stage_eval_dataset)} eval, "
                f"{len(stage_train_dataset)} train samples"
            )

        stage_train_loader = build_replacement_train_loader(
            stage_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        stage_eval_loader = build_eval_loader(
            stage_eval_dataset,
            batch_size=eval_batch_size,
            num_workers=0,
            pin_memory=False,
        )
        return (
            stage_dataset,
            stage_train_dataset,
            stage_eval_dataset,
            stage_train_loader,
            stage_eval_loader,
            int(stage_frame_height),
            int(stage_frame_width),
            int(stage_source_frame_height or stage_frame_height),
            int(stage_source_frame_width or stage_frame_width),
        )

    target_frame_size = int(args.frame_size)
    current_frame_size = target_frame_size

    (
        dataset,
        train_dataset,
        eval_dataset,
        train_loader,
        eval_loader,
        frame_height,
        frame_width,
        source_frame_height,
        source_frame_width,
    ) = build_data_for_frame_size(target_frame_size, stage_label="train")

    palette_path = Path(args.data_dir) / "palette.json"
    palette = load_palette_tensor(args.data_dir)
    num_colors = palette.shape[0]
    if is_main_process:
        console.print(f"[palette] Loaded {num_colors} colours from {palette_path}")
    class_weight = None
    if args.use_class_weights:
        class_weight = load_class_weights(
            args.data_dir,
            num_classes=num_colors,
            filename=args.class_weights_file,
            soften=args.class_weight_soften,
            device=device,
        )
        if is_main_process:
            console.print(
                "[class-weights] Enabled from "
                f"{Path(args.data_dir) / args.class_weights_file} "
                f"(soften={args.class_weight_soften:.2f}, "
                f"min={class_weight.min().item():.3f}, max={class_weight.max().item():.3f})"
            )
            if args.class_weight_radius >= 0.5:
                console.print(
                    f"[class-weights] Spatial LogSumExp pooling: "
                    f"radius={args.class_weight_radius:.1f}, hardness={args.class_weight_hardness:.1f}"
                    + (f", temporal_ema={args.class_weight_temporal_ema:.2f}" if args.class_weight_temporal_ema > 0 else "")
                )
    if args.temporal_change_boost > 0 and is_main_process:
        console.print(f"[temporal-change] Boost={args.temporal_change_boost:.2f}")
    palette_aug_probs = None
    if args.palette_aug_sample_prob > 0.0 and args.palette_aug_prob > 0.0:
        palette_aug_probs = load_palette_probabilities(
            args.data_dir,
            num_classes=num_colors,
            filename=args.palette_aug_file,
            device=device,
        )
        if is_main_process:
            console.print(
                "[palette-aug] Enabled input corruption from "
                f"{Path(args.data_dir) / args.palette_aug_file} "
                f"(sample_p={args.palette_aug_sample_prob:.3f}, pixel_p={args.palette_aug_prob:.3f}, "
                f"min={palette_aug_probs.min().item():.4f}, max={palette_aug_probs.max().item():.4f})"
            )
    model = VideoVAE(
        num_colors=num_colors,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        temporal_downsample=args.temporal_downsample,
        dropout=args.dropout,
    ).to(device)
    discriminator = None
    discriminator_optimizer = None
    lecam_ema = None
    discriminator_num_parameters = 0
    if args.use_gan:
        discriminator = build_palette_discriminator(
            num_colors,
            target_size=args.gan_target_size,
        ).to(device)
        discriminator_num_parameters = count_trainable_parameters(discriminator)
        if args.use_lecam:
            lecam_ema = LeCAMEMA(decay=args.lecam_decay)
        if is_main_process:
            console.print(
                f"[gan] Enabled discriminator {args.gan_target_size} "
                f"({discriminator_num_parameters:,} params), "
                f"gan_weight={args.gan_weight}, gan_lr={args.gan_lr:.2e}, "
                f"gan_start_step={args.gan_start_step}, "
                f"lecam={'on' if args.use_lecam else 'off'}"
            )

    # --- Auxiliary heads ---
    next_frame_predictor: NextFramePredictor | None = None
    ram_align_head: RAMAlignmentHead | None = None
    frozen_ram_vae: RAMVAE | None = None
    if use_aux_actions:
        next_frame_predictor = NextFramePredictor(
            latent_dim=args.latent_channels,
            num_actions=args.num_actions,
            action_embed_dim=args.action_embed_dim,
            hidden_dim=args.next_frame_hidden_dim,
        ).to(device)
        if is_main_process:
            nfp_params = sum(p.numel() for p in next_frame_predictor.parameters())
            console.print(
                f"[aux] Next-frame predictor enabled ({nfp_params:,} params), "
                f"weight={args.next_frame_weight}"
            )
    if use_aux_ram:
        frozen_ram_vae = load_frozen_ram_vae(
            args.ram_vae_checkpoint, args.data_dir, device=device,
        )
        ram_align_head = RAMAlignmentHead(
            video_latent_dim=args.latent_channels,
            ram_latent_dim=frozen_ram_vae.latent_dim,
        ).to(device)
        if is_main_process:
            rah_params = sum(p.numel() for p in ram_align_head.parameters())
            console.print(
                f"[aux] RAM alignment enabled ({rah_params:,} params), "
                f"weight={args.ram_align_weight}"
            )
    if args.temporal_smooth_weight > 0 and is_main_process:
        console.print(f"[aux] Temporal smoothness enabled, weight={args.temporal_smooth_weight}")

    # Compilation strategy:
    # - Single-GPU: compile here before accelerator.prepare().
    # - Multi-GPU: skip manual compile; use `accelerate launch --dynamo_backend inductor`
    #   which applies torch.compile during accelerator.prepare() in the correct DDP order.
    do_compile = args.compile and accelerator.num_processes == 1
    if args.compile and accelerator.num_processes > 1 and is_main_process:
        console.print(
            "[compile] Multi-GPU: skipping manual torch.compile. "
            "Use `--dynamo_backend inductor` in the accelerate launch command instead."
        )
    if do_compile:
        if is_main_process:
            console.print("[compile] Compiling the model with torch.compile()...")
        model = torch.compile(model)
        if discriminator is not None:
            if is_main_process:
                console.print("[compile] Compiling discriminator with torch.compile()...")
            discriminator = torch.compile(discriminator)
        if is_main_process:
            console.print("[compile] Compilation complete.")

    num_parameters = int(sum(parameter.numel() for parameter in model.parameters()))

    # Collect all trainable params: model + auxiliary heads
    all_params = list(model.parameters())
    if next_frame_predictor is not None:
        all_params += list(next_frame_predictor.parameters())
    if ram_align_head is not None:
        all_params += list(ram_align_head.parameters())
    optimizer = AdamW(all_params, lr=args.lr)
    if discriminator is not None:
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=args.gan_lr)
    warmup_steps = max(int(args.warmup_steps), 0)
    scheduler_min_lr_scale = 0.25

    best_eval = float("inf")
    metrics: list[dict[str, float]] = []

    if args.resume_from is not None:
        with accelerator.main_process_first():
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        load_model_state_dict(model, checkpoint["model"])
        if "optimizer" not in checkpoint:
            raise RuntimeError(
                f"[resume] Checkpoint is missing optimizer state: {args.resume_from}. "
                "Cannot resume training state."
            )
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            raise RuntimeError(
                _format_resume_optimizer_mismatch(
                    error=exc,
                    resume_from=args.resume_from,
                    optimizer=optimizer,
                    checkpoint=checkpoint,
                    args=args,
                    next_frame_predictor=next_frame_predictor,
                    ram_align_head=ram_align_head,
                )
            ) from exc
        if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
            if "discriminator" in checkpoint and "discriminator_optimizer" in checkpoint:
                load_model_state_dict(discriminator, checkpoint["discriminator"])
                try:
                    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
                except ValueError as exc:
                    raise RuntimeError(
                        _format_resume_discriminator_optimizer_mismatch(
                            error=exc,
                            resume_from=args.resume_from,
                            discriminator_optimizer=discriminator_optimizer,
                            checkpoint=checkpoint,
                            args=args,
                        )
                    ) from exc
            else:
                if is_main_process:
                    console.print("[resume] Checkpoint has no discriminator state; GAN starts from scratch.")
            if args.use_lecam and lecam_ema is not None:
                if "lecam_ema" in checkpoint:
                    lecam_ema.load_state_dict(checkpoint["lecam_ema"])
                else:
                    if is_main_process:
                        console.print("[resume] Checkpoint has no LeCAM EMA state; LeCAM EMA resets.")
        if next_frame_predictor is not None and "next_frame_predictor" in checkpoint:
            next_frame_predictor.load_state_dict(checkpoint["next_frame_predictor"])
        if ram_align_head is not None and "ram_align_head" in checkpoint:
            ram_align_head.load_state_dict(checkpoint["ram_align_head"])
        best_eval = float(checkpoint.get("best_eval", float("inf")))
        metrics = list(checkpoint.get("metrics", []))
        scheduler_setup = configure_resume_scheduler(
            optimizer,
            max_steps=args.max_steps,
            warmup_steps=warmup_steps,
            min_lr_scale=scheduler_min_lr_scale,
            checkpoint=checkpoint,
            resume_lr_mode=args.resume_lr_mode,
            resume_extra_steps=args.resume_extra_steps,
            resume_tail_final_lr_scale=args.resume_tail_final_lr_scale,
            restart_base_lr=args.lr,
        )
        scheduler = scheduler_setup.scheduler
        scheduler_metadata = scheduler_setup.scheduler_metadata
        args.max_steps = scheduler_setup.max_steps
        start_step = scheduler_setup.start_step
        if is_main_process:
            console.print(f"[resume] Loaded training state from {args.resume_from} at step {start_step}")
            for message in scheduler_setup.log_messages:
                console.print(message)

        if args.max_steps > 0 and start_step >= args.max_steps:
            if is_main_process:
                console.print(
                    f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do."
                )
            return
    else:
        scheduler_setup = configure_resume_scheduler(
            optimizer,
            max_steps=args.max_steps,
            warmup_steps=warmup_steps,
            min_lr_scale=scheduler_min_lr_scale,
        )
        scheduler = scheduler_setup.scheduler
        scheduler_metadata = scheduler_setup.scheduler_metadata
        args.max_steps = scheduler_setup.max_steps
        start_step = scheduler_setup.start_step
        if is_main_process:
            for message in scheduler_setup.log_messages:
                console.print(message)

    if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
        model, discriminator, optimizer, discriminator_optimizer, train_loader = accelerator.prepare(
            model,
            discriminator,
            optimizer,
            discriminator_optimizer,
            train_loader,
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    train_iter = infinite_batches(train_loader)

    config = build_trainer_config(
        model_name="video_vae",
        args=args,
        device=device,
        mixed_precision=args.mixed_precision,
        num_processes=accelerator.num_processes,
        data={
            "num_colors": int(num_colors),
            "clip_frames": int(args.clip_frames),
            "frame_height": int(frame_height),
            "frame_width": int(frame_width),
            "source_frame_height": int(source_frame_height or frame_height),
            "source_frame_width": int(source_frame_width or frame_width),
        },
        model={
            "num_parameters": int(num_parameters),
            "discriminator_parameters": int(discriminator_num_parameters),
            "base_channels": int(args.base_channels),
            "latent_channels": int(args.latent_channels),
            "temporal_downsample": int(args.temporal_downsample),
            "dropout": float(args.dropout),
            "next_frame_weight": float(args.next_frame_weight),
            "temporal_smooth_weight": float(args.temporal_smooth_weight),
            "ram_align_weight": float(args.ram_align_weight),
        },
    )
    mlflow_run = MLflowRun.create(
        enabled=is_main_process and args.mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name or args.run_name or output_dir.name,
        tracking_uri=args.mlflow_tracking_uri,
        tags={
            "model_name": "video_vae",
            "script": Path(__file__).name,
            "output_dir": str(output_dir),
            **args.mlflow_tags,
        },
    )
    if is_main_process:
        save_json(output_dir / "config.json", config, indent=2)
        mlflow_run.log_params(config)
        mlflow_run.log_artifact(output_dir / "config.json")
        console.print(f"Config saved to {output_dir / 'config.json'}")
        console.print(json.dumps(config, indent=2))

        console.print(f"Training Video VAE on {len(train_dataset)} samples")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Device: {device}")
        console.print(f"Model parameters: {num_parameters:,}")
        dist_type = getattr(accelerator.distributed_type, "value", str(accelerator.distributed_type))
        global_batch = args.batch_size * accelerator.num_processes
        effective_batch = global_batch * args.gradient_accumulation_steps
        console.print(
            f"Distributed runtime: type={dist_type}, processes={accelerator.num_processes}, "
            f"local_batch={args.batch_size}, global_batch={global_batch}, "
            f"accum_steps={args.gradient_accumulation_steps}, effective_batch={effective_batch}"
        )

    start_time = time.time()
    throughput_tracker = ThroughputTracker(window_steps=20)
    bytes_per_sample = None
    use_live_progress = sys.stdout.isatty() and is_main_process

    total_train_steps = max(args.max_steps - start_step, 0)
    with build_progress(use_live=use_live_progress) as progress:
        log_console = progress.console
        train_task = progress.add_task("Training", total=total_train_steps, status="")
        onehot_buffer: torch.Tensor | None = None
        for step in range(start_step, args.max_steps):
            gan_active = args.use_gan and step >= args.gan_start_step
            gan_discr_loss_value = 0.0
            gan_gen_loss_value = 0.0
            gan_real_logit_value = 0.0
            gan_fake_logit_value = 0.0
            gan_lecam_reg_value = 0.0
            next_frame_loss_value = 0.0
            temporal_smooth_loss_value = 0.0
            ram_align_loss_value = 0.0

            # Accumulators for logging (averaged across micro-steps).
            accum_steps = args.gradient_accumulation_steps
            accum_loss_sum = 0.0
            accum_recon_sum = 0.0
            accum_kl_sum = 0.0
            accum_nf_sum = 0.0
            accum_ts_sum = 0.0
            accum_ram_sum = 0.0
            accum_gen_adv_sum = 0.0
            grad_norm = 0.0
            fake_video_detached = None
            discriminator_real_inputs: torch.Tensor | None = None

            for _accum_idx in range(accum_steps):
                batch = next(train_iter)
                frames = batch["frames"].to(device, non_blocking=True)
                input_frames = frames
                if palette_aug_probs is not None:
                    input_frames = apply_palette_index_augmentation(
                        frames,
                        sample_prob=args.palette_aug_sample_prob,
                        replacement_prob=args.palette_aug_prob,
                        replacement_probs=palette_aug_probs,
                    )
                input_frames = frames_to_one_hot(input_frames, num_colors, dtype=onehot_dtype, out=onehot_buffer)
                onehot_buffer = input_frames
                if bytes_per_sample is None:
                    _, time_steps, height, width = frames.shape
                    bytes_per_sample = (
                        num_colors
                        * time_steps
                        * height
                        * width
                        * torch.tensor([], dtype=onehot_dtype).element_size()
                    )

                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        outputs = model(input_frames)
                    target_frames = frames.long()
                    recon_logits = outputs.logits
                    recon_targets = target_frames

                    # Build per-pixel weight combining class weights + temporal change boost,
                    # then spatially smooth.  Temporal change boost is added independently
                    # of class weight so that *any* pixel change is upweighted.
                    pixel_weight = None
                    if class_weight is not None or args.temporal_change_boost > 0:
                        # Start with per-pixel class weights (or ones if no class weight).
                        if class_weight is not None:
                            raw_cw = class_weight.to(target_frames.device)[target_frames]  # (B, T, H, W)
                        else:
                            raw_cw = torch.ones_like(target_frames, dtype=torch.float32)

                        # Add temporal change boost independently.
                        if args.temporal_change_boost > 0:
                            tc = temporal_change_weight(target_frames, boost=args.temporal_change_boost)
                            # tc has base 1 + boost*changed; extract just the delta
                            # and add it to the class weights.
                            raw_cw = raw_cw + args.temporal_change_boost * (tc - 1.0)

                        # Spatially smooth the combined map.
                        if class_weight is not None and args.class_weight_radius >= 0.5:
                            pixel_weight = spatial_weight_map(
                                target_frames, class_weight,
                                radius=args.class_weight_radius,
                                hardness=args.class_weight_hardness,
                                temporal_ema=args.class_weight_temporal_ema,
                                per_pixel_weight=raw_cw,
                            )
                        else:
                            pixel_weight = raw_cw

                    # Upcast to float32 for numerically stable loss computation.
                    recon_loss = focal_cross_entropy(
                        recon_logits.float(),
                        recon_targets,
                        gamma=args.focal_gamma,
                        class_weight=class_weight if args.class_weight_radius < 0.5 else None,
                        pixel_weight=pixel_weight,
                    )
                    kl_loss = VideoVAE.kl_loss(outputs.posterior_mean.float(), outputs.posterior_logvar.float())
                    loss = recon_loss + args.kl_weight * kl_loss

                    # --- Auxiliary losses ---
                    with accelerator.autocast():
                        if next_frame_predictor is not None:
                            actions = batch["actions"].to(device, non_blocking=True)
                            nf_pred, nf_target = next_frame_predictor(outputs.posterior_mean, actions)
                            next_frame_loss = F.mse_loss(nf_pred, nf_target)
                            loss = loss + args.next_frame_weight * next_frame_loss

                        if args.temporal_smooth_weight > 0:
                            temporal_loss = temporal_smoothness_loss(outputs.posterior_mean)
                            loss = loss + args.temporal_smooth_weight * temporal_loss

                        if ram_align_head is not None and frozen_ram_vae is not None:
                            ram = batch["ram"].to(device, non_blocking=True)
                            with torch.no_grad():
                                ram_mean, _ = frozen_ram_vae.encode(ram)
                            ram_loss = ram_align_head.loss(outputs.posterior_mean, ram_mean)
                            loss = loss + args.ram_align_weight * ram_loss

                    # GAN generator adversarial loss.
                    if gan_active and discriminator is not None:
                        discriminator_real_inputs = frames_to_one_hot(
                            target_frames,
                            num_colors,
                            dtype=onehot_dtype,
                            out=onehot_buffer,
                        )
                        onehot_buffer = discriminator_real_inputs
                        set_requires_grad(discriminator, False)
                        with accelerator.autocast():
                            fake_video = outputs.logits.softmax(dim=1)
                            gen_adv_loss = hinge_generator_loss(discriminator(fake_video))
                        fake_video_detached = fake_video.detach()
                        loss = loss + args.gan_weight * gen_adv_loss
                        accum_gen_adv_sum += gen_adv_loss.item()
                        del fake_video, gen_adv_loss

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if args.max_grad_norm > 0:
                            grad_norm = accelerator.clip_grad_norm_(all_params, max_norm=args.max_grad_norm).item()

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Track per-micro-step scalars for logging (outside accumulate context).
                accum_loss_sum += loss.item()
                accum_recon_sum += recon_loss.item()
                accum_kl_sum += kl_loss.item()
                if next_frame_predictor is not None:
                    accum_nf_sum += next_frame_loss.item()  # type: ignore[possibly-undefined]
                if args.temporal_smooth_weight > 0:
                    accum_ts_sum += temporal_loss.item()  # type: ignore[possibly-undefined]
                if ram_align_head is not None:
                    accum_ram_sum += ram_loss.item()  # type: ignore[possibly-undefined]

                del outputs, recon_logits, recon_targets, target_frames

            # --- End of micro-step loop ---
            scheduler.step()

            # Compute averaged loss values for logging.
            avg_loss = accum_loss_sum / accum_steps
            avg_recon = accum_recon_sum / accum_steps
            avg_kl = accum_kl_sum / accum_steps
            next_frame_loss_value = accum_nf_sum / accum_steps
            temporal_smooth_loss_value = accum_ts_sum / accum_steps
            ram_align_loss_value = accum_ram_sum / accum_steps
            gan_gen_loss_value = accum_gen_adv_sum / accum_steps

            # GAN discriminator step (once per optimizer step, using last micro-batch data).
            if gan_active and discriminator is not None and discriminator_optimizer is not None:
                set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    real_scores = discriminator(discriminator_real_inputs)
                    fake_scores = discriminator(fake_video_detached)
                    discr_loss = hinge_discriminator_loss(real_scores, fake_scores)
                    if args.use_lecam and lecam_ema is not None:
                        lecam_reg = lecam_ema.regularizer(real_scores, fake_scores)
                        gan_lecam_reg_value = lecam_reg.item()
                        discr_loss = discr_loss + args.lecam_weight * lecam_reg
                accelerator.backward(discr_loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(discriminator.parameters(), max_norm=args.max_grad_norm)
                discriminator_optimizer.step()

                if args.use_lecam and lecam_ema is not None:
                    lecam_ema.update(real_scores.mean(), fake_scores.mean())

                gan_discr_loss_value = discr_loss.item()
                gan_real_logit_value = real_scores.mean().item()
                gan_fake_logit_value = fake_scores.mean().item()
                del real_scores, fake_scores, discr_loss, fake_video_detached

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=args.batch_size * accelerator.num_processes * accum_steps,
            )

            if is_main_process:
                status = (
                    f"res={current_frame_size} "
                    f"loss={avg_loss:.4f} recon={avg_recon:.4f} "
                    f"kl={avg_kl:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"gnorm={grad_norm:.2f} "
                    f"samp/s={samples_per_second:.0f} step/s={steps_per_second:.2f}"
                )
                stats = gpu_stats(device)
                if "gpu_util_pct" in stats:
                    status += f" gpu={stats['gpu_util_pct']:.0f}%"
                if "gpu_mem_pct" in stats:
                    status += f" mem={stats['gpu_mem_pct']:.0f}%"
                if "gpu_temp_c" in stats:
                    status += f" temp={stats['gpu_temp_c']:.0f}C"
                if bytes_per_sample is not None:
                    status += f" MB/s={(samples_per_second * bytes_per_sample) / 2**20:.0f}"
                if gan_active:
                    status += f" g_adv={gan_gen_loss_value:.4f} d={gan_discr_loss_value:.4f}"
                    if args.use_lecam:
                        status += f" lecam={gan_lecam_reg_value:.4f}"
                if next_frame_predictor is not None:
                    status += f" nf={next_frame_loss_value:.4f}"
                if args.temporal_smooth_weight > 0:
                    status += f" ts={temporal_smooth_loss_value:.4f}"
                if ram_align_head is not None:
                    status += f" ram={ram_align_loss_value:.4f}"

                advance_progress(progress, train_task, status=status)

            log_due = should_log_step(
                step,
                start_step=start_step,
                log_interval=args.log_interval,
                max_steps=args.max_steps,
            )
            if log_due and is_main_process:
                train_row = {
                    "type": "train",
                    "step": step,
                    "frame_size": int(current_frame_size),
                    "loss": avg_loss,
                    "recon_loss": avg_recon,
                    "kl_loss": avg_kl,
                    "grad_norm": grad_norm,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "samples_per_sec": round(samples_per_second, 1),
                    "steps_per_sec": round(steps_per_second, 2),
                }
                train_row.update(gpu_stats(device))
                if bytes_per_sample is not None:
                    train_row["throughput_mb_per_sec"] = round((samples_per_second * bytes_per_sample) / 2**20, 1)
                if gan_active:
                    train_row.update(
                        {
                            "gan_generator_loss": gan_gen_loss_value,
                            "gan_discriminator_loss": gan_discr_loss_value,
                            "gan_real_logit": gan_real_logit_value,
                            "gan_fake_logit": gan_fake_logit_value,
                        }
                    )
                    if args.use_lecam:
                        train_row["gan_lecam_reg"] = gan_lecam_reg_value
                if next_frame_predictor is not None:
                    train_row["next_frame_loss"] = next_frame_loss_value
                if args.temporal_smooth_weight > 0:
                    train_row["temporal_smooth_loss"] = temporal_smooth_loss_value
                if ram_align_head is not None:
                    train_row["ram_align_loss"] = ram_align_loss_value
                metrics.append(train_row)
                mlflow_run.log_metrics(train_row)

            if log_due and not use_live_progress and is_main_process:
                elapsed = time.time() - start_time
                examples_per_second = (
                    (step - start_step + 1)
                    * args.batch_size
                    * accelerator.num_processes
                    * accum_steps
                    / max(elapsed, 1e-6)
                )
                log_console.print(
                    f"step={step:06d} loss={avg_loss:.4f} recon={avg_recon:.4f} "
                    f"kl={avg_kl:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"gnorm={grad_norm:.2f} ex/s={examples_per_second:.1f}"
                    + (
                        f" g_adv={gan_gen_loss_value:.4f} d={gan_discr_loss_value:.4f} "
                        f"real={gan_real_logit_value:.4f} fake={gan_fake_logit_value:.4f}"
                        if gan_active
                        else ""
                    )
                )

            eval_due = eval_loader is not None and is_periodic_event_due(
                step,
                interval=args.eval_interval,
                max_steps=args.max_steps,
            )
            should_checkpoint = is_periodic_event_due(
                step,
                interval=args.checkpoint_interval,
                max_steps=args.max_steps,
            )
            if eval_due or should_checkpoint:
                accelerator.wait_for_everyone()

            eval_line: str | None = None
            should_eval = eval_due
            if should_eval:
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                try:
                    eval_metrics, preview = evaluate_video_vae(
                        model,
                        eval_loader,
                        device=device,
                        num_colors=num_colors,
                        kl_weight=args.kl_weight,
                        onehot_dtype=onehot_dtype,
                        focal_gamma=args.focal_gamma,
                        class_weight=class_weight,
                        class_weight_radius=args.class_weight_radius,
                        class_weight_hardness=args.class_weight_hardness,
                        class_weight_temporal_ema=args.class_weight_temporal_ema,
                        temporal_change_boost=args.temporal_change_boost,
                        accelerator=accelerator,
                        aggregate_losses=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    if is_main_process:
                        log_console.print("[eval][OOM] Eval pass OOM; skipping this eval window.")
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    if is_main_process:
                        eval_metrics["type"] = "eval"
                        eval_metrics["step"] = step
                        eval_metrics["frame_size"] = int(current_frame_size)
                        eval_metrics["lr"] = float(scheduler.get_last_lr()[0])
                        eval_metrics["train_grad_norm"] = grad_norm
                        metrics.append(eval_metrics)
                        mlflow_run.log_metrics(eval_metrics)
                        save_metrics_json(output_dir / "metrics.json", metrics)

                        eval_line = (
                            f"eval step={step:06d} loss={eval_metrics['loss']:.4f} "
                            f"recon={eval_metrics['recon_loss']:.4f} kl={eval_metrics['kl_loss']:.4f}"
                        )

                        if preview is not None:
                            save_video_preview(
                                preview_path(output_dir, split="eval", step=step),
                                preview[0],
                                preview[1],
                                palette,
                                max_frames=args.clip_frames,
                            )

                        # Avoid desynchronizing data iteration across processes during distributed training.
                        if accelerator.num_processes == 1:
                            with torch.no_grad():
                                try:
                                    train_batch = next(train_iter)
                                    train_frames = train_batch["frames"][:1].to(device, non_blocking=True).long()
                                    train_input = frames_to_one_hot(train_frames, num_colors, dtype=onehot_dtype)
                                    with accelerator.autocast():
                                        train_outputs = model(train_input, sample_posterior=False)
                                    save_video_preview(
                                        preview_path(output_dir, split="train", step=step),
                                        train_frames.detach().cpu(),
                                        train_outputs.logits.detach().cpu(),
                                        palette,
                                        max_frames=args.clip_frames,
                                    )
                                    del train_outputs, train_frames, train_batch
                                except torch.cuda.OutOfMemoryError:
                                    log_console.print("[eval][OOM] Train preview OOM; skipping preview image.")
                                    if torch.cuda.is_available():
                                        gc.collect()
                                        torch.cuda.empty_cache()

                        if eval_metrics["loss"] < best_eval:
                            best_eval = eval_metrics["loss"]
                            save_training_state(
                                output_dir / "best.pt",
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scheduler_metadata=scheduler_metadata,
                                step=step,
                                best_eval=best_eval,
                                metrics=metrics,
                                accelerator=accelerator,
                                discriminator=discriminator,
                                discriminator_optimizer=discriminator_optimizer,
                                lecam_ema=lecam_ema,
                                next_frame_predictor=next_frame_predictor,
                                ram_align_head=ram_align_head,
                            )
                            eval_line += f" | best={best_eval:.6f}"

                    del preview
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()

            if should_checkpoint and is_main_process:
                if eval_line is not None:
                    eval_line += " | saved latest"
                else:
                    log_console.print(f"[checkpoint] Saving latest weights at step {step}")
                save_training_state(
                    output_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scheduler_metadata=scheduler_metadata,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                    accelerator=accelerator,
                    discriminator=discriminator,
                    discriminator_optimizer=discriminator_optimizer,
                    lecam_ema=lecam_ema,
                    next_frame_predictor=next_frame_predictor,
                    ram_align_head=ram_align_head,
                )

            if eval_line is not None and is_main_process:
                log_console.print(eval_line)

            if eval_due or should_checkpoint:
                accelerator.wait_for_everyone()

    if is_main_process:
        save_metrics_json(output_dir / "metrics.json", metrics)
        mlflow_run.log_artifact(output_dir / "metrics.json")
        if args.mlflow_log_artifacts:
            mlflow_run.log_artifacts(output_dir)
        mlflow_run.finish()


if __name__ == "__main__":
    main()
