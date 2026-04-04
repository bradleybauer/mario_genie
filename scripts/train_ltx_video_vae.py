#!/usr/bin/env python3
"""Train the LTX Video VAE (learned-upsample decoder)."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.gan_discriminator import build_palette_discriminator, count_trainable_parameters
from mario_world_model.gan_training import LeCAMEMA, hinge_discriminator_loss, hinge_generator_loss, set_requires_grad
from mario_world_model.losses import focal_cross_entropy, softened_inverse_frequency_weights
from mario_world_model.ltx_video_vae import LTXVideoVAE
from mario_world_model.normalized_dataset import NormalizedSequenceDataset, load_palette_tensor
from mario_world_model.system_info import collect_system_info, print_system_info


console = Console()


def _format_bytes(num_bytes: int) -> str:
    num_bytes = max(int(num_bytes), 0)
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LTX Video VAE (learned-upsample decoder).")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument(
        "--context-frames",
        type=int,
        default=4,
        help="Extra context frames prepended to each clip; recon loss is masked to non-context frames.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear warmup from 0 to --lr over this many steps (default: 1000)")
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for eval only (defaults to --batch-size).",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4,
                        help="Encoder patchify size (decoder uses learned upsampling).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf16", action="store_true", help="Enable TF16 matmul precision")
    parser.add_argument(
        "--onehot-dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype for one-hot input tensors (reduces memory vs float32 when using float16/bfloat16).",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Enable CUDA autocast for model forward paths.",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Compute dtype for --autocast.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=1.0,
                        help="Focal loss gamma (0 = standard cross-entropy, 2 = typical focal)")
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
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
        default=0.2,
        help="Softening exponent for inverse-frequency weights (0=uniform, 1=full inverse-frequency).",
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
        default=0,
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
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()
    if args.context_frames < 0:
        parser.error("--context-frames must be >= 0")
    if args.clip_frames <= 0:
        parser.error("--clip-frames must be > 0")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)
    if args.resume_from is not None:
        return Path(args.resume_from).resolve().parent
    run_name = args.run_name or datetime.now().strftime("ltx_video_vae_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "checkpoints" / run_name


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def frames_to_one_hot(
    frames: torch.Tensor,
    num_colors: int,
    *,
    dtype: torch.dtype = torch.float32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (B, T, H, W), got {tuple(frames.shape)}")

    if frames.dtype != torch.long:
        frames = frames.long()
    expected_shape = (frames.shape[0], num_colors, frames.shape[1], frames.shape[2], frames.shape[3])
    if (
        out is None
        or out.shape != expected_shape
        or out.dtype != dtype
        or out.device != frames.device
    ):
        out = torch.empty(expected_shape, dtype=dtype, device=frames.device)
    out.zero_()
    out.scatter_(1, frames.unsqueeze(1), 1)
    return out


def split_context_targets(
    logits: torch.Tensor,
    frames: torch.Tensor,
    context_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_frames <= 0:
        return logits, frames
    if context_frames >= frames.shape[1]:
        raise ValueError(
            f"context_frames ({context_frames}) must be smaller than total clip length ({frames.shape[1]})"
        )
    return logits[:, :, context_frames:], frames[:, context_frames:]


def save_video_preview(path: Path, frames: torch.Tensor, logits: torch.Tensor, palette: torch.Tensor, max_frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames[0, :max_frames].detach().cpu()
    recon = logits[0, :, :max_frames].argmax(dim=0).detach().cpu()
    palette_u8 = (palette.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    target_rgb = palette_u8[frames.numpy()]
    recon_rgb = palette_u8[recon.numpy()]
    rows = [np.concatenate([target_rgb[idx], recon_rgb[idx]], axis=1) for idx in range(target_rgb.shape[0])]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def evaluate(
    model: LTXVideoVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    num_colors: int,
    kl_weight: float,
    context_frames: int = 0,
    onehot_dtype: torch.dtype = torch.float32,
    autocast_enabled: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    focal_gamma: float = 0.0,
    class_weight: torch.Tensor | None = None,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    preview: tuple[torch.Tensor, torch.Tensor] | None = None
    onehot_buffer: torch.Tensor | None = None

    def autocast_context():
        if not autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).long()
            inputs = frames_to_one_hot(frames, num_colors, dtype=onehot_dtype, out=onehot_buffer)
            onehot_buffer = inputs
            with autocast_context():
                outputs = model(inputs, sample_posterior=False)
            recon_logits, recon_targets = split_context_targets(outputs.logits, frames, context_frames)
            recon_loss = focal_cross_entropy(
                recon_logits,
                recon_targets,
                gamma=focal_gamma,
                class_weight=class_weight,
            )
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            if preview is None:
                with autocast_context():
                    preview_out = model(inputs[:1], sample_posterior=False)
                preview = (frames[:1].detach().cpu(), preview_out.logits.detach().cpu())
                del preview_out
            del recon_logits, recon_targets, recon_loss, kl_loss, outputs, inputs, frames
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not recon_losses:
        return {"recon_loss": 0.0, "kl_loss": 0.0, "loss": 0.0}, preview

    mean_recon = float(np.mean(recon_losses))
    mean_kl = float(np.mean(kl_losses))
    return {
        "recon_loss": mean_recon,
        "kl_loss": mean_kl,
        "loss": mean_recon + kl_weight * mean_kl,
    }, preview


def save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    best_eval: float,
    metrics: list[dict[str, float]],
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: AdamW | None = None,
    lecam_ema: LeCAMEMA | None = None,
) -> None:
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_eval": best_eval,
        "metrics": metrics,
    }
    if discriminator is not None and discriminator_optimizer is not None:
        state["discriminator"] = discriminator.state_dict()
        state["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        if lecam_ema is not None:
            state["lecam_ema"] = lecam_ema.state_dict()
    torch.save(state, path)


def gpu_stats(device: torch.device) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    index = device.index or 0
    stats: dict[str, float] = {}
    try:
        mem_used = torch.cuda.memory_allocated(index) / 2**30
        mem_total = torch.cuda.get_device_properties(index).total_memory / 2**30
        stats["gpu_mem_pct"] = round(100.0 * mem_used / max(mem_total, 1e-9), 1)
    except Exception:
        pass
    try:
        stats["gpu_util_pct"] = float(torch.cuda.utilization(index))
    except Exception:
        pass
    try:
        stats["gpu_temp_c"] = float(torch.cuda.temperature(index))
    except Exception:
        pass
    return stats


def main() -> None:
    args = parse_args()
    if args.use_lecam and not args.use_gan:
        raise SystemExit("--use-lecam requires --use-gan")
    if not (0.0 < args.lecam_decay < 1.0):
        raise SystemExit("--lecam-decay must be in (0, 1)")
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        if args.tf16:
            torch.set_float32_matmul_precision("medium")
            console.print("[tf16] TF16 matmul precision enabled")
        else:
            torch.set_float32_matmul_precision("high")
            console.print("[tf32] TF32 matmul precision enabled")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    onehot_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    onehot_dtype = onehot_dtype_map[args.onehot_dtype]

    autocast_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    autocast_enabled = bool(args.autocast and device.type == "cuda")
    autocast_dtype = autocast_dtype_map[args.autocast_dtype]
    if args.autocast and device.type != "cuda":
        console.print("[autocast] Requested but CUDA is unavailable; disabling autocast.")
    if autocast_enabled and autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        console.print("[autocast] bfloat16 unsupported on this GPU; falling back to float16.")
        autocast_dtype = torch.float16
    if autocast_enabled:
        label = "bf16" if autocast_dtype == torch.bfloat16 else "fp16"
        console.print(f"[autocast] Enabled ({label})")

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and autocast_dtype == torch.float16,
    )

    def autocast_context():
        if not autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    system_info = collect_system_info()
    print_system_info(system_info)

    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_clip_frames = args.clip_frames + args.context_frames

    console.print("[dataset] Building sequence index (can take a few minutes on large runs)...")
    dataset = NormalizedSequenceDataset(
        data_dir=args.data_dir,
        clip_frames=total_clip_frames,
        num_workers=args.num_workers,
        system_info=system_info,
    )
    console.print("[dataset] Index build complete.")
    if len(dataset) == 0:
        raise RuntimeError("No training samples were found.")
    console.print(
        f"Found {len(dataset)} sequence segments of {total_clip_frames} frames "
        f"({args.context_frames} context + {args.clip_frames} target)."
    )
    console.print(
        f"Dataset: {dataset.num_files} files, {len(dataset):,} samples, "
        f"{_format_bytes(dataset.dataset_bytes)} on disk."
    )

    eval_dataset = None
    train_dataset = dataset
    if args.eval_samples > 0 and len(dataset) > args.eval_samples:
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(dataset), generator=generator)
        eval_indices = permutation[:args.eval_samples].tolist()
        train_indices = permutation[args.eval_samples:].tolist()
        eval_dataset = Subset(dataset, eval_indices)
        train_dataset = Subset(dataset, train_indices)
        console.print(f"Eval split: {len(eval_dataset)} eval, {len(train_dataset)} train samples")

    num_workers = max(args.num_workers, 0)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=10**7)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    def batch_iter():
        while True:
            yield from train_loader

    train_iter = batch_iter()

    eval_loader = None
    if eval_dataset is not None:
        eval_batch_size = args.eval_batch_size or args.batch_size
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    palette_path = Path(args.data_dir) / "palette.json"
    palette = load_palette_tensor(args.data_dir)
    num_colors = palette.shape[0]
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
        console.print(
            "[class-weights] Enabled from "
            f"{Path(args.data_dir) / args.class_weights_file} "
            f"(soften={args.class_weight_soften:.2f}, "
            f"min={class_weight.min().item():.3f}, max={class_weight.max().item():.3f})"
        )
    model = LTXVideoVAE(
        num_colors=num_colors,
        patch_size=args.patch_size,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
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
        console.print(
            f"[gan] Enabled discriminator {args.gan_target_size} "
            f"({discriminator_num_parameters:,} params), "
            f"gan_weight={args.gan_weight}, gan_lr={args.gan_lr:.2e}, "
            f"gan_start_step={args.gan_start_step}, "
            f"lecam={'on' if args.use_lecam else 'off'}"
        )

    if args.compile:
        console.print("[compile] Compiling the model with torch.compile()...")
        model = torch.compile(model)
        if discriminator is not None:
            console.print("[compile] Compiling discriminator with torch.compile()...")
            discriminator = torch.compile(discriminator)
        console.print("[compile] Compilation complete.")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    if discriminator is not None:
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=args.gan_lr)
    warmup_steps = max(int(args.warmup_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / float(warmup_steps)
        if args.max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(args.max_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.25 + 0.75 * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    if warmup_steps > 0:
        console.print(f"[lr] Warmup: {warmup_steps} steps → cosine decay to 25% of peak")
    else:
        console.print("[lr] Cosine decay (no warmup)")

    start_step = 0
    best_eval = float("inf")
    metrics: list[dict[str, float]] = []

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
            if "discriminator" in checkpoint and "discriminator_optimizer" in checkpoint:
                discriminator.load_state_dict(checkpoint["discriminator"])
                discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
            else:
                console.print("[resume] Checkpoint has no discriminator state; GAN starts from scratch.")
            if args.use_lecam and lecam_ema is not None:
                if "lecam_ema" in checkpoint:
                    lecam_ema.load_state_dict(checkpoint["lecam_ema"])
                else:
                    console.print("[resume] Checkpoint has no LeCAM EMA state; LeCAM EMA resets.")
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"]) + 1
        best_eval = float(checkpoint.get("best_eval", float("inf")))
        metrics = list(checkpoint.get("metrics", []))
        console.print(f"[resume] Loaded training state from {args.resume_from} at step {start_step}")

        if args.max_steps > 0 and start_step >= args.max_steps:
            console.print(
                f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do."
            )
            return

    config = vars(args).copy()
    config.update(
        {
            "model_name": "ltx_video_vae",
            "num_colors": int(num_colors),
            "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "discriminator_parameters": int(discriminator_num_parameters),
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
        }
    )
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    console.print(f"Config saved to {output_dir / 'config.json'}")
    console.print(json.dumps(config, indent=2))

    console.print(f"Training LTX Video VAE on {len(train_dataset)} samples")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Device: {device}")
    console.print(f"Model parameters: {config['num_parameters']:,}")

    start_time = time.time()
    perf_window_start = time.time()
    perf_window_steps = 0
    perf_window_samples = 0
    samples_per_second = 0.0
    steps_per_second = 0.0
    perf_window_size = 20
    bytes_per_sample = None
    use_live_progress = sys.stdout.isatty()

    total_train_steps = max(args.max_steps - start_step, 0)
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        disable=not use_live_progress,
        refresh_per_second=2,
    ) as progress:
        log_console = progress.console
        train_task = progress.add_task("Training", total=total_train_steps, status="")
        onehot_buffer: torch.Tensor | None = None
        for step in range(start_step, args.max_steps):
            batch = next(train_iter)
            frames = batch["frames"].to(device, non_blocking=True).long()
            inputs = frames_to_one_hot(frames, num_colors, dtype=onehot_dtype, out=onehot_buffer)
            onehot_buffer = inputs
            gan_active = args.use_gan and step >= args.gan_start_step
            gan_discr_loss_value = 0.0
            gan_gen_loss_value = 0.0
            gan_real_logit_value = 0.0
            gan_fake_logit_value = 0.0
            gan_lecam_reg_value = 0.0
            if bytes_per_sample is None:
                _, time_steps, height, width = frames.shape
                bytes_per_sample = (
                    num_colors
                    * time_steps
                    * height
                    * width
                    * torch.tensor([], dtype=onehot_dtype).element_size()
                )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                outputs = model(inputs)
            recon_logits, recon_targets = split_context_targets(outputs.logits, frames, args.context_frames)
            recon_loss = focal_cross_entropy(
                recon_logits,
                recon_targets,
                gamma=args.focal_gamma,
                class_weight=class_weight,
            )
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            loss = recon_loss + args.kl_weight * kl_loss

            fake_video_detached = None
            if gan_active and discriminator is not None:
                set_requires_grad(discriminator, False)
                with autocast_context():
                    fake_video = outputs.logits.softmax(dim=1)
                    gen_adv_loss = hinge_generator_loss(discriminator(fake_video))
                fake_video_detached = fake_video.detach()
                gan_gen_loss_value = gen_adv_loss.item()
                loss = loss + args.gan_weight * gen_adv_loss
                del fake_video, gen_adv_loss

            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            if grad_scaler.is_enabled():
                grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            del outputs, recon_logits, recon_targets

            if gan_active and discriminator is not None and discriminator_optimizer is not None:
                set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with autocast_context():
                    real_scores = discriminator(inputs)
                    fake_scores = discriminator(fake_video_detached)
                    discr_loss = hinge_discriminator_loss(real_scores, fake_scores)
                    if args.use_lecam and lecam_ema is not None:
                        lecam_reg = lecam_ema.regularizer(real_scores, fake_scores)
                        gan_lecam_reg_value = lecam_reg.item()
                        discr_loss = discr_loss + args.lecam_weight * lecam_reg
                discr_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                discriminator_optimizer.step()

                if args.use_lecam and lecam_ema is not None:
                    lecam_ema.update(real_scores.mean(), fake_scores.mean())

                gan_discr_loss_value = discr_loss.item()
                gan_real_logit_value = real_scores.mean().item()
                gan_fake_logit_value = fake_scores.mean().item()
                del real_scores, fake_scores, discr_loss, fake_video_detached

            perf_window_steps += 1
            perf_window_samples += frames.shape[0]
            now = time.time()
            elapsed_window = max(now - perf_window_start, 1e-9)
            if perf_window_steps >= perf_window_size:
                samples_per_second = perf_window_samples / elapsed_window
                steps_per_second = perf_window_steps / elapsed_window
                perf_window_start = now
                perf_window_steps = 0
                perf_window_samples = 0
            else:
                samples_per_second = perf_window_samples / elapsed_window
                steps_per_second = perf_window_steps / elapsed_window

            status = (
                f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                f"kl={kl_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
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

            progress.update(
                train_task,
                advance=1,
                status=status,
            )

            if (
                ((args.log_interval > 0 and step % args.log_interval == 0) or step == args.max_steps - 1)
                and not use_live_progress
            ):
                elapsed = time.time() - start_time
                examples_per_second = ((step - start_step + 1) * args.batch_size) / max(elapsed, 1e-6)
                log_console.print(
                    f"step={step:06d} loss={loss.item():.4f} recon={recon_loss.item():.4f} "
                    f"kl={kl_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"gnorm={grad_norm:.2f} ex/s={examples_per_second:.1f}"
                    + (
                        f" g_adv={gan_gen_loss_value:.4f} d={gan_discr_loss_value:.4f} "
                        f"real={gan_real_logit_value:.4f} fake={gan_fake_logit_value:.4f}"
                        if gan_active
                        else ""
                    )
                )

            should_eval = (
                eval_loader is not None
                and (
                    (args.eval_interval > 0 and (step + 1) % args.eval_interval == 0)
                    or step == args.max_steps - 1
                )
            )
            if should_eval:
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                try:
                    eval_metrics, preview = evaluate(
                        model,
                        eval_loader,
                        device=device,
                        num_colors=num_colors,
                        kl_weight=args.kl_weight,
                        context_frames=args.context_frames,
                        onehot_dtype=onehot_dtype,
                        autocast_enabled=autocast_enabled,
                        autocast_dtype=autocast_dtype,
                        focal_gamma=args.focal_gamma,
                        class_weight=class_weight,
                    )
                except torch.cuda.OutOfMemoryError:
                    log_console.print("[eval][OOM] Eval pass OOM; skipping this eval window.")
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue

                eval_metrics["step"] = step
                eval_metrics["lr"] = float(scheduler.get_last_lr()[0])
                metrics.append(eval_metrics)
                with (output_dir / "metrics.json").open("w") as handle:
                    json.dump(metrics, handle, indent=2)

                eval_line = (
                    f"eval step={step:06d} loss={eval_metrics['loss']:.4f} "
                    f"recon={eval_metrics['recon_loss']:.4f} kl={eval_metrics['kl_loss']:.4f}"
                )

                if preview is not None:
                    save_video_preview(
                        output_dir / f"preview_step_{step:06d}.png",
                        preview[0],
                        preview[1],
                        palette,
                        max_frames=total_clip_frames,
                    )

                # Random training sample preview
                with torch.no_grad():
                    try:
                        train_batch = next(train_iter)
                        train_frames = train_batch["frames"][:1].to(device, non_blocking=True).long()
                        train_inputs = frames_to_one_hot(train_frames, num_colors, dtype=onehot_dtype)
                        with autocast_context():
                            train_outputs = model(train_inputs, sample_posterior=False)
                        save_video_preview(
                            output_dir / f"train_preview_step_{step:06d}.png",
                            train_frames.detach().cpu(),
                            train_outputs.logits.detach().cpu(),
                            palette,
                            max_frames=total_clip_frames,
                        )
                        del train_outputs, train_inputs, train_frames, train_batch
                    except torch.cuda.OutOfMemoryError:
                        log_console.print("[eval][OOM] Train preview OOM; skipping preview image.")
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()

                del preview
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                if eval_metrics["loss"] < best_eval:
                    best_eval = eval_metrics["loss"]
                    save_training_state(
                        output_dir / "video_vae_best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                        discriminator=discriminator,
                        discriminator_optimizer=discriminator_optimizer,
                        lecam_ema=lecam_ema,
                    )
                    eval_line += f" | best={best_eval:.6f}"

            should_checkpoint = (
                (args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0)
                or step == args.max_steps - 1
            )
            if should_checkpoint:
                if should_eval:
                    eval_line += " | saved latest"
                else:
                    log_console.print(f"[checkpoint] Saving latest weights at step {step}")
                save_training_state(
                    output_dir / "video_vae_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                    discriminator=discriminator,
                    discriminator_optimizer=discriminator_optimizer,
                    lecam_ema=lecam_ema,
                )

            if should_eval:
                log_console.print(eval_line)


if __name__ == "__main__":
    main()
