#!/usr/bin/env python3
"""Train a joint VAE that encodes RAM and decodes both RAM and video.

Usage:
    python scripts/train/train_ram_video_vae.py --max-steps 10000
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from rich.console import Console
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.normalized_dataset import NormalizedSequenceDataset, load_palette_tensor
from src.data.video_frames import SUPPORTED_FRAME_SIZES
from src.models.gan_discriminator import build_palette_discriminator, count_trainable_parameters
from src.models.ram_video_vae import RAMVideoVAE
from src.models.ram_video_vae_v2 import RAMVideoVAEv2
from src.system_info import collect_system_info, print_system_info
from src.training.gan_training import LeCAMEMA, hinge_discriminator_loss, hinge_generator_loss, set_requires_grad
from src.training.losses import focal_cross_entropy
from src.training.palette_video_vae_training import frames_to_one_hot, save_video_preview
from src.training.trainer_common import (
    build_trainer_config,
    build_warmup_cosine_scheduler,
    configure_cuda_runtime,
    format_bytes,
    gpu_stats,
    is_periodic_event_due,
    make_output_dir,
    preview_path,
    seed_everything,
    should_log_step,
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
    unwrap_model,
)


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RAM-to-video joint VAE.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Model architecture to train.",
    )
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        choices=SUPPORTED_FRAME_SIZES,
        help="Output frame size used for training and eval previews.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--video-loss-weight", type=float, default=1.0)
    parser.add_argument("--ram-loss-weight", type=float, default=.5)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
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
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-fc-blocks", type=int, default=2)
    parser.add_argument("--n-temporal-blocks", type=int, default=2)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--video-base-channels", type=int, default=24)
    parser.add_argument("--video-latent-channels", type=int, default=16)
    parser.add_argument(
        "--video-adapter-dim",
        type=int,
        default=256,
        help="Hidden size for the v2 query-based video renderer.",
    )
    parser.add_argument(
        "--video-adapter-heads",
        type=int,
        default=8,
        help="Number of attention heads in the v2 query-based video renderer.",
    )
    parser.add_argument(
        "--n-ram-groups",
        type=int,
        default=128,
        help="Number of causal RAM address groups exposed to the v2 video renderer.",
    )
    parser.add_argument(
        "--n-video-temporal-blocks",
        type=int,
        default=2,
        help="Temporal refinement blocks in the v2 video-memory adapter.",
    )
    parser.add_argument(
        "--n-video-renderer-blocks",
        type=int,
        default=2,
        help="Cross-attention renderer blocks in the v2 video branch.",
    )
    parser.add_argument(
        "--temporal-downsample",
        type=int,
        default=1,
        choices=[0, 1],
        help="Number of temporal downsamples in the video decoder head (0 or 1).",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode passed to Accelerator.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def save_ram_preview(
    path: Path,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    title_prefix: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tgt = target[0].detach().cpu().numpy()
    rec = reconstruction[0].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

    im0 = axes[0].imshow(tgt.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"{title_prefix}Target RAM" if title_prefix else "Target RAM")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Byte index")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)

    im1 = axes[1].imshow(rec.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(f"{title_prefix}Reconstructed RAM" if title_prefix else "Reconstructed RAM")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Byte index")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)

    diff = np.abs(tgt - rec)
    im2 = axes[2].imshow(diff.T, origin="lower", aspect="auto", cmap="hot", vmin=0, vmax=0.5)
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Byte index")
    fig.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_joint_losses(
    model: torch.nn.Module,
    outputs,
    *,
    frames: torch.Tensor,
    ram: torch.Tensor,
    focal_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    video_recon_loss = focal_cross_entropy(
        outputs.video_logits.float(),
        frames,
        gamma=focal_gamma,
    )
    ram_recon_loss = F.mse_loss(outputs.ram_reconstruction.float(), ram)
    kl_loss = model.kl_loss(outputs.posterior_mean.float(), outputs.posterior_logvar.float())
    return video_recon_loss, ram_recon_loss, kl_loss


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    focal_gamma: float,
    video_loss_weight: float,
    ram_loss_weight: float,
    kl_weight: float,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    model.eval()
    video_recon_losses: list[float] = []
    ram_recon_losses: list[float] = []
    kl_losses: list[float] = []
    preview: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).long()
            ram = batch["ram"].to(device, non_blocking=True).float() / 255.0
            outputs = model(
                ram,
                output_video_shape=(frames.shape[1], frames.shape[2], frames.shape[3]),
                sample_posterior=False,
            )
            video_recon_loss, ram_recon_loss, kl_loss = compute_joint_losses(
                model,
                outputs,
                frames=frames,
                ram=ram,
                focal_gamma=focal_gamma,
            )
            video_recon_losses.append(video_recon_loss.item())
            ram_recon_losses.append(ram_recon_loss.item())
            kl_losses.append(kl_loss.item())

            if preview is None:
                preview = (
                    frames[:1].detach().cpu(),
                    outputs.video_logits[:1].detach().cpu(),
                    ram[:1].detach().cpu(),
                    outputs.ram_reconstruction[:1].detach().float().cpu(),
                )

    model.train()

    if not video_recon_losses:
        return {
            "video_recon_loss": 0.0,
            "ram_recon_loss": 0.0,
            "kl_loss": 0.0,
            "loss": 0.0,
        }, preview

    mean_video_recon = float(np.mean(video_recon_losses))
    mean_ram_recon = float(np.mean(ram_recon_losses))
    mean_kl = float(np.mean(kl_losses))
    return {
        "video_recon_loss": mean_video_recon,
        "ram_recon_loss": mean_ram_recon,
        "kl_loss": mean_kl,
        "loss": (
            video_loss_weight * mean_video_recon
            + ram_loss_weight * mean_ram_recon
            + kl_weight * mean_kl
        ),
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
    accelerator: Accelerator | None = None,
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
    lecam_ema: LeCAMEMA | None = None,
) -> None:
    state = {
        "model": get_model_state_dict(model, accelerator=accelerator),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_eval": best_eval,
        "metrics": metrics,
    }
    if discriminator is not None and discriminator_optimizer is not None:
        state["discriminator"] = get_model_state_dict(discriminator, accelerator=accelerator)
        state["discriminator_optimizer"] = discriminator_optimizer.state_dict()
        if lecam_ema is not None:
            state["lecam_ema"] = lecam_ema.state_dict()
    torch.save(state, path)


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
        default_prefix="ram_video_vae_v2" if args.model_version == "v2" else "ram_video_vae",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = create_accelerator_runtime(output_dir=output_dir, mixed_precision=args.mixed_precision)
    accelerator = runtime.accelerator
    device = runtime.device
    is_main_process = runtime.is_main_process

    if is_main_process:
        console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        configure_cuda_runtime()
        if is_main_process:
            console.print("[tf32] TF32 matmul precision enabled")
    if args.mixed_precision != "no" and is_main_process:
        console.print(f"[mixed-precision] Enabled ({args.mixed_precision})")

    system_info = collect_system_info()
    if is_main_process:
        print_system_info(system_info)

    if is_main_process:
        console.print("[dataset] Building sequence index...")
    with accelerator.main_process_first():
        dataset = NormalizedSequenceDataset(
            data_dir=args.data_dir,
            clip_frames=args.clip_frames,
            frame_size=args.frame_size,
            include_frames=True,
            include_audio=False,
            include_actions=False,
            include_ram=True,
            num_workers=args.num_workers,
            system_info=system_info,
        )
    if is_main_process:
        console.print("[dataset] Index build complete.")
    if len(dataset) == 0:
        raise RuntimeError("No RAM/video training samples were found.")

    n_bytes = dataset.ram_n_bytes
    if n_bytes is None:
        with np.load(dataset.data_files[0], mmap_mode="r") as npz:
            n_bytes = int(npz["ram"].shape[1])

    probe_sample = dataset[0]
    frame_height = int(probe_sample["frames"].shape[-2])
    frame_width = int(probe_sample["frames"].shape[-1])
    source_frame_height = int(dataset.source_frame_height or frame_height)
    source_frame_width = int(dataset.source_frame_width or frame_width)
    del probe_sample

    palette_path = Path(args.data_dir) / "palette.json"
    palette = load_palette_tensor(args.data_dir)
    num_colors = int(palette.shape[0])

    if is_main_process:
        console.print(
            f"Found {len(dataset):,} sequence segments of {args.clip_frames} frames, "
            f"{n_bytes} RAM bytes per frame."
        )
        console.print(
            f"Dataset: {dataset.num_files} files, {len(dataset):,} samples, "
            f"{format_bytes(dataset.dataset_bytes)} on disk."
        )
        console.print(
            f"[palette] Loaded {num_colors} colours from {palette_path} "
            f"for frames of {frame_height}x{frame_width}."
        )

    train_dataset, eval_dataset = split_train_eval_dataset(
        dataset,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )
    if eval_dataset is not None and is_main_process:
        console.print(f"Eval split: {len(eval_dataset)} eval, {len(train_dataset)} train samples")

    train_loader = build_replacement_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = build_eval_loader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    if args.model_version == "v2":
        model = RAMVideoVAEv2(
            n_bytes=n_bytes,
            num_colors=num_colors,
            frame_height=frame_height,
            frame_width=frame_width,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            n_fc_blocks=args.n_fc_blocks,
            n_temporal_blocks=args.n_temporal_blocks,
            temporal_kernel_size=args.temporal_kernel_size,
            video_base_channels=args.video_base_channels,
            video_latent_channels=args.video_latent_channels,
            temporal_downsample=args.temporal_downsample,
            video_adapter_dim=args.video_adapter_dim,
            video_adapter_heads=args.video_adapter_heads,
            n_ram_groups=args.n_ram_groups,
            n_video_temporal_blocks=args.n_video_temporal_blocks,
            n_video_renderer_blocks=args.n_video_renderer_blocks,
        ).to(device)
    else:
        model = RAMVideoVAE(
            n_bytes=n_bytes,
            num_colors=num_colors,
            frame_height=frame_height,
            frame_width=frame_width,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            n_fc_blocks=args.n_fc_blocks,
            n_temporal_blocks=args.n_temporal_blocks,
            temporal_kernel_size=args.temporal_kernel_size,
            video_base_channels=args.video_base_channels,
            video_latent_channels=args.video_latent_channels,
            temporal_downsample=args.temporal_downsample,
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

    if args.compile:
        if is_main_process:
            console.print("[compile] Compiling model with torch.compile()...")
        model = torch.compile(model)
        if discriminator is not None:
            if is_main_process:
                console.print("[compile] Compiling discriminator with torch.compile()...")
            discriminator = torch.compile(discriminator)
        if is_main_process:
            console.print("[compile] Compilation complete.")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    if discriminator is not None:
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=args.gan_lr)
    warmup_steps = max(int(args.warmup_steps), 0)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        min_lr_scale=0.1,
    )

    start_step = 0
    best_eval = float("inf")
    metrics: list[dict[str, float]] = []

    if args.resume_from is not None:
        with accelerator.main_process_first():
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        load_model_state_dict(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
            if "discriminator" in checkpoint and "discriminator_optimizer" in checkpoint:
                load_model_state_dict(discriminator, checkpoint["discriminator"])
                discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
            elif is_main_process:
                console.print("[resume] Checkpoint has no discriminator state; GAN starts from scratch.")
            if args.use_lecam and lecam_ema is not None:
                if "lecam_ema" in checkpoint:
                    lecam_ema.load_state_dict(checkpoint["lecam_ema"])
                elif is_main_process:
                    console.print("[resume] Checkpoint has no LeCAM EMA state; LeCAM EMA resets.")
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"]) + 1
        best_eval = float(checkpoint.get("best_eval", float("inf")))
        metrics = list(checkpoint.get("metrics", []))
        if is_main_process:
            console.print(f"[resume] Loaded training state from {args.resume_from} at step {start_step}")
        if args.max_steps > 0 and start_step >= args.max_steps:
            if is_main_process:
                console.print(f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do.")
            return

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

    num_params = sum(p.numel() for p in model.parameters())
    model_name = "ram_video_vae_v2" if args.model_version == "v2" else "ram_video_vae"
    model_config = {
        "num_parameters": int(num_params),
        "discriminator_parameters": int(discriminator_num_parameters),
        "hidden_dim": int(args.hidden_dim),
        "latent_dim": int(args.latent_dim),
        "n_fc_blocks": int(args.n_fc_blocks),
        "n_temporal_blocks": int(args.n_temporal_blocks),
        "temporal_kernel_size": int(args.temporal_kernel_size),
        "video_base_channels": int(args.video_base_channels),
        "video_latent_channels": int(args.video_latent_channels),
        "temporal_downsample": int(args.temporal_downsample),
    }
    if args.model_version == "v2":
        model_config.update(
            {
                "video_adapter_dim": int(args.video_adapter_dim),
                "video_adapter_heads": int(args.video_adapter_heads),
                "n_ram_groups": int(args.n_ram_groups),
                "n_video_temporal_blocks": int(args.n_video_temporal_blocks),
                "n_video_renderer_blocks": int(args.n_video_renderer_blocks),
            }
        )
    config = build_trainer_config(
        model_name=model_name,
        args=args,
        device=device,
        mixed_precision=args.mixed_precision,
        num_processes=accelerator.num_processes,
        data={
            "n_bytes": int(n_bytes),
            "num_colors": int(num_colors),
            "clip_frames": int(args.clip_frames),
            "frame_height": int(frame_height),
            "frame_width": int(frame_width),
            "source_frame_height": int(source_frame_height),
            "source_frame_width": int(source_frame_width),
        },
        model=model_config,
    )
    if is_main_process:
        save_json(output_dir / "config.json", config, indent=2)
        console.print(f"Training {model_name} on {len(train_dataset)} samples")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Device: {device}")
        console.print(f"Model parameters: {num_params:,}")
        if accelerator.num_processes > 1:
            console.print(
                f"Distributed training enabled with {accelerator.num_processes} processes "
                f"(global batch={args.batch_size * accelerator.num_processes})."
            )

    start_time = time.time()
    throughput_tracker = ThroughputTracker(window_steps=20)
    use_live_progress = sys.stdout.isatty() and is_main_process

    total_train_steps = max(args.max_steps - start_step, 0)
    with build_progress(use_live=use_live_progress) as progress:
        log_console = progress.console
        train_task = progress.add_task("Training", total=total_train_steps, status="")
        onehot_buffer: torch.Tensor | None = None

        for step in range(start_step, args.max_steps):
            batch = next(train_iter)
            frames = batch["frames"].to(device, non_blocking=True).long()
            ram = batch["ram"].to(device, non_blocking=True).float() / 255.0

            gan_active = args.use_gan and step >= args.gan_start_step
            gan_discr_loss_value = 0.0
            gan_gen_loss_value = 0.0
            gan_real_logit_value = 0.0
            gan_fake_logit_value = 0.0
            gan_lecam_reg_value = 0.0

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                outputs = model(
                    ram,
                    output_video_shape=(frames.shape[1], frames.shape[2], frames.shape[3]),
                )
                video_recon_loss, ram_recon_loss, kl_loss = compute_joint_losses(
                    model,
                    outputs,
                    frames=frames,
                    ram=ram,
                    focal_gamma=args.focal_gamma,
                )
                loss = (
                    args.video_loss_weight * video_recon_loss
                    + args.ram_loss_weight * ram_recon_loss
                    + args.kl_weight * kl_loss
                )

            fake_video_detached = None
            discriminator_real_inputs = None
            if gan_active and discriminator is not None:
                set_requires_grad(discriminator, False)
                with accelerator.autocast():
                    fake_video = outputs.video_logits.softmax(dim=1)
                    gen_adv_loss = hinge_generator_loss(discriminator(fake_video))
                discriminator_real_inputs = frames_to_one_hot(
                    frames,
                    num_colors,
                    dtype=fake_video.dtype,
                    out=onehot_buffer,
                )
                onehot_buffer = discriminator_real_inputs
                fake_video_detached = fake_video.detach()
                gan_gen_loss_value = gen_adv_loss.item()
                loss = loss + args.gan_weight * gen_adv_loss
                del fake_video, gen_adv_loss

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()

            if (
                gan_active
                and discriminator is not None
                and discriminator_optimizer is not None
                and discriminator_real_inputs is not None
                and fake_video_detached is not None
            ):
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
                accelerator.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                discriminator_optimizer.step()

                if args.use_lecam and lecam_ema is not None:
                    lecam_ema.update(real_scores.mean(), fake_scores.mean())

                gan_discr_loss_value = discr_loss.item()
                gan_real_logit_value = real_scores.mean().item()
                gan_fake_logit_value = fake_scores.mean().item()
                del real_scores, fake_scores, discr_loss, fake_video_detached

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=ram.shape[0] * accelerator.num_processes,
            )

            log_due = should_log_step(
                step,
                start_step=start_step,
                log_interval=args.log_interval,
                max_steps=args.max_steps,
            )
            if log_due:
                lr_current = scheduler.get_last_lr()[0]
                train_row = {
                    "type": "train",
                    "step": step,
                    "loss": loss.item(),
                    "video_recon_loss": video_recon_loss.item(),
                    "ram_recon_loss": ram_recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "grad_norm": grad_norm,
                    "lr": float(lr_current),
                    "samples_per_sec": round(samples_per_second, 1),
                    "steps_per_sec": round(steps_per_second, 2),
                }
                if is_main_process:
                    train_row.update(gpu_stats(device))
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
                    metrics.append(train_row)

                    status = (
                        f"loss={loss.item():.5f} video={video_recon_loss.item():.5f} "
                        f"ram={ram_recon_loss.item():.5f} kl={kl_loss.item():.3f} "
                        f"lr={lr_current:.2e} gnorm={grad_norm:.2f} "
                        f"{samples_per_second:.0f} samp/s"
                    )
                    if gan_active:
                        status += f" g_adv={gan_gen_loss_value:.4f} d={gan_discr_loss_value:.4f}"
                        if args.use_lecam:
                            status += f" lecam={gan_lecam_reg_value:.4f}"
                    advance_progress(progress, train_task, status=status)

                    if not use_live_progress:
                        elapsed = time.time() - start_time
                        log_console.print(f"[step {step:>6d}] {status} ({elapsed:.0f}s)")
                else:
                    advance_progress(progress, train_task)
            else:
                advance_progress(progress, train_task)

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
            if eval_due and is_main_process:
                eval_metrics, preview = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    focal_gamma=args.focal_gamma,
                    video_loss_weight=args.video_loss_weight,
                    ram_loss_weight=args.ram_loss_weight,
                    kl_weight=args.kl_weight,
                )
                eval_loss = eval_metrics["loss"]
                eval_metrics["type"] = "eval"
                eval_metrics["step"] = step
                eval_metrics["lr"] = float(scheduler.get_last_lr()[0])
                eval_metrics["train_grad_norm"] = grad_norm
                metrics.append(eval_metrics)
                save_metrics_json(output_dir / "metrics.json", metrics)

                eval_line = (
                    f"eval step={step:06d} loss={eval_loss:.5f} "
                    f"video={eval_metrics['video_recon_loss']:.5f} "
                    f"ram={eval_metrics['ram_recon_loss']:.5f} "
                    f"kl={eval_metrics['kl_loss']:.3f}"
                )

                if preview is not None:
                    save_video_preview(
                        preview_path(output_dir, split="eval_video", step=step),
                        preview[0],
                        preview[1],
                        palette,
                        max_frames=args.clip_frames,
                    )
                    save_ram_preview(
                        preview_path(output_dir, split="eval_ram", step=step),
                        preview[2],
                        preview[3],
                        title_prefix=f"Step {step} | ",
                    )

                if accelerator.num_processes == 1:
                    with torch.no_grad():
                        try:
                            train_index = int(np.random.randint(len(train_dataset)))
                            train_sample = train_dataset[train_index]
                            train_frames = train_sample["frames"].unsqueeze(0).to(device, non_blocking=True).long()
                            train_ram = train_sample["ram"].unsqueeze(0).to(device, non_blocking=True).float() / 255.0
                            with accelerator.autocast():
                                train_outputs = model(
                                    train_ram,
                                    output_video_shape=(
                                        train_frames.shape[1],
                                        train_frames.shape[2],
                                        train_frames.shape[3],
                                    ),
                                    sample_posterior=False,
                                )
                            save_video_preview(
                                preview_path(output_dir, split="train_video", step=step),
                                train_frames.detach().cpu(),
                                train_outputs.video_logits.detach().cpu(),
                                palette,
                                max_frames=args.clip_frames,
                            )
                            save_ram_preview(
                                preview_path(output_dir, split="train_ram", step=step),
                                train_ram.detach().cpu(),
                                train_outputs.ram_reconstruction.detach().float().cpu(),
                                title_prefix=f"Step {step} | ",
                            )
                            del train_outputs, train_ram, train_frames, train_sample
                        except torch.cuda.OutOfMemoryError:
                            log_console.print("[eval][OOM] Train preview OOM; skipping training preview images.")
                            if torch.cuda.is_available():
                                gc.collect()
                                torch.cuda.empty_cache()

                if eval_loss < best_eval:
                    best_eval = eval_loss
                    save_training_state(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                        accelerator=accelerator,
                        discriminator=discriminator,
                        discriminator_optimizer=discriminator_optimizer,
                        lecam_ema=lecam_ema,
                    )
                    eval_line += f" | best={best_eval:.6f}"

            if should_checkpoint and is_main_process:
                if eval_line is not None:
                    eval_line += " | saved latest"
                save_training_state(
                    output_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                    accelerator=accelerator,
                    discriminator=discriminator,
                    discriminator_optimizer=discriminator_optimizer,
                    lecam_ema=lecam_ema,
                )

            if eval_line is not None and is_main_process:
                log_console.print(eval_line)

            if eval_due or should_checkpoint:
                accelerator.wait_for_everyone()

    if is_main_process:
        save_metrics_json(output_dir / "metrics.json", metrics)
        console.print(f"Training complete. Best eval loss: {best_eval:.6f}")
        console.print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()