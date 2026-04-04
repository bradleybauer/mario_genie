#!/usr/bin/env python3
"""Train an action-conditioned world model over frozen video VAE latents.

Iteration one scope:
- Video latents only (no audio, no RAM)
- Frozen pretrained video VAE encoder/decoder
- Next-latent prediction with teacher forcing
"""

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
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.ltx_video_vae import LTXVideoVAE
from mario_world_model.ltx_video_vae_v2 import LTXVideoVAEv2
from mario_world_model.normalized_dataset import NormalizedSequenceDataset, load_palette_tensor
from mario_world_model.system_info import collect_system_info, print_system_info
from mario_world_model.video_latent_world_model import VideoLatentWorldModel


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a video-latent-only world model.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--video-vae-checkpoint", type=str, required=True)
    parser.add_argument("--video-vae-config", type=str, default=None)
    parser.add_argument(
        "--video-vae-version",
        type=str,
        default="auto",
        choices=["auto", "v1", "v2"],
        help="VAE architecture to load. auto reads config model_version when available.",
    )

    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))

    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)

    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=64)

    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--onehot-dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--autocast-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tf16", action="store_true", help="Enable lower-precision float32 matmul on NVIDIA GPUs")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None)

    args = parser.parse_args()
    if args.clip_frames < 2:
        parser.error("--clip-frames must be >= 2")
    if args.context_frames < 1:
        parser.error("--context-frames must be >= 1")
    if args.context_frames >= args.clip_frames:
        parser.error("--context-frames must be smaller than --clip-frames")
    if args.max_frames < args.clip_frames:
        parser.error("--max-frames must be >= --clip-frames")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
    return args


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
    run_name = args.run_name or datetime.now().strftime("video_latent_world_model_%Y%m%d_%H%M%S")
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


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def infer_video_vae_config_path(args: argparse.Namespace) -> Path:
    if args.video_vae_config is not None:
        return Path(args.video_vae_config)
    ckpt_parent = Path(args.video_vae_checkpoint).resolve().parent
    candidate = ckpt_parent / "config.json"
    if not candidate.is_file():
        raise FileNotFoundError(
            "Could not infer --video-vae-config from checkpoint directory; pass --video-vae-config explicitly"
        )
    return candidate


def load_num_actions(data_dir: str | Path) -> int:
    info_path = Path(data_dir) / "actions.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing actions metadata: {info_path}")
    info = _load_json(info_path)
    num_actions = int(info.get("num_actions", 0))
    if num_actions <= 0:
        raise ValueError(f"Invalid num_actions in {info_path}")
    return num_actions


def load_video_vae(
    *,
    checkpoint_path: Path,
    config_path: Path,
    version_hint: str,
    num_colors: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    cfg = _load_json(config_path)
    model_version = str(cfg.get("model_version", "")).lower()

    if version_hint == "auto":
        if model_version in {"v1", "v2"}:
            version = model_version
        elif "v2" in checkpoint_path.name.lower() or "v2" in config_path.name.lower():
            version = "v2"
        else:
            version = "v1"
    else:
        version = version_hint

    patch_size = int(cfg.get("patch_size", 4))
    base_channels = int(cfg.get("base_channels", 64))
    latent_channels = int(cfg.get("latent_channels", 64))
    vae_num_colors = int(cfg.get("num_colors", num_colors))
    if vae_num_colors != num_colors:
        raise ValueError(
            f"VAE config expects num_colors={vae_num_colors} but palette has {num_colors}. "
            "Ensure data normalization palette matches the trained VAE."
        )

    if version == "v2":
        vae: torch.nn.Module = LTXVideoVAEv2(
            num_colors=vae_num_colors,
            patch_size=patch_size,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )
    elif version == "v1":
        vae = LTXVideoVAE(
            num_colors=vae_num_colors,
            patch_size=patch_size,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )
    else:
        raise ValueError(f"Unknown video VAE version: {version}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    summary = {
        "version": version,
        "patch_size": patch_size,
        "base_channels": base_channels,
        "latent_channels": latent_channels,
        "num_colors": vae_num_colors,
    }
    return vae, summary


def encode_video_latents(
    vae: torch.nn.Module,
    frames: torch.Tensor,
    *,
    num_colors: int,
    onehot_dtype: torch.dtype,
    onehot_buffer: torch.Tensor | None,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    inputs = frames_to_one_hot(frames, num_colors, dtype=onehot_dtype, out=onehot_buffer)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_enabled
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        mean, _ = vae.encode(inputs)
    return mean.float(), inputs


def compute_loss(predicted: torch.Tensor, target: torch.Tensor, *, huber_delta: float) -> torch.Tensor:
    return F.smooth_l1_loss(predicted, target, beta=huber_delta)


def autoregressive_rollout(
    model: VideoLatentWorldModel,
    *,
    context_latents: torch.Tensor,
    transition_actions: torch.Tensor,
    total_frames: int,
) -> torch.Tensor:
    """Roll out latents autoregressively to total_frames.

    transition_actions has shape (B, total_frames - 1) where action[t] drives t -> t+1.
    """
    if context_latents.shape[2] >= total_frames:
        return context_latents[:, :, :total_frames]

    rollout = context_latents
    while rollout.shape[2] < total_frames:
        current_steps = rollout.shape[2]
        actions_prefix = transition_actions[:, :current_steps]
        predicted_seq = model(rollout, actions_prefix)
        next_latent = predicted_seq[:, :, -1:]
        rollout = torch.cat((rollout, next_latent), dim=2)
    return rollout


def latents_to_frame_indices(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_enabled
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        logits = vae.decode(latents)
    return logits.argmax(dim=1)


def save_preview_image(
    path: Path,
    *,
    frames_gt: torch.Tensor,
    frames_tf: torch.Tensor,
    frames_rollout: torch.Tensor,
    palette: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    palette_u8 = (palette.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    gt_np = frames_gt.detach().cpu().numpy()
    tf_np = frames_tf.detach().cpu().numpy()
    ro_np = frames_rollout.detach().cpu().numpy()

    gt_rgb = palette_u8[gt_np]
    tf_rgb = palette_u8[tf_np]
    ro_rgb = palette_u8[ro_np]

    rows = [
        np.concatenate((gt_rgb[t], tf_rgb[t], ro_rgb[t]), axis=1)
        for t in range(gt_rgb.shape[0])
    ]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def evaluate(
    model: VideoLatentWorldModel,
    *,
    loader: DataLoader,
    vae: torch.nn.Module,
    device: torch.device,
    num_colors: int,
    huber_delta: float,
    onehot_dtype: torch.dtype,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    model.eval()
    losses: list[float] = []
    onehot_buffer: torch.Tensor | None = None
    preview_payload: dict[str, torch.Tensor] | None = None

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).long()
            actions = batch["actions"].to(device, non_blocking=True).long()

            latents, onehot_buffer = encode_video_latents(
                vae,
                frames,
                num_colors=num_colors,
                onehot_dtype=onehot_dtype,
                onehot_buffer=onehot_buffer,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )

            pred = model(latents[:, :, :-1], actions[:, :-1])
            target = latents[:, :, 1:]
            loss = compute_loss(pred, target, huber_delta=huber_delta)
            losses.append(loss.item())

            if preview_payload is None:
                preview_payload = {
                    "frames": frames[:1].detach().cpu(),
                    "actions": actions[:1].detach().cpu(),
                    "latents": latents[:1].detach().cpu(),
                }

    model.train()
    if not losses:
        return {"latent_huber": 0.0}, preview_payload
    return {"latent_huber": float(np.mean(losses))}, preview_payload


def save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    best_eval: float,
    metrics: list[dict[str, float]],
) -> None:
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best_eval": best_eval,
        "metrics": metrics,
    }
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        if args.tf16:
            torch.set_float32_matmul_precision("medium")
            console.print("[tf16] TensorFloat matmul enabled")
        else:
            torch.set_float32_matmul_precision("high")
            console.print("[tf32] TensorFloat matmul enabled")
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
        console.print("[autocast] Requested but CUDA unavailable; disabling")
    if autocast_enabled and autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        console.print("[autocast] bfloat16 unsupported on this GPU; falling back to float16")
        autocast_dtype = torch.float16

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

    console.print("[dataset] Building sequence index...")
    dataset = NormalizedSequenceDataset(
        data_dir=args.data_dir,
        clip_frames=args.clip_frames,
        include_frames=True,
        include_actions=True,
        include_audio=False,
        num_workers=args.num_workers,
        system_info=system_info,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training samples found")
    console.print(f"[dataset] {len(dataset):,} clips loaded")

    eval_dataset = None
    train_dataset = dataset
    if args.eval_samples > 0 and len(dataset) > args.eval_samples:
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(dataset), generator=generator)
        eval_indices = permutation[: args.eval_samples].tolist()
        train_indices = permutation[args.eval_samples :].tolist()
        eval_dataset = Subset(dataset, eval_indices)
        train_dataset = Subset(dataset, train_indices)
        console.print(f"Eval split: {len(eval_dataset)} eval / {len(train_dataset)} train")

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
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    palette = load_palette_tensor(args.data_dir)
    num_colors = int(palette.shape[0])
    num_actions = load_num_actions(args.data_dir)
    console.print(f"[data] num_colors={num_colors}, num_actions={num_actions}")

    vae_config_path = infer_video_vae_config_path(args)
    vae_checkpoint_path = Path(args.video_vae_checkpoint)
    video_vae, video_vae_summary = load_video_vae(
        checkpoint_path=vae_checkpoint_path,
        config_path=vae_config_path,
        version_hint=args.video_vae_version,
        num_colors=num_colors,
        device=device,
    )
    latent_channels = int(video_vae_summary["latent_channels"])
    console.print(
        "[vae] loaded "
        f"version={video_vae_summary['version']} "
        f"latent_channels={latent_channels}"
    )

    model = VideoLatentWorldModel(
        latent_channels=latent_channels,
        num_actions=num_actions,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        max_frames=args.max_frames,
    ).to(device)

    if args.compile:
        console.print("[compile] Compiling world model with torch.compile()")
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    start_step = 0
    best_eval = float("inf")
    metrics: list[dict[str, float]] = []

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"]) + 1
        best_eval = float(checkpoint.get("best_eval", float("inf")))
        metrics = list(checkpoint.get("metrics", []))
        console.print(f"[resume] Resumed at step {start_step}")
        if start_step >= args.max_steps:
            console.print("[resume] max steps already reached")
            return

    config = vars(args).copy()
    config.update(
        {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "num_colors": num_colors,
            "num_actions": num_actions,
            "latent_channels": latent_channels,
            "video_vae": video_vae_summary,
            "video_vae_checkpoint": str(vae_checkpoint_path),
            "video_vae_config": str(vae_config_path),
            "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        }
    )
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    console.print(f"Output directory: {output_dir}")
    console.print(f"World model parameters: {config['num_parameters']:,}")

    total_train_steps = max(args.max_steps - start_step, 0)
    use_live_progress = sys.stdout.isatty()
    start_time = time.time()

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
            actions = batch["actions"].to(device, non_blocking=True).long()

            latents, onehot_buffer = encode_video_latents(
                video_vae,
                frames,
                num_colors=num_colors,
                onehot_dtype=onehot_dtype,
                onehot_buffer=onehot_buffer,
                autocast_enabled=autocast_enabled,
                autocast_dtype=autocast_dtype,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                predicted_next = model(latents[:, :, :-1], actions[:, :-1])
                target_next = latents[:, :, 1:]
                loss = compute_loss(predicted_next, target_next, huber_delta=args.huber_delta)

            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()
                optimizer.step()
            scheduler.step()

            status = (
                f"loss={loss.item():.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"gnorm={grad_norm:.2f}"
            )
            progress.update(train_task, advance=1, status=status)

            if ((args.log_interval > 0 and step % args.log_interval == 0) or step == args.max_steps - 1) and not use_live_progress:
                elapsed = max(time.time() - start_time, 1e-6)
                samples_per_sec = ((step - start_step + 1) * args.batch_size) / elapsed
                log_console.print(
                    f"step={step:06d} loss={loss.item():.5f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} gnorm={grad_norm:.2f} "
                    f"samples/s={samples_per_sec:.1f}"
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

                eval_metrics, preview_payload = evaluate(
                    model,
                    loader=eval_loader,
                    vae=video_vae,
                    device=device,
                    num_colors=num_colors,
                    huber_delta=args.huber_delta,
                    onehot_dtype=onehot_dtype,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                )
                eval_metrics["step"] = step
                eval_metrics["lr"] = float(scheduler.get_last_lr()[0])
                metrics.append(eval_metrics)
                with (output_dir / "metrics.json").open("w") as handle:
                    json.dump(metrics, handle, indent=2)

                eval_line = f"eval step={step:06d} latent_huber={eval_metrics['latent_huber']:.5f}"

                if preview_payload is not None:
                    with torch.no_grad():
                        frames_preview = preview_payload["frames"].to(device=device, non_blocking=True).long()
                        actions_preview = preview_payload["actions"].to(device=device, non_blocking=True).long()
                        latents_preview = preview_payload["latents"].to(device=device, non_blocking=True)

                        teacher_next = model(latents_preview[:, :, :-1], actions_preview[:, :-1])
                        teacher_full = latents_preview.clone()
                        teacher_full[:, :, 1:] = teacher_next

                        transition_actions = actions_preview[:, : args.clip_frames - 1]
                        context = min(args.context_frames, args.clip_frames - 1)
                        rollout_full = autoregressive_rollout(
                            model,
                            context_latents=latents_preview[:, :, :context],
                            transition_actions=transition_actions,
                            total_frames=args.clip_frames,
                        )

                        teacher_frames = latents_to_frame_indices(
                            video_vae,
                            teacher_full,
                            autocast_enabled=autocast_enabled,
                            autocast_dtype=autocast_dtype,
                        )
                        rollout_frames = latents_to_frame_indices(
                            video_vae,
                            rollout_full,
                            autocast_enabled=autocast_enabled,
                            autocast_dtype=autocast_dtype,
                        )

                        save_preview_image(
                            output_dir / f"preview_step_{step:06d}.png",
                            frames_gt=frames_preview[0],
                            frames_tf=teacher_frames[0],
                            frames_rollout=rollout_frames[0],
                            palette=palette,
                        )

                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                if eval_metrics["latent_huber"] < best_eval:
                    best_eval = eval_metrics["latent_huber"]
                    save_training_state(
                        output_dir / "video_latent_world_model_best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                    )
                    eval_line += f" | best={best_eval:.5f}"

                log_console.print(eval_line)

            should_checkpoint = (
                (args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0)
                or step == args.max_steps - 1
            )
            if should_checkpoint:
                save_training_state(
                    output_dir / "video_latent_world_model_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                )


if __name__ == "__main__":
    main()
