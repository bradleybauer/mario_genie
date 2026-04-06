#!/usr/bin/env python3
"""Train the RAM VAE on normalized NES RAM sequences.

Usage:
    python scripts/train_ram_vae.py --max-steps 10000
    python scripts/train_ram_vae.py --max-steps 50000 --hidden-dim 256 --latent-dim 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
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
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.normalized_dataset import NormalizedSequenceDataset
from models.ram_vae import RAMVAE
from system_info import collect_system_info, print_system_info
from training.training_utils import (
    ThroughputTracker,
    build_eval_loader,
    build_progress,
    build_replacement_train_loader,
    create_accelerator_runtime,
    get_model_state_dict,
    infinite_batches,
    save_json,
    save_metrics_json,
    split_train_eval_dataset,
    unwrap_model,
)

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
    parser = argparse.ArgumentParser(description="Train the RAM VAE.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-fc-blocks", type=int, default=2)
    parser.add_argument("--n-temporal-blocks", type=int, default=2)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


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
    run_name = args.run_name or datetime.now().strftime("ram_vae_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "checkpoints" / run_name


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


def save_ram_preview(
    path: Path,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    title_prefix: str = "",
) -> None:
    """Save a visual comparison of target vs reconstructed RAM for one sample."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Take first sample from batch: (T, N_bytes)
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


def evaluate(
    model: RAMVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    kl_weight: float,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    preview: tuple[torch.Tensor, torch.Tensor] | None = None

    with torch.no_grad():
        for batch in loader:
            ram = batch["ram"].to(device, non_blocking=True).float() / 255.0
            outputs = model(ram)
            recon_loss = F.mse_loss(outputs.reconstruction, ram)
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

            if preview is None:
                preview_out = model(ram[:1], sample_posterior=False)
                preview = (ram[:1].detach().cpu(), preview_out.reconstruction.detach().cpu())
                del preview_out

    model.train()

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
    accelerator: Accelerator | None = None,
) -> None:
    torch.save(
        {
            "model": get_model_state_dict(model, accelerator=accelerator),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_eval": best_eval,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = create_accelerator_runtime(output_dir=output_dir, mixed_precision="no")
    accelerator = runtime.accelerator
    device = runtime.device
    is_main_process = runtime.is_main_process

    if is_main_process:
        console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if is_main_process:
            console.print("[tf32] TF32 matmul precision enabled")

    system_info = collect_system_info()
    if is_main_process:
        print_system_info(system_info)

    # --- Dataset ---
    if is_main_process:
        console.print("[dataset] Building sequence index...")
    with accelerator.main_process_first():
        dataset = NormalizedSequenceDataset(
            data_dir=args.data_dir,
            clip_frames=args.clip_frames,
            include_frames=False,
            include_audio=False,
            include_actions=False,
            include_ram=True,
            num_workers=args.num_workers,
            system_info=system_info,
        )
    if is_main_process:
        console.print("[dataset] Index build complete.")
    if len(dataset) == 0:
        raise RuntimeError("No RAM training samples were found.")

    n_bytes = dataset.ram_n_bytes
    if n_bytes is None:
        # Probe from first file
        with np.load(dataset.data_files[0], mmap_mode="r") as npz:
            n_bytes = npz["ram"].shape[1]
    if is_main_process:
        console.print(
            f"Found {len(dataset):,} sequence segments of {args.clip_frames} frames, "
            f"{n_bytes} RAM bytes per frame."
        )
        console.print(
            f"Dataset: {dataset.num_files} files, {len(dataset):,} samples, "
            f"{_format_bytes(dataset.dataset_bytes)} on disk."
        )

    # --- Train/eval split ---
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

    # --- Model ---
    model = RAMVAE(
        n_bytes=n_bytes,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_fc_blocks=args.n_fc_blocks,
        n_temporal_blocks=args.n_temporal_blocks,
        temporal_kernel_size=args.temporal_kernel_size,
    ).to(device)

    if args.compile:
        if is_main_process:
            console.print("[compile] Compiling model with torch.compile()...")
        model = torch.compile(model)
        if is_main_process:
            console.print("[compile] Compilation complete.")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_steps = max(int(args.warmup_steps), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / float(warmup_steps)
        if args.max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(args.max_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_step = 0
    best_eval = float("inf")
    metrics: list[dict[str, float]] = []

    if args.resume_from is not None:
        with accelerator.main_process_first():
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
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

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    train_iter = infinite_batches(train_loader)

    num_params = sum(p.numel() for p in model.parameters())
    config = vars(args).copy()
    config.update(
        {
            "n_bytes": n_bytes,
            "num_parameters": int(num_params),
            "device": str(device),
            "mixed_precision": "no",
            "num_processes": int(accelerator.num_processes),
            "timestamp": datetime.now().isoformat(),
        }
    )
    if is_main_process:
        save_json(output_dir / "config.json", config, indent=2)
        console.print(f"Training RAM VAE on {len(train_dataset)} samples")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Device: {device}")
        console.print(f"Model parameters: {num_params:,}")
        console.print(f"RAM bytes: {n_bytes}, latent_dim: {args.latent_dim}")
        if accelerator.num_processes > 1:
            console.print(
                f"Distributed training enabled with {accelerator.num_processes} processes "
                f"(global batch={args.batch_size * accelerator.num_processes})."
            )

    # --- Training loop ---
    start_time = time.time()
    throughput_tracker = ThroughputTracker(window_steps=20)
    use_live_progress = sys.stdout.isatty() and is_main_process

    total_train_steps = max(args.max_steps - start_step, 0)
    with build_progress(use_live=use_live_progress) as progress:
        log_console = progress.console
        train_task = progress.add_task("Training", total=total_train_steps, status="")

        for step in range(start_step, args.max_steps):
            batch = next(train_iter)
            ram = batch["ram"].to(device, non_blocking=True).float() / 255.0

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                outputs = model(ram)
            recon_loss = F.mse_loss(outputs.reconstruction, ram)
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            loss = recon_loss + args.kl_weight * kl_loss

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=ram.shape[0] * accelerator.num_processes,
            )

            # --- Logging ---
            if step % args.log_interval == 0 or step == start_step:
                lr_current = scheduler.get_last_lr()[0]
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "grad_norm": grad_norm,
                    "lr": lr_current,
                    "samples_per_sec": round(samples_per_second, 1),
                    "steps_per_sec": round(steps_per_second, 2),
                }
                if is_main_process:
                    row.update(gpu_stats(device))
                    metrics.append(row)

                    status = (
                        f"loss={loss.item():.5f} recon={recon_loss.item():.5f} "
                        f"kl={kl_loss.item():.3f} lr={lr_current:.2e} "
                        f"gnorm={grad_norm:.2f} "
                        f"{samples_per_second:.0f} samp/s"
                    )
                    progress.update(train_task, advance=1, status=status)

                    if not use_live_progress:
                        elapsed = time.time() - start_time
                        log_console.print(
                            f"[step {step:>6d}] {status} ({elapsed:.0f}s)"
                        )
                else:
                    progress.update(train_task, advance=1, status="")
            else:
                progress.update(train_task, advance=1, status="")

            # --- Eval ---
            eval_due = eval_loader is not None and step > 0 and step % args.eval_interval == 0
            should_checkpoint = step > 0 and step % args.checkpoint_interval == 0
            if eval_due or should_checkpoint:
                accelerator.wait_for_everyone()

            if eval_due and is_main_process:
                eval_metrics, preview = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    kl_weight=args.kl_weight,
                )
                eval_loss = eval_metrics["loss"]
                eval_metrics["step"] = step
                eval_metrics["lr"] = float(scheduler.get_last_lr()[0])
                eval_metrics["train_gnorm"] = grad_norm
                metrics.append(eval_metrics)
                save_metrics_json(output_dir / "metrics.json", metrics)
                log_console.print(
                    f"[eval step {step}] loss={eval_loss:.5f} "
                    f"recon={eval_metrics['recon_loss']:.5f} "
                    f"kl={eval_metrics['kl_loss']:.3f}"
                )

                if preview is not None:
                    save_ram_preview(
                        output_dir / "previews" / f"step_{step:06d}.png",
                        preview[0],
                        preview[1],
                        title_prefix=f"Step {step} | ",
                    )

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
                    )
                    log_console.print(f"[eval] New best: {best_eval:.6f}")

            # --- Checkpoint ---
            if should_checkpoint and is_main_process:
                save_training_state(
                    output_dir / f"step_{step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                    accelerator=accelerator,
                )

            if eval_due or should_checkpoint:
                accelerator.wait_for_everyone()

    # --- Final save ---
    if is_main_process:
        save_training_state(
            output_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=args.max_steps - 1,
            best_eval=best_eval,
            metrics=metrics,
            accelerator=accelerator,
        )

        save_metrics_json(output_dir / "metrics.json", metrics)

        console.print(f"Training complete. Best eval loss: {best_eval:.6f}")
        console.print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
