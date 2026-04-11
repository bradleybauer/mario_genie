#!/usr/bin/env python3
"""Train the RAM VAE on normalized NES RAM sequences.

Usage:
    python scripts/train_ram_vae.py --max-steps 10000
    python scripts/train_ram_vae.py --max-steps 50000 --hidden-dim 256 --latent-dim 32
"""
from __future__ import annotations

import argparse
import json
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

from src.plot_style import apply_plot_style
apply_plot_style()

from src.data.normalized_dataset import NormalizedSequenceDataset
from src.models.ram_vae import RAMVAE
from src.system_info import collect_system_info, print_system_info
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
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RAM VAE.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0 = standard CE).")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=8, help="Per-address embedding dimension.")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability used in model residual blocks.",
    )
    parser.add_argument("--n-fc-blocks", type=int, default=2)
    parser.add_argument("--n-temporal-blocks", type=int, default=2)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument(
        "--temporal-downsample",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable 2× temporal downsampling in latent (0=off, 1=on).",
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
    args = parser.parse_args()
    if not (0.0 <= args.dropout < 1.0):
        parser.error("--dropout must be in [0, 1)")
    return args


def save_ram_preview(
    path: Path,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    title_prefix: str = "",
) -> None:
    """Save a visual comparison of target vs reconstructed RAM for one sample."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Take first sample from batch: (T, N_addr) — raw byte values
    tgt = target[0].detach().cpu().float().numpy()
    rec = reconstruction[0].detach().cpu().float().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

    im0 = axes[0].imshow(tgt.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=255)
    axes[0].set_title(f"{title_prefix}Target RAM" if title_prefix else "Target RAM")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Address index")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)

    im1 = axes[1].imshow(rec.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=255)
    axes[1].set_title(f"{title_prefix}Reconstructed RAM" if title_prefix else "Reconstructed RAM")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Address index")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)

    diff = np.abs(tgt - rec)
    im2 = axes[2].imshow(diff.T, origin="lower", aspect="auto", cmap="hot", vmin=0, vmax=128)
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Address index")
    fig.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.01)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_ram_vae_losses(
    model: RAMVAE,
    outputs,
    target_ram: torch.Tensor,
    *,
    focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute categorical (focal) cross-entropy reconstruction loss and KL."""
    recon_loss = model.categorical_loss(outputs.logits.float(), target_ram, gamma=focal_gamma)
    kl_loss = model.kl_loss(outputs.posterior_mean.float(), outputs.posterior_logvar.float())
    return recon_loss, kl_loss


def evaluate(
    model: RAMVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    kl_weight: float,
    focal_gamma: float = 0.0,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    preview: tuple[torch.Tensor, torch.Tensor] | None = None

    with torch.no_grad():
        for batch in loader:
            ram = batch["ram"].to(device, non_blocking=True)
            outputs = model(ram)
            recon_loss, kl_loss = compute_ram_vae_losses(model, outputs, ram, focal_gamma=focal_gamma)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

            if preview is None:
                preview_out = model(ram[:1], sample_posterior=False)
                preview = (
                    ram[:1].detach().cpu(),
                    preview_out.reconstruction.detach().cpu(),
                )
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

    output_dir = make_output_dir(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        run_name=args.run_name,
        default_prefix="ram_vae",
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

    # Load per-address value vocabularies from ram_addresses.json
    ram_json_path = Path(args.data_dir) / "ram_addresses.json"
    with open(ram_json_path) as f:
        ram_addr_info = json.load(f)
    values_per_address: list[list[int]] = ram_addr_info["values_per_address"]
    n_addresses = len(values_per_address)
    if is_main_process:
        cardinalities = [len(v) for v in values_per_address]
        console.print(
            f"Found {len(dataset):,} sequence segments of {args.clip_frames} frames, "
            f"{n_addresses} RAM addresses per frame "
            f"(total {sum(cardinalities)} categorical classes)."
        )
        console.print(
            f"Dataset: {dataset.num_files} files, {len(dataset):,} samples, "
            f"{format_bytes(dataset.dataset_bytes)} on disk."
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
        values_per_address=values_per_address,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_fc_blocks=args.n_fc_blocks,
        n_temporal_blocks=args.n_temporal_blocks,
        temporal_kernel_size=args.temporal_kernel_size,
        temporal_downsample=args.temporal_downsample,
        dropout=args.dropout,
    ).to(device)

    if args.compile:
        if is_main_process:
            console.print("[compile] Compiling model with torch.compile()...")
        model = torch.compile(model)
        if is_main_process:
            console.print("[compile] Compilation complete.")

    optimizer = AdamW(model.parameters(), lr=args.lr)
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
    config = build_trainer_config(
        model_name="ram_vae",
        args=args,
        device=device,
        mixed_precision=args.mixed_precision,
        num_processes=accelerator.num_processes,
        data={
            "n_addresses": n_addresses,
            "total_classes": int(sum(len(v) for v in values_per_address)),
            "clip_frames": int(args.clip_frames),
        },
        model={
            "num_parameters": int(num_params),
            "embed_dim": int(args.embed_dim),
            "hidden_dim": int(args.hidden_dim),
            "latent_dim": int(args.latent_dim),
            "dropout": float(args.dropout),
            "n_fc_blocks": int(args.n_fc_blocks),
            "n_temporal_blocks": int(args.n_temporal_blocks),
            "temporal_kernel_size": int(args.temporal_kernel_size),
            "temporal_downsample": int(args.temporal_downsample),
        },
    )
    if is_main_process:
        save_json(output_dir / "config.json", config, indent=2)
        console.print(f"Training RAM VAE on {len(train_dataset)} samples")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Device: {device}")
        console.print(f"Model parameters: {num_params:,}")
        console.print(f"RAM addresses: {n_addresses}, embed_dim: {args.embed_dim}, latent_dim: {args.latent_dim}")
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
            ram = batch["ram"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                outputs = model(ram)
            recon_loss, kl_loss = compute_ram_vae_losses(model, outputs, ram, focal_gamma=args.focal_gamma)
            loss = recon_loss + args.kl_weight * kl_loss

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=ram.shape[0] * accelerator.num_processes,
            )

            # --- Logging ---
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
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "grad_norm": grad_norm,
                    "lr": float(lr_current),
                    "samples_per_sec": round(samples_per_second, 1),
                    "steps_per_sec": round(steps_per_second, 2),
                }
                if is_main_process:
                    train_row.update(gpu_stats(device))
                    metrics.append(train_row)

                    status = (
                        f"loss={loss.item():.5f} recon={recon_loss.item():.5f} "
                        f"kl={kl_loss.item():.3f} lr={lr_current:.2e} "
                        f"gnorm={grad_norm:.2f} "
                        f"{samples_per_second:.0f} samp/s"
                    )
                    advance_progress(progress, train_task, status=status)

                    if not use_live_progress:
                        elapsed = time.time() - start_time
                        log_console.print(
                            f"[step {step:>6d}] {status} ({elapsed:.0f}s)"
                        )
                else:
                    advance_progress(progress, train_task)
            else:
                advance_progress(progress, train_task)

            # --- Eval ---
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
                    kl_weight=args.kl_weight,
                    focal_gamma=args.focal_gamma,
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
                    f"recon={eval_metrics['recon_loss']:.5f} "
                    f"kl={eval_metrics['kl_loss']:.3f}"
                )

                if preview is not None:
                    save_ram_preview(
                        preview_path(output_dir, split="eval", step=step),
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
                    eval_line += f" | best={best_eval:.6f}"

            # --- Checkpoint ---
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
