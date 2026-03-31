#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.audio_features import LogMelSpectrogram, frame_audio_to_waveform, mel_time_frequency_shape
from mario_world_model.config import AUDIO_FMAX, AUDIO_FMIN, AUDIO_HOP_LENGTH, AUDIO_N_FFT, AUDIO_N_MELS, AUDIO_SAMPLE_RATE
from mario_world_model.ltx_audio_vae import LTXAudioVAE
from mario_world_model.normalized_dataset import NormalizedSequenceDataset
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
    parser = argparse.ArgumentParser(description="Train the LTX-style mel audio VAE.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=AUDIO_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=AUDIO_N_FFT)
    parser.add_argument("--hop-length", type=int, default=AUDIO_HOP_LENGTH)
    parser.add_argument("--n-mels", type=int, default=AUDIO_N_MELS)
    parser.add_argument("--fmin", type=float, default=AUDIO_FMIN)
    parser.add_argument("--fmax", type=float, default=AUDIO_FMAX)
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
    run_name = args.run_name or datetime.now().strftime("ltx_audio_vae_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "checkpoints" / run_name


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(prediction, target)
    diff = (prediction - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def build_mel_mask(
    waveform_lengths: torch.Tensor,
    *,
    max_time_steps: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    mask = torch.zeros((waveform_lengths.shape[0], 1, max_time_steps, 1), dtype=torch.float32, device=waveform_lengths.device)
    for batch_idx, length in enumerate(waveform_lengths.tolist()):
        valid_steps, _ = mel_time_frequency_shape(int(length), n_fft=n_fft, hop_length=hop_length, center=False)
        mask[batch_idx, :, : min(valid_steps, max_time_steps)] = 1.0
    return mask


def save_mel_preview(path: Path, mel: torch.Tensor, reconstruction: torch.Tensor, *, title_prefix: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mel_np = mel[0, 0].detach().cpu().numpy().T
    recon_np = reconstruction[0, 0].detach().cpu().numpy().T

    shared_min = min(mel_np.min(), recon_np.min())
    shared_max = max(mel_np.max(), recon_np.max())
    if shared_max - shared_min < 1e-6:
        shared_max = shared_min + 1.0

    target_title = f"{title_prefix}Target log-mel" if title_prefix else "Target log-mel"
    recon_title = f"{title_prefix}Reconstructed log-mel" if title_prefix else "Reconstructed log-mel"

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    im0 = axes[0].imshow(mel_np, origin="lower", aspect="auto", cmap="magma", vmin=shared_min, vmax=shared_max)
    axes[0].set_title(target_title)
    figure.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.01)
    im1 = axes[1].imshow(recon_np, origin="lower", aspect="auto", cmap="magma", vmin=shared_min, vmax=shared_max)
    axes[1].set_title(recon_title)
    figure.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.01)
    for axis in axes:
        axis.set_xlabel("Time")
        axis.set_ylabel("Mel bin")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluate(
    model: LTXAudioVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    mel_extractor: LogMelSpectrogram,
    n_fft: int,
    hop_length: int,
    kl_weight: float,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    preview: tuple[torch.Tensor, torch.Tensor] | None = None
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True).float() / 32768.0
            audio_lengths = batch["audio_lengths"]
            all_full_length = bool(torch.all(audio_lengths == audio.shape[-1]))
            needs_mask = False

            if all_full_length:
                waveform = audio.reshape(audio.shape[0], -1)
                waveform_lengths = audio_lengths.sum(dim=1)
                mask = None
            else:
                waveform = frame_audio_to_waveform(audio, audio_lengths)
                waveform_lengths = audio_lengths.sum(dim=1)
                needs_mask = True
                mask = None

            mel = mel_extractor(waveform)
            if needs_mask:
                mask = build_mel_mask(
                    waveform_lengths.to(device),
                    max_time_steps=mel.shape[2],
                    n_fft=n_fft,
                    hop_length=hop_length,
                )
            outputs = model(mel)
            recon_loss = masked_l1_loss(
                outputs.reconstruction,
                mel,
                None if mask is None else mask.expand_as(mel),
            )
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            if preview is None:
                preview = (mel.detach().cpu(), outputs.reconstruction.detach().cpu())
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
    model: LTXAudioVAE,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    step: int,
    best_eval: float,
    metrics: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "best_eval": best_eval,
            "metrics": metrics,
        },
        path,
    )


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
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        console.print("[tf32] TF32 matmul precision enabled")

    system_info = collect_system_info()
    print_system_info(system_info)

    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[dataset] Building sequence index (can take a few minutes on large runs)...")
    dataset = NormalizedSequenceDataset(
        data_dir=args.data_dir,
        clip_frames=args.clip_frames,
        include_audio=True,
        num_workers=args.num_workers,
        system_info=system_info,
    )
    console.print("[dataset] Index build complete.")
    if len(dataset) == 0:
        raise RuntimeError("No audio training samples were found.")
    console.print(
        f"Found {len(dataset)} sequence segments of {args.clip_frames} frames "
        f"(audio_frame_size={int(dataset.audio_frame_size or 0)})."
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
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    mel_extractor = LogMelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=False,
    ).to(device)
    model = LTXAudioVAE(
        in_channels=1,
        n_mels=args.n_mels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
    ).to(device)

    if args.compile:
        console.print("[compile] Compiling the model with torch.compile()...")
        model = torch.compile(model)
        console.print("[compile] Compilation complete.")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    start_step = 0
    best_eval = float("inf")
    metrics: list[dict[str, float]] = []

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
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
            "audio_frame_size": int(dataset.audio_frame_size or 0),
            "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
        }
    )
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    console.print(f"Config saved to {output_dir / 'config.json'}")
    console.print(json.dumps(config, indent=2))

    console.print(f"Training LTX audio VAE on {len(train_dataset)} samples")
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
        for step in range(start_step, args.max_steps):
            batch = next(train_iter)
            audio = batch["audio"].to(device, non_blocking=True).float() / 32768.0
            audio_lengths = batch["audio_lengths"]
            all_full_length = bool(torch.all(audio_lengths == audio.shape[-1]))

            # Fast path: if all frame lengths are full-width, a reshape is enough.
            if all_full_length:
                waveform = audio.reshape(audio.shape[0], -1)
                waveform_lengths = audio_lengths.sum(dim=1)
            else:
                waveform = frame_audio_to_waveform(audio, audio_lengths)
                waveform_lengths = audio_lengths.sum(dim=1)

            mel = mel_extractor(waveform)

            mask = None
            if not all_full_length:
                mask = build_mel_mask(
                    waveform_lengths.to(device),
                    max_time_steps=mel.shape[2],
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(mel)
            recon_loss = masked_l1_loss(
                outputs.reconstruction,
                mel,
                None if mask is None else mask.expand_as(mel),
            )
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            loss = recon_loss + args.kl_weight * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            perf_window_steps += 1
            perf_window_samples += audio.shape[0]
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
                f"samp/s={samples_per_second:.0f} step/s={steps_per_second:.2f}"
            )
            stats = gpu_stats(device)
            if "gpu_util_pct" in stats:
                status += f" gpu={stats['gpu_util_pct']:.0f}%"
            if "gpu_mem_pct" in stats:
                status += f" mem={stats['gpu_mem_pct']:.0f}%"
            if "gpu_temp_c" in stats:
                status += f" temp={stats['gpu_temp_c']:.0f}C"

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
                    f"ex/s={examples_per_second:.1f}"
                )

            should_eval = (
                eval_loader is not None
                and (
                    (args.eval_interval > 0 and (step + 1) % args.eval_interval == 0)
                    or step == args.max_steps - 1
                )
            )
            if should_eval:
                eval_metrics, preview = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    mel_extractor=mel_extractor,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    kl_weight=args.kl_weight,
                )
                eval_metrics["step"] = step
                metrics.append(eval_metrics)
                with (output_dir / "metrics.json").open("w") as handle:
                    json.dump(metrics, handle, indent=2)

                log_console.print(
                    f"eval step={step:06d} loss={eval_metrics['loss']:.4f} "
                    f"recon={eval_metrics['recon_loss']:.4f} kl={eval_metrics['kl_loss']:.4f}"
                )

                if preview is not None:
                    save_mel_preview(output_dir / f"preview_step_{step:06d}.png", preview[0], preview[1])

                # Random training sample preview
                with torch.no_grad():
                    train_batch = next(train_iter)
                    train_audio = train_batch["audio"].to(device, non_blocking=True).float() / 32768.0
                    train_audio_lengths = train_batch["audio_lengths"]
                    if bool(torch.all(train_audio_lengths == train_audio.shape[-1])):
                        train_waveform = train_audio.reshape(train_audio.shape[0], -1)
                    else:
                        train_waveform = frame_audio_to_waveform(train_audio, train_audio_lengths)
                    train_mel = mel_extractor(train_waveform)
                    train_outputs = model(train_mel)
                    save_mel_preview(
                        output_dir / f"train_preview_step_{step:06d}.png",
                        train_mel.detach().cpu(),
                        train_outputs.reconstruction.detach().cpu(),
                        title_prefix="[Train] ",
                    )

                if eval_metrics["loss"] < best_eval:
                    best_eval = eval_metrics["loss"]
                    save_training_state(
                        output_dir / "audio_vae_best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                    )
                    log_console.print(
                        f"[checkpoint] New best eval_loss={best_eval:.6f} -> saved audio_vae_best.pt"
                    )

            should_checkpoint = (
                (args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0)
                or step == args.max_steps - 1
            )
            if should_checkpoint:
                log_console.print(f"[checkpoint] Saving latest weights at step {step}")
                save_training_state(
                    output_dir / "audio_vae_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                )


if __name__ == "__main__":
    main()