#!/usr/bin/env python3
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
from accelerate import Accelerator
from rich.console import Console
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.gan_discriminator import build_mel_discriminator, count_trainable_parameters
from src.training.gan_training import LeCAMEMA, hinge_discriminator_loss, hinge_generator_loss, set_requires_grad
from src.data.audio_features import LogMelSpectrogram, frame_audio_to_waveform
from src.config import AUDIO_FMAX, AUDIO_FMIN, AUDIO_HOP_LENGTH, AUDIO_N_FFT, AUDIO_N_MELS, AUDIO_SAMPLE_RATE
from src.models.audio_vae import AudioVAE
from src.data.normalized_dataset import NormalizedSequenceDataset
from src.system_info import collect_system_info, print_system_info
from src.training.audio_training_helpers import build_mel_mask, context_waveform_lengths, masked_l1_loss
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
    save_json,
    save_metrics_json,
    split_train_eval_dataset,
    unwrap_model,
)


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mel audio VAE.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument(
        "--context-frames",
        type=int,
        default=0,
        help="Number of leading clip frames used as context; reconstruction loss ignores this prefix.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear warmup from 0 to --lr over this many steps (default: 1000)")
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
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode passed to Accelerator.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--use-gan",
        action="store_true",
        help="Enable adversarial training with a compact 2D mel-spectrogram discriminator.",
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
        parser.error("--context-frames must be non-negative")
    if args.context_frames >= args.clip_frames:
        parser.error("--context-frames must be smaller than --clip-frames")

    return args


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
    model: AudioVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    mel_extractor: LogMelSpectrogram,
    n_fft: int,
    hop_length: int,
    kl_weight: float,
    context_frames: int = 0,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    preview: tuple[torch.Tensor, torch.Tensor] | None = None
    preview_energy = -1.0
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True).float() / 32768.0
            audio_lengths = batch["audio_lengths"]
            all_full_length = bool(torch.all(audio_lengths == audio.shape[-1]))

            if all_full_length:
                waveform = audio.reshape(audio.shape[0], -1)
            else:
                waveform = frame_audio_to_waveform(audio, audio_lengths)

            waveform_lengths = audio_lengths.sum(dim=1).to(device)
            context_lengths = context_waveform_lengths(
                audio_lengths,
                context_frames=context_frames,
            ).to(device)

            mel = mel_extractor(waveform)
            valid_mask = None
            if not all_full_length:
                valid_mask = build_mel_mask(
                    waveform_lengths,
                    max_time_steps=mel.shape[2],
                    n_fft=n_fft,
                    hop_length=hop_length,
                )

            recon_mask = valid_mask
            if context_frames > 0:
                recon_mask = build_mel_mask(
                    waveform_lengths,
                    max_time_steps=mel.shape[2],
                    n_fft=n_fft,
                    hop_length=hop_length,
                    context_lengths=context_lengths,
                )

            outputs = model(mel)
            recon_loss = masked_l1_loss(
                outputs.reconstruction,
                mel,
                None if recon_mask is None else recon_mask.expand_as(mel),
            )
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            energy = float(mel[0].mean())
            if energy > preview_energy:
                preview_energy = energy
                preview_out = model(mel[:1], sample_posterior=False)
                preview = (mel[:1].detach().cpu(), preview_out.reconstruction.detach().cpu())
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
    discriminator: torch.nn.Module | None = None,
    discriminator_optimizer: AdamW | None = None,
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
        default_prefix="audio_vae",
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
        console.print("[dataset] Building sequence index (can take a few minutes on large runs)...")
    with accelerator.main_process_first():
        dataset = NormalizedSequenceDataset(
            data_dir=args.data_dir,
            clip_frames=args.clip_frames,
            include_frames=False,
            include_audio=True,
            num_workers=args.num_workers,
            system_info=system_info,
        )
    if is_main_process:
        console.print("[dataset] Index build complete.")
    if len(dataset) == 0:
        raise RuntimeError("No audio training samples were found.")

    target_frames = args.clip_frames - args.context_frames
    if is_main_process:
        if args.context_frames > 0:
            console.print(
                f"Found {len(dataset)} sequence segments of {args.clip_frames} frames "
                f"({args.context_frames} context + {target_frames} target, "
                f"audio_frame_size={int(dataset.audio_frame_size or 0)})."
            )
            console.print(
                f"[context] Reconstruction losses ignore the first {args.context_frames} frame(s) of each clip."
            )
        else:
            console.print(
                f"Found {len(dataset)} sequence segments of {args.clip_frames} frames "
                f"(audio_frame_size={int(dataset.audio_frame_size or 0)})."
            )
        console.print(
            f"Dataset: {dataset.num_files} files, {len(dataset):,} samples, "
            f"{format_bytes(dataset.dataset_bytes)} on disk."
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

    mel_extractor = LogMelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        center=False,
    ).to(device)
    model = AudioVAE(
        in_channels=1,
        n_mels=args.n_mels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
    ).to(device)
    discriminator = None
    discriminator_optimizer = None
    lecam_ema = None
    discriminator_num_parameters = 0
    if args.use_gan:
        discriminator = build_mel_discriminator(
            in_channels=1,
            target_size=args.gan_target_size,
        ).to(device)
        discriminator_num_parameters = count_trainable_parameters(discriminator)
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=args.gan_lr)
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
            console.print("[compile] Compiling the model with torch.compile()...")
        model = torch.compile(model)
        if discriminator is not None:
            if is_main_process:
                console.print("[compile] Compiling discriminator with torch.compile()...")
            discriminator = torch.compile(discriminator)
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
    if warmup_steps > 0:
        if is_main_process:
            console.print(f"[lr] Warmup: {warmup_steps} steps -> cosine decay to 10% of peak")
    else:
        if is_main_process:
            console.print("[lr] Cosine decay (no warmup)")

    start_step = 0
    best_eval = float("inf")
    metrics: list[dict[str, float]] = []

    if args.resume_from is not None:
        with accelerator.main_process_first():
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
            if "discriminator" in checkpoint and "discriminator_optimizer" in checkpoint:
                discriminator.load_state_dict(checkpoint["discriminator"])
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
                console.print(
                    f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do."
                )
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

    num_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    config = build_trainer_config(
        model_name="audio_vae",
        args=args,
        device=device,
        mixed_precision=args.mixed_precision,
        num_processes=accelerator.num_processes,
        data={
            "audio_frame_size": int(dataset.audio_frame_size or 0),
            "clip_frames": int(args.clip_frames),
            "context_frames": int(args.context_frames),
        },
        model={
            "num_parameters": num_parameters,
            "discriminator_parameters": int(discriminator_num_parameters),
            "base_channels": int(args.base_channels),
            "latent_channels": int(args.latent_channels),
        },
    )
    if is_main_process:
        save_json(output_dir / "config.json", config, indent=2)
        console.print(f"Config saved to {output_dir / 'config.json'}")
        console.print(json.dumps(config, indent=2))

        console.print(f"Training audio VAE on {len(train_dataset)} samples")
        console.print(f"Output directory: {output_dir}")
        console.print(f"Device: {device}")
        console.print(f"Model parameters: {num_parameters:,}")
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
        for step in range(start_step, args.max_steps):
            batch = next(train_iter)
            audio = batch["audio"].to(device, non_blocking=True).float() / 32768.0
            audio_lengths = batch["audio_lengths"]
            all_full_length = bool(torch.all(audio_lengths == audio.shape[-1]))
            context_lengths = context_waveform_lengths(
                audio_lengths,
                context_frames=args.context_frames,
            ).to(device)
            gan_active = args.use_gan and step >= args.gan_start_step
            gan_discr_loss_value = 0.0
            gan_gen_loss_value = 0.0
            gan_real_logit_value = 0.0
            gan_fake_logit_value = 0.0
            gan_lecam_reg_value = 0.0

            if all_full_length:
                waveform = audio.reshape(audio.shape[0], -1)
            else:
                waveform = frame_audio_to_waveform(audio, audio_lengths)

            waveform_lengths = audio_lengths.sum(dim=1).to(device)
            mel = mel_extractor(waveform)

            valid_mask = None
            if not all_full_length:
                valid_mask = build_mel_mask(
                    waveform_lengths,
                    max_time_steps=mel.shape[2],
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                )

            recon_mask = valid_mask
            if args.context_frames > 0:
                recon_mask = build_mel_mask(
                    waveform_lengths,
                    max_time_steps=mel.shape[2],
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    context_lengths=context_lengths,
                )

            mel_for_discriminator = mel
            if valid_mask is not None:
                mel_for_discriminator = (
                    mel_for_discriminator
                    * valid_mask.to(dtype=mel_for_discriminator.dtype).expand_as(mel_for_discriminator)
                )

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                outputs = model(mel)
            recon_loss = masked_l1_loss(
                outputs.reconstruction,
                mel,
                None if recon_mask is None else recon_mask.expand_as(mel),
            )
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            loss = recon_loss + args.kl_weight * kl_loss

            fake_mel_detached = None
            if gan_active and discriminator is not None:
                fake_mel = outputs.reconstruction
                if valid_mask is not None:
                    fake_mel = fake_mel * valid_mask.to(dtype=fake_mel.dtype).expand_as(fake_mel)
                fake_mel_detached = fake_mel.detach()

                set_requires_grad(discriminator, False)
                with accelerator.autocast():
                    gen_adv_loss = hinge_generator_loss(discriminator(fake_mel))
                gan_gen_loss_value = gen_adv_loss.item()
                loss = loss + args.gan_weight * gen_adv_loss
                del fake_mel, gen_adv_loss

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()

            if gan_active and discriminator is not None and discriminator_optimizer is not None:
                set_requires_grad(discriminator, True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    real_scores = discriminator(mel_for_discriminator)
                    fake_scores = discriminator(fake_mel_detached)
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
                del real_scores, fake_scores, discr_loss, fake_mel_detached

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=audio.shape[0] * accelerator.num_processes,
            )

            if is_main_process:
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
                if gan_active:
                    status += f" g_adv={gan_gen_loss_value:.4f} d={gan_discr_loss_value:.4f}"
                    if args.use_lecam:
                        status += f" lecam={gan_lecam_reg_value:.4f}"

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
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "grad_norm": grad_norm,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "samples_per_sec": round(samples_per_second, 1),
                    "steps_per_sec": round(steps_per_second, 2),
                }
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

            if log_due and not use_live_progress and is_main_process:
                elapsed = time.time() - start_time
                examples_per_second = (
                    (step - start_step + 1)
                    * args.batch_size
                    * accelerator.num_processes
                    / max(elapsed, 1e-6)
                )
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
            should_eval = eval_due and is_main_process
            if should_eval:
                eval_metrics, preview = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    mel_extractor=mel_extractor,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    kl_weight=args.kl_weight,
                    context_frames=args.context_frames,
                )
                eval_metrics["type"] = "eval"
                eval_metrics["step"] = step
                eval_metrics["lr"] = float(scheduler.get_last_lr()[0])
                eval_metrics["train_grad_norm"] = grad_norm
                metrics.append(eval_metrics)
                save_metrics_json(output_dir / "metrics.json", metrics)

                eval_line = (
                    f"eval step={step:06d} loss={eval_metrics['loss']:.4f} "
                    f"recon={eval_metrics['recon_loss']:.4f} kl={eval_metrics['kl_loss']:.4f}"
                )

                if preview is not None:
                    save_mel_preview(preview_path(output_dir, split="eval", step=step), preview[0], preview[1])

                if accelerator.num_processes == 1:
                    with torch.no_grad():
                        best_train_mel = None
                        best_train_recon = None
                        best_train_energy = -1.0
                        for _ in range(5):
                            train_batch = next(train_iter)
                            train_audio = train_batch["audio"].to(device, non_blocking=True).float() / 32768.0
                            train_audio_lengths = train_batch["audio_lengths"]
                            if bool(torch.all(train_audio_lengths == train_audio.shape[-1])):
                                train_waveform = train_audio.reshape(train_audio.shape[0], -1)
                            else:
                                train_waveform = frame_audio_to_waveform(train_audio, train_audio_lengths)
                            train_mel = mel_extractor(train_waveform)
                            energy = float(train_mel[0].mean())
                            if energy > best_train_energy:
                                best_train_energy = energy
                                with accelerator.autocast():
                                    train_outputs = model(train_mel, sample_posterior=False)
                                best_train_mel = train_mel.detach().cpu()
                                best_train_recon = train_outputs.reconstruction.detach().cpu()
                        if best_train_mel is not None:
                            save_mel_preview(
                                preview_path(output_dir, split="train", step=step),
                                best_train_mel,
                                best_train_recon,
                                title_prefix="[Train] ",
                            )

                if eval_metrics["loss"] < best_eval:
                    best_eval = eval_metrics["loss"]
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
                else:
                    log_console.print(f"[checkpoint] Saving latest weights at step {step}")
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


if __name__ == "__main__":
    main()