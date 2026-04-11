#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from rich.console import Console
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.plot_style import apply_plot_style
apply_plot_style()

from src.data.audio_features import LogMelSpectrogram, frame_audio_to_waveform, mel_time_frequency_shape
from src.config import AUDIO_FMAX, AUDIO_FMIN, AUDIO_HOP_LENGTH, AUDIO_N_FFT, AUDIO_N_MELS, AUDIO_SAMPLE_RATE
from src.models.audio_vocoder import AudioVocoder
from src.data.normalized_dataset import NormalizedSequenceDataset
from src.system_info import collect_system_info, print_system_info
from src.training.audio_training_helpers import (
    build_mel_mask,
    build_waveform_mask,
    context_waveform_lengths,
    masked_l1_loss,
)
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


def parse_resblock_dilation_sizes(value: str) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for group in value.split(";"):
        tokens = [token.strip() for token in group.split(",") if token.strip()]
        if not tokens:
            continue
        groups.append(tuple(int(token) for token in tokens))
    if not groups:
        raise ValueError("resblock dilation groups cannot be empty")
    return tuple(groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact audio vocoder.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument(
        "--context-frames",
        type=int,
        default=0,
        help="Number of leading clip frames used as context; reconstruction losses ignore this prefix.",
    )
    parser.add_argument("--overfit-n", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--constant-lr", action="store_true")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--upsample-initial-channel", type=int, default=512)
    parser.add_argument("--upsample-rates", type=int, nargs="+", default=[5, 5, 4])
    parser.add_argument("--upsample-kernel-sizes", type=int, nargs="+", default=[11, 11, 8])
    parser.add_argument("--resblock-kernel-sizes", type=int, nargs="+", default=[3, 7])
    parser.add_argument(
        "--resblock-dilation-sizes",
        type=str,
        default="1,3,9;1,3,9",
        help="Semicolon-separated dilation groups. Example: 1,3,9;1,3,9",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability used in vocoder residual blocks.",
    )
    parser.add_argument("--target-max-params", type=int, default=5_000_000)

    parser.add_argument("--sample-rate", type=int, default=AUDIO_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=AUDIO_N_FFT)
    parser.add_argument("--hop-length", type=int, default=AUDIO_HOP_LENGTH)
    parser.add_argument("--n-mels", type=int, default=AUDIO_N_MELS)
    parser.add_argument("--fmin", type=float, default=AUDIO_FMIN)
    parser.add_argument("--fmax", type=float, default=AUDIO_FMAX)

    parser.add_argument("--waveform-l1-weight", type=float, default=1.0)
    parser.add_argument("--mel-l1-weight", type=float, default=20.0)
    parser.add_argument("--stft-sc-weight", type=float, default=1.0)
    parser.add_argument("--stft-mag-weight", type=float, default=1.0)
    parser.add_argument("--stft-fft-sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--stft-hop-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--stft-win-lengths", type=int, nargs="+", default=[256, 512, 1024])

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

    if args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")

    if len(args.upsample_rates) != len(args.upsample_kernel_sizes):
        parser.error("--upsample-rates and --upsample-kernel-sizes must have the same length")
    if len(args.stft_fft_sizes) != len(args.stft_win_lengths):
        parser.error("--stft-fft-sizes and --stft-win-lengths must have the same length")
    if len(args.stft_fft_sizes) != len(args.stft_hop_sizes):
        parser.error("--stft-fft-sizes and --stft-hop-sizes must have the same length")

    try:
        args.resblock_dilation_sizes = parse_resblock_dilation_sizes(args.resblock_dilation_sizes)
    except ValueError as error:
        parser.error(f"Invalid --resblock-dilation-sizes: {error}")

    if len(args.resblock_kernel_sizes) != len(args.resblock_dilation_sizes):
        parser.error("--resblock-kernel-sizes must match parsed --resblock-dilation-sizes group count")

    if args.context_frames < 0:
        parser.error("--context-frames must be non-negative")
    if args.context_frames >= args.clip_frames:
        parser.error("--context-frames must be smaller than --clip-frames")
    if not (0.0 <= args.dropout < 1.0):
        parser.error("--dropout must be in [0, 1)")

    return args


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        *,
        fft_sizes: Sequence[int],
        hop_sizes: Sequence[int],
        win_lengths: Sequence[int],
    ) -> None:
        super().__init__()
        if len(fft_sizes) != len(hop_sizes) or len(fft_sizes) != len(win_lengths):
            raise ValueError("fft_sizes, hop_sizes, and win_lengths must have the same length")
        if not fft_sizes:
            raise ValueError("At least one STFT resolution is required")

        self.fft_sizes = tuple(int(size) for size in fft_sizes)
        self.hop_sizes = tuple(int(size) for size in hop_sizes)
        self.win_lengths = tuple(int(size) for size in win_lengths)

        for index, win_length in enumerate(self.win_lengths):
            self.register_buffer(f"window_{index}", torch.hann_window(win_length), persistent=False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction and target must have matching shape, got {prediction.shape} vs {target.shape}")
        if prediction.ndim != 3:
            raise ValueError(f"Expected (B, C, T) waveform tensors, got {tuple(prediction.shape)}")

        pred = prediction.reshape(-1, prediction.shape[-1])
        tgt = target.reshape(-1, target.shape[-1])

        sc_total = prediction.new_zeros(())
        mag_total = prediction.new_zeros(())

        for index, (n_fft, hop, win_length) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_lengths, strict=True)
        ):
            window = getattr(self, f"window_{index}").to(device=prediction.device, dtype=prediction.dtype)

            pred_spec = torch.stft(
                pred,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            tgt_spec = torch.stft(
                tgt,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )

            pred_mag = pred_spec.abs().clamp_min(1e-7)
            tgt_mag = tgt_spec.abs().clamp_min(1e-7)

            sc = torch.linalg.norm(pred_mag - tgt_mag) / torch.linalg.norm(tgt_mag).clamp_min(1e-7)
            mag = F.l1_loss(torch.log(pred_mag), torch.log(tgt_mag))

            sc_total = sc_total + sc
            mag_total = mag_total + mag

        denom = float(len(self.fft_sizes))
        return sc_total / denom, mag_total / denom


def save_vocoder_preview(
    path: Path,
    target_waveform: torch.Tensor,
    predicted_waveform: torch.Tensor,
    target_mel: torch.Tensor,
    predicted_mel: torch.Tensor,
    *,
    sample_rate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    target_wave = target_waveform[0, 0].detach().cpu().numpy()
    predicted_wave = predicted_waveform[0, 0].detach().cpu().numpy()
    target_mel_np = target_mel[0, 0].detach().cpu().numpy().T
    pred_mel_np = predicted_mel[0, 0].detach().cpu().numpy().T

    target_time = np.arange(target_wave.shape[0], dtype=np.float32) / float(sample_rate)
    pred_time = np.arange(predicted_wave.shape[0], dtype=np.float32) / float(sample_rate)

    mel_min = min(float(target_mel_np.min()), float(pred_mel_np.min()))
    mel_max = max(float(target_mel_np.max()), float(pred_mel_np.max()))
    if mel_max - mel_min < 1e-6:
        mel_max = mel_min + 1.0

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].plot(target_time, target_wave, label="Target", linewidth=1.0)
    axes[0].plot(pred_time, predicted_wave, label="Predicted", linewidth=1.0, alpha=0.8)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")

    im0 = axes[1].imshow(target_mel_np, origin="lower", aspect="auto", cmap="magma", vmin=mel_min, vmax=mel_max)
    axes[1].set_title("Target log-mel")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mel bin")
    figure.colorbar(im0, ax=axes[1], fraction=0.02, pad=0.01)

    im1 = axes[2].imshow(pred_mel_np, origin="lower", aspect="auto", cmap="magma", vmin=mel_min, vmax=mel_max)
    axes[2].set_title("Predicted log-mel")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Mel bin")
    figure.colorbar(im1, ax=axes[2], fraction=0.02, pad=0.01)

    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_waveform_wav(path: Path, waveform: torch.Tensor, *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = waveform[0, 0].detach().cpu().clamp(-1.0, 1.0).numpy()
    audio_i16 = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(audio_i16.tobytes())


def save_training_state(
    path: Path,
    *,
    model: nn.Module,
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


def evaluate(
    model: AudioVocoder,
    loader: DataLoader,
    *,
    device: torch.device,
    mel_extractor: LogMelSpectrogram,
    stft_loss: MultiResolutionSTFTLoss,
    n_fft: int,
    hop_length: int,
    waveform_l1_weight: float,
    mel_l1_weight: float,
    stft_sc_weight: float,
    stft_mag_weight: float,
    context_frames: int = 0,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    model.eval()
    base_model = unwrap_model(model)

    total_losses: list[float] = []
    wave_losses: list[float] = []
    mel_losses: list[float] = []
    stft_sc_losses: list[float] = []
    stft_mag_losses: list[float] = []

    preview: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    preview_energy = -1.0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True).float() / 32768.0
            audio_lengths = batch["audio_lengths"]

            waveform = frame_audio_to_waveform(audio, audio_lengths)
            waveform_lengths = audio_lengths.sum(dim=1).to(device)
            context_lengths = context_waveform_lengths(
                audio_lengths,
                context_frames=context_frames,
            ).to(device)
            mel = mel_extractor(waveform)

            mel_mask = build_mel_mask(
                waveform_lengths,
                max_time_steps=mel.shape[2],
                n_fft=n_fft,
                hop_length=hop_length,
                context_lengths=context_lengths if context_frames > 0 else None,
            )

            target_length = base_model.expected_output_length(mel.shape[2])
            waveform_target = waveform[:, :target_length].unsqueeze(1)
            valid_lengths = waveform_lengths.clamp_max(target_length)
            masked_context_lengths = context_lengths.clamp_max(valid_lengths) if context_frames > 0 else None
            waveform_mask = build_waveform_mask(
                valid_lengths,
                max_samples=target_length,
                context_lengths=masked_context_lengths,
            )

            prediction = model(mel, output_length=target_length)
            pred_mel = mel_extractor(prediction.squeeze(1))

            wave_l1 = masked_l1_loss(prediction, waveform_target, waveform_mask)
            mel_l1 = masked_l1_loss(pred_mel, mel, mel_mask.expand_as(mel))
            if context_frames > 0:
                stft_prediction = prediction * waveform_mask
                stft_target = waveform_target * waveform_mask
                stft_sc, stft_mag = stft_loss(stft_prediction, stft_target)
            else:
                stft_sc, stft_mag = stft_loss(prediction, waveform_target)

            total_loss = (
                waveform_l1_weight * wave_l1
                + mel_l1_weight * mel_l1
                + stft_sc_weight * stft_sc
                + stft_mag_weight * stft_mag
            )

            total_losses.append(float(total_loss.item()))
            wave_losses.append(float(wave_l1.item()))
            mel_losses.append(float(mel_l1.item()))
            stft_sc_losses.append(float(stft_sc.item()))
            stft_mag_losses.append(float(stft_mag.item()))

            energy = float(mel[0].mean())
            if energy > preview_energy:
                preview_energy = energy
                preview = (
                    waveform_target.detach().cpu(),
                    prediction.detach().cpu(),
                    mel.detach().cpu(),
                    pred_mel.detach().cpu(),
                )

    model.train()

    if not total_losses:
        return {
            "loss": 0.0,
            "wave_l1": 0.0,
            "mel_l1": 0.0,
            "stft_sc": 0.0,
            "stft_mag": 0.0,
        }, preview

    return {
        "loss": float(np.mean(total_losses)),
        "wave_l1": float(np.mean(wave_losses)),
        "mel_l1": float(np.mean(mel_losses)),
        "stft_sc": float(np.mean(stft_sc_losses)),
        "stft_mag": float(np.mean(stft_mag_losses)),
    }, preview


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = make_output_dir(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        run_name=args.run_name,
        default_prefix="audio_vocoder",
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
            subset_n=args.overfit_n,
            seed=args.seed,
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

    if args.overfit_n > 0:
        train_dataset = dataset
        eval_dataset = None
        if is_main_process:
            console.print(f"[overfit] Enabled on {len(dataset)} sample(s)")
    else:
        train_dataset, eval_dataset = split_train_eval_dataset(
            dataset,
            eval_samples=args.eval_samples,
            seed=args.seed,
        )
        if eval_dataset is not None and is_main_process:
            console.print(f"Eval split: {len(eval_dataset)} eval, {len(train_dataset)} train samples")

    if len(train_dataset) > 0 and args.batch_size > len(train_dataset):
        if is_main_process:
            console.print(f"[batch-size] Capping batch_size {args.batch_size} -> {len(train_dataset)}")
        args.batch_size = len(train_dataset)

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

    model = AudioVocoder(
        in_channels=1,
        n_mels=args.n_mels,
        out_channels=1,
        upsample_initial_channel=args.upsample_initial_channel,
        upsample_rates=tuple(args.upsample_rates),
        upsample_kernel_sizes=tuple(args.upsample_kernel_sizes),
        resblock_kernel_sizes=tuple(args.resblock_kernel_sizes),
        resblock_dilation_sizes=args.resblock_dilation_sizes,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        dropout=args.dropout,
    ).to(device)

    num_parameters = model.num_parameters
    if num_parameters > args.target_max_params:
        if is_main_process:
            console.print(
                "[warning] Model size exceeds target budget: "
                f"{num_parameters:,} > {args.target_max_params:,} parameters"
            )
    else:
        if is_main_process:
            console.print(
                "[model] Parameter budget check passed: "
                f"{num_parameters:,} <= {args.target_max_params:,}"
            )

    if args.compile:
        if is_main_process:
            console.print("[compile] Compiling the model with torch.compile()...")
        model = torch.compile(model)
        if is_main_process:
            console.print("[compile] Compilation complete.")

    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=args.stft_fft_sizes,
        hop_sizes=args.stft_hop_sizes,
        win_lengths=args.stft_win_lengths,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    if args.constant_lr:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    else:
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
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            if is_main_process:
                console.print("[resume] Scheduler state was incompatible; continuing with the current scheduler")
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

    config = build_trainer_config(
        model_name="audio_vocoder",
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
            "num_parameters": int(num_parameters),
            "upsample_initial_channel": int(args.upsample_initial_channel),
            "upsample_rates": list(args.upsample_rates),
            "upsample_kernel_sizes": list(args.upsample_kernel_sizes),
            "resblock_kernel_sizes": list(args.resblock_kernel_sizes),
            "resblock_dilation_sizes": [list(group) for group in args.resblock_dilation_sizes],
            "dropout": float(args.dropout),
        },
    )
    if is_main_process:
        save_json(output_dir / "config.json", config, indent=2)
        console.print(f"Config saved to {output_dir / 'config.json'}")
        console.print(json.dumps(config, indent=2))

        console.print(f"Training audio vocoder on {len(train_dataset)} samples")
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
    bytes_per_sample = None
    use_live_progress = sys.stdout.isatty() and is_main_process
    final_preview_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    total_train_steps = max(args.max_steps - start_step, 0)
    with build_progress(use_live=use_live_progress) as progress:
        log_console = progress.console
        train_task = progress.add_task("Training", total=total_train_steps, status="")
        for step in range(start_step, args.max_steps):
            batch = next(train_iter)

            audio = batch["audio"].to(device, non_blocking=True).float() / 32768.0
            audio_lengths = batch["audio_lengths"]

            waveform = frame_audio_to_waveform(audio, audio_lengths)
            waveform_lengths = audio_lengths.sum(dim=1).to(device)
            context_lengths = context_waveform_lengths(
                audio_lengths,
                context_frames=args.context_frames,
            ).to(device)

            mel = mel_extractor(waveform)
            mel_mask = build_mel_mask(
                waveform_lengths,
                max_time_steps=mel.shape[2],
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                context_lengths=context_lengths if args.context_frames > 0 else None,
            )

            target_length = unwrap_model(model).expected_output_length(mel.shape[2])
            waveform_target = waveform[:, :target_length].unsqueeze(1)
            valid_lengths = waveform_lengths.clamp_max(target_length)
            masked_context_lengths = context_lengths.clamp_max(valid_lengths) if args.context_frames > 0 else None
            waveform_mask = build_waveform_mask(
                valid_lengths,
                max_samples=target_length,
                context_lengths=masked_context_lengths,
            )

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                prediction = model(mel, output_length=target_length)
            pred_mel = mel_extractor(prediction.squeeze(1))
            final_preview_payload = (
                waveform_target.detach().cpu(),
                prediction.detach().cpu(),
                mel.detach().cpu(),
                pred_mel.detach().cpu(),
            )

            wave_l1 = masked_l1_loss(prediction, waveform_target, waveform_mask)
            mel_l1 = masked_l1_loss(pred_mel, mel, mel_mask.expand_as(mel))
            if args.context_frames > 0:
                stft_prediction = prediction * waveform_mask
                stft_target = waveform_target * waveform_mask
                stft_sc, stft_mag = stft_loss(stft_prediction, stft_target)
            else:
                stft_sc, stft_mag = stft_loss(prediction, waveform_target)

            loss = (
                args.waveform_l1_weight * wave_l1
                + args.mel_l1_weight * mel_l1
                + args.stft_sc_weight * stft_sc
                + args.stft_mag_weight * stft_mag
            )

            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()

            if bytes_per_sample is None:
                bytes_per_sample = prediction.shape[1] * prediction.shape[-1] * 4

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=audio.shape[0] * accelerator.num_processes,
            )

            if is_main_process:
                status = (
                    f"loss={loss.item():.4f} wave={wave_l1.item():.4f} "
                    f"mel={mel_l1.item():.4f} sc={stft_sc.item():.4f} mag={stft_mag.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} gnorm={grad_norm:.2f} "
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
                    "wave_l1": wave_l1.item(),
                    "mel_l1": mel_l1.item(),
                    "stft_sc": stft_sc.item(),
                    "stft_mag": stft_mag.item(),
                    "grad_norm": grad_norm,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "samples_per_sec": round(samples_per_second, 1),
                    "steps_per_sec": round(steps_per_second, 2),
                }
                train_row.update(gpu_stats(device))
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
                    f"step={step:06d} loss={loss.item():.4f} wave={wave_l1.item():.4f} "
                    f"mel={mel_l1.item():.4f} sc={stft_sc.item():.4f} mag={stft_mag.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} gnorm={grad_norm:.2f} ex/s={examples_per_second:.1f}"
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
                    stft_loss=stft_loss,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    waveform_l1_weight=args.waveform_l1_weight,
                    mel_l1_weight=args.mel_l1_weight,
                    stft_sc_weight=args.stft_sc_weight,
                    stft_mag_weight=args.stft_mag_weight,
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
                    f"wave={eval_metrics['wave_l1']:.4f} mel={eval_metrics['mel_l1']:.4f} "
                    f"sc={eval_metrics['stft_sc']:.4f} mag={eval_metrics['stft_mag']:.4f}"
                )

                if preview is not None:
                    save_vocoder_preview(
                        preview_path(output_dir, split="eval", step=step),
                        preview[0],
                        preview[1],
                        preview[2],
                        preview[3],
                        sample_rate=args.sample_rate,
                    )

                save_vocoder_preview(
                    preview_path(output_dir, split="train", step=step),
                    waveform_target.detach().cpu(),
                    prediction.detach().cpu(),
                    mel.detach().cpu(),
                    pred_mel.detach().cpu(),
                    sample_rate=args.sample_rate,
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
                )

            if eval_line is not None and is_main_process:
                log_console.print(eval_line)

            if eval_due or should_checkpoint:
                accelerator.wait_for_everyone()

    if final_preview_payload is not None and is_main_process:
        save_vocoder_preview(
            output_dir / "previews" / "final_reconstruction.png",
            final_preview_payload[0],
            final_preview_payload[1],
            final_preview_payload[2],
            final_preview_payload[3],
            sample_rate=args.sample_rate,
        )
        save_waveform_wav(
            output_dir / "previews" / "final_target.wav",
            final_preview_payload[0],
            sample_rate=args.sample_rate,
        )
        save_waveform_wav(
            output_dir / "previews" / "final_reconstruction.wav",
            final_preview_payload[1],
            sample_rate=args.sample_rate,
        )
        console.print(f"[final] Saved final reconstruction artifacts to {output_dir}")


if __name__ == "__main__":
    main()