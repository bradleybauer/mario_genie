#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Sequence

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
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.audio_features import LogMelSpectrogram, frame_audio_to_waveform, mel_time_frequency_shape
from mario_world_model.config import AUDIO_FMAX, AUDIO_FMIN, AUDIO_HOP_LENGTH, AUDIO_N_FFT, AUDIO_N_MELS, AUDIO_SAMPLE_RATE
from mario_world_model.ltx_audio_vocoder import LTXAudioVocoder
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
    parser = argparse.ArgumentParser(description="Train the LTX-style compact audio vocoder.")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--clip-frames", type=int, default=16)
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
    run_name = args.run_name or datetime.now().strftime("ltx_audio_vocoder_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "checkpoints" / run_name


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


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


def build_waveform_mask(lengths: torch.Tensor, *, max_samples: int) -> torch.Tensor:
    mask = torch.zeros((lengths.shape[0], 1, max_samples), dtype=torch.float32, device=lengths.device)
    for batch_idx, length in enumerate(lengths.tolist()):
        mask[batch_idx, :, : min(int(length), max_samples)] = 1.0
    return mask


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
) -> None:
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
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


def evaluate(
    model: LTXAudioVocoder,
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
            mel = mel_extractor(waveform)

            mel_mask = build_mel_mask(
                waveform_lengths,
                max_time_steps=mel.shape[2],
                n_fft=n_fft,
                hop_length=hop_length,
            )

            target_length = base_model.expected_output_length(mel.shape[2])
            waveform_target = waveform[:, :target_length].unsqueeze(1)
            valid_lengths = waveform_lengths.clamp_max(target_length)
            waveform_mask = build_waveform_mask(valid_lengths, max_samples=target_length)

            prediction = model(mel, output_length=target_length)
            pred_mel = mel_extractor(prediction.squeeze(1))

            wave_l1 = masked_l1_loss(prediction, waveform_target, waveform_mask)
            mel_l1 = masked_l1_loss(pred_mel, mel, mel_mask.expand_as(mel))
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
        include_frames=False,
        include_audio=True,
        subset_n=args.overfit_n,
        seed=args.seed,
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
    if args.overfit_n > 0:
        console.print(f"[overfit] Enabled on {len(dataset)} sample(s)")
    elif args.eval_samples > 0 and len(dataset) > args.eval_samples:
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(dataset), generator=generator)
        eval_indices = permutation[:args.eval_samples].tolist()
        train_indices = permutation[args.eval_samples:].tolist()
        eval_dataset = Subset(dataset, eval_indices)
        train_dataset = Subset(dataset, train_indices)
        console.print(f"Eval split: {len(eval_dataset)} eval, {len(train_dataset)} train samples")

    if len(train_dataset) > 0 and args.batch_size > len(train_dataset):
        console.print(f"[batch-size] Capping batch_size {args.batch_size} -> {len(train_dataset)}")
        args.batch_size = len(train_dataset)

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

    model = LTXAudioVocoder(
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
    ).to(device)

    num_parameters = model.num_parameters
    if num_parameters > args.target_max_params:
        console.print(
            "[warning] Model size exceeds target budget: "
            f"{num_parameters:,} > {args.target_max_params:,} parameters"
        )
    else:
        console.print(
            "[model] Parameter budget check passed: "
            f"{num_parameters:,} <= {args.target_max_params:,}"
        )

    if args.compile:
        console.print("[compile] Compiling the model with torch.compile()...")
        model = torch.compile(model)
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
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            console.print("[resume] Scheduler state was incompatible; continuing with the current scheduler")
        start_step = int(checkpoint["step"]) + 1
        best_eval = float(checkpoint.get("best_eval", float("inf")))
        metrics = list(checkpoint.get("metrics", []))
        console.print(f"[resume] Loaded training state from {args.resume_from} at step {start_step}")

        if args.max_steps > 0 and start_step >= args.max_steps:
            console.print(f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do.")
            return

    config = vars(args).copy()
    config.update(
        {
            "audio_frame_size": int(dataset.audio_frame_size or 0),
            "num_parameters": int(num_parameters),
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
        }
    )
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    console.print(f"Config saved to {output_dir / 'config.json'}")
    console.print(json.dumps(config, indent=2))

    console.print(f"Training LTX audio vocoder on {len(train_dataset)} samples")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Device: {device}")
    console.print(f"Model parameters: {num_parameters:,}")

    start_time = time.time()
    perf_window_start = time.time()
    perf_window_steps = 0
    perf_window_samples = 0
    samples_per_second = 0.0
    steps_per_second = 0.0
    perf_window_size = 20
    bytes_per_sample = None
    use_live_progress = sys.stdout.isatty()
    final_preview_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

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

            waveform = frame_audio_to_waveform(audio, audio_lengths)
            waveform_lengths = audio_lengths.sum(dim=1).to(device)

            mel = mel_extractor(waveform)
            mel_mask = build_mel_mask(
                waveform_lengths,
                max_time_steps=mel.shape[2],
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )

            target_length = unwrap_model(model).expected_output_length(mel.shape[2])
            waveform_target = waveform[:, :target_length].unsqueeze(1)
            valid_lengths = waveform_lengths.clamp_max(target_length)
            waveform_mask = build_waveform_mask(valid_lengths, max_samples=target_length)

            optimizer.zero_grad(set_to_none=True)
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
            stft_sc, stft_mag = stft_loss(prediction, waveform_target)

            loss = (
                args.waveform_l1_weight * wave_l1
                + args.mel_l1_weight * mel_l1
                + args.stft_sc_weight * stft_sc
                + args.stft_mag_weight * stft_mag
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if bytes_per_sample is None:
                bytes_per_sample = prediction.shape[1] * prediction.shape[-1] * 4

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
                f"loss={loss.item():.4f} wave={wave_l1.item():.4f} "
                f"mel={mel_l1.item():.4f} sc={stft_sc.item():.4f} mag={stft_mag.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
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

            progress.update(train_task, advance=1, status=status)

            if (
                ((args.log_interval > 0 and step % args.log_interval == 0) or step == args.max_steps - 1)
                and not use_live_progress
            ):
                elapsed = time.time() - start_time
                examples_per_second = ((step - start_step + 1) * args.batch_size) / max(elapsed, 1e-6)
                log_console.print(
                    f"step={step:06d} loss={loss.item():.4f} wave={wave_l1.item():.4f} "
                    f"mel={mel_l1.item():.4f} sc={stft_sc.item():.4f} mag={stft_mag.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} ex/s={examples_per_second:.1f}"
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
                    stft_loss=stft_loss,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    waveform_l1_weight=args.waveform_l1_weight,
                    mel_l1_weight=args.mel_l1_weight,
                    stft_sc_weight=args.stft_sc_weight,
                    stft_mag_weight=args.stft_mag_weight,
                )
                eval_metrics["step"] = step
                metrics.append(eval_metrics)
                with (output_dir / "metrics.json").open("w") as handle:
                    json.dump(metrics, handle, indent=2)

                log_console.print(
                    f"eval step={step:06d} loss={eval_metrics['loss']:.4f} "
                    f"wave={eval_metrics['wave_l1']:.4f} mel={eval_metrics['mel_l1']:.4f} "
                    f"sc={eval_metrics['stft_sc']:.4f} mag={eval_metrics['stft_mag']:.4f}"
                )

                if preview is not None:
                    save_vocoder_preview(
                        output_dir / f"preview_step_{step:06d}.png",
                        preview[0],
                        preview[1],
                        preview[2],
                        preview[3],
                        sample_rate=args.sample_rate,
                    )

                save_vocoder_preview(
                    output_dir / f"train_preview_step_{step:06d}.png",
                    waveform_target.detach().cpu(),
                    prediction.detach().cpu(),
                    mel.detach().cpu(),
                    pred_mel.detach().cpu(),
                    sample_rate=args.sample_rate,
                )

                if eval_metrics["loss"] < best_eval:
                    best_eval = eval_metrics["loss"]
                    save_training_state(
                        output_dir / "audio_vocoder_best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                    )
                    log_console.print(
                        f"[checkpoint] New best eval_loss={best_eval:.6f} -> saved audio_vocoder_best.pt"
                    )

            should_checkpoint = (
                (args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0)
                or step == args.max_steps - 1
            )
            if should_checkpoint:
                log_console.print(f"[checkpoint] Saving latest weights at step {step}")
                save_training_state(
                    output_dir / "audio_vocoder_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                )

    if final_preview_payload is not None:
        save_vocoder_preview(
            output_dir / "final_reconstruction.png",
            final_preview_payload[0],
            final_preview_payload[1],
            final_preview_payload[2],
            final_preview_payload[3],
            sample_rate=args.sample_rate,
        )
        save_waveform_wav(
            output_dir / "final_target.wav",
            final_preview_payload[0],
            sample_rate=args.sample_rate,
        )
        save_waveform_wav(
            output_dir / "final_reconstruction.wav",
            final_preview_payload[1],
            sample_rate=args.sample_rate,
        )
        console.print(f"[final] Saved final reconstruction artifacts to {output_dir}")


if __name__ == "__main__":
    main()