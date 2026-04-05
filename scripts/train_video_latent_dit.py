#!/usr/bin/env python3
"""Train a video-latent DiT world model with flow matching.

This trainer consumes pre-encoded latent windows and learns to
denoise/predict future latent segments conditioned on history and actions.
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
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
from mario_world_model.latent_dataset import LatentSequenceDataset
from mario_world_model.path_utils import resolve_workspace_path, serialize_project_path
from mario_world_model.system_info import collect_system_info, print_system_info
from mario_world_model.video_latent_dit import VideoLatentDiT


console = Console()


@dataclass
class LatentNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    stats_path: str

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.mean) / self.std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.std + self.mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a video-latent DiT with flow matching.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--video-vae-checkpoint", type=str, default=None)
    parser.add_argument("--video-vae-config", type=str, default=None)
    parser.add_argument(
        "--latent-stats",
        type=str,
        default=None,
        help=(
            "Path to latent_stats.json for per-channel normalization "
            "(defaults to dataset latent_stats.json when available)."
        ),
    )
    parser.add_argument(
        "--disable-latent-normalization",
        action="store_true",
        help="Disable latent normalization even if latent_stats.json is available.",
    )

    parser.add_argument("--clip-frames", type=int, default=20)
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))

    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)

    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Convenience: sets --num-encoder-layers and --num-decoder-layers to "
                             "half this value each if neither is specified explicitly.")
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--num-decoder-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=64)

    parser.add_argument("--flow-loss", type=str, choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--grad-norm-skip",
        type=float,
        default=100.0,
        help=(
            "Skip optimizer step when pre-clip gradient norm exceeds this value "
            "(<=0 disables)."
        ),
    )

    parser.add_argument(
        "--error-buffer-size",
        type=int,
        default=0,
        help="Number of residual clips to retain for history perturbation.",
    )
    parser.add_argument(
        "--error-inject-prob",
        type=float,
        default=0.0,
        help="Probability of perturbing latent history with sampled residual noise.",
    )
    parser.add_argument(
        "--error-inject-scale",
        type=float,
        default=0.05,
        help="Scale factor for sampled residual perturbations.",
    )
    parser.add_argument(
        "--error-inject-start-step",
        type=int,
        default=20000,
        help="Delay residual history perturbation until this global training step.",
    )
    parser.add_argument(
        "--error-buffer-clip",
        type=float,
        default=3.0,
        help="Clip residual values to +/- this magnitude before storing in the error buffer (0 disables clipping).",
    )

    parser.add_argument(
        "--preview-ode-steps",
        type=int,
        default=24,
        help="Euler steps for denoising preview rollout.",
    )

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
    if args.action_dim <= 0:
        parser.error("--action-dim must be > 0")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("--eval-batch-size must be > 0")
    if args.error_buffer_size < 0:
        parser.error("--error-buffer-size must be >= 0")
    if args.grad_norm_skip < 0.0:
        parser.error("--grad-norm-skip must be >= 0")
    if not (0.0 <= args.error_inject_prob <= 1.0):
        parser.error("--error-inject-prob must be in [0, 1]")
    if args.error_inject_start_step < 0:
        parser.error("--error-inject-start-step must be >= 0")
    if args.error_buffer_clip < 0.0:
        parser.error("--error-buffer-clip must be >= 0")
    if args.preview_ode_steps < 1:
        parser.error("--preview-ode-steps must be >= 1")
    # Resolve encoder/decoder layer counts.
    default_half = (args.num_layers // 2) if args.num_layers is not None else 6
    if args.num_encoder_layers is None:
        args.num_encoder_layers = default_half
    if args.num_decoder_layers is None:
        args.num_decoder_layers = default_half
    if args.num_encoder_layers <= 0:
        parser.error("--num-encoder-layers must be > 0")
    if args.num_decoder_layers <= 0:
        parser.error("--num-decoder-layers must be > 0")
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
    run_name = args.run_name or datetime.now().strftime("video_latent_dit_%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "checkpoints" / run_name


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_palette_tensor(data_dir: str | Path) -> torch.Tensor:
    palette_path = Path(data_dir) / "palette.json"
    if not palette_path.is_file():
        raise FileNotFoundError(f"Missing palette.json in {palette_path.parent}")
    palette_info = _load_json(palette_path)
    return torch.tensor(palette_info["colors_rgb"], dtype=torch.float32) / 255.0


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        return data_dir

    candidates = (PROJECT_ROOT / "data" / "latents",)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not infer --data-dir. Checked data/latents. Pass --data-dir explicitly."
    )


def _resolve_path(value: str | None, *, config_dir: Path | None = None) -> Path | None:
    return resolve_workspace_path(value, project_root=PROJECT_ROOT, config_dir=config_dir)


def _is_readable_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return path.is_file()
    except (PermissionError, OSError):
        return False


def _load_optional_latent_metadata(data_dir: Path) -> dict[str, Any] | None:
    latent_config = data_dir / "latent_config.json"
    if not latent_config.is_file():
        return None
    return _load_json(latent_config)


def _resolve_latent_stats_path(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    latent_meta: dict[str, Any] | None,
) -> Path | None:
    if args.disable_latent_normalization:
        return None

    if args.latent_stats is not None:
        stats_path = _resolve_path(args.latent_stats, config_dir=data_dir)
        if not _is_readable_file(stats_path):
            raise FileNotFoundError(f"Latent stats file not found: {args.latent_stats}")
        return stats_path

    if latent_meta is not None:
        meta_stats_path = latent_meta.get("latent_stats_path")
        if isinstance(meta_stats_path, str):
            stats_path = _resolve_path(meta_stats_path, config_dir=data_dir)
            if _is_readable_file(stats_path):
                return stats_path

        meta_stats_file = latent_meta.get("latent_stats_file")
        if isinstance(meta_stats_file, str):
            candidate = data_dir / meta_stats_file
            if candidate.is_file():
                return candidate.resolve()

    default_stats = data_dir / "latent_stats.json"
    if default_stats.is_file():
        return default_stats.resolve()
    return None


def load_latent_normalization(
    *,
    stats_path: Path,
    latent_channels: int,
    device: torch.device,
) -> LatentNormalization:
    stats = _load_json(stats_path)

    mean_values = stats.get("channel_mean")
    std_values = stats.get("channel_std_clamped")
    if std_values is None:
        std_values = stats.get("channel_std")

    if mean_values is None or std_values is None:
        raise ValueError(
            f"Latent stats file {stats_path} must contain channel_mean and channel_std(_clamped)."
        )
    if len(mean_values) != latent_channels or len(std_values) != latent_channels:
        raise ValueError(
            "Latent stats channels mismatch: "
            f"expected {latent_channels}, got mean={len(mean_values)} std={len(std_values)}"
        )

    eps = float(stats.get("std_epsilon", 1e-6))
    eps = max(eps, 1e-6)

    mean = torch.tensor(mean_values, dtype=torch.float32, device=device).view(1, latent_channels, 1, 1, 1)
    std = torch.tensor(std_values, dtype=torch.float32, device=device).view(1, latent_channels, 1, 1, 1)
    std = torch.nan_to_num(std, nan=eps, posinf=eps, neginf=eps).clamp_min(eps)

    if not torch.isfinite(mean).all():
        raise ValueError(f"Latent stats mean has non-finite values in {stats_path}")

    return LatentNormalization(mean=mean, std=std, stats_path=str(stats_path))


def _resolve_video_vae_sources(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    latent_meta: dict[str, Any] | None,
) -> tuple[Path | None, Path | None, str | None, str | None]:
    checkpoint_value = args.video_vae_checkpoint
    config_value = args.video_vae_config

    if latent_meta is not None:
        checkpoint_value = checkpoint_value or latent_meta.get("video_vae_checkpoint")
        config_value = config_value or latent_meta.get("video_vae_config_path")

    checkpoint_path = _resolve_path(checkpoint_value, config_dir=data_dir)
    config_path = _resolve_path(config_value, config_dir=data_dir)

    copied_config = data_dir / "video_vae_config.json"
    if not _is_readable_file(config_path) and copied_config.is_file():
        config_path = copied_config.resolve()

    if not _is_readable_file(config_path) and _is_readable_file(checkpoint_path):
        sibling = checkpoint_path.parent / "config.json"
        if sibling.is_file():
            config_path = sibling.resolve()

    if not _is_readable_file(checkpoint_path):
        checkpoint_path = None
    if not _is_readable_file(config_path):
        config_path = None

    if checkpoint_path is not None:
        checkpoint_value = str(checkpoint_path)
    if config_path is not None:
        config_value = str(config_path)

    return checkpoint_path, config_path, checkpoint_value, config_value


def load_video_vae(
    *,
    checkpoint_path: Path,
    config_path: Path,
    num_colors: int | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int]]:
    cfg = _load_json(config_path)
    patch_size = int(cfg.get("patch_size", 4))
    base_channels = int(cfg.get("base_channels", 64))
    latent_channels = int(cfg.get("latent_channels", 64))
    fallback_num_colors = int(num_colors) if num_colors is not None else 0
    vae_num_colors = int(cfg.get("num_colors", fallback_num_colors))

    if vae_num_colors <= 0:
        raise ValueError(
            "Could not determine VAE num_colors. Provide a readable palette.json or a VAE config with num_colors."
        )

    if num_colors is not None and vae_num_colors != num_colors:
        raise ValueError(
            f"VAE config expects num_colors={vae_num_colors} but palette has {num_colors}."
        )

    vae: torch.nn.Module = LTXVideoVAE(
        num_colors=vae_num_colors,
        patch_size=patch_size,
        base_channels=base_channels,
        latent_channels=latent_channels,
    )

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    summary = {
        "patch_size": patch_size,
        "base_channels": base_channels,
        "latent_channels": latent_channels,
        "num_colors": vae_num_colors,
    }
    return vae, summary


def shift_actions_causal(actions: torch.Tensor, *, default_action_index: int) -> torch.Tensor:
    """Build causal action sequence so action token t uses action from frame t-1."""
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (B, T), got {tuple(actions.shape)}")

    shifted = torch.empty_like(actions)
    shifted[:, 0] = int(default_action_index)
    shifted[:, 1:] = actions[:, :-1]
    return shifted


def prepare_batch_latents(
    batch: dict[str, Any],
    *,
    device: torch.device,
    default_action_index: int,
    latent_normalization: LatentNormalization | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = batch["actions"].to(device, non_blocking=True).long()
    actions = shift_actions_causal(actions, default_action_index=default_action_index)
    latents = batch["latents"].to(device, non_blocking=True).float()
    if latent_normalization is not None:
        latents = latent_normalization.normalize(latents)
    return latents, actions


def sample_flow_target(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample noisy latents and velocity targets for linear flow matching.

    x_t = (1 - t) * x0 + t * eps
    v*  = eps - x0
    """
    batch = x0.shape[0]
    t = torch.rand(batch, device=x0.device, dtype=x0.dtype)
    eps = torch.randn_like(x0)

    t_broadcast = t.view(batch, 1, 1, 1, 1)
    x_t = (1.0 - t_broadcast) * x0 + t_broadcast * eps
    v_target = eps - x0
    return x_t, v_target, t, eps


def flow_loss(pred: torch.Tensor, target: torch.Tensor, *, loss_type: str, huber_delta: float) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target, beta=huber_delta)


def _match_frame_count(x: torch.Tensor, frames: int) -> torch.Tensor:
    if x.shape[2] == frames:
        return x
    if x.shape[2] > frames:
        return x[:, :, :frames]
    repeat_count = math.ceil(frames / x.shape[2])
    return x.repeat(1, 1, repeat_count, 1, 1)[:, :, :frames]


class ResidualErrorBuffer:
    """FIFO residual bank for lightweight error-aware history perturbation."""

    def __init__(self, capacity: int, *, clip_abs: float = 0.0) -> None:
        self.capacity = int(capacity)
        self.clip_abs = max(float(clip_abs), 0.0)
        self._bank: deque[torch.Tensor] = deque(maxlen=max(self.capacity, 1))

    def __len__(self) -> int:
        return len(self._bank)

    def enabled(self) -> bool:
        return self.capacity > 0

    def clear(self) -> None:
        self._bank.clear()

    def add_batch(self, residual_batch: torch.Tensor) -> None:
        if not self.enabled():
            return
        residual_batch = residual_batch.detach()
        if self.clip_abs > 0.0:
            residual_batch = residual_batch.clamp(min=-self.clip_abs, max=self.clip_abs)

        finite_mask = torch.isfinite(residual_batch.flatten(start_dim=1)).all(dim=1)
        if not finite_mask.any():
            return

        residual_batch = residual_batch[finite_mask].to("cpu", dtype=torch.float16)
        for sample in residual_batch:
            self._bank.append(sample.unsqueeze(0))

    def sample(self, batch_size: int, *, device: torch.device, dtype: torch.dtype, frames: int) -> torch.Tensor | None:
        if len(self._bank) == 0:
            return None
        choices = random.choices(list(self._bank), k=batch_size)
        stacked = torch.cat(choices, dim=0).to(device=device, dtype=dtype)
        return _match_frame_count(stacked, frames)


@torch.no_grad()
def denoise_future_segment(
    model: VideoLatentDiT,
    *,
    history_latents: torch.Tensor,
    actions: torch.Tensor,
    future_frames: int,
    ode_steps: int,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    """Denoise a future latent segment from Gaussian noise via Euler steps."""
    batch, channels, context_frames, height, width = history_latents.shape
    future = torch.randn(
        batch,
        channels,
        future_frames,
        height,
        width,
        device=history_latents.device,
        dtype=history_latents.dtype,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_enabled
        else nullcontext()
    )

    with autocast_ctx:
        encoded_history = model.encode_history(history_latents, actions[:, :context_frames])

    dt = 1.0 / float(ode_steps)
    for step in range(ode_steps, 0, -1):
        t_val = (step - 0.5) / float(ode_steps)
        t = torch.full((batch,), t_val, device=history_latents.device, dtype=history_latents.dtype)
        with autocast_ctx:
            velocity_future = model.decode_future(future, actions, t, encoded_history, context_frames)
        future = future - dt * velocity_future

    return future


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
    with torch.inference_mode(), autocast_ctx:
        logits = vae.decode(latents)
    return logits.argmax(dim=1)


def save_preview_image(
    path: Path,
    *,
    frames_gt: torch.Tensor,
    frames_sample: torch.Tensor,
    palette: torch.Tensor,
) -> None:
    """Save a 2-column preview: GT | sampled rollout."""
    path.parent.mkdir(parents=True, exist_ok=True)

    palette_u8 = (palette.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    gt_np = frames_gt.detach().cpu().numpy()
    sample_np = frames_sample.detach().cpu().numpy()

    gt_rgb = palette_u8[gt_np]
    sample_rgb = palette_u8[sample_np]

    rows = [np.concatenate((gt_rgb[t], sample_rgb[t]), axis=1) for t in range(gt_rgb.shape[0])]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def evaluate(
    model: VideoLatentDiT,
    *,
    loader: DataLoader,
    device: torch.device,
    context_frames: int,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
    default_action_index: int,
    loss_type: str,
    huber_delta: float,
    latent_normalization: LatentNormalization | None,
) -> dict[str, float]:
    model.eval()
    flow_losses: list[float] = []
    clean_losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            latents, actions = prepare_batch_latents(
                batch,
                device=device,
                default_action_index=default_action_index,
                latent_normalization=latent_normalization,
            )

            history = latents[:, :, :context_frames]
            future = latents[:, :, context_frames:]
            noisy_future, v_target, t, eps = sample_flow_target(future)
            model_input = torch.cat((history, noisy_future), dim=2)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_enabled
                else nullcontext()
            )
            with autocast_ctx:
                velocity_future = model(model_input, actions, t, context_frames)

            flow_l = flow_loss(velocity_future, v_target, loss_type=loss_type, huber_delta=huber_delta)
            flow_losses.append(flow_l.item())

            x0_hat = eps - velocity_future
            clean_losses.append(F.mse_loss(x0_hat, future).item())

    model.train()
    if not flow_losses:
        return {"flow_loss": 0.0, "x0_mse": 0.0}
    return {
        "flow_loss": float(np.mean(flow_losses)),
        "x0_mse": float(np.mean(clean_losses)),
    }


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
    data_dir = _resolve_data_dir(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}")
    console.print(f"[data] Using latent dataset at {data_dir}")
    if torch.cuda.is_available():
        if args.tf16:
            torch.set_float32_matmul_precision("medium")
        else:
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
    dataset = LatentSequenceDataset(
        data_dir=data_dir,
        clip_frames=args.clip_frames,
        include_actions=True,
        num_workers=args.num_workers,
        system_info=system_info,
    )
    console.print("[dataset] Index build complete.")
    if len(dataset) == 0:
        raise RuntimeError("No training samples were found.")
    console.print(f"Found {len(dataset)} sequence segments of {args.clip_frames} frames.")

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

    latent_meta = _load_optional_latent_metadata(data_dir)

    palette: torch.Tensor | None = None
    num_colors: int | None = None
    palette_path = data_dir / "palette.json"
    if palette_path.is_file():
        palette = load_palette_tensor(data_dir)
        num_colors = int(palette.shape[0])
        console.print(f"[palette] Loaded {num_colors} colors")
    elif latent_meta is not None and latent_meta.get("num_colors") is not None:
        num_colors = int(latent_meta["num_colors"])

    vae_checkpoint, vae_config, video_vae_checkpoint_value, video_vae_config_value = _resolve_video_vae_sources(
        args,
        data_dir=data_dir,
        latent_meta=latent_meta,
    )

    video_vae: torch.nn.Module | None = None
    latent_channels = int(dataset.latent_channels)
    console.print(f"[latents] loaded latent_channels={latent_channels}")

    latent_normalization: LatentNormalization | None = None
    latent_stats_path = _resolve_latent_stats_path(
        args,
        data_dir=data_dir,
        latent_meta=latent_meta,
    )
    if latent_stats_path is not None:
        latent_normalization = load_latent_normalization(
            stats_path=latent_stats_path,
            latent_channels=latent_channels,
            device=device,
        )
        console.print(f"[latents] Applying per-channel normalization from {latent_stats_path}")
    elif args.disable_latent_normalization:
        console.print("[latents] Latent normalization disabled by flag.")
    else:
        console.print("[latents] No latent stats found; using raw latents.")

    if vae_checkpoint is not None and vae_config is not None:
        video_vae, vae_summary = load_video_vae(
            checkpoint_path=vae_checkpoint,
            config_path=vae_config,
            num_colors=num_colors,
            device=device,
        )
        if int(vae_summary["latent_channels"]) != latent_channels:
            raise ValueError(
                f"Latent dataset has latent_channels={latent_channels} but VAE config expects {vae_summary['latent_channels']}"
            )
        console.print("[vae] Loaded video VAE for preview generation")
    else:
        console.print("[vae] No readable VAE checkpoint found; preview images will be skipped")

    actions_path = data_dir / "actions.json"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing actions metadata: {actions_path}")
    actions_info = _load_json(actions_path)
    num_actions = int(actions_info.get("num_actions", 0))
    if num_actions <= 0:
        raise ValueError(f"Invalid num_actions in {actions_path}")

    action_values_meta = actions_info.get("reduced_to_original_value")
    if action_values_meta is None:
        action_values = list(range(num_actions))
    else:
        action_values = [int(v) for v in action_values_meta]
        if len(action_values) != num_actions:
            raise ValueError(
                "actions.json reduced_to_original_value length does not match num_actions "
                f"({len(action_values)} != {num_actions})"
            )

    action_to_index = {value: idx for idx, value in enumerate(action_values)}
    default_action_index = int(action_to_index.get(0, 0))
    if not (0 <= default_action_index < num_actions):
        raise ValueError(
            f"Default action index {default_action_index} out of range [0, {num_actions - 1}]"
        )

    model: torch.nn.Module = VideoLatentDiT(
        latent_channels=latent_channels,
        num_actions=num_actions,
        action_dim=args.action_dim,
        action_values=action_values,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        max_frames=args.max_frames,
    ).to(device)
    console.print(
        f"[actions] num_actions={num_actions} default_index={default_action_index} "
        f"action_dim={args.action_dim}"
    )

    if args.compile:
        console.print("[compile] Compiling model with torch.compile()...")
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
        return 0.1 + 0.9 * cosine

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
        console.print(f"[resume] Loaded training state from {args.resume_from} at step {start_step}")

        if args.max_steps > 0 and start_step >= args.max_steps:
            console.print(f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do.")
            return

    config = vars(args).copy()
    config.update(
        {
            "data_dir":serialize_project_path(data_dir, project_root=PROJECT_ROOT),
            "dataset_format": "latents",
            "model_name": "video_latent_dit",
            "num_colors": int(num_colors) if num_colors is not None else None,
            "num_actions": int(num_actions),
            "action_values": [int(v) for v in action_values],
            "default_action_index": int(default_action_index),
            "latent_channels": int(latent_channels),
            "num_parameters": int(sum(p.numel() for p in model.parameters())),
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
            "video_vae_checkpoint": (
                serialize_project_path(video_vae_checkpoint_value, project_root=PROJECT_ROOT)
                if video_vae_checkpoint_value is not None
                else None
            ),
            "video_vae_config": (
                serialize_project_path(video_vae_config_value, project_root=PROJECT_ROOT)
                if video_vae_config_value is not None
                else None
            ),
            "latent_normalization_enabled": latent_normalization is not None,
            "latent_stats_path": (
                serialize_project_path(latent_stats_path, project_root=PROJECT_ROOT)
                if latent_stats_path is not None
                else None
            ),
        }
    )
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    console.print(f"Training video latent DiT on {len(train_dataset)} samples")
    console.print(f"Output directory: {output_dir}")

    error_buffer = ResidualErrorBuffer(
        args.error_buffer_size,
        clip_abs=args.error_buffer_clip,
    )

    start_time = time.time()
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
            latents, actions = prepare_batch_latents(
                batch,
                device=device,
                default_action_index=default_action_index,
                latent_normalization=latent_normalization,
            )

            history = latents[:, :, : args.context_frames]
            future = latents[:, :, args.context_frames :]

            if (
                error_buffer.enabled()
                and step >= args.error_inject_start_step
                and len(error_buffer) > 0
                and random.random() < args.error_inject_prob
            ):
                perturb = error_buffer.sample(
                    batch_size=history.shape[0],
                    device=history.device,
                    dtype=history.dtype,
                    frames=args.context_frames,
                )
                if perturb is not None:
                    history = history + args.error_inject_scale * perturb

            noisy_future, velocity_target, t, eps = sample_flow_target(future)
            model_input = torch.cat((history, noisy_future), dim=2)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                velocity_pred_future = model(model_input, actions, t, args.context_frames)
                loss = flow_loss(
                    velocity_pred_future,
                    velocity_target,
                    loss_type=args.flow_loss,
                    huber_delta=args.huber_delta,
                )

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                if error_buffer.enabled() and len(error_buffer) > 0:
                    error_buffer.clear()
                status = (
                    "loss=nonfinite(skip,bufclr) "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"gnorm=nan errbuf={len(error_buffer)}"
                )
                progress.update(train_task, advance=1, status=status)
                continue

            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()
                if (
                    args.grad_norm_skip > 0.0
                    and math.isfinite(grad_norm)
                    and grad_norm > args.grad_norm_skip
                ):
                    optimizer.zero_grad(set_to_none=True)
                    if error_buffer.enabled() and len(error_buffer) > 0:
                        error_buffer.clear()
                    grad_scaler.update()
                    status = (
                        f"loss={loss.item():.5f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"gnorm={grad_norm:.2f} skip=gradnorm(bufclr) errbuf={len(error_buffer)}"
                    )
                    progress.update(train_task, advance=1, status=status)
                    continue
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()
                if (
                    args.grad_norm_skip > 0.0
                    and math.isfinite(grad_norm)
                    and grad_norm > args.grad_norm_skip
                ):
                    optimizer.zero_grad(set_to_none=True)
                    if error_buffer.enabled() and len(error_buffer) > 0:
                        error_buffer.clear()
                    status = (
                        f"loss={loss.item():.5f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"gnorm={grad_norm:.2f} skip=gradnorm(bufclr) errbuf={len(error_buffer)}"
                    )
                    progress.update(train_task, advance=1, status=status)
                    continue
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                x0_hat = eps - velocity_pred_future
                residual = x0_hat - future
                error_buffer.add_batch(residual)

            status = (
                f"loss={loss.item():.5f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"gnorm={grad_norm:.2f} "
                f"errbuf={len(error_buffer)}"
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

                eval_metrics = evaluate(
                    model,
                    loader=eval_loader,
                    device=device,
                    context_frames=args.context_frames,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                    default_action_index=default_action_index,
                    loss_type=args.flow_loss,
                    huber_delta=args.huber_delta,
                    latent_normalization=latent_normalization,
                )
                eval_metrics["step"] = step
                eval_metrics["lr"] = float(scheduler.get_last_lr()[0])
                metrics.append(eval_metrics)
                with (output_dir / "metrics.json").open("w") as handle:
                    json.dump(metrics, handle, indent=2)

                eval_line = (
                    f"eval step={step:06d} flow_loss={eval_metrics['flow_loss']:.6f} "
                    f"x0_mse={eval_metrics['x0_mse']:.6f}"
                )

                preview_enabled = video_vae is not None and palette is not None
                with torch.no_grad():
                    try:
                        if preview_enabled:
                            train_batch = next(train_iter)
                            latents_preview, train_actions = prepare_batch_latents(
                                train_batch,
                                device=device,
                                default_action_index=default_action_index,
                                latent_normalization=latent_normalization,
                            )

                            sample_idx = random.randrange(latents_preview.shape[0])
                            latents_preview = latents_preview[sample_idx : sample_idx + 1]
                            actions_preview = train_actions[sample_idx : sample_idx + 1]

                            history_preview = latents_preview[:, :, : args.context_frames]
                            sampled_future = denoise_future_segment(
                                unwrap_model(model),
                                history_latents=history_preview,
                                actions=actions_preview,
                                future_frames=args.clip_frames - args.context_frames,
                                ode_steps=args.preview_ode_steps,
                                autocast_enabled=autocast_enabled,
                                autocast_dtype=autocast_dtype,
                            )
                            sampled_full = torch.cat((history_preview, sampled_future), dim=2)

                            decode_sampled_full = sampled_full
                            decode_latents_preview = latents_preview
                            if latent_normalization is not None:
                                decode_sampled_full = latent_normalization.denormalize(decode_sampled_full)
                                decode_latents_preview = latent_normalization.denormalize(decode_latents_preview)

                            sampled_frames = latents_to_frame_indices(
                                video_vae,
                                decode_sampled_full,
                                autocast_enabled=autocast_enabled,
                                autocast_dtype=autocast_dtype,
                            )
                            frames_preview = latents_to_frame_indices(
                                video_vae,
                                decode_latents_preview,
                                autocast_enabled=autocast_enabled,
                                autocast_dtype=autocast_dtype,
                            )

                            save_preview_image(
                                output_dir / f"preview_step_{step:06d}.png",
                                frames_gt=frames_preview[0],
                                frames_sample=sampled_frames[0],
                                palette=palette,
                            )
                    except torch.cuda.OutOfMemoryError:
                        log_console.print("[eval][OOM] Preview generation OOM; skipping preview image.")
                        if torch.cuda.is_available():
                            gc.collect()
                            torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

                if eval_metrics["flow_loss"] < best_eval:
                    best_eval = eval_metrics["flow_loss"]
                    save_training_state(
                        output_dir / "video_latent_dit_best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                    )
                    eval_line += f" | best={best_eval:.6f}"

                log_console.print(eval_line)

            should_checkpoint = (
                (args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0)
                or step == args.max_steps - 1
            )
            if should_checkpoint:
                save_training_state(
                    output_dir / "video_latent_dit_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                )


if __name__ == "__main__":
    main()
