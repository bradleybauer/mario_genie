#!/usr/bin/env python3
"""Train VideoLatentDiT (Diffusers backend) with Accelerate.

- Accelerator replaces manual torch.autocast / GradScaler boilerplate.
    Pass --mixed-precision bf16 (default), fp16, or no.
- Safetensors (.safetensors) for model-only checkpoints; accelerator.save_state()
    for full training-state (optimizer + scheduler) checkpoints.
- Optional shifted logit-normal timestep sampler (--timestep-sampler logit_normal)
    concentrates training on mid-range noise levels, which improves sample quality.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import diffusers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from rich.console import Console
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.latent_dataset import LatentSequenceDataset
from src.models.video_latent_dit_diffusers import VideoLatentDiTDiffusers
from src.models.video_vae import VideoVAE
from src.path_utils import resolve_workspace_path
from src.system_info import collect_system_info, print_system_info
from src.training.trainer_common import (
    build_trainer_config,
    build_warmup_cosine_scheduler,
    configure_cuda_runtime,
    gpu_stats,
    is_periodic_event_due,
    make_output_dir,
    preview_path,
    should_log_step,
)
from src.training.training_utils import (
    ThroughputTracker,
    advance_progress,
    build_eval_loader,
    build_progress,
    build_replacement_train_loader,
    create_accelerator_runtime,
    infinite_batches,
    save_json,
    save_metrics_json,
    split_train_eval_dataset,
    unwrap_model,
)

console = Console()


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------

class UniformTimestepSampler:
    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype)


class LogitNormalTimestepSampler:
    """Shifted logit-normal sampler that concentrates on mid-range noise levels.

    Ported from LTX-2/packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py.
    """

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1) -> None:
        self.std = std
        self.eps = eps
        self.uniform_prob = uniform_prob
        self._p999 = 3.0902 * std
        self._p005 = -2.5758 * std

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        normal = torch.randn(batch_size, device=device) * self.std
        logitnormal = torch.sigmoid(normal)

        p999 = torch.sigmoid(torch.tensor(self._p999, device=device))
        p005 = torch.sigmoid(torch.tensor(self._p005, device=device))

        raw = (logitnormal - p005) / (p999 - p005)
        stretched = torch.where(raw >= self.eps, raw, 2 * self.eps - raw).clamp(0, 1)

        uniform = (1 - self.eps) * torch.rand(batch_size, device=device) + self.eps
        mask = torch.rand(batch_size, device=device) > self.uniform_prob
        return torch.where(mask, stretched, uniform).to(dtype)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VideoLatentDiT with Accelerate.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--disable-latent-normalization", action="store_true")

    parser.add_argument("--clip-latents", type=int, default=256)
    parser.add_argument("--context-latents", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))

    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)

    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Sets encoder and decoder layers to half this value each.")
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--num-decoder-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--flow-loss", type=str, choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=10.0)

    parser.add_argument("--timestep-sampler", type=str, choices=["uniform", "logit_normal"],
                        default="uniform")

    parser.add_argument(
        "--error-buffer-size", type=int, default=1024,
        help="Residual clips retained for history perturbation.",
    )
    parser.add_argument("--error-inject-prob", type=float, default=0.2)
    parser.add_argument("--error-inject-scale", type=float, default=0.05)
    parser.add_argument("--error-inject-start-step", type=int, default=20000)
    parser.add_argument("--error-buffer-clip", type=float, default=3.0)

    parser.add_argument("--preview-ode-steps", type=int, default=8)
    parser.add_argument("--preview-interval", type=int, default=None,
                        help="Steps between rollout previews (default: eval-interval).")
    parser.add_argument("--preview-rollout-latents", type=int, default=64,
                        help="Total latents to generate via autoregressive rollout.")

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument("--mixed-precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode passed to Accelerator.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None)

    args = parser.parse_args()

    if args.clip_latents < 2:
        parser.error("--clip-latents must be >= 2")
    if args.context_latents < 1:
        parser.error("--context-latents must be >= 1")
    if args.context_latents >= args.clip_latents:
        parser.error("--context-latents must be smaller than --clip-latents")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")

    default_half = (args.num_layers // 2)
    if args.num_encoder_layers is None:
        args.num_encoder_layers = default_half
    if args.num_decoder_layers is None:
        args.num_decoder_layers = default_half
    if args.num_encoder_layers <= 0:
        parser.error("--num-encoder-layers must be > 0")
    if args.num_decoder_layers <= 0:
        parser.error("--num-decoder-layers must be > 0")

    return args


# ---------------------------------------------------------------------------
# Utilities carried over from the original trainer
# ---------------------------------------------------------------------------

@dataclass
class LatentNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    stats_path: str
    scheme: str
    version: int

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        p = Path(args.data_dir).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {p}")
        return p
    candidate = PROJECT_ROOT / "data" / "latents"
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError("Could not infer --data-dir. Pass it explicitly.")


def _resolve_path(value: str | None, *, config_dir: Path | None = None) -> Path | None:
    return resolve_workspace_path(value, project_root=PROJECT_ROOT, config_dir=config_dir)


def _is_readable(p: Path | None) -> bool:
    try:
        return p is not None and p.is_file()
    except (PermissionError, OSError):
        return False


def load_latent_normalization(
    *,
    stats_path: Path,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
) -> LatentNormalization:
    stats = _load_json(stats_path)
    version_raw = stats.get("latent_stats_version")
    if version_raw is None:
        raise ValueError(
            f"{stats_path}: missing latent_stats_version. Regenerate with "
            "scripts/collect/encode_latents.py --stats-only."
        )
    try:
        version = int(version_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{stats_path}: latent_stats_version must be an integer, got {version_raw!r}."
        ) from exc
    if version != 2:
        raise ValueError(
            f"{stats_path}: latent_stats_version={version} is not supported. "
            "Expected version 2."
        )

    scheme = stats.get("normalization_scheme")
    if scheme != "component_chw_shared_time":
        raise ValueError(
            f"Unsupported latent normalization scheme '{scheme}' in {stats_path}; "
            "expected 'component_chw_shared_time'."
        )

    mean_v = stats.get("component_mean")
    std_v = stats.get("component_std_clamped")
    if mean_v is None or std_v is None:
        raise ValueError(
            f"{stats_path}: missing component_mean/component_std_clamped. "
            "Legacy latent stats are no longer supported; regenerate with "
            "scripts/collect/encode_latents.py --stats-only."
        )

    expected_shape = (latent_channels, latent_height, latent_width)
    mean_np = np.asarray(mean_v, dtype=np.float32)
    std_np = np.asarray(std_v, dtype=np.float32)
    if mean_np.shape != expected_shape or std_np.shape != expected_shape:
        raise ValueError(
            f"Latent component stats shape mismatch: expected {expected_shape}, "
            f"got mean={mean_np.shape}, std={std_np.shape}"
        )

    eps = max(float(stats.get("std_epsilon", 1e-6)), 1e-6)
    mean = torch.from_numpy(mean_np).to(device=device, dtype=torch.float32).view(
        1, latent_channels, 1, latent_height, latent_width
    )
    std = torch.from_numpy(std_np).to(device=device, dtype=torch.float32).view(
        1, latent_channels, 1, latent_height, latent_width
    )
    std = torch.nan_to_num(std, nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    return LatentNormalization(
        mean=mean,
        std=std,
        stats_path=str(stats_path),
        scheme="component_chw_shared_time",
        version=version,
    )


def load_latent_stats_path(
    *, data_dir: Path, latent_meta: dict | None, disable: bool = False
) -> Path | None:
    if disable:
        return None
    if latent_meta is not None:
        for key in ("latent_stats_path", "latent_stats_file"):
            v = latent_meta.get(key)
            if isinstance(v, str):
                p = _resolve_path(v, config_dir=data_dir)
                if _is_readable(p):
                    return p
    default = data_dir / "latent_stats.json"
    return default.resolve() if default.is_file() else None


def load_palette_tensor(data_dir: Path) -> torch.Tensor:
    info = _load_json(data_dir / "palette.json")
    return torch.tensor(info["colors_rgb"], dtype=torch.float32) / 255.0


def load_video_vae(
    *, checkpoint_path: Path, config_path: Path, num_colors: int | None, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    cfg = _load_json(config_path)
    model_cfg = dict(cfg.get("model", {}))
    data_cfg = dict(cfg.get("data", {}))
    base_channels = int(model_cfg.get("base_channels", cfg.get("base_channels", 64)))
    latent_channels = int(model_cfg.get("latent_channels", cfg.get("latent_channels", 64)))
    temporal_downsample = int(model_cfg.get("temporal_downsample", cfg.get("temporal_downsample", 0)))
    dropout = float(model_cfg.get("dropout", cfg.get("dropout", 0.0)))
    global_bottleneck_attn = bool(
        model_cfg.get("global_bottleneck_attn", cfg.get("global_bottleneck_attn", False))
    )
    global_bottleneck_attn_heads = int(
        model_cfg.get("global_bottleneck_attn_heads", cfg.get("global_bottleneck_attn_heads", 8))
    )
    vae_num_colors = int(data_cfg.get("num_colors", cfg.get("num_colors", num_colors or 0)))
    if vae_num_colors <= 0:
        raise ValueError("Cannot determine VAE num_colors.")
    vae = VideoVAE(
        num_colors=vae_num_colors,
        base_channels=base_channels,
        latent_channels=latent_channels,
        temporal_downsample=temporal_downsample,
        dropout=dropout,
        global_bottleneck_attn=global_bottleneck_attn,
        global_bottleneck_attn_heads=global_bottleneck_attn_heads,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, {
        "latent_channels": latent_channels,
        "num_colors": vae_num_colors,
        "temporal_downsample": temporal_downsample,
    }


def shift_actions_causal(actions: torch.Tensor, *, default_action_index: int) -> torch.Tensor:
    shifted = torch.empty_like(actions)
    shifted[:, 0] = default_action_index
    shifted[:, 1:] = actions[:, :-1]
    return shifted


def prepare_batch(
    batch: dict,
    *,
    device: torch.device,
    default_action_index: int,
    latent_normalization: LatentNormalization | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = shift_actions_causal(
        batch["actions"].to(device, non_blocking=True).long(),
        default_action_index=default_action_index,
    )
    latents = batch["latents"].to(device, non_blocking=True).float()
    if latent_normalization is not None:
        latents = latent_normalization.normalize(latents)
    return latents, actions


def sample_flow_target(
    x0: torch.Tensor, t_sampler: UniformTimestepSampler | LogitNormalTimestepSampler
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear flow matching: x_t = (1-t)*x0 + t*eps,  v* = eps - x0."""
    t = t_sampler.sample(x0.shape[0], device=x0.device, dtype=x0.dtype)
    eps = torch.randn_like(x0)
    t_b = t.view(x0.shape[0], 1, 1, 1, 1)
    x_t = (1.0 - t_b) * x0 + t_b * eps
    return x_t, eps - x0, t, eps


def compute_flow_loss(
    pred: torch.Tensor, target: torch.Tensor, *, loss_type: str, huber_delta: float
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target, beta=huber_delta)


def build_video_latent_dit(*, model_config: dict) -> torch.nn.Module:
    return VideoLatentDiTDiffusers(**model_config)


# ---------------------------------------------------------------------------
# Residual error buffer (unchanged from original)
# ---------------------------------------------------------------------------

class ResidualErrorBuffer:
    def __init__(self, capacity: int, *, clip_abs: float = 0.0) -> None:
        self.capacity = int(capacity)
        self.clip_abs = max(float(clip_abs), 0.0)
        self._bank: deque[torch.Tensor] = deque(maxlen=max(self.capacity, 1))

    def enabled(self) -> bool:
        return self.capacity > 0

    def __len__(self) -> int:
        return len(self._bank)

    def clear(self) -> None:
        self._bank.clear()

    def add_batch(self, residual: torch.Tensor) -> None:
        if not self.enabled():
            return
        residual = residual.detach()
        if self.clip_abs > 0:
            residual = residual.clamp(-self.clip_abs, self.clip_abs)
        finite = torch.isfinite(residual.flatten(start_dim=1)).all(dim=1)
        for sample in residual[finite]:
            self._bank.append(sample.unsqueeze(0))

    def sample(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype, latents: int
    ) -> torch.Tensor | None:
        if len(self._bank) == 0:
            return None
        indices = [random.randrange(len(self._bank)) for _ in range(batch_size)]
        samples = [self._bank[i] for i in indices]
        out = torch.cat(samples, dim=0).to(device=device, dtype=dtype)
        B, C, T, H, W = out.shape
        if T < latents:
            out = out.repeat(1, 1, math.ceil(latents / T), 1, 1)[:, :, :latents]
        elif T > latents:
            out = out[:, :, :latents]
        return out


# ---------------------------------------------------------------------------
# Checkpoint save / load (safetensors + JSON config)
# ---------------------------------------------------------------------------

def save_model_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    model_config: dict,
    step: int,
    best_eval: float,
    metrics: list,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)
    safetensors_save(raw.state_dict(), path / "model.safetensors")

    # Persist a native Diffusers bundle when using a ModelMixin backend.
    if hasattr(raw, "save_pretrained"):
        try:
            raw.save_pretrained(path / "diffusers", safe_serialization=True)
        except TypeError:
            raw.save_pretrained(path / "diffusers")

    meta = {
        "model_config": model_config,
        "step": step,
        "best_eval": best_eval,
    }
    with (path / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    save_metrics_json(path / "metrics.json", metrics)


def load_model_weights(model: torch.nn.Module, path: Path) -> None:
    raw = unwrap_model(model)
    raw.load_state_dict(safetensors_load(path / "model.safetensors"))


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_future_segment(
    model: torch.nn.Module,
    *,
    history_latents: torch.Tensor,
    actions: torch.Tensor,
    future_latents: int,
    ode_steps: int,
    accelerator: Accelerator,
) -> torch.Tensor:
    batch, channels, context_latents, height, width = history_latents.shape
    future = torch.randn(batch, channels, future_latents, height, width,
                         device=history_latents.device, dtype=history_latents.dtype)

    raw_model = accelerator.unwrap_model(model)
    with accelerator.autocast():
        encoded = raw_model.encode_history(history_latents, actions[:, :context_latents])

    dt = 1.0 / float(ode_steps)
    for step in range(ode_steps, 0, -1):
        t_val = (step - 0.5) / float(ode_steps)
        t = torch.full((batch,), t_val, device=history_latents.device, dtype=history_latents.dtype)
        with accelerator.autocast():
            velocity = raw_model.decode_future(future, actions, t, encoded, context_latents)
        future = future - dt * velocity
    return future


@torch.no_grad()
def autoregressive_rollout(
    model: torch.nn.Module,
    *,
    seed_latents: torch.Tensor,
    all_actions: torch.Tensor,
    context_latents: int,
    future_latents: int,
    total_latents: int,
    ode_steps: int,
    accelerator: Accelerator,
) -> torch.Tensor:
    """Autoregressively roll out latent timesteps.

    *seed_latents* provides the initial ``context_latents`` of real data.
    At each step, the last ``context_latents`` generated latents become the
    new history and ``future_latents`` are predicted via ODE integration.
    *all_actions* must cover ``total_latents``; pad with the default action
    for latents beyond the dataset clip.
    """
    generated = seed_latents  # (B, C, context_latents, H, W)
    while generated.shape[2] < total_latents:
        step_future = min(future_latents, total_latents - generated.shape[2])
        history = generated[:, :, -context_latents:]
        abs_start = generated.shape[2] - context_latents
        abs_end = generated.shape[2] + step_future
        step_actions = all_actions[:, abs_start:abs_end]
        pred = denoise_future_segment(
            model,
            history_latents=history,
            actions=step_actions,
            future_latents=step_future,
            ode_steps=ode_steps,
            accelerator=accelerator,
        )
        generated = torch.cat([generated, pred], dim=2)
    return generated


@torch.inference_mode()
def latents_to_frame_indices(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    accelerator: Accelerator,
) -> torch.Tensor:
    with accelerator.autocast():
        logits = vae.decode(latents)
    return logits.argmax(dim=1)


def save_preview_image(
    path: Path,
    *,
    frames_gt: torch.Tensor,
    frames_sample: torch.Tensor,
    palette: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pal = (palette.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    gt_rgb = pal[frames_gt.detach().cpu().numpy()]
    sa_rgb = pal[frames_sample.detach().cpu().numpy()]
    rows = [np.concatenate((gt_rgb[t], sa_rgb[t]), axis=1) for t in range(gt_rgb.shape[0])]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def save_rollout_preview(
    path: Path,
    *,
    frames_gt: torch.Tensor,
    frames_pred: torch.Tensor,
    palette: torch.Tensor,
) -> None:
    """Save a rollout preview: GT (left column) and prediction (right column).

    Each row shows one predicted timestep with ground truth on the left and
    the model's prediction on the right.  GT timesteps beyond the available
    length are rendered as dark-gray placeholders.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pal = (palette.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    gt = pal[frames_gt.cpu().numpy()]        # (T_gt, H, W, 3)
    pred = pal[frames_pred.cpu().numpy()]    # (T, H, W, 3)
    T = pred.shape[0]
    T_gt = gt.shape[0]
    H, W = pred.shape[1], pred.shape[2]
    blank = np.full((H, W, 3), 32, dtype=np.uint8)

    rows = []
    for t in range(T):
        gt_frame = gt[t] if t < T_gt else blank
        rows.append(np.concatenate([gt_frame, pred[t]], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    *,
    loader: DataLoader,
    device: torch.device,
    context_latents: int,
    accelerator: Accelerator,
    default_action_index: int,
    loss_type: str,
    huber_delta: float,
    latent_normalization: LatentNormalization | None,
    t_sampler: UniformTimestepSampler | LogitNormalTimestepSampler,
) -> dict[str, float]:
    # Unwrap to avoid DDP collective ops (broadcast_buffers) that would
    # deadlock when only rank 0 runs evaluation.
    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()
    flow_losses: list[float] = []
    x0_mses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            latents, actions = prepare_batch(
                batch,
                device=device,
                default_action_index=default_action_index,
                latent_normalization=latent_normalization,
            )
            history = latents[:, :, :context_latents]
            future = latents[:, :, context_latents:]
            noisy_future, v_target, t, eps = sample_flow_target(future, t_sampler)
            model_input = torch.cat((history, noisy_future), dim=2)

            with accelerator.autocast():
                v_pred = raw_model(model_input, actions, t, context_latents)

            flow_losses.append(
                compute_flow_loss(v_pred, v_target, loss_type=loss_type, huber_delta=huber_delta).item()
            )
            x0_mses.append(F.mse_loss(eps - v_pred, future).item())

    raw_model.train()
    if not flow_losses:
        return {"flow_loss": 0.0, "x0_mse": 0.0}
    return {"flow_loss": float(np.mean(flow_losses)), "x0_mse": float(np.mean(x0_mses))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = _resolve_data_dir(args)
    output_dir = make_output_dir(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        run_name=args.run_name,
        default_prefix="video_latent_dit",
        resume_parent=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = create_accelerator_runtime(
        output_dir=output_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator = runtime.accelerator
    device = accelerator.device

    if torch.cuda.is_available():
        configure_cuda_runtime(matmul_precision="high")

    system_info = collect_system_info()
    if accelerator.is_main_process:
        print_system_info(system_info)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    latent_meta: dict | None = None
    latent_meta_path = data_dir / "latent_config.json"
    if latent_meta_path.is_file():
        latent_meta = _load_json(latent_meta_path)

    with accelerator.main_process_first():
        dataset = LatentSequenceDataset(
            data_dir=data_dir,
            clip_latents=args.clip_latents,
            include_actions=True,
            num_workers=args.num_workers,
            system_info=system_info,
        )
    if len(dataset) == 0:
        raise RuntimeError("No training samples found.")

    latent_frame_height = int(latent_meta.get("frame_height", 0)) if latent_meta is not None else 0
    latent_frame_width = int(latent_meta.get("frame_width", 0)) if latent_meta is not None else 0
    latent_source_height = int(latent_meta.get("source_frame_height", latent_frame_height or 0)) if latent_meta is not None else 0
    latent_source_width = int(latent_meta.get("source_frame_width", latent_frame_width or 0)) if latent_meta is not None else 0
    expected_latent_height = int(latent_meta.get("latent_height", 0)) if latent_meta is not None else 0
    expected_latent_width = int(latent_meta.get("latent_width", 0)) if latent_meta is not None else 0
    if expected_latent_height > 0 and expected_latent_height != int(dataset.latent_height):
        raise ValueError(
            f"Latent dataset height mismatch: metadata says {expected_latent_height}, indexed dataset says {dataset.latent_height}"
        )
    if expected_latent_width > 0 and expected_latent_width != int(dataset.latent_width):
        raise ValueError(
            f"Latent dataset width mismatch: metadata says {expected_latent_width}, indexed dataset says {dataset.latent_width}"
        )

    train_dataset, eval_dataset = split_train_eval_dataset(
        dataset,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )

    if accelerator.is_main_process:
        console.print(f"Found {len(dataset)} segments of {args.clip_latents} latents.")
        if eval_dataset:
            console.print(f"Eval split: {len(eval_dataset)} eval, {len(train_dataset)} train")

    train_loader = build_replacement_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = build_eval_loader(
        eval_dataset,
        batch_size=args.eval_batch_size or args.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Latent normalization
    # ------------------------------------------------------------------
    latent_channels = int(dataset.latent_channels)
    latent_stats_path = load_latent_stats_path(
        data_dir=data_dir, latent_meta=latent_meta,
        disable=args.disable_latent_normalization,
    )
    latent_normalization: LatentNormalization | None = None
    if latent_stats_path is not None:
        latent_normalization = load_latent_normalization(
            stats_path=latent_stats_path,
            latent_channels=latent_channels,
            latent_height=int(dataset.latent_height),
            latent_width=int(dataset.latent_width),
            device=device,
        )
        if accelerator.is_main_process:
            console.print(
                f"[latents] Component normalization ({latent_normalization.scheme}, "
                f"v{latent_normalization.version}) from {latent_stats_path}"
            )
            if latent_frame_height > 0 and latent_frame_width > 0:
                console.print(
                    f"[latents] Source frames {latent_source_height}x{latent_source_width} -> "
                    f"encoded frames {latent_frame_height}x{latent_frame_width} -> "
                    f"latent grid {dataset.latent_height}x{dataset.latent_width}"
                )

    # ------------------------------------------------------------------
    # Palette and preview VAE
    # ------------------------------------------------------------------
    palette: torch.Tensor | None = None
    video_vae: torch.nn.Module | None = None
    palette_path = data_dir / "palette.json"
    if palette_path.is_file():
        palette = load_palette_tensor(data_dir)
    elif accelerator.is_main_process:
        console.print(
            f"[preview] Palette not found at {palette_path}; previews disabled",
            markup=False,
        )

    vae_ckpt = latent_meta.get("video_vae_checkpoint") if latent_meta is not None else None
    vae_cfg = latent_meta.get("video_vae_config_path") if latent_meta is not None else None
    vae_ckpt_path = _resolve_path(vae_ckpt, config_dir=data_dir)
    vae_cfg_path = _resolve_path(vae_cfg, config_dir=data_dir)
    if not _is_readable(vae_cfg_path) and _is_readable(vae_ckpt_path):
        sibling = vae_ckpt_path.parent / "config.json"
        if sibling.is_file():
            vae_cfg_path = sibling
    if accelerator.is_main_process:
        console.print(
            f"[preview] Resolved VAE checkpoint: {vae_ckpt_path}",
            markup=False,
        )
        console.print(
            f"[preview] Resolved VAE config: {vae_cfg_path}",
            markup=False,
        )
    if _is_readable(vae_ckpt_path) and _is_readable(vae_cfg_path):
        try:
            video_vae, vae_summary = load_video_vae(
                checkpoint_path=vae_ckpt_path, config_path=vae_cfg_path,
                num_colors=int(palette.shape[0]) if palette is not None else None,
                device=device,
            )
            if accelerator.is_main_process:
                console.print("[preview] Loaded video VAE for preview generation", markup=False)
        except Exception as exc:
            if accelerator.is_main_process:
                console.print(
                    f"[preview] Failed to load VAE ({exc}); previews disabled",
                    markup=False,
                )
    elif accelerator.is_main_process:
        console.print(
            "[preview] Missing readable VAE checkpoint/config; previews disabled",
            markup=False,
        )

    # ------------------------------------------------------------------
    # Actions metadata
    # ------------------------------------------------------------------
    actions_path = data_dir / "actions.json"
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing actions metadata: {actions_path}")
    actions_info = _load_json(actions_path)
    num_actions = int(actions_info.get("num_actions", 0))
    if num_actions <= 0:
        raise ValueError(f"Invalid num_actions in {actions_path}")
    action_values_meta = actions_info.get("reduced_to_original_value")
    action_values = (
        [int(v) for v in action_values_meta]
        if action_values_meta is not None
        else list(range(num_actions))
    )
    default_action_index = int({v: i for i, v in enumerate(action_values)}.get(0, 0))
    if accelerator.is_main_process:
        console.print(f"[actions] num_actions={num_actions} default_index={default_action_index}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = dict(
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
        max_latents=args.clip_latents,
    )
    model: torch.nn.Module = build_video_latent_dit(model_config=model_config).to(device)
    diffusers_version = str(diffusers.__version__)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        console.print(f"[model] backend=diffusers diffusers={diffusers_version}")
        console.print(f"[model] {num_params:,} trainable parameters")

    # Compilation strategy:
    # - Single-GPU: compile here before accelerator.prepare().
    # - Multi-GPU: skip manual compile; use `accelerate launch --dynamo_backend inductor`
    #   which applies torch.compile during accelerator.prepare() in the correct DDP order.
    do_compile = args.compile and accelerator.num_processes == 1
    if args.compile and accelerator.num_processes > 1 and accelerator.is_main_process:
        console.print(
            "[compile] Multi-GPU: skipping manual torch.compile. "
            "Use `--dynamo_backend inductor` in the accelerate launch command instead."
        )
    if do_compile:
        if accelerator.is_main_process:
            console.print("[compile] torch.compile()...")
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    warmup_steps = max(args.warmup_steps, 0)
    lr_scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        min_lr_scale=0.1,
    )

    # ------------------------------------------------------------------
    # Accelerator prepare
    # ------------------------------------------------------------------
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Disable buffer broadcasting — the only buffer (action_bit_lut) is a
    # constant lookup table identical on every rank.  DDP broadcasts buffers
    # on each forward pass, which causes NCCL deadlocks when rank-0-only
    # code (eval/checkpoint) runs between forward calls.
    if hasattr(model, "broadcast_buffers"):
        model.broadcast_buffers = False

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    best_eval = float("inf")
    metrics: list[dict] = []

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        loaded_full_state = False
        loaded_model_weights = False
        if (resume_path / "training_state").is_dir():
            accelerator.load_state(str(resume_path / "training_state"))
            loaded_full_state = True
        elif (resume_path / "model.safetensors").is_file():
            load_model_weights(model, resume_path)
            loaded_model_weights = True
        meta_path = resume_path / "meta.json"
        if meta_path.is_file():
            meta = _load_json(meta_path)
            start_step = int(meta.get("step", 0)) + 1
            best_eval = float(meta.get("best_eval", float("inf")))
        metrics_path = resume_path / "metrics.json"
        if metrics_path.is_file():
            metrics = _load_json(metrics_path)
        if accelerator.is_main_process:
            if loaded_full_state:
                console.print(f"[resume] Loaded full training_state; resuming from step {start_step}")
            elif loaded_model_weights:
                console.print(f"[resume] Loaded model-only checkpoint; resuming from step {start_step}")
            else:
                console.print(f"[resume] Found metadata only; resuming from step {start_step}")

        if args.max_steps > 0 and start_step >= args.max_steps:
            if accelerator.is_main_process:
                console.print(
                    f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do."
                )
            return

    # ------------------------------------------------------------------
    # Timestep sampler
    # ------------------------------------------------------------------
    t_sampler: UniformTimestepSampler | LogitNormalTimestepSampler = (
        LogitNormalTimestepSampler() if args.timestep_sampler == "logit_normal"
        else UniformTimestepSampler()
    )

    # ------------------------------------------------------------------
    # Save training config
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        save_json(
            output_dir / "config.json",
            build_trainer_config(
                model_name="video_latent_dit",
                args=args,
                device=device,
                mixed_precision=args.mixed_precision,
                num_processes=accelerator.num_processes,
                data={
                    "latent_channels": int(latent_channels),
                    "latent_height": int(dataset.latent_height),
                    "latent_width": int(dataset.latent_width),
                    "num_actions": int(num_actions),
                    "clip_latents": int(args.clip_latents),
                    "context_latents": int(args.context_latents),
                    "frame_height": int(latent_frame_height) if latent_frame_height > 0 else None,
                    "frame_width": int(latent_frame_width) if latent_frame_width > 0 else None,
                    "source_frame_height": int(latent_source_height) if latent_source_height > 0 else None,
                    "source_frame_width": int(latent_source_width) if latent_source_width > 0 else None,
                    "latent_stats_path": str(latent_stats_path) if latent_stats_path is not None else None,
                    "latent_stats_version": int(latent_normalization.version) if latent_normalization is not None else None,
                    "latent_normalization_scheme": latent_normalization.scheme if latent_normalization is not None else None,
                },
                model={
                    "num_parameters": int(num_params),
                    "config": model_config,
                    "backend": "diffusers",
                    "diffusers_version": diffusers_version,
                },
            ),
            indent=2,
        )

    error_buffer = ResidualErrorBuffer(args.error_buffer_size, clip_abs=args.error_buffer_clip)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _component_gnorms(raw_model: torch.nn.Module) -> dict[str, float]:
        groups: dict[str, list[torch.Tensor]] = {
            "enc_attn": [
                p for b in raw_model.encoder_blocks
                for name, p in b.named_parameters()
                if "attn" in name and p.grad is not None
            ],
            "dec_attn": [
                p for b in raw_model.decoder_blocks
                for name, p in b.named_parameters()
                if "attn" in name and p.grad is not None
            ],
            "enc_mlp": [
                p for b in raw_model.encoder_blocks
                for name, p in b.named_parameters()
                if "mlp" in name and p.grad is not None
            ],
            "dec_mlp": [
                p for b in raw_model.decoder_blocks
                for name, p in b.named_parameters()
                if "mlp" in name and p.grad is not None
            ],
            "out_proj": [p for p in raw_model.output_proj.parameters() if p.grad is not None],
            "in_proj":  [p for p in raw_model.input_proj.parameters()  if p.grad is not None],
        }
        return {
            k: torch.stack([p.grad.detach().float().norm() for p in ps]).norm().item()
            for k, ps in groups.items() if ps
        }

    train_iter = infinite_batches(train_loader)

    # Cache a single preview batch so we don't desync the training
    # iterator when running multi-GPU.
    _preview_batch = next(train_iter) if (video_vae is not None and palette is not None) else None
    if accelerator.is_main_process and _preview_batch is None:
        console.print(
            "[preview] Preview batch unavailable because preview prerequisites were not met",
            markup=False,
        )

    use_live = sys.stdout.isatty() and accelerator.is_main_process
    start_time = time.time()
    throughput_tracker = ThroughputTracker(window_steps=20)
    loss_window: deque[float] = deque(maxlen=100)
    train_log_path = output_dir / "train_log.jsonl"

    if accelerator.is_main_process:
        console.print(f"Training on {len(train_dataset)} samples | output: {output_dir}")

    with build_progress(use_live=use_live) as progress:
        log_console = progress.console
        task = progress.add_task("Training", total=max(args.max_steps - start_step, 0), status="")

        accum_steps = args.gradient_accumulation_steps

        for step in range(start_step, args.max_steps):
            accum_loss_sum = 0.0
            accum_v_pred = None
            accum_v_target = None
            accum_t = None
            accum_eps = None
            accum_future = None
            accum_batch_samples = 0
            skip_step = False
            grad_norm = 0.0

            for _accum_idx in range(accum_steps):
                batch = next(train_iter)
                latents, actions = prepare_batch(
                    batch,
                    device=device,
                    default_action_index=default_action_index,
                    latent_normalization=latent_normalization,
                )

                history = latents[:, :, :args.context_latents]
                future = latents[:, :, args.context_latents:]

                # Optional history perturbation
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
                        latents=args.context_latents,
                    )
                    if perturb is not None:
                        history = history + args.error_inject_scale * perturb

                noisy_future, v_target, t, eps = sample_flow_target(future, t_sampler)
                model_input = torch.cat((history, noisy_future), dim=2)

                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        v_pred = model(model_input, actions, t, args.context_latents)
                        loss = compute_flow_loss(v_pred, v_target, loss_type=args.flow_loss, huber_delta=args.huber_delta)

                    if not torch.isfinite(loss):
                        optimizer.zero_grad(set_to_none=True)
                        error_buffer.clear()
                        skip_step = True
                        break

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                accum_loss_sum += loss.item()
                accum_batch_samples += latents.shape[0]
                # Keep last micro-batch tensors for error buffer / logging.
                accum_v_pred = v_pred
                accum_v_target = v_target
                accum_t = t
                accum_eps = eps
                accum_future = future

            if skip_step:
                advance_progress(
                    progress,
                    task,
                    status=f"loss=nonfinite(skip) lr={lr_scheduler.get_last_lr()[0]:.2e}",
                )
                continue

            lr_scheduler.step()
            avg_loss = accum_loss_sum / accum_steps
            with torch.no_grad():
                error_buffer.add_batch(accum_eps - accum_v_pred - accum_future)
            loss_window.append(avg_loss)

            log_due = should_log_step(
                step,
                start_step=start_step,
                log_interval=args.log_interval,
                max_steps=args.max_steps,
            )
            should_log = accelerator.is_main_process and log_due
            if should_log:
                raw = accelerator.unwrap_model(model)
                cgnorms = _component_gnorms(raw)

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=accum_batch_samples * accelerator.num_processes,
            )

            advance_progress(
                progress,
                task,
                status=(
                    f"loss={avg_loss:.5f} lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    + f" errbuf={len(error_buffer)}"
                    + f" sps={samples_per_second:.1f}"
                ),
            )

            if should_log:
                with torch.no_grad():
                    vp = accum_v_pred.detach().float()
                    vt = accum_v_target.detach().float()
                    v_pred_norm = vp.norm(dim=1).mean().item()
                    v_target_norm = vt.norm(dim=1).mean().item()

                    # x0 reconstruction MSE (same formula as eval)
                    x0_mse = F.mse_loss(accum_eps.detach().float() - vp, accum_future.detach().float()).item()

                    # Cosine similarity between predicted and target velocity
                    vp_flat = vp.flatten(start_dim=1)
                    vt_flat = vt.flatten(start_dim=1)
                    cos_sim = F.cosine_similarity(vp_flat, vt_flat, dim=1).mean().item()

                    # Loss stratified by timestep third (early/mid/late)
                    t_vals = accum_t.detach().float()
                    loss_per_sample = (vp - vt).pow(2).flatten(start_dim=1).mean(dim=1)
                    t_low = t_vals < 1.0 / 3.0
                    t_mid = (t_vals >= 1.0 / 3.0) & (t_vals < 2.0 / 3.0)
                    t_high = t_vals >= 2.0 / 3.0
                    loss_t_early = loss_per_sample[t_low].mean().item() if t_low.any() else None
                    loss_t_mid = loss_per_sample[t_mid].mean().item() if t_mid.any() else None
                    loss_t_late = loss_per_sample[t_high].mean().item() if t_high.any() else None

                elapsed_s = time.time() - start_time
                train_entry: dict = {
                    "type": "train",
                    "step": step,
                    "elapsed_s": round(elapsed_s, 1),
                    "loss": avg_loss,
                    "loss_smooth": float(np.mean(loss_window)) if loss_window else avg_loss,
                    "x0_mse": x0_mse,
                    "v_cos_sim": cos_sim,
                    "loss_t_early": loss_t_early,
                    "loss_t_mid": loss_t_mid,
                    "loss_t_late": loss_t_late,
                    "grad_norm": grad_norm,
                    **{f"grad_norm_{k}": v for k, v in cgnorms.items()},
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                    "t_mean": float(accum_t.mean().item()),
                    "t_std": float(accum_t.std().item()),
                    "v_pred_norm": v_pred_norm,
                    "v_target_norm": v_target_norm,
                    "error_buffer_size": len(error_buffer),
                    "samples_per_sec": round(samples_per_second, 1),
                    "steps_per_sec": round(steps_per_second, 2),
                }
                train_entry.update(gpu_stats(device))
                metrics.append(train_entry)
                with train_log_path.open("a") as fh:
                    fh.write(json.dumps(train_entry) + "\n")
                if not use_live:
                    log_console.print(
                        f"step={step:06d} loss={train_entry['loss']:.5f} smooth={train_entry['loss_smooth']:.5f} "
                        f"lr={train_entry['lr']:.2e}"
                        + f" sps={samples_per_second:.1f}"
                    )

            # ------------------------------------------------------------------
            # Evaluation
            # ------------------------------------------------------------------
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
            preview_interval = args.preview_interval or args.eval_interval
            preview_due = is_periodic_event_due(
                step,
                interval=preview_interval,
                max_steps=args.max_steps,
            )
            if eval_due or should_checkpoint or preview_due:
                accelerator.wait_for_everyone()

            should_eval = eval_due and accelerator.is_main_process
            eval_line: str | None = None
            if should_eval:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                eval_metrics = evaluate(
                    model,
                    loader=eval_loader,
                    device=device,
                    context_latents=args.context_latents,
                    accelerator=accelerator,
                    default_action_index=default_action_index,
                    loss_type=args.flow_loss,
                    huber_delta=args.huber_delta,
                    latent_normalization=latent_normalization,
                    t_sampler=t_sampler,
                )
                eval_metrics["type"] = "eval"
                eval_metrics["step"] = step
                eval_metrics["elapsed_s"] = round(time.time() - start_time, 1)
                eval_metrics["lr"] = float(lr_scheduler.get_last_lr()[0])
                eval_metrics["train_grad_norm"] = grad_norm
                eval_cgnorms = _component_gnorms(accelerator.unwrap_model(model))
                eval_metrics.update({f"grad_norm_{k}": v for k, v in eval_cgnorms.items()})
                metrics.append(eval_metrics)

                eval_line = (
                    f"eval step={step:06d} flow_loss={eval_metrics['flow_loss']:.6f} "
                    f"x0_mse={eval_metrics['x0_mse']:.6f} "
                    f"gnorm={grad_norm:.2f} "
                    + " ".join(f"{k}={v:.2f}" for k, v in eval_cgnorms.items())
                )

                is_new_best = eval_metrics["flow_loss"] < best_eval
                if is_new_best:
                    best_eval = eval_metrics["flow_loss"]
                    save_model_checkpoint(
                        output_dir / "best",
                        model=model,
                        model_config=model_config,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                    )
                    eval_line += f" | best={best_eval:.6f}"

                save_metrics_json(output_dir / "metrics.json", metrics)

                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # ------------------------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------------------------
            if should_checkpoint and accelerator.is_main_process:
                latest_dir = output_dir / "latest"
                save_model_checkpoint(
                    latest_dir,
                    model=model,
                    model_config=model_config,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                )
                if eval_line is not None:
                    eval_line += " | saved latest"

            # save_state is collective — all ranks call it together.
            if should_checkpoint:
                accelerator.save_state(str(output_dir / "latest" / "training_state"))

            if eval_line is not None and accelerator.is_main_process:
                log_console.print(eval_line)

            # ------------------------------------------------------------------
            # Autoregressive rollout preview
            # ------------------------------------------------------------------
            if preview_due and accelerator.is_main_process:
                with torch.no_grad():
                    try:
                        if video_vae is not None and palette is not None and _preview_batch is not None:
                            t0_preview = time.perf_counter()
                            prev_latents, prev_actions = prepare_batch(
                                _preview_batch,
                                device=device,
                                default_action_index=default_action_index,
                                latent_normalization=latent_normalization,
                            )
                            idx = random.randrange(prev_latents.shape[0])
                            prev_latents = prev_latents[idx : idx + 1]
                            prev_actions = prev_actions[idx : idx + 1]

                            ctx = min(args.context_latents, prev_latents.shape[2])
                            future_per_step = max(int(args.clip_latents) - ctx, 1)
                            rollout_total = ctx + max(args.preview_rollout_latents, future_per_step)

                            # Pad actions to cover the full rollout
                            clip_actions = prev_actions.shape[1]
                            if clip_actions < rollout_total:
                                pad = torch.full(
                                    (1, rollout_total - clip_actions),
                                    default_action_index,
                                    device=prev_actions.device,
                                    dtype=prev_actions.dtype,
                                )
                                prev_actions = torch.cat([prev_actions, pad], dim=1)

                            seed = prev_latents[:, :, :ctx]

                            t0_denoise = time.perf_counter()
                            rollout_latents = autoregressive_rollout(
                                model,
                                seed_latents=seed,
                                all_actions=prev_actions[:, :rollout_total],
                                context_latents=ctx,
                                future_latents=future_per_step,
                                total_latents=rollout_total,
                                ode_steps=args.preview_ode_steps,
                                accelerator=accelerator,
                            )
                            torch.cuda.synchronize()
                            t_denoise = time.perf_counter() - t0_denoise

                            gt_full = prev_latents[:, :, : min(rollout_total, prev_latents.shape[2])]

                            def maybe_denorm(x):
                                return latent_normalization.denormalize(x) if latent_normalization else x

                            t0_decode = time.perf_counter()
                            sampled_frames = latents_to_frame_indices(
                                video_vae, maybe_denorm(rollout_latents), accelerator=accelerator
                            )
                            gt_frames = latents_to_frame_indices(
                                video_vae, maybe_denorm(gt_full), accelerator=accelerator
                            )
                            torch.cuda.synchronize()
                            t_decode = time.perf_counter() - t0_decode

                            t0_save = time.perf_counter()
                            out_path = preview_path(output_dir, split="rollout", step=step)
                            n_pred = rollout_total - ctx
                            pred_frames = sampled_frames[0, ctx:rollout_total]
                            gt_pred = gt_frames[0, ctx:min(gt_frames.shape[1], rollout_total)]
                            save_rollout_preview(
                                out_path,
                                frames_gt=gt_pred,
                                frames_pred=pred_frames,
                                palette=palette,
                            )
                            t_save = time.perf_counter() - t0_save

                            t_preview = time.perf_counter() - t0_preview
                            log_console.print(
                                f"[rollout] {t_preview:.1f}s total "
                                f"(latents={rollout_total} ctx={ctx} pred={n_pred} "
                                f"denoise={t_denoise:.1f}s "
                                f"decode={t_decode:.1f}s save={t_save:.1f}s) -> {out_path}",
                                markup=False,
                            )
                            del rollout_latents, sampled_frames, gt_frames, gt_full, pred_frames, gt_pred
                    except torch.cuda.OutOfMemoryError:
                        log_console.print("[rollout] Skipped due to CUDA OOM", markup=False)
                        gc.collect()
                        torch.cuda.empty_cache()

            if eval_due or should_checkpoint or preview_due:
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        console.print(f"Training complete. Best eval flow_loss={best_eval:.6f}")


if __name__ == "__main__":
    main()
