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
    is_periodic_event_due,
    make_output_dir,
    preview_path,
    should_log_step,
)
from src.training.training_utils import (
    ThroughputTracker,
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

    parser.add_argument("--video-vae-checkpoint", type=str, default=None)
    parser.add_argument("--video-vae-config", type=str, default=None)
    parser.add_argument("--latent-stats", type=str, default=None)
    parser.add_argument("--disable-latent-normalization", action="store_true")

    parser.add_argument("--clip-frames", type=int, default=20)
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))

    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
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
                        help="Sets encoder and decoder layers to half this value each.")
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--num-decoder-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=64)

    parser.add_argument("--flow-loss", type=str, choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=10.0)

    parser.add_argument("--timestep-sampler", type=str, choices=["uniform", "logit_normal"],
                        default="uniform")

    parser.add_argument(
        "--error-buffer-size", type=int, default=0,
        help="Residual clips retained for history perturbation.",
    )
    parser.add_argument("--error-inject-prob", type=float, default=0.0)
    parser.add_argument("--error-inject-scale", type=float, default=0.05)
    parser.add_argument("--error-inject-start-step", type=int, default=20000)
    parser.add_argument("--error-buffer-clip", type=float, default=3.0)

    parser.add_argument("--preview-ode-steps", type=int, default=24)

    parser.add_argument("--mixed-precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode passed to Accelerator.")
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


# ---------------------------------------------------------------------------
# Utilities carried over from the original trainer
# ---------------------------------------------------------------------------

@dataclass
class LatentNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    stats_path: str

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
    *, stats_path: Path, latent_channels: int, device: torch.device
) -> LatentNormalization:
    stats = _load_json(stats_path)
    mean_v = stats.get("channel_mean")
    std_v = stats.get("channel_std_clamped") or stats.get("channel_std")
    if mean_v is None or std_v is None:
        raise ValueError(f"{stats_path}: must contain channel_mean and channel_std(_clamped)")
    if len(mean_v) != latent_channels or len(std_v) != latent_channels:
        raise ValueError("Latent stats channel count mismatch")
    eps = max(float(stats.get("std_epsilon", 1e-6)), 1e-6)
    mean = torch.tensor(mean_v, dtype=torch.float32, device=device).view(1, latent_channels, 1, 1, 1)
    std = torch.tensor(std_v, dtype=torch.float32, device=device).view(1, latent_channels, 1, 1, 1)
    std = torch.nan_to_num(std, nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    return LatentNormalization(mean=mean, std=std, stats_path=str(stats_path))


def load_latent_stats_path(
    args: argparse.Namespace, *, data_dir: Path, latent_meta: dict | None
) -> Path | None:
    if args.disable_latent_normalization:
        return None
    if args.latent_stats is not None:
        p = _resolve_path(args.latent_stats, config_dir=data_dir)
        if not _is_readable(p):
            raise FileNotFoundError(f"Latent stats not found: {args.latent_stats}")
        return p
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
    patch_size = int(cfg.get("patch_size", 4))
    base_channels = int(cfg.get("base_channels", 64))
    latent_channels = int(cfg.get("latent_channels", 64))
    vae_num_colors = int(cfg.get("num_colors", num_colors or 0))
    if vae_num_colors <= 0:
        raise ValueError("Cannot determine VAE num_colors.")
    vae = VideoVAE(
        num_colors=vae_num_colors,
        patch_size=patch_size,
        base_channels=base_channels,
        latent_channels=latent_channels,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, {"latent_channels": latent_channels, "num_colors": vae_num_colors}


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
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype, frames: int
    ) -> torch.Tensor | None:
        if len(self._bank) == 0:
            return None
        indices = [random.randrange(len(self._bank)) for _ in range(batch_size)]
        samples = [self._bank[i] for i in indices]
        out = torch.cat(samples, dim=0).to(device=device, dtype=dtype)
        B, C, T, H, W = out.shape
        if T < frames:
            out = out.repeat(1, 1, math.ceil(frames / T), 1, 1)[:, :, :frames]
        elif T > frames:
            out = out[:, :, :frames]
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
    future_frames: int,
    ode_steps: int,
    accelerator: Accelerator,
) -> torch.Tensor:
    batch, channels, context_frames, height, width = history_latents.shape
    future = torch.randn(batch, channels, future_frames, height, width,
                         device=history_latents.device, dtype=history_latents.dtype)

    raw_model = accelerator.unwrap_model(model)
    with accelerator.autocast():
        encoded = raw_model.encode_history(history_latents, actions[:, :context_frames])

    dt = 1.0 / float(ode_steps)
    for step in range(ode_steps, 0, -1):
        t_val = (step - 0.5) / float(ode_steps)
        t = torch.full((batch,), t_val, device=history_latents.device, dtype=history_latents.dtype)
        with accelerator.autocast():
            velocity = raw_model.decode_future(future, actions, t, encoded, context_frames)
        future = future - dt * velocity
    return future


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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    *,
    loader: DataLoader,
    device: torch.device,
    context_frames: int,
    accelerator: Accelerator,
    default_action_index: int,
    loss_type: str,
    huber_delta: float,
    latent_normalization: LatentNormalization | None,
    t_sampler: UniformTimestepSampler | LogitNormalTimestepSampler,
) -> dict[str, float]:
    model.eval()
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
            history = latents[:, :, :context_frames]
            future = latents[:, :, context_frames:]
            noisy_future, v_target, t, eps = sample_flow_target(future, t_sampler)
            model_input = torch.cat((history, noisy_future), dim=2)

            with accelerator.autocast():
                v_pred = model(model_input, actions, t, context_frames)

            flow_losses.append(
                compute_flow_loss(v_pred, v_target, loss_type=loss_type, huber_delta=huber_delta).item()
            )
            x0_mses.append(F.mse_loss(eps - v_pred, future).item())

    model.train()
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
            clip_frames=args.clip_frames,
            include_actions=True,
            num_workers=args.num_workers,
            system_info=system_info,
        )
    if len(dataset) == 0:
        raise RuntimeError("No training samples found.")

    train_dataset, eval_dataset = split_train_eval_dataset(
        dataset,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )

    if accelerator.is_main_process:
        console.print(f"Found {len(dataset)} segments of {args.clip_frames} frames.")
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
    latent_stats_path = load_latent_stats_path(args, data_dir=data_dir, latent_meta=latent_meta)
    latent_normalization: LatentNormalization | None = None
    if latent_stats_path is not None:
        latent_normalization = load_latent_normalization(
            stats_path=latent_stats_path, latent_channels=latent_channels, device=device
        )
        if accelerator.is_main_process:
            console.print(f"[latents] Per-channel normalization from {latent_stats_path}")

    # ------------------------------------------------------------------
    # Palette and preview VAE
    # ------------------------------------------------------------------
    palette: torch.Tensor | None = None
    video_vae: torch.nn.Module | None = None
    palette_path = data_dir / "palette.json"
    if palette_path.is_file():
        palette = load_palette_tensor(data_dir)

    vae_ckpt = args.video_vae_checkpoint
    vae_cfg = args.video_vae_config
    if latent_meta is not None:
        vae_ckpt = vae_ckpt or latent_meta.get("video_vae_checkpoint")
        vae_cfg = vae_cfg or latent_meta.get("video_vae_config_path")
    vae_ckpt_path = _resolve_path(vae_ckpt, config_dir=data_dir)
    vae_cfg_path = _resolve_path(vae_cfg, config_dir=data_dir)
    if not _is_readable(vae_cfg_path) and _is_readable(vae_ckpt_path):
        sibling = vae_ckpt_path.parent / "config.json"
        if sibling.is_file():
            vae_cfg_path = sibling
    if _is_readable(vae_ckpt_path) and _is_readable(vae_cfg_path):
        try:
            video_vae, vae_summary = load_video_vae(
                checkpoint_path=vae_ckpt_path, config_path=vae_cfg_path,
                num_colors=int(palette.shape[0]) if palette is not None else None,
                device=device,
            )
            if accelerator.is_main_process:
                console.print("[vae] Loaded video VAE for preview generation")
        except Exception as exc:
            if accelerator.is_main_process:
                console.print(f"[vae] Failed to load VAE ({exc}); previews disabled")

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
        max_frames=args.max_frames,
    )
    model: torch.nn.Module = build_video_latent_dit(model_config=model_config).to(device)
    diffusers_version = str(diffusers.__version__)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        console.print(f"[model] backend=diffusers diffusers={diffusers_version}")
        console.print(f"[model] {num_params:,} trainable parameters")

    if args.compile:
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
                    "num_actions": int(num_actions),
                    "clip_frames": int(args.clip_frames),
                    "context_frames": int(args.context_frames),
                    "latent_stats_path": str(latent_stats_path) if latent_stats_path is not None else None,
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
    use_live = sys.stdout.isatty() and accelerator.is_main_process
    throughput_tracker = ThroughputTracker(window_steps=20)
    loss_window: deque[float] = deque(maxlen=100)
    train_log_path = output_dir / "train_log.jsonl"

    if accelerator.is_main_process:
        console.print(f"Training on {len(train_dataset)} samples | output: {output_dir}")

    with build_progress(use_live=use_live) as progress:
        log_console = progress.console
        task = progress.add_task("Training", total=max(args.max_steps - start_step, 0), status="")

        for step in range(start_step, args.max_steps):
            batch = next(train_iter)
            latents, actions = prepare_batch(
                batch,
                device=device,
                default_action_index=default_action_index,
                latent_normalization=latent_normalization,
            )

            history = latents[:, :, :args.context_frames]
            future = latents[:, :, args.context_frames:]

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
                    frames=args.context_frames,
                )
                if perturb is not None:
                    history = history + args.error_inject_scale * perturb

            noisy_future, v_target, t, eps = sample_flow_target(future, t_sampler)
            model_input = torch.cat((history, noisy_future), dim=2)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                v_pred = model(model_input, actions, t, args.context_frames)
                loss = compute_flow_loss(v_pred, v_target, loss_type=args.flow_loss, huber_delta=args.huber_delta)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                error_buffer.clear()
                progress.update(task, advance=1, status=f"loss=nonfinite(skip) lr={lr_scheduler.get_last_lr()[0]:.2e}")
                continue

            accelerator.backward(loss)

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

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip).item()
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                error_buffer.add_batch(eps - v_pred - future)
            loss_window.append(loss.item())

            samples_per_second, steps_per_second = throughput_tracker.update(
                samples=latents.shape[0] * accelerator.num_processes,
            )

            progress.update(task, advance=1, status=(
                f"loss={loss.item():.5f} lr={lr_scheduler.get_last_lr()[0]:.2e}"
                + f" errbuf={len(error_buffer)}"
                + f" sps={samples_per_second:.1f}"
            ))

            if should_log:
                with torch.no_grad():
                    v_pred_norm = v_pred.detach().float().norm(dim=1).mean().item()
                    v_target_norm = v_target.detach().float().norm(dim=1).mean().item()
                train_entry: dict = {
                    "type": "train",
                    "step": step,
                    "loss": loss.item(),
                    "loss_smooth": float(np.mean(loss_window)) if loss_window else loss.item(),
                    "grad_norm": grad_norm,
                    **{f"grad_norm_{k}": v for k, v in cgnorms.items()},
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                    "t_mean": float(t.mean().item()),
                    "t_std": float(t.std().item()),
                    "v_pred_norm": v_pred_norm,
                    "v_target_norm": v_target_norm,
                    "samples_per_sec": samples_per_second,
                    "steps_per_sec": steps_per_second,
                }
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
            if eval_due or should_checkpoint:
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
                    context_frames=args.context_frames,
                    accelerator=accelerator,
                    default_action_index=default_action_index,
                    loss_type=args.flow_loss,
                    huber_delta=args.huber_delta,
                    latent_normalization=latent_normalization,
                    t_sampler=t_sampler,
                )
                eval_metrics["type"] = "eval"
                eval_metrics["step"] = step
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

                # Preview
                with torch.no_grad():
                    try:
                        if video_vae is not None and palette is not None:
                            preview_batch = next(train_iter)
                            prev_latents, prev_actions = prepare_batch(
                                preview_batch,
                                device=device,
                                default_action_index=default_action_index,
                                latent_normalization=latent_normalization,
                            )
                            idx = random.randrange(prev_latents.shape[0])
                            prev_latents = prev_latents[idx:idx+1]
                            prev_actions = prev_actions[idx:idx+1]

                            history_prev = prev_latents[:, :, :args.context_frames]
                            raw_model = accelerator.unwrap_model(model)
                            sampled_future = denoise_future_segment(
                                raw_model,
                                history_latents=history_prev,
                                actions=prev_actions,
                                future_frames=args.clip_frames - args.context_frames,
                                ode_steps=args.preview_ode_steps,
                                accelerator=accelerator,
                            )
                            sampled_full = torch.cat((history_prev, sampled_future), dim=2)

                            def maybe_denorm(x):
                                return latent_normalization.denormalize(x) if latent_normalization else x

                            sampled_frames = latents_to_frame_indices(
                                video_vae, maybe_denorm(sampled_full), accelerator=accelerator
                            )
                            gt_frames = latents_to_frame_indices(
                                video_vae, maybe_denorm(prev_latents), accelerator=accelerator
                            )
                            save_preview_image(
                                preview_path(output_dir, split="eval", step=step),
                                frames_gt=gt_frames[0],
                                frames_sample=sampled_frames[0],
                                palette=palette,
                            )
                    except torch.cuda.OutOfMemoryError:
                        log_console.print("[eval][OOM] Preview skipped")
                        gc.collect()
                        torch.cuda.empty_cache()

                if eval_metrics["flow_loss"] < best_eval:
                    best_eval = eval_metrics["flow_loss"]
                    save_model_checkpoint(
                        output_dir / "best",
                        model=model,
                        model_config=model_config,
                        step=step,
                        best_eval=best_eval,
                        metrics=metrics,
                    )
                    accelerator.save_state(str(output_dir / "best" / "training_state"))
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
                accelerator.save_state(str(latest_dir / "training_state"))

                step_dir = output_dir / f"step_{step:06d}"
                save_model_checkpoint(
                    step_dir,
                    model=model,
                    model_config=model_config,
                    step=step,
                    best_eval=best_eval,
                    metrics=metrics,
                )
                accelerator.save_state(str(step_dir / "training_state"))
                if eval_line is not None:
                    eval_line += " | saved latest"

            if eval_line is not None and accelerator.is_main_process:
                log_console.print(eval_line)

            if eval_due or should_checkpoint:
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        console.print(f"Training complete. Best eval flow_loss={best_eval:.6f}")


if __name__ == "__main__":
    main()
