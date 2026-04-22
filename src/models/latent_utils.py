"""Shared utilities for latent normalization, VAE loading, and ODE denoising.

Used by both the DiT trainer and the play-world-model evaluation script.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.video_vae import VideoVAE
from src.path_utils import resolve_workspace_path


# ---------------------------------------------------------------------------
# JSON / path helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def is_readable(p: Path | None) -> bool:
    try:
        return p is not None and p.is_file()
    except (PermissionError, OSError):
        return False


# ---------------------------------------------------------------------------
# Latent normalization
# ---------------------------------------------------------------------------

@dataclass
class LatentNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    stats_path: str = ""
    scheme: str = "component_chw_shared_time"
    version: int = 2

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def load_latent_normalization(
    *,
    stats_path: Path,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
) -> LatentNormalization:
    stats = load_json(stats_path)
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
    *,
    data_dir: Path,
    latent_meta: dict | None,
    project_root: Path,
    disable: bool = False,
) -> Path | None:
    if disable:
        return None
    if latent_meta is not None:
        for key in ("latent_stats_path", "latent_stats_file"):
            v = latent_meta.get(key)
            if isinstance(v, str):
                p = resolve_workspace_path(v, project_root=project_root, config_dir=data_dir)
                if is_readable(p):
                    return p
    default = data_dir / "latent_stats.json"
    return default.resolve() if default.is_file() else None


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def load_video_vae(
    *, checkpoint_path: Path, config_path: Path, num_colors: int | None, device: torch.device,
) -> tuple[VideoVAE, dict]:
    cfg = load_json(config_path)
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


# ---------------------------------------------------------------------------
# ODE denoising
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_future_segment(
    model: torch.nn.Module,
    *,
    history_latents: torch.Tensor,
    actions: torch.Tensor,
    future_latents: int,
    ode_steps: int,
    action_cfg_scale: float = 1.0,
    autocast_ctx=None,
) -> torch.Tensor:
    """Denoise future latents via midpoint ODE integration.

    Parameters
    ----------
    model:
        A raw (unwrapped) VideoLatentDiTDiffusers model.
    autocast_ctx:
        A callable returning a context manager for mixed-precision autocast.
        E.g. ``accelerator.autocast`` or
        ``lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)``.
        If *None*, no autocasting is applied.
    """
    batch, channels, ctx, height, width = history_latents.shape
    future = torch.randn(
        batch, channels, future_latents, height, width,
        device=history_latents.device, dtype=history_latents.dtype,
    )

    if autocast_ctx is not None:
        with autocast_ctx():
            encoded_cond = model.encode_history(history_latents, actions[:, :ctx])
            encoded_uncond = None
            if not math.isclose(float(action_cfg_scale), 1.0):
                encoded_uncond = model.encode_history(
                    history_latents,
                    actions[:, :ctx],
                    action_cond_scale=0.0,
                )
    else:
        encoded_cond = model.encode_history(history_latents, actions[:, :ctx])
        encoded_uncond = None
        if not math.isclose(float(action_cfg_scale), 1.0):
            encoded_uncond = model.encode_history(
                history_latents,
                actions[:, :ctx],
                action_cond_scale=0.0,
            )

    dt = 1.0 / float(ode_steps)
    for step in range(ode_steps, 0, -1):
        t_val = (step - 0.5) / float(ode_steps)
        t = torch.full((batch,), t_val, device=history_latents.device, dtype=history_latents.dtype)
        if autocast_ctx is not None:
            with autocast_ctx():
                velocity_cond = model.decode_future(future, actions, t, encoded_cond, ctx)
                if encoded_uncond is None:
                    velocity = velocity_cond
                else:
                    velocity_uncond = model.decode_future(
                        future,
                        actions,
                        t,
                        encoded_uncond,
                        ctx,
                        action_cond_scale=0.0,
                    )
                    velocity = velocity_uncond + float(action_cfg_scale) * (
                        velocity_cond - velocity_uncond
                    )
        else:
            velocity_cond = model.decode_future(future, actions, t, encoded_cond, ctx)
            if encoded_uncond is None:
                velocity = velocity_cond
            else:
                velocity_uncond = model.decode_future(
                    future,
                    actions,
                    t,
                    encoded_uncond,
                    ctx,
                    action_cond_scale=0.0,
                )
                velocity = velocity_uncond + float(action_cfg_scale) * (
                    velocity_cond - velocity_uncond
                )
        future = future - dt * velocity
    return future
