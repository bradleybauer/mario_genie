"""Shared utilities for latent normalization, VAE loading, and ODE denoising.

Used by both the DiT trainer and the play-world-model evaluation script.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    vae_num_colors = int(data_cfg.get("num_colors", cfg.get("num_colors", num_colors or 0)))
    if vae_num_colors <= 0:
        raise ValueError("Cannot determine VAE num_colors.")
    vae = VideoVAE(
        num_colors=vae_num_colors,
        base_channels=base_channels,
        latent_channels=latent_channels,
        temporal_downsample=temporal_downsample,
        dropout=dropout,
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

FLOW_SAMPLERS = ("euler", "heun")


def normalize_flow_sampler(sampler: str) -> str:
    normalized = str(sampler).strip().lower()
    if normalized not in FLOW_SAMPLERS:
        choices = ", ".join(FLOW_SAMPLERS)
        raise ValueError(f"Unknown flow sampler {sampler!r}; expected one of: {choices}")
    return normalized


def integrate_flow_ode(
    initial_state: torch.Tensor,
    *,
    ode_steps: int,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sampler: str = "heun",
) -> torch.Tensor:
    if ode_steps < 1:
        raise ValueError("ode_steps must be >= 1")

    sampler = normalize_flow_sampler(sampler)
    state = initial_state
    batch = state.shape[0]
    dt = 1.0 / float(ode_steps)

    for step in range(ode_steps, 0, -1):
        if sampler == "euler":
            t_val = (step - 0.5) / float(ode_steps)
            t = torch.full((batch,), t_val, device=state.device, dtype=state.dtype)
            state = state - dt * velocity_fn(state, t)
            continue

        t_now = torch.full(
            (batch,),
            step / float(ode_steps),
            device=state.device,
            dtype=state.dtype,
        )
        velocity_now = velocity_fn(state, t_now)
        predicted_state = state - dt * velocity_now
        t_next = torch.full(
            (batch,),
            (step - 1) / float(ode_steps),
            device=state.device,
            dtype=state.dtype,
        )
        velocity_next = velocity_fn(predicted_state, t_next)
        state = state - 0.5 * dt * (velocity_now + velocity_next)

    return state

@torch.no_grad()
def denoise_future_segment(
    model: torch.nn.Module,
    *,
    history_latents: torch.Tensor,
    actions: torch.Tensor,
    future_latents: int,
    ode_steps: int,
    ode_sampler: str = "heun",
    action_cfg_scale: float = 1.0,
    autocast_ctx=nullcontext,
) -> torch.Tensor:
    """Denoise future latents via explicit flow ODE integration.

    Parameters
    ----------
    model:
        A raw (unwrapped) VideoLatentDiTDiffusers model.
    autocast_ctx:
        A callable returning a context manager for mixed-precision autocast.
        E.g. ``accelerator.autocast`` or
        ``lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)``.
        Defaults to ``contextlib.nullcontext`` when no autocasting is applied.
    """
    batch, channels, ctx, height, width = history_latents.shape
    future = torch.randn(
        batch, channels, future_latents, height, width,
        device=history_latents.device, dtype=history_latents.dtype,
    )

    with autocast_ctx():
        encoded_cond = model.encode_history(history_latents, actions[:, :ctx])
        encoded_uncond = None
        if not math.isclose(float(action_cfg_scale), 1.0):
            encoded_uncond = model.encode_history(
                history_latents,
                actions[:, :ctx],
                action_cond_scale=0.0,
            )

    ode_sampler = normalize_flow_sampler(ode_sampler)

    def velocity_fn(current_future: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with autocast_ctx():
            velocity_cond = model.decode_future(current_future, actions, t, encoded_cond, ctx)
            if encoded_uncond is None:
                return velocity_cond
            velocity_uncond = model.decode_future(
                current_future,
                actions,
                t,
                encoded_uncond,
                ctx,
                action_cond_scale=0.0,
            )
            return velocity_uncond + float(action_cfg_scale) * (
                velocity_cond - velocity_uncond
            )

    return integrate_flow_ode(
        future,
        ode_steps=ode_steps,
        velocity_fn=velocity_fn,
        sampler=ode_sampler,
    )
