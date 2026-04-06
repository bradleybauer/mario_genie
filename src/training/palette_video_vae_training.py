from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader

from src.training.losses import focal_cross_entropy


def frames_to_one_hot(
    frames: torch.Tensor,
    num_colors: int,
    *,
    dtype: torch.dtype = torch.float32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (B, T, H, W), got {tuple(frames.shape)}")

    if frames.dtype != torch.long:
        frames = frames.long()
    expected_shape = (frames.shape[0], num_colors, frames.shape[1], frames.shape[2], frames.shape[3])
    if (
        out is None
        or out.shape != expected_shape
        or out.dtype != dtype
        or out.device != frames.device
    ):
        out = torch.empty(expected_shape, dtype=dtype, device=frames.device)
    out.zero_()
    out.scatter_(1, frames.unsqueeze(1), 1)
    return out


def split_context_targets(
    logits: torch.Tensor,
    frames: torch.Tensor,
    context_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context_frames <= 0:
        return logits, frames
    if context_frames >= frames.shape[1]:
        raise ValueError(
            f"context_frames ({context_frames}) must be smaller than total clip length ({frames.shape[1]})"
        )
    return logits[:, :, context_frames:], frames[:, context_frames:]


def save_video_preview(
    path: Path,
    frames: torch.Tensor,
    logits: torch.Tensor,
    palette: torch.Tensor,
    max_frames: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = frames[0, :max_frames].detach().cpu()
    recon = logits[0, :, :max_frames].argmax(dim=0).detach().cpu()
    palette_u8 = (palette.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    target_rgb = palette_u8[frames.numpy()]
    recon_rgb = palette_u8[recon.numpy()]
    rows = [np.concatenate([target_rgb[idx], recon_rgb[idx]], axis=1) for idx in range(target_rgb.shape[0])]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def evaluate_video_vae(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_colors: int,
    kl_weight: float,
    context_frames: int = 0,
    onehot_dtype: torch.dtype = torch.float32,
    focal_gamma: float = 0.0,
    class_weight: torch.Tensor | None = None,
    accelerator: Accelerator | None = None,
    autocast_enabled: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()
    recon_losses: list[float] = []
    kl_losses: list[float] = []
    preview: tuple[torch.Tensor, torch.Tensor] | None = None
    onehot_buffer: torch.Tensor | None = None

    def autocast_context():
        if accelerator is not None:
            return accelerator.autocast()
        if not autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).long()
            inputs = frames_to_one_hot(frames, num_colors, dtype=onehot_dtype, out=onehot_buffer)
            onehot_buffer = inputs
            with autocast_context():
                outputs = model(inputs, sample_posterior=False)
            recon_logits, recon_targets = split_context_targets(outputs.logits, frames, context_frames)
            recon_loss = focal_cross_entropy(
                recon_logits,
                recon_targets,
                gamma=focal_gamma,
                class_weight=class_weight,
            )
            kl_loss = model.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            if preview is None:
                with autocast_context():
                    preview_out = model(inputs[:1], sample_posterior=False)
                preview = (frames[:1].detach().cpu(), preview_out.logits.detach().cpu())
                del preview_out
            del recon_logits, recon_targets, recon_loss, kl_loss, outputs, inputs, frames

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not recon_losses:
        return {"recon_loss": 0.0, "kl_loss": 0.0, "loss": 0.0}, preview

    mean_recon = float(np.mean(recon_losses))
    mean_kl = float(np.mean(kl_losses))
    return {
        "recon_loss": mean_recon,
        "kl_loss": mean_kl,
        "loss": mean_recon + kl_weight * mean_kl,
    }, preview
