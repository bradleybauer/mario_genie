from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader

from src.models.video_vae import VideoVAE
from src.training.losses import focal_cross_entropy, spatial_weight_map, temporal_change_weight


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


def apply_palette_index_augmentation(
    frames: torch.Tensor,
    *,
    sample_prob: float = 1.0,
    replacement_prob: float,
    replacement_probs: torch.Tensor,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (B, T, H, W), got {tuple(frames.shape)}")
    if not (0.0 <= sample_prob <= 1.0):
        raise ValueError("sample_prob must be in [0, 1]")
    if not (0.0 <= replacement_prob <= 1.0):
        raise ValueError("replacement_prob must be in [0, 1]")
    if sample_prob == 0.0 or replacement_prob == 0.0:
        return frames
    if replacement_probs.ndim != 1:
        raise ValueError("replacement_probs must be a 1D tensor")

    num_classes = int(replacement_probs.numel())
    if num_classes <= 0:
        raise ValueError("replacement_probs must not be empty")
    probs = replacement_probs.to(device=frames.device, dtype=torch.float32)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        raise ValueError("replacement_probs must sum to a positive value")
    probs = probs / probs_sum

    if frames.dtype.is_floating_point or frames.dtype.is_complex:
        raise ValueError("frames must contain integer palette indices")
    if int(frames.max().item()) >= num_classes or int(frames.min().item()) < 0:
        raise ValueError("frames contain palette indices outside the augmentation distribution")

    sample_mask = torch.rand((frames.shape[0], 1, 1, 1), device=frames.device) < sample_prob
    if not bool(sample_mask.any()):
        return frames

    replace_mask = sample_mask & (torch.rand(frames.shape, device=frames.device) < replacement_prob)
    replace_count = int(replace_mask.sum().item())
    if replace_count == 0:
        return frames

    augmented = frames.clone()
    sampled = torch.multinomial(probs, num_samples=replace_count, replacement=True)
    augmented[replace_mask] = sampled.to(device=frames.device, dtype=frames.dtype)
    return augmented


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
    onehot_conv: bool = False,
    focal_gamma: float = 0.0,
    class_weight: torch.Tensor | None = None,
    class_weight_radius: float = 0.0,
    class_weight_hardness: float = 5.0,
    class_weight_temporal_ema: float = 0.0,
    temporal_change_boost: float = 0.0,
    accelerator: Accelerator | None = None,
    aggregate_losses: bool = False,
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
            if onehot_conv:
                model_input = frames.byte()
            else:
                model_input = frames_to_one_hot(frames, num_colors, dtype=onehot_dtype, out=onehot_buffer)
                onehot_buffer = model_input
            with autocast_context():
                outputs = model(model_input, sample_posterior=False)
            recon_logits, recon_targets = split_context_targets(outputs.logits, frames, context_frames)

            pixel_weight = None
            if class_weight is not None or temporal_change_boost > 0:
                # Start with per-pixel class weights (or ones).
                if class_weight is not None:
                    raw_cw = class_weight.to(frames.device)[frames]  # (B, T, H, W)
                else:
                    raw_cw = torch.ones_like(frames, dtype=torch.float32)

                # Add temporal change boost independently of class weight.
                if temporal_change_boost > 0:
                    tc = temporal_change_weight(
                        frames, boost=temporal_change_boost,
                        context_frames=context_frames,
                    )
                    if context_frames > 0:
                        raw_cw = raw_cw[:, context_frames:]
                    raw_cw = raw_cw + temporal_change_boost * (tc - 1.0)

                # Spatially smooth the combined map.
                if class_weight is not None and class_weight_radius >= 0.5:
                    if context_frames > 0 and temporal_change_boost > 0:
                        ctx_cw = class_weight.to(frames.device)[frames[:, :context_frames]]
                        combined = torch.cat([ctx_cw, raw_cw], dim=1)
                    else:
                        combined = raw_cw
                    pixel_weight = spatial_weight_map(
                        frames, class_weight,
                        radius=class_weight_radius,
                        hardness=class_weight_hardness,
                        temporal_ema=class_weight_temporal_ema,
                        per_pixel_weight=combined,
                    )
                    if context_frames > 0:
                        pixel_weight = pixel_weight[:, context_frames:]
                else:
                    pixel_weight = raw_cw

            recon_loss = focal_cross_entropy(
                recon_logits,
                recon_targets,
                gamma=focal_gamma,
                class_weight=class_weight if class_weight_radius < 0.5 else None,
                pixel_weight=pixel_weight,
            )
            kl_loss = VideoVAE.kl_loss(outputs.posterior_mean, outputs.posterior_logvar)
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            if preview is None:
                with autocast_context():
                    preview_input = frames[:1].byte() if onehot_conv else model_input[:1]
                    preview_out = model(preview_input, sample_posterior=False)
                preview = (frames[:1].detach().cpu(), preview_out.logits.detach().cpu())
                del preview_out
            del recon_logits, recon_targets, recon_loss, kl_loss, outputs, model_input, frames

    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not recon_losses:
        return {"recon_loss": 0.0, "kl_loss": 0.0, "loss": 0.0}, preview

    mean_recon = float(np.mean(recon_losses))
    mean_kl = float(np.mean(kl_losses))
    if aggregate_losses and accelerator is not None and accelerator.num_processes > 1:
        local_stats = torch.tensor(
            [float(np.sum(recon_losses)), float(np.sum(kl_losses)), float(len(recon_losses))],
            device=device,
            dtype=torch.float64,
        )
        reduced_stats = accelerator.reduce(local_stats, reduction="sum")
        denom = max(float(reduced_stats[2].item()), 1.0)
        mean_recon = float(reduced_stats[0].item() / denom)
        mean_kl = float(reduced_stats[1].item() / denom)
    return {
        "recon_loss": mean_recon,
        "kl_loss": mean_kl,
        "loss": mean_recon + kl_weight * mean_kl,
    }, preview
