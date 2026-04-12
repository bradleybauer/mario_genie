"""Custom loss functions."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def softened_inverse_frequency_weights(
    counts: Tensor,
    *,
    soften: float = 0.5,
    eps: float = 1e-12,
) -> Tensor:
    """Build softened inverse-frequency class weights from per-class counts.

    ``soften`` controls how strongly rare classes are up-weighted:
    * ``0.0`` -> uniform weights
    * ``1.0`` -> full inverse-frequency weighting
    """
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D tensor")
    if counts.numel() == 0:
        raise ValueError("counts must not be empty")
    if soften < 0.0:
        raise ValueError("soften must be >= 0")

    counts = counts.to(dtype=torch.float32)
    total = counts.sum()
    if total <= 0:
        raise ValueError("counts must sum to a positive value")

    probs = (counts / total).clamp_min(eps)
    weights = probs.pow(-soften)
    return weights / weights.mean().clamp_min(eps)


def focal_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 1.0,
    class_weight: Tensor | None = None,
    pixel_weight: Tensor | None = None,
) -> Tensor:
    """Focal loss for multi-class classification.

    Equivalent to cross-entropy when ``gamma=0``.

    Args:
        logits: ``(B, C, ...)`` unnormalised class scores.
        targets: ``(B, ...)`` integer class labels.
        gamma: focusing parameter.  Higher values down-weight easy examples more.
        class_weight: optional per-class multiplicative weight vector of shape ``(C,)``.
        pixel_weight: optional per-pixel multiplicative weight of same shape as *targets*.
    """
    if gamma <= 0.0 and pixel_weight is None:
        return F.cross_entropy(logits, targets, weight=class_weight)

    num_classes = logits.shape[1]
    logits_flat = logits.movedim(1, -1).reshape(-1, num_classes)
    targets_flat = targets.reshape(-1).long()

    if gamma <= 0.0:
        loss = F.cross_entropy(logits_flat, targets_flat, weight=class_weight, reduction="none")
    else:
        log_probs = F.log_softmax(logits_flat, dim=-1)
        log_p_t = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        loss = -((1.0 - p_t) ** gamma) * log_p_t

        if class_weight is not None:
            if class_weight.ndim != 1 or class_weight.numel() != num_classes:
                raise ValueError("class_weight must be 1D with length equal to logits.shape[1]")
            loss = loss * class_weight.to(logits_flat.device, dtype=logits_flat.dtype)[targets_flat]

    if pixel_weight is not None:
        loss = loss * pixel_weight.reshape(-1).to(loss.device, dtype=loss.dtype)

    return loss.mean()


def temporal_change_weight(
    targets: Tensor,
    boost: float = 1.0,
    context_frames: int = 0,
) -> Tensor:
    """Build per-pixel weight that upweights pixels that changed from the previous frame.

    Args:
        targets: ``(B, T, H, W)`` integer class labels.
        boost: additive weight for changed pixels (0 = disabled).
        context_frames: number of leading context frames already stripped from loss.
            The diff is computed on the *full* sequence, then the context prefix
            is removed so the result aligns with the loss targets.

    Returns:
        Weight tensor of shape ``(B, T', H, W)`` where ``T' = T - context_frames``.
        Static pixels get weight 1, changed pixels get ``1 + boost``.
    """
    # Diff with previous frame; first frame has no predecessor → no change.
    changed = torch.zeros_like(targets[:, :1])  # (B, 1, H, W)
    changed = torch.cat([changed, (targets[:, 1:] != targets[:, :-1]).float()], dim=1)
    # Strip context prefix to match loss target shape.
    if context_frames > 0:
        changed = changed[:, context_frames:]
    return 1.0 + boost * changed


def spatial_weight_map(
    targets: Tensor,
    class_weight: Tensor,
    radius: float,
    hardness: float = 5.0,
    temporal_ema: float = 0.0,
    per_pixel_weight: Tensor | None = None,
) -> Tensor:
    """Spatially pool per-pixel class weights with LogSumExp (soft local max).

    Args:
        targets: ``(B, T, H, W)`` integer class labels.
        class_weight: ``(C,)`` per-class weight vector.
        radius: spatial radius for the circular pooling region.
        hardness: β for LogSumExp.  Higher → closer to hard max.
        temporal_ema: causal max-decay persistence factor in [0, 1).
            At each time step the accumulator is ``max(current, ema * previous)``,
            so high-weight regions persist and decay through subsequent frames
            (e.g. through star-power color cycling).  0 disables.
        per_pixel_weight: optional ``(B, T, H, W)`` pre-built weight map.
            When provided, this is used directly instead of looking up from
            *class_weight* and *targets*.  Smoothing, temporal EMA, etc. are
            still applied.

    Returns:
        Weight tensor of shape ``(B, T, H, W)``.
    """
    # Look up per-pixel weight from class labels, or use the caller-provided map.
    if per_pixel_weight is not None:
        w = per_pixel_weight
    else:
        w = class_weight.to(targets.device)[targets.long()]  # (B, T, H, W)
    if radius < 0.5:
        return w

    r = int(torch.tensor(radius).ceil().item())
    y = torch.arange(-r, r + 1, device=targets.device, dtype=torch.float32)
    x = torch.arange(-r, r + 1, device=targets.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mask = (xx * xx + yy * yy) <= radius * radius
    kernel = mask.float()
    kernel = kernel / kernel.sum()  # (k, k)

    # Reshape to (B*T, 1, H, W) for conv2d.
    B, T, H, W = w.shape
    w_flat = w.reshape(B * T, 1, H, W)

    # Numerically stable LogSumExp via local-max subtraction.
    # Use max-pool for the local max.
    k = 2 * r + 1
    local_max = F.max_pool2d(w_flat, kernel_size=k, stride=1, padding=r)
    shifted = hardness * (w_flat - local_max)
    exp_shifted = shifted.exp()
    # Conv with circular kernel.
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    avg_exp = F.conv2d(exp_shifted, kernel_4d, padding=r)
    avg_exp = avg_exp.clamp_min(1e-30)
    out = local_max + avg_exp.log() / hardness
    out = out.reshape(B, T, H, W)

    # Causal max-decay temporal persistence.
    if temporal_ema > 0:
        for t in range(1, T):
            out[:, t] = torch.maximum(out[:, t], temporal_ema * out[:, t - 1])

    return out
