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
) -> Tensor:
    """Focal loss for multi-class classification.

    Equivalent to cross-entropy when ``gamma=0``.

    Args:
        logits: ``(B, C, ...)`` unnormalised class scores.
        targets: ``(B, ...)`` integer class labels.
        gamma: focusing parameter.  Higher values down-weight easy examples more.
        class_weight: optional per-class multiplicative weight vector of shape ``(C,)``.
    """
    if gamma <= 0.0:
        return F.cross_entropy(logits, targets, weight=class_weight)

    num_classes = logits.shape[1]
    logits_flat = logits.movedim(1, -1).reshape(-1, num_classes)
    targets_flat = targets.reshape(-1).long()

    log_probs = F.log_softmax(logits_flat, dim=-1)
    log_p_t = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp()

    loss = -((1.0 - p_t) ** gamma) * log_p_t

    if class_weight is not None:
        if class_weight.ndim != 1 or class_weight.numel() != num_classes:
            raise ValueError("class_weight must be 1D with length equal to logits.shape[1]")
        loss = loss * class_weight.to(logits_flat.device, dtype=logits_flat.dtype)[targets_flat]

    return loss.mean()
