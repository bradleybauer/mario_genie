"""Automatic batch-size finder for maximising GPU utilisation.

Runs a binary search over batch sizes using synthetic data to find the largest
batch that fits in VRAM.  A configurable safety margin is subtracted so that
the validation forward pass (which also allocates memory) doesn't OOM.

Usage
-----
>>> from mario_world_model.auto_batch import find_max_batch_size
>>> bs = find_max_batch_size(model, image_size=128, seq_len=16, device="cuda")
>>> print(f"Using batch size {bs}")
"""

from __future__ import annotations

import gc
import math

import torch
import torch.nn as nn


def _clear_gpu(device: torch.device) -> None:
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _try_batch(
    model: nn.Module,
    batch_size: int,
    image_size: int,
    seq_len: int,
    device: torch.device,
) -> bool:
    """Attempt a forward + backward pass with *batch_size* synthetic frames.

    Returns ``True`` if it succeeds, ``False`` on OOM.
    """
    _clear_gpu(device)
    try:
        dummy = torch.randn(
            batch_size, 3, seq_len, image_size, image_size, device=device
        )
        model.train()
        loss, _ = model(dummy, return_loss=True)
        loss.backward()
        model.zero_grad(set_to_none=True)
        del dummy, loss
        _clear_gpu(device)
        return True
    except torch.cuda.OutOfMemoryError:
        print("[auto-batch-sizer] OOM at batch_size =", batch_size)
        model.zero_grad(set_to_none=True)
        _clear_gpu(device)
        return False


def find_max_batch_size(
    model: nn.Module,
    image_size: int,
    seq_len: int,
    device: torch.device | str = "cuda",
    floor: int = 1,
    ceiling: int = 256,
    safety_fraction: float = 0.85,
) -> int:
    """Find the largest batch size that fits in GPU memory.

    Parameters
    ----------
    model:
        The model to probe (must already be on *device*).
    image_size:
        Spatial resolution of input frames.
    seq_len:
        Number of frames per clip (temporal dimension).
    device:
        CUDA device to probe.
    floor:
        Smallest batch size to consider (guaranteed to be returned if even
        ``floor`` OOMs — but that would indicate a model too large for the GPU).
    ceiling:
        Upper bound on the search range.  Kept reasonable to avoid wasting time
        on impossibly large batches.
    safety_fraction:
        After finding the raw maximum, the returned batch size is
        ``max(floor, int(raw_max * safety_fraction))`` so that peaks during
        validation or codebook updates don't cause OOM.

    Returns
    -------
    int
        Optimal batch size.
    """
    if isinstance(device, str):
        device = torch.device(device)

    was_training = model.training

    # ── Phase 1: exponential growth to find an upper bound ───────────
    hi = floor
    while hi <= ceiling:
        if _try_batch(model, hi, image_size, seq_len, device):
            hi *= 2
        else:
            break
    # hi is now the first power-of-2 that OOM'd (or > ceiling)
    upper = min(hi, ceiling)
    lower = max(floor, upper // 2)

    # If even floor fails, nothing we can do — return floor and hope for the
    # best (the real training loop will OOM on its own, with a clear message).
    if upper == floor and not _try_batch(model, floor, image_size, seq_len, device):
        print(f"[auto-batch] WARNING: even batch_size={floor} OOMs — returning {floor} anyway")
        model.train(was_training)
        return floor

    # ── Phase 2: binary search between lower (fits) and upper (OOMs) ─
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if _try_batch(model, mid, image_size, seq_len, device):
            lower = mid
        else:
            upper = mid

    raw_max = lower
    safe = max(floor, int(raw_max * safety_fraction))

    # Round down to the nearest power of 2 for DataLoader efficiency
    safe_pow2 = 2 ** int(math.log2(safe)) if safe >= 2 else safe

    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(
        f"[auto-batch] GPU {torch.cuda.get_device_name(device)} "
        f"({vram_gb:.1f} GB)  raw_max={raw_max}  safe={safe}  "
        f"using={safe_pow2}  (safety={safety_fraction:.0%})"
    )

    _clear_gpu(device)
    model.train(was_training)
    return safe_pow2
