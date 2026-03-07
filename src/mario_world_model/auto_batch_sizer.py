"""Automatic batch-size finder for maximising GPU utilisation.

Binary-searches over batch sizes using synthetic data to find the largest
batch that fits in VRAM.  A configurable safety margin is subtracted so that
the validation forward pass (which also allocates memory) doesn't OOM.

Usage
-----
>>> from mario_world_model.auto_batch_sizer import find_max_batch_size
>>> bs = find_max_batch_size(model, image_size=128, seq_len=16, device="cuda")
>>> print(f"Using batch size {bs}")
"""

from __future__ import annotations

import gc

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
        print(f"[auto-batch]   batch_size={batch_size:>4d}  ✓ fits")
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"[auto-batch]   batch_size={batch_size:>4d}  ✗ OOM")
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
        Smallest batch size to consider (returned even if it OOMs).
    ceiling:
        Upper bound on the search range.
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

    # Doubling from 64 to bracket the OOM boundary, then binary search.
    probe = min(64, ceiling)
    print(f"[auto-batch] Searching [{floor}, {ceiling}], starting at {probe} …")

    if _try_batch(model, probe, image_size, seq_len, device):
        lower = probe
        while probe * 2 <= ceiling:
            probe *= 2
            if not _try_batch(model, probe, image_size, seq_len, device):
                upper = probe
                break
            lower = probe
        else:
            # Doubling couldn't reach ceiling — test it directly
            if probe < ceiling:
                if _try_batch(model, ceiling, image_size, seq_len, device):
                    raw_max = ceiling
                    lower = upper = ceiling
                else:
                    # OOM boundary is between last fit and ceiling
                    upper = ceiling
            else:
                # probe == ceiling and it fit
                raw_max = ceiling
                lower = upper = ceiling
    else:
        upper = probe
        lower = floor

    if lower != upper:
        while upper - lower > 1:
            mid = (lower + upper) // 2
            if _try_batch(model, mid, image_size, seq_len, device):
                lower = mid
            else:
                upper = mid
        raw_max = lower

    if raw_max < floor or not _try_batch(model, max(raw_max, floor), image_size, seq_len, device):
        print(f"[auto-batch] WARNING: even batch_size={floor} OOMs — returning {floor} anyway")
        model.train(was_training)
        return floor

    if raw_max >= ceiling:
        safe = ceiling
    else:
        safe = max(floor, int(raw_max * safety_fraction))

    safe_round = (safe // 8) * 8 if safe >= 8 else safe

    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(
        f"[auto-batch] GPU {torch.cuda.get_device_name(device)} "
        f"({vram_gb:.1f} GB)  raw_max={raw_max}  safe={safe}  "
        f"using={safe_round}  (safety={safety_fraction:.0%})"
    )

    _clear_gpu(device)
    model.train(was_training)
    return safe_round
