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
    video_contains_first_frame: bool,
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
        loss, _ = model(
            dummy,
            return_loss=True,
            video_contains_first_frame=video_contains_first_frame,
        )
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
    video_contains_first_frame: bool = True,
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
    lower_fit: int | None = None

    # Start near 64 for efficiency, but never below the requested floor.
    probe = min(max(floor, 64), ceiling)
    print(f"[auto-batch] Searching [{floor}, {ceiling}], starting at {probe} …")

    if _try_batch(model, probe, image_size, seq_len, device, video_contains_first_frame):
        lower = probe
        lower_fit = probe
        while probe * 2 <= ceiling:
            probe *= 2
            if not _try_batch(model, probe, image_size, seq_len, device, video_contains_first_frame):
                upper = probe
                break
            lower = probe
            lower_fit = probe
        else:
            # Doubling couldn't reach ceiling — test it directly
            if probe < ceiling:
                if _try_batch(model, ceiling, image_size, seq_len, device, video_contains_first_frame):
                    lower_fit = ceiling
                    lower = upper = ceiling
                else:
                    # OOM boundary is between last fit and ceiling
                    upper = ceiling
            else:
                # probe == ceiling and it fit
                lower_fit = ceiling
                lower = upper = ceiling
    else:
        upper = probe
        lower = floor

    if lower_fit is not None and lower == upper:
        raw_max = lower_fit
    else:
        while upper - lower > 1:
            mid = (lower + upper) // 2
            if _try_batch(model, mid, image_size, seq_len, device, video_contains_first_frame):
                lower = mid
                lower_fit = mid
            else:
                upper = mid
        if lower_fit is None:
            if not _try_batch(model, floor, image_size, seq_len, device, video_contains_first_frame):
                print(f"[auto-batch] WARNING: even batch_size={floor} OOMs — returning {floor} anyway")
                model.train(was_training)
                return floor
            raw_max = floor
        else:
            raw_max = lower_fit

    if raw_max < floor:
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
