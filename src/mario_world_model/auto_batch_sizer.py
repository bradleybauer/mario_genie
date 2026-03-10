"""Automatic batch-size finder for maximising GPU utilisation.

Binary-searches over batch sizes using synthetic data to find the largest
batch that fits in VRAM.  A configurable safety margin is subtracted so that
the validation forward pass (which also allocates memory) doesn't OOM.

Before probing, analyses GPU memory, model parameters, and per-sample
tensor dimensions to compute a realistic search ceiling — avoiding
expensive (or freezing) probes with batch sizes that obviously cannot fit.

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


# ---------------------------------------------------------------------------
# Memory-aware ceiling estimation
# ---------------------------------------------------------------------------

# Heuristic multiplier: total per-sample memory (input + activations stored
# for backward) is roughly this many times the raw input tensor size.
# For a multi-layer 3D ConvNet without gradient checkpointing, 20× is a
# reasonable conservative default (overestimate → tighter ceiling → faster
# search; the binary search will still find the true max).
_ACTIVATION_RATIO = 20


def _estimate_memory_ceiling(
    model: nn.Module,
    image_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[int | None, dict[str, object]]:
    """Analytically estimate max batch size from GPU memory and model/data dims.

    Returns ``(estimated_ceiling, info)`` where *info* is a dict of
    display-friendly diagnostics, or ``(None, {})`` if estimation is not
    possible (e.g. non-CUDA device, missing attributes).
    """
    info: dict[str, object] = {}

    # --- GPU memory ---
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["free_gb"] = free_mem / 1024**3
        info["total_gb"] = total_mem / 1024**3
    except Exception:
        return None, info

    # --- Model parameters ---
    try:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_bytes = 0
        param_count = 0
    info["param_count_m"] = param_count / 1e6
    info["param_mb"] = param_bytes / 1024**2

    # --- Fixed overhead (not yet allocated) ---
    # Gradients (same size as params) + Adam optimizer states (2× params).
    fixed_overhead = param_bytes * 3
    info["fixed_overhead_mb"] = fixed_overhead / 1024**2

    available = max(free_mem - fixed_overhead, 0) * 0.80  # 20 % headroom
    info["available_mb"] = available / 1024**2

    # --- Per-sample variable cost ---
    channels = getattr(model, "channels", 3)
    input_bytes = channels * seq_len * image_size * image_size * 4  # float32
    per_sample = input_bytes * _ACTIVATION_RATIO
    info["per_sample_mb"] = per_sample / 1024**2
    info["input_mb"] = input_bytes / 1024**2

    if per_sample <= 0:
        return None, info

    estimated = max(1, int(available / per_sample))
    info["estimated"] = estimated
    return estimated, info


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
        channels = getattr(model, 'channels', 3)
        dummy = torch.randn(
            batch_size, channels, seq_len, image_size, image_size, device=device
        )
        model.train()
        num_palette = getattr(model, 'num_palette_colors', 0)
        if num_palette > 0:
            targets = torch.randint(
                0, num_palette, (batch_size, seq_len, image_size, image_size),
                device=device,
            )
            loss, _ = model(
                dummy,
                targets=targets,
                return_loss=True,
                video_contains_first_frame=video_contains_first_frame,
            )
        else:
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

    # --- Memory-aware ceiling -------------------------------------------
    estimated, mem_info = _estimate_memory_ceiling(
        model, image_size, seq_len, device,
    )
    if estimated is not None:
        print(
            f"[auto-batch] GPU: {mem_info['gpu_name']} "
            f"({mem_info['free_gb']:.1f} / {mem_info['total_gb']:.1f} GB free/total)"
        )
        print(
            f"[auto-batch] Model: {mem_info['param_count_m']:.1f}M params "
            f"({mem_info['param_mb']:.0f} MB)  "
            f"grad+optim reserve: {mem_info['fixed_overhead_mb']:.0f} MB"
        )
        print(
            f"[auto-batch] Per-sample estimate: {mem_info['per_sample_mb']:.1f} MB "
            f"(input {mem_info['input_mb']:.1f} MB × {_ACTIVATION_RATIO})"
        )
        if estimated < ceiling:
            print(
                f"[auto-batch] Memory estimate → max ~{estimated} samples; "
                f"clamping ceiling {ceiling} → {estimated}"
            )
            ceiling = estimated

    probe = min(max(floor, (ceiling - floor)//2), ceiling)
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
