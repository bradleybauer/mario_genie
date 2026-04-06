from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.audio_features import mel_time_frequency_shape


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(prediction, target)
    diff = (prediction - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def build_mel_mask(
    waveform_lengths: torch.Tensor,
    *,
    max_time_steps: int,
    n_fft: int,
    hop_length: int,
    context_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if context_lengths is not None and context_lengths.shape != waveform_lengths.shape:
        raise ValueError(
            "context_lengths must have same shape as waveform_lengths: "
            f"got {tuple(context_lengths.shape)} vs {tuple(waveform_lengths.shape)}"
        )

    mask = torch.zeros(
        (waveform_lengths.shape[0], 1, max_time_steps, 1),
        dtype=torch.float32,
        device=waveform_lengths.device,
    )
    context_iter = context_lengths.tolist() if context_lengths is not None else [0] * waveform_lengths.shape[0]
    for batch_idx, (length, context_length) in enumerate(zip(waveform_lengths.tolist(), context_iter, strict=True)):
        valid_steps, _ = mel_time_frequency_shape(int(length), n_fft=n_fft, hop_length=hop_length, center=False)
        context_steps = 0
        if context_length > 0:
            context_steps, _ = mel_time_frequency_shape(
                int(context_length),
                n_fft=n_fft,
                hop_length=hop_length,
                center=False,
            )

        start_step = min(context_steps, max_time_steps)
        end_step = min(valid_steps, max_time_steps)
        if end_step > start_step:
            mask[batch_idx, :, start_step:end_step] = 1.0

    return mask


def build_waveform_mask(
    lengths: torch.Tensor,
    *,
    max_samples: int,
    context_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if context_lengths is not None and context_lengths.shape != lengths.shape:
        raise ValueError(
            "context_lengths must have same shape as lengths: "
            f"got {tuple(context_lengths.shape)} vs {tuple(lengths.shape)}"
        )

    mask = torch.zeros((lengths.shape[0], 1, max_samples), dtype=torch.float32, device=lengths.device)
    context_iter = context_lengths.tolist() if context_lengths is not None else [0] * lengths.shape[0]
    for batch_idx, (length, context_length) in enumerate(zip(lengths.tolist(), context_iter, strict=True)):
        start_sample = min(max(int(context_length), 0), max_samples)
        end_sample = min(max(int(length), 0), max_samples)
        if end_sample > start_sample:
            mask[batch_idx, :, start_sample:end_sample] = 1.0

    return mask


def context_waveform_lengths(audio_lengths: torch.Tensor, *, context_frames: int) -> torch.Tensor:
    if audio_lengths.ndim != 2:
        raise ValueError(f"Expected audio_lengths with shape (B, T), got {tuple(audio_lengths.shape)}")
    if context_frames <= 0:
        return torch.zeros(audio_lengths.shape[0], dtype=torch.int64, device=audio_lengths.device)

    capped_context = min(context_frames, audio_lengths.shape[1])
    return audio_lengths[:, :capped_context].to(dtype=torch.int64).sum(dim=1)