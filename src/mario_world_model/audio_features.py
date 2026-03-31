from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from mario_world_model.config import (
    AUDIO_DB_FLOOR,
    AUDIO_FMAX,
    AUDIO_FMIN,
    AUDIO_HOP_LENGTH,
    AUDIO_N_FFT,
    AUDIO_N_MELS,
    AUDIO_SAMPLE_RATE,
)


def hz_to_mel(freq_hz: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(freq_mel: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, freq_mel / 2595.0) - 1.0)


def build_mel_filter_bank(
    sample_rate: int = AUDIO_SAMPLE_RATE,
    n_fft: int = AUDIO_N_FFT,
    n_mels: int = AUDIO_N_MELS,
    fmin: float = AUDIO_FMIN,
    fmax: float | None = AUDIO_FMAX,
) -> Tensor:
    max_freq = float(sample_rate) / 2.0 if fmax is None else float(fmax)
    if fmin < 0 or max_freq <= fmin:
        raise ValueError(f"Invalid mel range: fmin={fmin}, fmax={max_freq}")

    mel_min = hz_to_mel(torch.tensor(float(fmin), dtype=torch.float32))
    mel_max = hz_to_mel(torch.tensor(float(max_freq), dtype=torch.float32))
    mel_edges = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=torch.float32)
    hz_edges = mel_to_hz(mel_edges)
    fft_freqs = torch.linspace(0.0, float(sample_rate) / 2.0, n_fft // 2 + 1, dtype=torch.float32)
    filter_bank = torch.zeros((n_mels, fft_freqs.numel()), dtype=torch.float32)

    for mel_idx in range(n_mels):
        left_hz = hz_edges[mel_idx]
        center_hz = hz_edges[mel_idx + 1]
        right_hz = hz_edges[mel_idx + 2]

        left_mask = (fft_freqs >= left_hz) & (fft_freqs <= center_hz)
        right_mask = (fft_freqs >= center_hz) & (fft_freqs <= right_hz)

        if center_hz > left_hz:
            filter_bank[mel_idx, left_mask] = (
                (fft_freqs[left_mask] - left_hz) / (center_hz - left_hz)
            )
        if right_hz > center_hz:
            filter_bank[mel_idx, right_mask] = (
                (right_hz - fft_freqs[right_mask]) / (right_hz - center_hz)
            )

    return filter_bank


def frame_audio_to_waveform(audio_frames: Tensor, audio_lengths: Tensor | None = None) -> Tensor:
    squeeze = False
    if audio_frames.ndim == 2:
        audio_frames = audio_frames.unsqueeze(0)
        squeeze = True

    if audio_frames.ndim != 3:
        raise ValueError(f"Expected audio frames with shape (B, T, S) or (T, S), got {tuple(audio_frames.shape)}")

    if audio_lengths is None:
        waveform = audio_frames.reshape(audio_frames.shape[0], -1)
    else:
        if audio_lengths.ndim == 1:
            audio_lengths = audio_lengths.unsqueeze(0)
        if audio_lengths.shape[:2] != audio_frames.shape[:2]:
            raise ValueError(
                "audio_lengths must match the first two dimensions of audio_frames: "
                f"got {tuple(audio_lengths.shape)} vs {tuple(audio_frames.shape)}"
            )
        total_lengths = audio_lengths.sum(dim=1)
        max_length = int(total_lengths.max().item())
        waveform = audio_frames.new_zeros((audio_frames.shape[0], max_length))
        for batch_idx in range(audio_frames.shape[0]):
            cursor = 0
            for frame_idx in range(audio_frames.shape[1]):
                length = int(audio_lengths[batch_idx, frame_idx].item())
                if length <= 0:
                    continue
                waveform[batch_idx, cursor:cursor + length] = audio_frames[batch_idx, frame_idx, :length]
                cursor += length

    if squeeze:
        waveform = waveform.squeeze(0)
    return waveform


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        n_fft: int = AUDIO_N_FFT,
        hop_length: int = AUDIO_HOP_LENGTH,
        n_mels: int = AUDIO_N_MELS,
        fmin: float = AUDIO_FMIN,
        fmax: float | None = AUDIO_FMAX,
        db_floor: float = AUDIO_DB_FLOOR,
        center: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.db_floor = db_floor
        self.center = center

        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "mel_filter_bank",
            build_mel_filter_bank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
            ),
            persistent=False,
        )

    def normalize_db(self, db_values: Tensor) -> Tensor:
        db_values = torch.clamp(db_values, min=-self.db_floor, max=0.0)
        return (db_values + self.db_floor) / self.db_floor

    def denormalize(self, normalized: Tensor) -> Tensor:
        return normalized * self.db_floor - self.db_floor

    def forward(self, waveform: Tensor) -> Tensor:
        squeeze = False
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform with shape (B, S) or (S,), got {tuple(waveform.shape)}")

        if waveform.shape[-1] < self.n_fft:
            pad = self.n_fft - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=self.center,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        mel = torch.einsum(
            "mf,bft->bmt",
            self.mel_filter_bank.to(device=power.device, dtype=power.dtype),
            power,
        )
        mel_db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        mel_norm = self.normalize_db(mel_db)
        mel_norm = mel_norm.transpose(1, 2).unsqueeze(1)

        if squeeze:
            mel_norm = mel_norm.squeeze(0)
        return mel_norm


def mel_time_frequency_shape(
    num_samples: int,
    *,
    n_fft: int = AUDIO_N_FFT,
    hop_length: int = AUDIO_HOP_LENGTH,
    center: bool = False,
) -> Tuple[int, int]:
    padded_samples = num_samples
    if center:
        padded_samples += n_fft
    if padded_samples < n_fft:
        padded_samples = n_fft
    time_steps = 1 + (padded_samples - n_fft) // hop_length
    return time_steps, AUDIO_N_MELS