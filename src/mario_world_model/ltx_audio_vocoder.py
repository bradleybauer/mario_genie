from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SnakeBeta(nn.Module):
    """Periodic activation used by BigVGAN/HiFi-GAN-style vocoders."""

    def __init__(self, channels: int, *, alpha: float = 1.0, trainable: bool = True) -> None:
        super().__init__()
        init = math.log(max(alpha, 1e-4))
        self.log_alpha = nn.Parameter(torch.full((channels,), init, dtype=torch.float32), requires_grad=trainable)
        self.log_beta = nn.Parameter(torch.full((channels,), init, dtype=torch.float32), requires_grad=trainable)
        self.eps = 1e-9

    def forward(self, x: Tensor) -> Tensor:
        alpha = torch.exp(self.log_alpha).view(1, -1, 1)
        beta = torch.exp(self.log_beta).view(1, -1, 1)
        return x + (1.0 / (beta + self.eps)) * torch.sin(alpha * x).pow(2)


class VocoderResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilations: Sequence[int],
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if not dilations:
            raise ValueError("dilations must be non-empty")

        self.acts = nn.ModuleList([SnakeBeta(channels) for _ in dilations])
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=int(dilation),
                    padding=((kernel_size - 1) // 2) * int(dilation),
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for act, conv in zip(self.acts, self.convs, strict=True):
            residual = x
            x = conv(act(x))
            x = x + residual
        return x


class LTXAudioVocoder(nn.Module):
    """Compact mel-to-waveform vocoder for the mario_world_model audio stack.

    Input mel formats:
    - (B, T, F) for mono
    - (B, C, T, F) for multi-channel conditioning (for example stereo)

    Output waveform format:
    - (B, out_channels, S)
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        n_mels: int = 64,
        out_channels: int = 1,
        upsample_initial_channel: int = 512,
        upsample_rates: Sequence[int] = (5, 5, 4),
        upsample_kernel_sizes: Sequence[int] = (11, 11, 8),
        resblock_kernel_sizes: Sequence[int] = (3, 7),
        resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 9), (1, 3, 9)),
        hop_length: int = 100,
        n_fft: int = 400,
        input_prepad_frames: int | None = None,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")
        if upsample_initial_channel <= 0:
            raise ValueError(f"upsample_initial_channel must be positive, got {upsample_initial_channel}")
        if len(upsample_rates) == 0:
            raise ValueError("upsample_rates must be non-empty")
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must have the same length")
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError("resblock_kernel_sizes and resblock_dilation_sizes must have the same length")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {n_fft}")

        upsample_factor = int(math.prod(int(rate) for rate in upsample_rates))
        if upsample_factor != hop_length:
            raise ValueError(
                f"Product of upsample_rates ({upsample_factor}) must equal hop_length ({hop_length})"
            )

        self.in_channels = int(in_channels)
        self.n_mels = int(n_mels)
        self.out_channels = int(out_channels)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.upsample_rates = tuple(int(rate) for rate in upsample_rates)

        if input_prepad_frames is None:
            # For center=False STFT, this recovers the common target length formula:
            #   n_fft + (T - 1) * hop_length
            self.input_prepad_frames = max(self.n_fft // self.hop_length - 1, 0)
        else:
            self.input_prepad_frames = max(int(input_prepad_frames), 0)

        self.conv_pre = nn.Conv1d(
            in_channels=self.in_channels * self.n_mels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        current_channels = int(upsample_initial_channel)
        for idx, (rate, kernel_size) in enumerate(zip(self.upsample_rates, upsample_kernel_sizes, strict=True)):
            stride = int(rate)
            kernel = int(kernel_size)
            if stride <= 0:
                raise ValueError(f"upsample stride must be positive, got {stride} at index {idx}")
            if kernel < stride:
                raise ValueError(
                    f"upsample kernel_size must be >= stride, got kernel_size={kernel}, stride={stride}"
                )
            if (kernel - stride) % 2 != 0:
                raise ValueError(
                    "upsample_kernel_sizes must satisfy (kernel_size - stride) % 2 == 0 "
                    f"for exact-length upsampling. Got kernel_size={kernel}, stride={stride}"
                )

            next_channels = max(current_channels // 2, out_channels)
            padding = (kernel - stride) // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    current_channels,
                    next_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

            stage_blocks = nn.ModuleList()
            for block_kernel, block_dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                stage_blocks.append(
                    VocoderResidualBlock(
                        next_channels,
                        kernel_size=int(block_kernel),
                        dilations=tuple(int(dilation) for dilation in block_dilations),
                    )
                )
            self.resblocks.append(stage_blocks)
            current_channels = next_channels

        self.act_post = SnakeBeta(current_channels)
        self.conv_post = nn.Conv1d(
            in_channels=current_channels,
            out_channels=self.out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

    @property
    def num_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def expected_output_length(self, mel_time_steps: int) -> int:
        if mel_time_steps <= 0:
            raise ValueError(f"mel_time_steps must be positive, got {mel_time_steps}")
        return (int(mel_time_steps) + self.input_prepad_frames) * self.hop_length

    def _prepare_mel(self, mel: Tensor) -> Tensor:
        if mel.ndim == 3:
            # (B, T, F) -> (B, 1, T, F)
            mel = mel.unsqueeze(1)

        if mel.ndim != 4:
            raise ValueError(f"Expected mel with shape (B, T, F) or (B, C, T, F), got {tuple(mel.shape)}")

        if mel.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} mel channel(s), got {mel.shape[1]}")
        if mel.shape[-1] != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {mel.shape[-1]}")

        # (B, C, T, F) -> (B, C * F, T)
        mel = mel.permute(0, 1, 3, 2).reshape(mel.shape[0], self.in_channels * self.n_mels, mel.shape[2])
        if self.input_prepad_frames > 0:
            # Replicate the earliest frame to provide left context for the first sample region.
            prefix = mel[:, :, :1].expand(-1, -1, self.input_prepad_frames)
            mel = torch.cat([prefix, mel], dim=-1)
        return mel

    def forward(self, mel: Tensor, *, output_length: int | None = None) -> Tensor:
        x = self._prepare_mel(mel)
        x = self.conv_pre(x)

        for upsample, stage_blocks in zip(self.ups, self.resblocks, strict=True):
            x = F.leaky_relu(x, negative_slope=0.1)
            x = upsample(x)
            block_outputs = torch.stack([block(x) for block in stage_blocks], dim=0)
            x = block_outputs.mean(dim=0)

        waveform = torch.tanh(self.conv_post(self.act_post(x)))

        if output_length is not None:
            target_length = int(output_length)
            if target_length <= 0:
                raise ValueError(f"output_length must be positive, got {target_length}")
            current_length = waveform.shape[-1]
            if current_length > target_length:
                waveform = waveform[..., :target_length]
            elif current_length < target_length:
                waveform = F.pad(waveform, (0, target_length - current_length))

        return waveform