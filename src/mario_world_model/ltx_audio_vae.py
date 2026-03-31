from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _num_groups(channels: int, preferred: int = 8) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


@dataclass
class AudioVAEOutput:
    reconstruction: Tensor
    posterior_mean: Tensor
    posterior_logvar: Tensor
    latents: Tensor


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        pad_t = self.kernel_size[0] - 1
        pad_f = self.kernel_size[1] // 2
        if pad_t > 0 or pad_f > 0:
            x = F.pad(x, (pad_f, pad_f, pad_t, 0), mode="replicate")
        return self.conv(x)


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class Upsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv2d(in_channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=(2.0, 2.0), mode="nearest")
        return self.conv(x)


class LTXAudioVAE(nn.Module):
    """Causal mel-spectrogram VAE with 4x temporal compression."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        n_mels: int = 64,
        base_channels: int = 64,
        latent_channels: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_mels = n_mels
        self.latent_channels = latent_channels

        hidden_2 = base_channels * 2
        hidden_4 = base_channels * 4

        self.encoder_in = CausalConv2d(in_channels, base_channels, kernel_size=3)
        self.encoder_block1 = ResidualBlock2d(base_channels)
        self.encoder_down1 = CausalConv2d(base_channels, hidden_2, kernel_size=3, stride=(2, 2))
        self.encoder_block2 = ResidualBlock2d(hidden_2)
        self.encoder_down2 = CausalConv2d(hidden_2, hidden_4, kernel_size=3, stride=(2, 2))
        self.encoder_block3 = ResidualBlock2d(hidden_4)
        self.encoder_mid = ResidualBlock2d(hidden_4)
        self.encoder_out = nn.Conv2d(hidden_4, latent_channels * 2, kernel_size=1)

        self.decoder_in = nn.Conv2d(latent_channels, hidden_4, kernel_size=1)
        self.decoder_mid = ResidualBlock2d(hidden_4)
        self.decoder_block3 = ResidualBlock2d(hidden_4)
        self.decoder_up1 = Upsample2d(hidden_4, hidden_2)
        self.decoder_block2 = ResidualBlock2d(hidden_2)
        self.decoder_up2 = Upsample2d(hidden_2, base_channels)
        self.decoder_block1 = ResidualBlock2d(base_channels)
        self.decoder_norm = nn.GroupNorm(_num_groups(base_channels), base_channels)
        self.decoder_out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def encode(self, mel: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder_in(mel)
        x = self.encoder_block1(x)
        x = self.encoder_down1(x)
        x = self.encoder_block2(x)
        x = self.encoder_down2(x)
        x = self.encoder_block3(x)
        x = self.encoder_mid(x)
        mean, logvar = self.encoder_out(x).chunk(2, dim=1)
        return mean, torch.clamp(logvar, min=-30.0, max=20.0)

    def reparameterize(self, mean: Tensor, logvar: Tensor, sample_posterior: bool = True) -> Tensor:
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, latents: Tensor, output_shape: tuple[int, int] | None = None) -> Tensor:
        x = self.decoder_in(latents)
        x = self.decoder_mid(x)
        x = self.decoder_block3(x)
        x = self.decoder_up1(x)
        x = self.decoder_block2(x)
        x = self.decoder_up2(x)
        x = self.decoder_block1(x)
        x = torch.sigmoid(self.decoder_out(F.silu(self.decoder_norm(x))))
        if output_shape is not None:
            target_t, target_f = output_shape
            x = x[:, :, :target_t, :target_f]
        return x

    def forward(self, mel: Tensor, *, sample_posterior: bool = True) -> AudioVAEOutput:
        if mel.ndim != 4:
            raise ValueError(f"Expected mel tensor with shape (B, C, T, F), got {tuple(mel.shape)}")
        if mel.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channel(s), got {mel.shape[1]}")
        if mel.shape[-1] != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {mel.shape[-1]}")

        mean, logvar = self.encode(mel)
        latents = self.reparameterize(mean, logvar, sample_posterior=sample_posterior)
        reconstruction = self.decode(latents, output_shape=mel.shape[-2:])
        return AudioVAEOutput(
            reconstruction=reconstruction,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
        )