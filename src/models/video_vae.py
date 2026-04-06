"""Video VAE with a fully learned symmetric encoder/decoder.

Default spatial path: 5 learned 2x spatial downsamples for a total
compression of 32x. On 224x224 inputs this produces a 7x7 latent grid.

Temporal compression is configurable: 0 or 1 learned temporal downsamples. When
enabled, the deepest encoder stage packs adjacent frame pairs before a causal
conv, so the latent time length becomes ``ceil(T / 2)`` and the decoder upsamples
time back to the original length during ``forward``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F


def _num_groups(channels: int, preferred: int = 8) -> int:
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


@dataclass
class VideoVAEOutput:
    logits: Tensor
    posterior_mean: Tensor
    posterior_logvar: Tensor
    latents: Tensor


class SpatialPatchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected video tensor with shape (B, C, T, H, W), got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input resolution {(h, w)} must be divisible by patch_size={self.patch_size}"
            )
        return rearrange(
            x,
            "b c t (h ph) (w pw) -> b (c ph pw) t h w",
            ph=self.patch_size,
            pw=self.patch_size,
        )


class SpatialUnpatchify(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor, *, out_channels: int) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected video tensor with shape (B, C, T, H, W), got {tuple(x.shape)}")
        expected_channels = out_channels * self.patch_size * self.patch_size
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels for out_channels={out_channels}, got {x.shape[1]}"
            )
        return rearrange(
            x,
            "b (c ph pw) t h w -> b c t (h ph) (w pw)",
            c=out_channels,
            ph=self.patch_size,
            pw=self.patch_size,
        )


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        pad_t = self.kernel_size[0] - 1
        pad_h = self.kernel_size[1] // 2
        pad_w = self.kernel_size[2] // 2

        if pad_t > 0:
            first_frame = x[:, :, :1].expand(-1, -1, pad_t, -1, -1)
            x = torch.cat((first_frame, x), dim=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h, 0, 0), mode="replicate")
        return self.conv(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, downsample_time: bool = False) -> None:
        super().__init__()
        self.downsample_time = downsample_time
        conv_in_channels = in_channels * (2 if downsample_time else 1)
        self.conv = CausalConv3d(conv_in_channels, out_channels, kernel_size=3, stride=(1, 2, 2))

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample_time:
            if x.shape[2] % 2 != 0:
                x = torch.cat((x, x[:, :, -1:]), dim=2)
            x = rearrange(x, "b c (t pair) h w -> b (c pair) t h w", pair=2)
        return self.conv(x)


class SpatialUpsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, upsample_time: bool = False) -> None:
        super().__init__()
        self.scale_factor = (2.0 if upsample_time else 1.0, 2.0, 2.0)
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


class VideoVAE(nn.Module):
    """Palette-aware 3D convolutional VAE with symmetric spatial hierarchy.

    Encoder: learned stem -> 5 learned spatial downsamples -> latent
    Decoder: latent -> 5 learned upsamples -> full resolution logits

    Spatial compression: 32x.
    Temporal compression: 1x or 2x depending on ``temporal_downsample``.
    """

    def __init__(
        self,
        *,
        num_colors: int,
        base_channels: int = 64,
        latent_channels: int = 64,
        temporal_downsample: int = 0,
    ) -> None:
        super().__init__()
        if temporal_downsample not in (0, 1):
            raise ValueError("temporal_downsample must be 0 or 1")
        self.num_colors = num_colors
        self.latent_channels = latent_channels
        self.temporal_downsample = temporal_downsample

        hidden_2 = base_channels * 2
        hidden_4 = base_channels * 4
        hidden_8 = base_channels * 8

        self.encoder_in = CausalConv3d(num_colors, base_channels, kernel_size=3)
        self.encoder_block1 = ResidualBlock3D(base_channels)
        self.encoder_down1 = Downsample3D(base_channels, base_channels)
        self.encoder_block2 = ResidualBlock3D(base_channels)
        self.encoder_down2 = Downsample3D(base_channels, hidden_2)
        self.encoder_block3 = ResidualBlock3D(hidden_2)
        self.encoder_down3 = Downsample3D(hidden_2, hidden_4)
        self.encoder_block4 = ResidualBlock3D(hidden_4)
        self.encoder_down4 = Downsample3D(hidden_4, hidden_8)
        self.encoder_block5 = ResidualBlock3D(hidden_8)
        self.encoder_down5 = Downsample3D(hidden_8, hidden_8, downsample_time=temporal_downsample == 1)
        self.encoder_block6 = ResidualBlock3D(hidden_8)
        self.encoder_mid = ResidualBlock3D(hidden_8)
        self.encoder_out = nn.Conv3d(hidden_8, latent_channels * 2, kernel_size=1)

        self.decoder_in = nn.Conv3d(latent_channels, hidden_8, kernel_size=1)
        self.decoder_mid = ResidualBlock3D(hidden_8)
        self.decoder_block6 = ResidualBlock3D(hidden_8)

        self.decoder_up1 = SpatialUpsample3D(hidden_8, hidden_8, upsample_time=temporal_downsample == 1)
        self.decoder_block5 = ResidualBlock3D(hidden_8)

        self.decoder_up2 = SpatialUpsample3D(hidden_8, hidden_4)
        self.decoder_block4 = ResidualBlock3D(hidden_4)

        self.decoder_up3 = SpatialUpsample3D(hidden_4, hidden_2)
        self.decoder_block3 = ResidualBlock3D(hidden_2)

        self.decoder_up4 = SpatialUpsample3D(hidden_2, base_channels)
        self.decoder_block2 = ResidualBlock3D(base_channels)

        self.decoder_up5 = SpatialUpsample3D(base_channels, base_channels)
        self.decoder_block1 = ResidualBlock3D(base_channels)

        self.decoder_norm = nn.GroupNorm(_num_groups(base_channels), base_channels)
        self.decoder_out = CausalConv3d(base_channels, num_colors, kernel_size=3)

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def encode(self, video: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder_in(video)
        x = self.encoder_block1(x)
        x = self.encoder_down1(x)
        x = self.encoder_block2(x)
        x = self.encoder_down2(x)
        x = self.encoder_block3(x)
        x = self.encoder_down3(x)
        x = self.encoder_block4(x)
        x = self.encoder_down4(x)
        x = self.encoder_block5(x)
        x = self.encoder_down5(x)
        x = self.encoder_block6(x)
        x = self.encoder_mid(x)
        mean, logvar = self.encoder_out(x).chunk(2, dim=1)
        return mean, torch.clamp(logvar, min=-30.0, max=10.0)

    def reparameterize(self, mean: Tensor, logvar: Tensor, sample_posterior: bool = True) -> Tensor:
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, latents: Tensor, *, output_shape: tuple[int, int, int] | None = None) -> Tensor:
        x = self.decoder_in(latents)
        x = self.decoder_mid(x)
        x = self.decoder_block6(x)
        x = self.decoder_up1(x)
        x = self.decoder_block5(x)
        x = self.decoder_up2(x)
        x = self.decoder_block4(x)
        x = self.decoder_up3(x)
        x = self.decoder_block3(x)
        x = self.decoder_up4(x)
        x = self.decoder_block2(x)
        x = self.decoder_up5(x)
        x = self.decoder_block1(x)
        x = self.decoder_out(F.silu(self.decoder_norm(x)))
        if output_shape is not None:
            output_frames, output_height, output_width = output_shape
            if (
                x.shape[2] < output_frames
                or x.shape[3] < output_height
                or x.shape[4] < output_width
            ):
                raise ValueError(
                    f"Decoded tensor shape {tuple(x.shape)} is smaller than requested output shape {output_shape}"
                )
            x = x[:, :, :output_frames, :output_height, :output_width]
        return x

    def forward(self, video: Tensor, *, sample_posterior: bool = True) -> VideoVAEOutput:
        if video.ndim != 5:
            raise ValueError(f"Expected video tensor with shape (B, C, T, H, W), got {tuple(video.shape)}")
        if video.shape[1] != self.num_colors:
            raise ValueError(
                f"Expected {self.num_colors} channels, got {video.shape[1]}. "
                "Pass palette one-hot frames into the video VAE."
            )
        mean, logvar = self.encode(video)
        latents = self.reparameterize(mean, logvar, sample_posterior=sample_posterior)
        logits = self.decode(latents, output_shape=(video.shape[2], video.shape[3], video.shape[4]))
        return VideoVAEOutput(
            logits=logits,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
        )
