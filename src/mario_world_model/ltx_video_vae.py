"""LTX Video VAE — learned upsampling decoder for pixel-perfect reconstruction.

The encoder is patchify 4× + 2 strided downsamples = 16× spatial
compressions.  The decoder replaces the 1×1-conv + unpatchify with four learned
``SpatialUpsample3D`` stages (each 2×), so every pixel is generated with full
spatial context from its neighbors.

Latent shape is unchanged: ``(B, latent_ch, T, H/16, W/16)``.
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


class SpatialUpsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="nearest")
        return self.conv(x)


class LTXVideoVAE(nn.Module):
    """Palette-aware 3D convolutional VAE with learned-upsample decoder.

    Encoder: patchify(4×) → conv → 2 strided downsamples → latent
    Decoder: latent → 4 learned upsamples (each 2×) → full resolution logits

    The 4 upsamples replace the old 2-upsample + unpatchify pattern, giving the
    decoder spatial context at every resolution level for sharp reconstruction.

    Spatial compression: 16× (same as v1).
    Temporal compression: none (same as v1).
    """

    def __init__(
        self,
        *,
        num_colors: int,
        patch_size: int = 4,
        base_channels: int = 64,
        latent_channels: int = 64,
    ) -> None:
        super().__init__()
        self.num_colors = num_colors
        self.patch_size = patch_size
        self.latent_channels = latent_channels

        patched_channels = num_colors * patch_size * patch_size
        hidden_2 = base_channels * 2
        hidden_4 = base_channels * 4

        # --- Encoder (identical to v1) ---
        self.patchify = SpatialPatchify(patch_size)

        self.encoder_in = CausalConv3d(patched_channels, base_channels, kernel_size=3)
        self.encoder_block1 = ResidualBlock3D(base_channels)
        self.encoder_down1 = CausalConv3d(base_channels, hidden_2, kernel_size=3, stride=(1, 2, 2))
        self.encoder_block2 = ResidualBlock3D(hidden_2)
        self.encoder_down2 = CausalConv3d(hidden_2, hidden_4, kernel_size=3, stride=(1, 2, 2))
        self.encoder_block3 = ResidualBlock3D(hidden_4)
        self.encoder_mid = ResidualBlock3D(hidden_4)
        self.encoder_out = nn.Conv3d(hidden_4, latent_channels * 2, kernel_size=1)

        # --- Decoder (learned upsampling — replaces unpatchify) ---
        # Latent is at H/16, W/16.  We need 4 × 2× upsamples to reach full res.
        # Channel plan: hidden_4 → hidden_4 → hidden_2 → base → base//2 → logits
        half_base = max(base_channels // 2, num_colors)

        self.decoder_in = nn.Conv3d(latent_channels, hidden_4, kernel_size=1)
        self.decoder_mid = ResidualBlock3D(hidden_4)
        self.decoder_block3 = ResidualBlock3D(hidden_4)

        # Stage 1: H/16 → H/8  (matches v1 decoder_up1)
        self.decoder_up1 = SpatialUpsample3D(hidden_4, hidden_2)
        self.decoder_block2 = ResidualBlock3D(hidden_2)

        # Stage 2: H/8 → H/4  (matches v1 decoder_up2)
        self.decoder_up2 = SpatialUpsample3D(hidden_2, base_channels)
        self.decoder_block_up2 = ResidualBlock3D(base_channels)

        # Stage 3: H/4 → H/2  (NEW — replaces part of unpatchify)
        self.decoder_up3 = SpatialUpsample3D(base_channels, half_base)
        self.decoder_block_up3 = ResidualBlock3D(half_base)

        # Stage 4: H/2 → H  (NEW — replaces rest of unpatchify)
        self.decoder_up4 = SpatialUpsample3D(half_base, half_base)
        self.decoder_block_up4 = ResidualBlock3D(half_base)

        # Final projection to palette logits at full resolution
        self.decoder_norm = nn.GroupNorm(_num_groups(half_base), half_base)
        self.decoder_out = CausalConv3d(half_base, num_colors, kernel_size=3)

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def encode(self, video: Tensor) -> tuple[Tensor, Tensor]:
        x = self.patchify(video)
        x = self.encoder_in(x)
        x = self.encoder_block1(x)
        x = self.encoder_down1(x)
        x = self.encoder_block2(x)
        x = self.encoder_down2(x)
        x = self.encoder_block3(x)
        x = self.encoder_mid(x)
        mean, logvar = self.encoder_out(x).chunk(2, dim=1)
        return mean, torch.clamp(logvar, min=-30.0, max=10.0)

    def reparameterize(self, mean: Tensor, logvar: Tensor, sample_posterior: bool = True) -> Tensor:
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, latents: Tensor) -> Tensor:
        x = self.decoder_in(latents)
        x = self.decoder_mid(x)
        x = self.decoder_block3(x)
        # Stage 1: H/16 → H/8
        x = self.decoder_up1(x)
        x = self.decoder_block2(x)
        # Stage 2: H/8 → H/4
        x = self.decoder_up2(x)
        x = self.decoder_block_up2(x)
        # Stage 3: H/4 → H/2
        x = self.decoder_up3(x)
        x = self.decoder_block_up3(x)
        # Stage 4: H/2 → H
        x = self.decoder_up4(x)
        x = self.decoder_block_up4(x)
        # Project to num_colors logits at full resolution
        x = self.decoder_out(F.silu(self.decoder_norm(x)))
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
        logits = self.decode(latents)
        return VideoVAEOutput(
            logits=logits,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
        )
