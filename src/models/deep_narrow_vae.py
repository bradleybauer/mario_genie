"""Deep narrow causal 3D VAE for palette-indexed NES video.

Uses many residual blocks per spatial level with very few channels,
producing a high depth-to-width ratio that keeps parameter count low
while maintaining good representational capacity.

Default architecture (patch_size=4, base_channels=32, blocks_per_level=6):
  224x224 input → patchify → 56x56
  Encoder: 4 spatial levels with 3 downsamples → 7x7 latent
  26 residual blocks in encoder, 26 in decoder = 52 total (104 conv3d layers)
  ~4M total parameters
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.video_vae import (
    CausalConv3d,
    ResidualBlock3D,
    SpatialPatchify,
    SpatialUnpatchify,
    SpatialUpsample3D,
    VideoVAEOutput,
    _num_groups,
)


class DeepNarrowVideoVAE(nn.Module):
    """A deep but narrow causal 3D VAE for palette-indexed video.

    Architecture: many residual blocks per spatial level with few channels.
    With default settings on 224x224 input:
      patchify(4): 56x56 → downsample 3x: 28→14→7
      channels: [32, 32, 48, 48] across 4 levels
      6 res blocks per level + 2 mid blocks = 26 encoder blocks
      Latent: 32 channels at 7x7 spatial
    """

    def __init__(
        self,
        *,
        num_colors: int,
        patch_size: int = 4,
        base_channels: int = 32,
        latent_channels: int = 32,
        blocks_per_level: int = 6,
        channel_mult: tuple[float, ...] = (1, 1, 1.5, 1.5),
    ) -> None:
        super().__init__()
        self.num_colors = num_colors
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.blocks_per_level = blocks_per_level
        self.channel_mult = channel_mult

        patched_channels = num_colors * patch_size * patch_size
        num_levels = len(channel_mult)
        channels = [max(int(base_channels * m), 1) for m in channel_mult]

        self.patchify = SpatialPatchify(patch_size)
        self.unpatchify = SpatialUnpatchify(patch_size)

        # --- Encoder ---
        self.encoder_in = CausalConv3d(patched_channels, channels[0], kernel_size=3)

        self.encoder_levels = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        for i in range(num_levels):
            level = nn.ModuleList(
                [ResidualBlock3D(channels[i]) for _ in range(blocks_per_level)]
            )
            self.encoder_levels.append(level)
            if i < num_levels - 1:
                self.encoder_downsamples.append(
                    CausalConv3d(channels[i], channels[i + 1], kernel_size=3, stride=(1, 2, 2))
                )

        self.encoder_mid = nn.Sequential(
            ResidualBlock3D(channels[-1]),
            ResidualBlock3D(channels[-1]),
        )
        self.encoder_out = nn.Conv3d(channels[-1], latent_channels * 2, kernel_size=1)

        # --- Decoder (mirror) ---
        self.decoder_in = nn.Conv3d(latent_channels, channels[-1], kernel_size=1)
        self.decoder_mid = nn.Sequential(
            ResidualBlock3D(channels[-1]),
            ResidualBlock3D(channels[-1]),
        )

        self.decoder_levels = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        for i in reversed(range(num_levels)):
            level = nn.ModuleList(
                [ResidualBlock3D(channels[i]) for _ in range(blocks_per_level)]
            )
            self.decoder_levels.append(level)
            if i > 0:
                self.decoder_upsamples.append(
                    SpatialUpsample3D(channels[i], channels[i - 1])
                )

        self.decoder_norm = nn.GroupNorm(_num_groups(channels[0]), channels[0])
        self.decoder_out = nn.Conv3d(
            channels[0],
            num_colors * patch_size * patch_size,
            kernel_size=1,
        )

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def encode(self, video: Tensor) -> tuple[Tensor, Tensor]:
        x = self.patchify(video)
        x = self.encoder_in(x)
        for i, level in enumerate(self.encoder_levels):
            for block in level:
                x = block(x)
            if i < len(self.encoder_downsamples):
                x = self.encoder_downsamples[i](x)
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
        for i, level in enumerate(self.decoder_levels):
            for block in level:
                x = block(x)
            if i < len(self.decoder_upsamples):
                x = self.decoder_upsamples[i](x)
        x = self.decoder_out(F.silu(self.decoder_norm(x)))
        return self.unpatchify(x, out_channels=self.num_colors)

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
