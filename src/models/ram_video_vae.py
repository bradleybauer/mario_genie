"""Joint VAE that encodes RAM and decodes both RAM and video.

The encoder operates on RAM byte sequences and produces a per-frame latent.
One decoder reconstructs the input RAM, while a second head projects the same
latent sequence into a compact 3D video latent grid and upsamples it into video
logits over the palette.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.ram_vae import ResidualFC, TemporalResBlock
from src.models.video_vae import CausalConv3d, ResidualBlock3D, SpatialUpsample3D, _num_groups


@dataclass
class RAMVideoVAEOutput:
    video_logits: Tensor
    ram_reconstruction: Tensor
    posterior_mean: Tensor
    posterior_logvar: Tensor
    latents: Tensor
    video_latents: Tensor


class RAMVideoVAE(nn.Module):
    """Encode RAM into a shared latent, then decode to RAM and video."""

    spatial_downsample_factor = 32

    def __init__(
        self,
        *,
        n_bytes: int,
        num_colors: int,
        frame_height: int,
        frame_width: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        n_fc_blocks: int = 2,
        n_temporal_blocks: int = 2,
        temporal_kernel_size: int = 3,
        video_base_channels: int = 24,
        video_latent_channels: int = 16,
        temporal_downsample: int = 0,
    ) -> None:
        super().__init__()
        if n_bytes <= 0:
            raise ValueError("n_bytes must be positive")
        if num_colors <= 0:
            raise ValueError("num_colors must be positive")
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("frame_height and frame_width must be positive")
        if temporal_downsample not in (0, 1):
            raise ValueError("temporal_downsample must be 0 or 1")

        self.n_bytes = n_bytes
        self.num_colors = num_colors
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.video_latent_channels = video_latent_channels
        self.temporal_downsample = temporal_downsample
        self.latent_height = max(1, math.ceil(frame_height / self.spatial_downsample_factor))
        self.latent_width = max(1, math.ceil(frame_width / self.spatial_downsample_factor))

        self.encoder_in = nn.Linear(n_bytes, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim) for _ in range(n_fc_blocks)]
        )
        self.encoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size) for _ in range(n_temporal_blocks)]
        )
        self.encoder_out = nn.Conv1d(hidden_dim, latent_dim * 2, kernel_size=1)

        self.ram_decoder_in = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.ram_decoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size) for _ in range(n_temporal_blocks)]
        )
        self.ram_decoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim) for _ in range(n_fc_blocks)]
        )
        self.ram_decoder_out = nn.Linear(hidden_dim, n_bytes)

        video_projection_in_dim = latent_dim * (2 if temporal_downsample == 1 else 1)
        self.video_latent_proj = nn.Linear(
            video_projection_in_dim,
            video_latent_channels * self.latent_height * self.latent_width,
        )

        hidden_2 = video_base_channels * 2
        hidden_4 = video_base_channels * 4
        hidden_8 = video_base_channels * 8
        self.video_decoder_in = nn.Conv3d(video_latent_channels, hidden_8, kernel_size=1)
        self.video_decoder_mid = ResidualBlock3D(hidden_8)
        self.video_decoder_block6 = ResidualBlock3D(hidden_8)

        self.video_decoder_up1 = SpatialUpsample3D(hidden_8, hidden_8, upsample_time=temporal_downsample == 1)
        self.video_decoder_block5 = ResidualBlock3D(hidden_8)

        self.video_decoder_up2 = SpatialUpsample3D(hidden_8, hidden_4)
        self.video_decoder_block4 = ResidualBlock3D(hidden_4)

        self.video_decoder_up3 = SpatialUpsample3D(hidden_4, hidden_2)
        self.video_decoder_block3 = ResidualBlock3D(hidden_2)

        self.video_decoder_up4 = SpatialUpsample3D(hidden_2, video_base_channels)
        self.video_decoder_block2 = ResidualBlock3D(video_base_channels)

        self.video_decoder_up5 = SpatialUpsample3D(video_base_channels, video_base_channels)
        self.video_decoder_block1 = ResidualBlock3D(video_base_channels)

        self.video_decoder_norm = nn.GroupNorm(_num_groups(video_base_channels), video_base_channels)
        self.video_decoder_out = CausalConv3d(video_base_channels, num_colors, kernel_size=3)

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def encode(self, ram: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, time_steps, n_bytes = ram.shape
        x = ram.reshape(batch_size * time_steps, n_bytes)
        x = F.silu(self.encoder_in(x))
        for block in self.encoder_blocks:
            x = block(x)

        x = x.reshape(batch_size, time_steps, self.hidden_dim).permute(0, 2, 1)
        for temporal_block in self.encoder_temporal:
            x = temporal_block(x)

        x = self.encoder_out(x).permute(0, 2, 1)
        mean, logvar = x.chunk(2, dim=-1)
        return mean, torch.clamp(logvar, min=-30.0, max=10.0)

    def reparameterize(
        self,
        mean: Tensor,
        logvar: Tensor,
        sample_posterior: bool = True,
    ) -> Tensor:
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode_ram(self, latents: Tensor) -> Tensor:
        batch_size, time_steps, _ = latents.shape
        x = latents.permute(0, 2, 1)
        x = self.ram_decoder_in(x)
        for temporal_block in self.ram_decoder_temporal:
            x = temporal_block(x)

        x = x.permute(0, 2, 1).reshape(batch_size * time_steps, self.hidden_dim)
        for block in self.ram_decoder_blocks:
            x = block(x)

        x = torch.sigmoid(self.ram_decoder_out(x))
        return x.reshape(batch_size, time_steps, self.n_bytes)

    def to_video_latents(self, latents: Tensor) -> Tensor:
        video_tokens = latents
        if self.temporal_downsample == 1:
            if video_tokens.shape[1] % 2 != 0:
                video_tokens = torch.cat((video_tokens, video_tokens[:, -1:]), dim=1)
            video_tokens = rearrange(video_tokens, "b (t pair) d -> b t (d pair)", pair=2)

        video_latents = self.video_latent_proj(video_tokens)
        return rearrange(
            video_latents,
            "b t (c h w) -> b c t h w",
            c=self.video_latent_channels,
            h=self.latent_height,
            w=self.latent_width,
        )

    def decode_video(
        self,
        video_latents: Tensor,
        *,
        output_shape: tuple[int, int, int] | None = None,
    ) -> Tensor:
        x = self.video_decoder_in(video_latents)
        x = self.video_decoder_mid(x)
        x = self.video_decoder_block6(x)
        x = self.video_decoder_up1(x)
        x = self.video_decoder_block5(x)
        x = self.video_decoder_up2(x)
        x = self.video_decoder_block4(x)
        x = self.video_decoder_up3(x)
        x = self.video_decoder_block3(x)
        x = self.video_decoder_up4(x)
        x = self.video_decoder_block2(x)
        x = self.video_decoder_up5(x)
        x = self.video_decoder_block1(x)
        x = self.video_decoder_out(F.silu(self.video_decoder_norm(x)))
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

    def forward(
        self,
        ram: Tensor,
        *,
        output_video_shape: tuple[int, int, int] | None = None,
        sample_posterior: bool = True,
    ) -> RAMVideoVAEOutput:
        if ram.dtype == torch.uint8:
            ram = ram.float() / 255.0
        if ram.ndim != 3:
            raise ValueError(
                f"Expected RAM tensor with shape (B, T, N_bytes), got {tuple(ram.shape)}"
            )
        if ram.shape[-1] != self.n_bytes:
            raise ValueError(f"Expected {self.n_bytes} RAM bytes, got {ram.shape[-1]}")

        mean, logvar = self.encode(ram)
        latents = self.reparameterize(mean, logvar, sample_posterior=sample_posterior)
        video_latents = self.to_video_latents(latents)
        if output_video_shape is None:
            output_video_shape = (ram.shape[1], self.frame_height, self.frame_width)
        video_logits = self.decode_video(video_latents, output_shape=output_video_shape)
        ram_reconstruction = self.decode_ram(latents)
        return RAMVideoVAEOutput(
            video_logits=video_logits,
            ram_reconstruction=ram_reconstruction,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
            video_latents=video_latents,
        )