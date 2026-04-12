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

from src.models.onehot_conv3d import OneHotConv3d


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
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
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
        stride_t = 2 if upsample_time else 1
        self.time_scale = stride_t
        # Fuse nearest-neighbor upsample + conv into one transposed-conv op.
        # Temporal padding stays left-aligned (no symmetric padding), then we trim
        # any extra tail frames so each stage matches the expected scale exactly.
        self.deconv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(3, 4, 4),
            stride=(stride_t, 2, 2),
            padding=(0, 1, 1),
            output_padding=(0, 0, 0),
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        target_t = x.shape[2] * self.time_scale
        x = self.deconv(x)
        if x.shape[2] < target_t:
            raise ValueError(
                f"Upsample produced too few frames: got {x.shape[2]}, expected >= {target_t}"
            )
        if x.shape[2] > target_t:
            x = x[:, :, :target_t]
        return x


class GlobalBottleneckAttention3D(nn.Module):
    """Lightweight global self-attention over ``(T * H * W)`` bottleneck tokens."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        hidden = max(channels, int(channels * mlp_ratio))
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        spatial = h * w
        tokens = rearrange(x, "b c t h w -> b (t h w) c")
        attn_in = self.norm1(tokens)
        # Causal mask: each token can attend to tokens at the same or earlier
        # time step.  Tokens are laid out as (t0_hw, t1_hw, …), so the block
        # index for token i is i // spatial.
        seq_len = t * spatial
        idx = torch.arange(seq_len, device=x.device)
        time_idx = idx // spatial  # (seq_len,)
        # attn_mask: True means *blocked*.  Block positions where query time < key time.
        attn_mask = time_idx.unsqueeze(1) < time_idx.unsqueeze(0)  # (seq_len, seq_len)
        attn_out, _ = self.attn(
            attn_in, attn_in, attn_in, attn_mask=attn_mask, need_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return rearrange(tokens, "b (t h w) c -> b c t h w", t=t, h=h, w=w)


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
        base_channels: int = 24,
        latent_channels: int = 16,
        temporal_downsample: int = 0,
        dropout: float = 0.01,
        onehot_conv: bool = False,
        global_bottleneck_attn: bool = False,
        global_bottleneck_attn_heads: int = 8,
    ) -> None:
        super().__init__()
        if temporal_downsample not in (0, 1):
            raise ValueError("temporal_downsample must be 0 or 1")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if global_bottleneck_attn_heads <= 0:
            raise ValueError("global_bottleneck_attn_heads must be positive")
        self.num_colors = num_colors
        self.latent_channels = latent_channels
        self.temporal_downsample = temporal_downsample
        self.onehot_conv = onehot_conv
        self.global_bottleneck_attn = global_bottleneck_attn
        self.global_bottleneck_attn_heads = global_bottleneck_attn_heads

        hidden_2 = base_channels * 2
        hidden_4 = base_channels * 4
        hidden_8 = base_channels * 8

        if onehot_conv:
            self.encoder_in = OneHotConv3d(num_colors, base_channels, kernel_size=3)
        else:
            self.encoder_in = CausalConv3d(num_colors, base_channels, kernel_size=3)
        self.encoder_block1 = ResidualBlock3D(base_channels, dropout=dropout)
        self.encoder_down1 = Downsample3D(base_channels, base_channels)
        self.encoder_block2 = ResidualBlock3D(base_channels, dropout=dropout)
        self.encoder_down2 = Downsample3D(base_channels, hidden_2)
        self.encoder_block3 = ResidualBlock3D(hidden_2, dropout=dropout)
        self.encoder_down3 = Downsample3D(hidden_2, hidden_4)
        self.encoder_block4 = ResidualBlock3D(hidden_4, dropout=dropout)
        self.encoder_down4 = Downsample3D(hidden_4, hidden_8)
        self.encoder_block5 = ResidualBlock3D(hidden_8, dropout=dropout)
        self.encoder_down5 = Downsample3D(hidden_8, hidden_8, downsample_time=temporal_downsample == 1)
        self.encoder_block6 = ResidualBlock3D(hidden_8, dropout=dropout)
        self.encoder_mid = ResidualBlock3D(hidden_8, dropout=dropout)
        self.encoder_global_attn = (
            GlobalBottleneckAttention3D(
                hidden_8,
                num_heads=global_bottleneck_attn_heads,
                dropout=dropout,
            )
            if global_bottleneck_attn
            else None
        )
        self.encoder_out = nn.Conv3d(hidden_8, latent_channels * 2, kernel_size=1)

        self.decoder_in = nn.Conv3d(latent_channels, hidden_8, kernel_size=1)
        self.decoder_mid = ResidualBlock3D(hidden_8, dropout=dropout)
        self.decoder_block6 = ResidualBlock3D(hidden_8, dropout=dropout)
        self.decoder_global_attn = (
            GlobalBottleneckAttention3D(
                hidden_8,
                num_heads=global_bottleneck_attn_heads,
                dropout=dropout,
            )
            if global_bottleneck_attn
            else None
        )

        self.decoder_up1 = SpatialUpsample3D(hidden_8, hidden_8, upsample_time=temporal_downsample == 1)
        self.decoder_block5 = ResidualBlock3D(hidden_8, dropout=dropout)

        self.decoder_up2 = SpatialUpsample3D(hidden_8, hidden_4)
        self.decoder_block4 = ResidualBlock3D(hidden_4, dropout=dropout)

        self.decoder_up3 = SpatialUpsample3D(hidden_4, hidden_2)
        self.decoder_block3 = ResidualBlock3D(hidden_2, dropout=dropout)

        self.decoder_up4 = SpatialUpsample3D(hidden_2, base_channels)
        self.decoder_block2 = ResidualBlock3D(base_channels, dropout=dropout)

        self.decoder_up5 = SpatialUpsample3D(base_channels, base_channels)
        self.decoder_block1 = ResidualBlock3D(base_channels, dropout=dropout)

        self.decoder_norm = nn.GroupNorm(_num_groups(base_channels), base_channels)
        self.decoder_out = CausalConv3d(base_channels, num_colors, kernel_size=3)

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, object],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Remap between CausalConv3d ↔ OneHotConv3d encoder_in keys.
        # CausalConv3d stores as encoder_in.conv.{weight,bias};
        # OneHotConv3d stores as encoder_in.{weight,bias}.
        causal_w = f"{prefix}encoder_in.conv.weight"
        causal_b = f"{prefix}encoder_in.conv.bias"
        onehot_w = f"{prefix}encoder_in.weight"
        onehot_b = f"{prefix}encoder_in.bias"
        if self.onehot_conv:
            # Loading into OneHotConv3d — remap CausalConv3d keys if present
            if causal_w in state_dict and onehot_w not in state_dict:
                state_dict[onehot_w] = state_dict.pop(causal_w)
            if causal_b in state_dict and onehot_b not in state_dict:
                state_dict[onehot_b] = state_dict.pop(causal_b)
        else:
            # Loading into CausalConv3d — remap OneHotConv3d keys if present
            if onehot_w in state_dict and causal_w not in state_dict:
                state_dict[causal_w] = state_dict.pop(onehot_w)
            if onehot_b in state_dict and causal_b not in state_dict:
                state_dict[causal_b] = state_dict.pop(onehot_b)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def encode(self, video: Tensor) -> tuple[Tensor, Tensor]:
        if self.onehot_conv:
            # OneHotConv3d expects 4D palette indices (B, T, H, W)
            if video.ndim == 5:
                video = video.argmax(dim=1).byte()
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
        if self.encoder_global_attn is not None:
            x = self.encoder_global_attn(x)
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
        if self.decoder_global_attn is not None:
            x = self.decoder_global_attn(x)
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
        """Accept palette indices ``(B, T, H, W)`` or one-hot ``(B, C, T, H, W)``."""
        if video.ndim == 4:
            # Palette indices (B, T, H, W)
            output_shape = (video.shape[1], video.shape[2], video.shape[3])
        elif video.ndim == 5:
            if video.shape[1] != self.num_colors:
                raise ValueError(
                    f"Expected {self.num_colors} channels, got {video.shape[1]}. "
                    "Pass palette one-hot frames or (B, T, H, W) index tensors."
                )
            output_shape = (video.shape[2], video.shape[3], video.shape[4])
        else:
            raise ValueError(f"Expected (B, T, H, W) or (B, C, T, H, W), got {tuple(video.shape)}")
        mean, logvar = self.encode(video)
        latents = self.reparameterize(mean, logvar, sample_posterior=sample_posterior)
        logits = self.decode(latents, output_shape=output_shape)
        return VideoVAEOutput(
            logits=logits,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
        )
