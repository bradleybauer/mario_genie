from __future__ import annotations

import torch
import torch.nn as nn


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class ResBlockDown2D(nn.Module):
    """StyleGAN-like residual downsampling block for 2D spectrogram tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int] = (2, 2),
        activation_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=activation_slope, inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        return (x + residual) * (2.0**-0.5)


class ResBlockDown3D(nn.Module):
    """StyleGAN-like residual downsampling block for 3D video tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int, int] = (1, 2, 2),
        activation_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=activation_slope, inplace=True)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.skip_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        return (x + residual) * (2.0**-0.5)


class CompactVideoDiscriminator3D(nn.Module):
    """Compact 3D discriminator for palette-video GAN training.

    Defaults target ~9-10M trainable params and outputs one logit per sample.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 72,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        temporal_downsample_layers: tuple[int, ...] = (1, 2),
        activation_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")

        self.stem = nn.Conv3d(in_channels, base_channels * channel_multipliers[0], kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=activation_slope, inplace=True)

        blocks: list[nn.Module] = []
        current_channels = base_channels * channel_multipliers[0]

        for layer_idx, multiplier in enumerate(channel_multipliers):
            next_channels = base_channels * multiplier
            stride_t = 2 if layer_idx in temporal_downsample_layers else 1
            blocks.append(
                ResBlockDown3D(
                    current_channels,
                    next_channels,
                    stride=(stride_t, 2, 2),
                    activation_slope=activation_slope,
                )
            )
            current_channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Sequential(
            nn.Linear(current_channels, current_channels),
            nn.LeakyReLU(negative_slope=activation_slope, inplace=True),
            nn.Linear(current_channels, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(x.shape)}")

        x = self.activation(self.stem(x))
        for block in self.blocks:
            x = block(x)

        x = self.pool(x).flatten(1)
        logits = self.head(x)
        return logits.squeeze(-1)


class CompactSpectrogramDiscriminator2D(nn.Module):
    """Compact 2D discriminator for mel-spectrogram GAN training."""

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 80,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        activation_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if not channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")

        self.stem = nn.Conv2d(in_channels, base_channels * channel_multipliers[0], kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=activation_slope, inplace=True)

        blocks: list[nn.Module] = []
        current_channels = base_channels * channel_multipliers[0]

        for multiplier in channel_multipliers:
            next_channels = base_channels * multiplier
            blocks.append(
                ResBlockDown2D(
                    current_channels,
                    next_channels,
                    stride=(2, 2),
                    activation_slope=activation_slope,
                )
            )
            current_channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(current_channels, current_channels),
            nn.LeakyReLU(negative_slope=activation_slope, inplace=True),
            nn.Linear(current_channels, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, T, F), got {tuple(x.shape)}")

        x = self.activation(self.stem(x))
        for block in self.blocks:
            x = block(x)

        x = self.pool(x).flatten(1)
        logits = self.head(x)
        return logits.squeeze(-1)


def build_palette_discriminator(
    num_palette_colors: int,
    *,
    target_size: str = "~10m",
) -> CompactVideoDiscriminator3D:
    """Factory for project defaults.

    Supported presets:
    - ``target_size='~10m'``: around 9-10M parameters.
    - ``target_size='~5m'``: around 5M parameters.
    """
    if target_size == "~10m":
        return CompactVideoDiscriminator3D(in_channels=num_palette_colors)

    if target_size == "~5m":
        return CompactVideoDiscriminator3D(in_channels=num_palette_colors, base_channels=53)

    raise ValueError("target_size must be one of: '~10m', '~5m'")


def build_mel_discriminator(
    in_channels: int = 1,
    *,
    target_size: str = "~10m",
) -> CompactSpectrogramDiscriminator2D:
    """Factory for project-default mel-spectrogram discriminators."""
    if target_size == "~10m":
        return CompactSpectrogramDiscriminator2D(in_channels=in_channels)

    if target_size == "~5m":
        return CompactSpectrogramDiscriminator2D(in_channels=in_channels, base_channels=56)

    raise ValueError("target_size must be one of: '~10m', '~5m'")
