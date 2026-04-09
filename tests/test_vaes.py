from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.gan_discriminator import build_mel_discriminator, build_palette_discriminator
from src.models.audio_vae import AudioVAE
from src.models.audio_vocoder import AudioVocoder
from src.models.ram_video_vae import RAMVideoVAE
from src.models.ram_video_vae_v2 import RAMVideoVAEv2
from src.models.onehot_conv3d import OneHotConv3d
from src.models.video_vae import CausalConv3d, VideoVAE


def test_video_vae_preserves_video_shape() -> None:
    model = VideoVAE(num_colors=8, base_channels=8, latent_channels=4)
    video = torch.randn(2, 8, 4, 32, 32)

    output = model(video, sample_posterior=False)

    assert output.logits.shape == video.shape
    assert output.posterior_mean.shape == (2, 4, 4, 1, 1)
    assert output.posterior_logvar.shape == (2, 4, 4, 1, 1)
    assert output.latents.shape == (2, 4, 4, 1, 1)


def test_video_vae_temporal_downsample_preserves_output_shape() -> None:
    model = VideoVAE(
        num_colors=8,
        base_channels=8,
        latent_channels=4,
        temporal_downsample=1,
    )
    video = torch.randn(2, 8, 5, 32, 32)

    output = model(video, sample_posterior=False)

    assert output.logits.shape == video.shape
    assert output.posterior_mean.shape == (2, 4, 3, 1, 1)
    assert output.posterior_logvar.shape == (2, 4, 3, 1, 1)
    assert output.latents.shape == (2, 4, 3, 1, 1)


def test_video_vae_global_bottleneck_attention_preserves_shape() -> None:
    model = VideoVAE(
        num_colors=8,
        base_channels=8,
        latent_channels=4,
        temporal_downsample=1,
        global_bottleneck_attn=True,
        global_bottleneck_attn_heads=4,
    )
    video = torch.randn(2, 8, 5, 32, 32)

    output = model(video, sample_posterior=False)

    assert output.logits.shape == video.shape
    assert output.posterior_mean.shape == (2, 4, 3, 1, 1)
    assert output.posterior_logvar.shape == (2, 4, 3, 1, 1)
    assert output.latents.shape == (2, 4, 3, 1, 1)


def test_ram_video_vae_preserves_ram_and_video_shapes() -> None:
    model = RAMVideoVAE(
        n_bytes=32,
        num_colors=8,
        frame_height=32,
        frame_width=32,
        hidden_dim=32,
        latent_dim=8,
        video_base_channels=8,
        video_latent_channels=4,
    )
    ram = torch.randint(0, 256, (2, 4, 32), dtype=torch.uint8)

    output = model(ram, sample_posterior=False)

    assert output.video_logits.shape == (2, 8, 4, 32, 32)
    assert output.ram_reconstruction.shape == (2, 4, 32)
    assert output.posterior_mean.shape == (2, 4, 8)
    assert output.posterior_logvar.shape == (2, 4, 8)
    assert output.latents.shape == (2, 4, 8)
    assert output.video_latents.shape == (2, 4, 4, 1, 1)


def test_ram_video_vae_temporal_downsample_preserves_output_shape() -> None:
    model = RAMVideoVAE(
        n_bytes=16,
        num_colors=8,
        frame_height=64,
        frame_width=96,
        hidden_dim=32,
        latent_dim=8,
        video_base_channels=8,
        video_latent_channels=4,
        temporal_downsample=1,
    )
    ram = torch.randint(0, 256, (2, 5, 16), dtype=torch.uint8)

    output = model(ram, sample_posterior=False)

    assert output.video_logits.shape == (2, 8, 5, 64, 96)
    assert output.ram_reconstruction.shape == (2, 5, 16)
    assert output.posterior_mean.shape == (2, 5, 8)
    assert output.posterior_logvar.shape == (2, 5, 8)
    assert output.latents.shape == (2, 5, 8)
    assert output.video_latents.shape == (2, 4, 3, 2, 3)


def test_ram_video_vae_v2_preserves_ram_and_video_shapes() -> None:
    model = RAMVideoVAEv2(
        n_bytes=32,
        num_colors=8,
        frame_height=32,
        frame_width=32,
        hidden_dim=32,
        latent_dim=8,
        video_base_channels=8,
        video_latent_channels=4,
        video_adapter_dim=32,
        video_adapter_heads=4,
    )
    ram = torch.randint(0, 256, (2, 4, 32), dtype=torch.uint8)

    output = model(ram, sample_posterior=False)

    assert output.video_logits.shape == (2, 8, 4, 32, 32)
    assert output.ram_reconstruction.shape == (2, 4, 32)
    assert output.posterior_mean.shape == (2, 4, 8)
    assert output.posterior_logvar.shape == (2, 4, 8)
    assert output.latents.shape == (2, 4, 8)
    assert output.video_latents.shape == (2, 4, 4, 1, 1)


def test_ram_video_vae_v2_temporal_downsample_preserves_output_shape() -> None:
    model = RAMVideoVAEv2(
        n_bytes=16,
        num_colors=8,
        frame_height=64,
        frame_width=96,
        hidden_dim=32,
        latent_dim=8,
        video_base_channels=8,
        video_latent_channels=4,
        temporal_downsample=1,
        video_adapter_dim=32,
        video_adapter_heads=4,
    )
    ram = torch.randint(0, 256, (2, 5, 16), dtype=torch.uint8)

    output = model(ram, sample_posterior=False)

    assert output.video_logits.shape == (2, 8, 5, 64, 96)
    assert output.ram_reconstruction.shape == (2, 5, 16)
    assert output.posterior_mean.shape == (2, 5, 8)
    assert output.posterior_logvar.shape == (2, 5, 8)
    assert output.latents.shape == (2, 5, 8)
    assert output.video_latents.shape == (2, 4, 3, 2, 3)


def test_audio_vae_preserves_mel_shape() -> None:
    model = AudioVAE(in_channels=1, n_mels=64, base_channels=8, latent_channels=4)
    mel = torch.rand(2, 1, 61, 64)

    output = model(mel, sample_posterior=False)

    assert output.reconstruction.shape == mel.shape
    assert output.posterior_mean.shape == (2, 4, 16, 16)
    assert output.posterior_logvar.shape == (2, 4, 16, 16)
    assert output.latents.shape == (2, 4, 16, 16)


def test_audio_vocoder_output_length_matches_expected_formula() -> None:
    model = AudioVocoder(
        in_channels=1,
        n_mels=64,
        out_channels=1,
        upsample_initial_channel=128,
        upsample_rates=(5, 5, 4),
        upsample_kernel_sizes=(11, 11, 8),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 9),),
        hop_length=100,
        n_fft=400,
    )
    mel = torch.rand(2, 1, 61, 64)

    waveform = model(mel)

    assert waveform.shape == (2, 1, model.expected_output_length(61))


def test_audio_vocoder_default_model_fits_project_size_budget() -> None:
    model = AudioVocoder()
    assert model.num_parameters <= 5_000_000


def test_video_discriminator_returns_one_logit_per_clip() -> None:
    discriminator = build_palette_discriminator(8, target_size="~5m")
    video = torch.rand(2, 8, 4, 32, 32)

    logits = discriminator(video)

    assert logits.shape == (2,)


def test_mel_discriminator_returns_one_logit_per_sample() -> None:
    discriminator = build_mel_discriminator(in_channels=1, target_size="~5m")
    mel = torch.rand(2, 1, 64, 64)

    logits = discriminator(mel)

    assert logits.shape == (2,)


# ---------- OneHotConv3d tests ----------


def test_onehot_conv3d_matches_causal_conv3d() -> None:
    """OneHotConv3d on palette indices must produce the same output as CausalConv3d on one-hot."""
    num_classes, out_channels = 8, 12
    causal = CausalConv3d(num_classes, out_channels, kernel_size=3)
    onehot = OneHotConv3d.from_causal_conv3d(causal)

    indices = torch.randint(0, num_classes, (2, 4, 32, 32))
    one_hot = torch.zeros(2, num_classes, 4, 32, 32)
    one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

    expected = causal(one_hot)
    actual = onehot(indices)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_onehot_conv3d_uint8_matches_int64() -> None:
    """OneHotConv3d produces identical output for uint8 and int64 indices."""
    num_classes, out_channels = 8, 12
    causal = CausalConv3d(num_classes, out_channels, kernel_size=3)
    onehot = OneHotConv3d.from_causal_conv3d(causal)

    indices_long = torch.randint(0, num_classes, (2, 4, 32, 32))
    indices_byte = indices_long.byte()

    out_long = onehot(indices_long)
    out_byte = onehot(indices_byte)

    torch.testing.assert_close(out_byte, out_long, atol=1e-5, rtol=1e-5)


def test_onehot_conv3d_different_kernel_sizes() -> None:
    """OneHotConv3d matches CausalConv3d for non-cubic kernels too."""
    num_classes, out_channels = 6, 10
    for ks in [(1, 1, 1), (3, 1, 1), (1, 3, 3)]:
        causal = CausalConv3d(num_classes, out_channels, kernel_size=ks)
        onehot = OneHotConv3d.from_causal_conv3d(causal)

        indices = torch.randint(0, num_classes, (1, 3, 16, 16))
        one_hot = torch.zeros(1, num_classes, 3, 16, 16)
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

        expected = causal(one_hot)
        actual = onehot(indices)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_video_vae_4d_indices_match_5d_onehot() -> None:
    """VideoVAE(onehot_conv=True) forward with 4D indices matches 5D one-hot."""
    num_colors = 8
    model = VideoVAE(num_colors=num_colors, base_channels=8, latent_channels=4, onehot_conv=True)
    model.eval()

    indices = torch.randint(0, num_colors, (2, 4, 32, 32))
    one_hot = torch.zeros(2, num_colors, 4, 32, 32)
    one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

    with torch.no_grad():
        out_idx = model(indices, sample_posterior=False)
        out_oh = model(one_hot, sample_posterior=False)

    torch.testing.assert_close(out_idx.logits, out_oh.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_idx.posterior_mean, out_oh.posterior_mean, atol=1e-5, rtol=1e-5)


def test_video_vae_loads_old_checkpoint_keys() -> None:
    """Checkpoints cross-load between onehot_conv=True and False."""
    num_colors = 8

    # CausalConv3d (default) -> OneHotConv3d
    default_model = VideoVAE(num_colors=num_colors, base_channels=8, latent_channels=4)
    default_state = default_model.state_dict()
    onehot_model = VideoVAE(num_colors=num_colors, base_channels=8, latent_channels=4, onehot_conv=True)
    onehot_model.load_state_dict(default_state, strict=True)

    # OneHotConv3d -> CausalConv3d (default)
    onehot_state = onehot_model.state_dict()
    fresh_default = VideoVAE(num_colors=num_colors, base_channels=8, latent_channels=4)
    fresh_default.load_state_dict(onehot_state, strict=True)