from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.gan_discriminator import build_mel_discriminator, build_palette_discriminator
from src.models.audio_vae import AudioVAE
from src.models.audio_vocoder import AudioVocoder
from src.models.ram_vae import RAMVAE
from src.models.ram_video_vae import RAMVideoVAE
from src.models.ram_video_vae_v2 import RAMVideoVAEv2
from src.models.video_vae import VideoVAE


def _encode_stream_in_chunks(
    model: VideoVAE,
    video: torch.Tensor,
    chunk_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    assert sum(chunk_sizes) == video.shape[2]

    state = model.init_encode_stream_state()
    mean_chunks: list[torch.Tensor] = []
    logvar_chunks: list[torch.Tensor] = []
    start = 0
    for index, chunk_size in enumerate(chunk_sizes):
        end = start + chunk_size
        mean_chunk, logvar_chunk, state = model.encode_stream(
            video[:, :, start:end],
            state,
            final=index == len(chunk_sizes) - 1,
        )
        mean_chunks.append(mean_chunk)
        logvar_chunks.append(logvar_chunk)
        start = end

    return torch.cat(mean_chunks, dim=2), torch.cat(logvar_chunks, dim=2)


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


@pytest.mark.parametrize(
    ("temporal_downsample", "frames", "chunk_sizes"),
    [
        (0, 6, (1, 2, 3)),
        (0, 6, (2, 1, 1, 2)),
        (1, 7, (1, 2, 1, 3)),
        (1, 7, (1, 1, 1, 1, 1, 1, 1)),
    ],
)
def test_video_vae_encode_stream_matches_full_encode(
    temporal_downsample: int,
    frames: int,
    chunk_sizes: tuple[int, ...],
) -> None:
    model = VideoVAE(
        num_colors=8,
        base_channels=8,
        latent_channels=4,
        temporal_downsample=temporal_downsample,
        dropout=0.0,
    )
    model.eval()
    video = torch.randn(2, 8, frames, 32, 32)

    full_mean, full_logvar = model.encode(video)
    stream_mean, stream_logvar = _encode_stream_in_chunks(model, video, chunk_sizes)

    torch.testing.assert_close(stream_mean, full_mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(stream_logvar, full_logvar, rtol=1e-5, atol=1e-6)


def test_video_vae_encode_stream_final_flush_emits_trailing_paired_latent() -> None:
    model = VideoVAE(
        num_colors=8,
        base_channels=8,
        latent_channels=4,
        temporal_downsample=1,
        dropout=0.0,
    )
    model.eval()
    video = torch.randn(2, 8, 5, 32, 32)

    state = model.init_encode_stream_state()
    mean_a, logvar_a, state = model.encode_stream(video[:, :, :4], state, final=False)
    mean_b, logvar_b, _ = model.encode_stream(video[:, :, 4:], state, final=True)
    full_mean, full_logvar = model.encode(video)

    assert mean_a.shape[2] == 2
    assert logvar_a.shape[2] == 2
    assert mean_b.shape[2] == 1
    assert logvar_b.shape[2] == 1
    torch.testing.assert_close(torch.cat((mean_a, mean_b), dim=2), full_mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.cat((logvar_a, logvar_b), dim=2), full_logvar, rtol=1e-5, atol=1e-6)


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


def test_ram_vae_expected_values_preserve_ram_shape() -> None:
    model = RAMVAE(
        values_per_address=[list(range(4)), [0, 64, 128, 255], [3, 9]],
        embed_dim=4,
        hidden_dim=16,
        latent_dim=8,
        dropout=0.0,
    )
    ram = torch.tensor(
        [
            [[0, 64, 3], [1, 128, 9], [2, 255, 3]],
            [[3, 0, 9], [2, 64, 3], [1, 128, 9]],
        ],
        dtype=torch.long,
    )

    output = model(ram, sample_posterior=False)
    expected = model.logits_to_expected_values(output.logits)

    assert output.logits.shape == (2, 3, 10)
    assert output.reconstruction.shape == ram.shape
    assert expected.shape == ram.shape
    assert expected.dtype == output.logits.dtype


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


