from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.gan_discriminator import build_mel_discriminator, build_palette_discriminator
from models.ltx_audio_vae import LTXAudioVAE
from models.ltx_audio_vocoder import LTXAudioVocoder
from models.ltx_video_vae import LTXVideoVAE


def test_video_vae_preserves_video_shape() -> None:
    model = LTXVideoVAE(num_colors=8, patch_size=4, base_channels=8, latent_channels=4)
    video = torch.randn(2, 8, 4, 32, 32)

    output = model(video, sample_posterior=False)

    assert output.logits.shape == video.shape
    assert output.posterior_mean.shape == (2, 4, 4, 2, 2)
    assert output.posterior_logvar.shape == (2, 4, 4, 2, 2)
    assert output.latents.shape == (2, 4, 4, 2, 2)


def test_audio_vae_preserves_mel_shape() -> None:
    model = LTXAudioVAE(in_channels=1, n_mels=64, base_channels=8, latent_channels=4)
    mel = torch.rand(2, 1, 61, 64)

    output = model(mel, sample_posterior=False)

    assert output.reconstruction.shape == mel.shape
    assert output.posterior_mean.shape == (2, 4, 16, 16)
    assert output.posterior_logvar.shape == (2, 4, 16, 16)
    assert output.latents.shape == (2, 4, 16, 16)


def test_audio_vocoder_output_length_matches_expected_formula() -> None:
    model = LTXAudioVocoder(
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
    model = LTXAudioVocoder()
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