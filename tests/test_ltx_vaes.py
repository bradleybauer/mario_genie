from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.ltx_audio_vae import LTXAudioVAE
from mario_world_model.ltx_video_vae import LTXVideoVAE


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