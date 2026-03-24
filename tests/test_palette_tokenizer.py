from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.palette_tokenizer import PaletteVideoTokenizer


def _build_tokenizer(
    *,
    pad_mode: str = "replicate",
    layers=("residual",),
    dim_cond: int | None = None,
) -> PaletteVideoTokenizer:
    return PaletteVideoTokenizer(
        num_palette_colors=4,
        image_size=16,
        init_dim=8,
        codebook_size=16,
        layers=layers,
        dim_cond=dim_cond,
        pad_mode=pad_mode,
        use_gan=True,
        perceptual_loss_weight=1.0,
    )


def test_palette_tokenizer_defaults_to_replicate_padding() -> None:
    tokenizer = _build_tokenizer()

    assert tokenizer.conv_in.pad_mode == "replicate"
    assert tokenizer.conv_out.pad_mode == "replicate"
    assert tokenizer.encoder_layers[0].fn[0].pad_mode == "replicate"
    assert tokenizer.decoder_layers[0].fn[0].pad_mode == "replicate"


def test_palette_tokenizer_propagates_explicit_pad_mode_override() -> None:
    tokenizer = _build_tokenizer(pad_mode="constant")

    assert tokenizer.conv_in.pad_mode == "constant"
    assert tokenizer.conv_out.pad_mode == "constant"
    assert tokenizer.encoder_layers[0].fn[0].pad_mode == "constant"
    assert tokenizer.decoder_layers[0].fn[0].pad_mode == "constant"


def test_palette_tokenizer_patches_conditioned_residual_blocks() -> None:
    tokenizer = _build_tokenizer(
        layers=("cond_residual",),
        dim_cond=4,
        pad_mode="replicate",
    )

    assert tokenizer.encoder_layers[0].conv.pad_mode == "replicate"
    assert tokenizer.decoder_layers[0].conv.pad_mode == "replicate"