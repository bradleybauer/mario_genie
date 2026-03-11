"""Central registry of MAGVIT tokenizer architecture configurations.

Every named model variant lives here so that sweep scripts, profiling tools,
and training scripts can all reference the same definitions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    residual_blocks: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class AttentionVariant:
    name: str
    suffix: str
    enabled: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: str
    init_dim: int
    codebook_size: int
    layers: str
    scale_name: str
    attention_name: str


def build_open_genie_layers(
    init_dim: int,
    residual_blocks: tuple[int, int, int, int, int],
    *,
    use_attention: bool = False,
) -> str:
    """Approximate the Open-Genie MAGVIT hierarchy with MAGVIT-2 layer primitives.

    With IMAGE_SIZE=256 the four compress_space stages yield a 16×16 latent grid
    (256 / 2⁴ = 16).
    """
    stage2_dim = init_dim * 2
    stage3_dim = init_dim * 4
    stage4_dim = init_dim * 4
    blocks_1, blocks_2, blocks_3, blocks_4, blocks_5 = residual_blocks
    attention_layers = ["attend_space", "attend_time"] if use_attention else []

    layers = [
        f"consecutive_residual:{blocks_1}",
        f"compress_space:{init_dim}",
        f"consecutive_residual:{blocks_2}",
        f"compress_time:{stage2_dim}",
        f"compress_space:{stage2_dim}",
        *attention_layers,
        f"consecutive_residual:{blocks_3}",
        f"compress_time:{stage3_dim}",
        f"compress_space:{stage3_dim}",
        *attention_layers,
        f"consecutive_residual:{blocks_4}",
        f"compress_space:{stage4_dim}",
        f"consecutive_residual:{blocks_5}",
    ]
    return ",".join(layers)


def build_vanilla_layers(init_dim: int) -> str:
    """Simple residual + compress_space + compress_time architecture.

    Four spatial compressions (256 → 16) and two temporal compressions
    (16 → 4 latent frames), with a single residual block between each stage.
    """
    dim2 = init_dim * 2
    dim4 = init_dim * 4
    layers = [
        f"residual:{init_dim}",
        f"compress_space:{init_dim}",
        f"residual:{dim2}",
        f"compress_time:{dim2}",
        f"compress_space:{dim2}",
        f"residual:{dim4}",
        f"compress_time:{dim4}",
        f"compress_space:{dim4}",
        f"residual:{dim4}",
        f"compress_space:{dim4}",
    ]
    return ",".join(layers)


def build_open_genie_layers_deep(
    init_dim: int,
    residual_blocks: tuple[int, int, int, int, int],
    *,
    use_attention: bool = False,
) -> str:
    """Open-Genie layout with one extra compress_space and compress_time.

    5 compress_space stages (256 → 8) and 3 compress_time stages (16 → 2).
    """
    stage2_dim = init_dim * 2
    stage3_dim = init_dim * 4
    stage4_dim = init_dim * 4
    stage5_dim = init_dim * 4
    blocks_1, blocks_2, blocks_3, blocks_4, blocks_5 = residual_blocks
    attention_layers = ["attend_space", "attend_time"] if use_attention else []

    layers = [
        f"consecutive_residual:{blocks_1}",
        f"compress_space:{init_dim}",
        f"consecutive_residual:{blocks_2}",
        f"compress_time:{stage2_dim}",
        f"compress_space:{stage2_dim}",
        *attention_layers,
        f"consecutive_residual:{blocks_3}",
        f"compress_time:{stage3_dim}",
        f"compress_space:{stage3_dim}",
        *attention_layers,
        f"consecutive_residual:{blocks_4}",
        f"compress_time:{stage4_dim}",
        f"compress_space:{stage4_dim}",
        f"consecutive_residual:{blocks_5}",
        f"compress_space:{stage5_dim}",
    ]
    return ",".join(layers)


def build_vanilla_layers_deep(init_dim: int) -> str:
    """Vanilla layout with one extra compress_space and compress_time.

    5 spatial compressions (256 → 8) and 3 temporal compressions (16 → 2).
    """
    dim2 = init_dim * 2
    dim4 = init_dim * 4
    layers = [
        f"residual:{init_dim}",
        f"compress_space:{init_dim}",
        f"residual:{dim2}",
        f"compress_time:{dim2}",
        f"compress_space:{dim2}",
        f"residual:{dim4}",
        f"compress_time:{dim4}",
        f"compress_space:{dim4}",
        f"residual:{dim4}",
        f"compress_time:{dim4}",
        f"compress_space:{dim4}",
        f"residual:{dim4}",
        f"compress_space:{dim4}",
    ]
    return ",".join(layers)


# ── Scale presets ──────────────────────────────────────────────────

SCALE_CONFIGS = {
    "genie_small": ScaleConfig(name="genie_small", residual_blocks=(2, 4, 4, 4, 2)),
    "genie_base": ScaleConfig(name="genie_base", residual_blocks=(4, 4, 4, 8, 2)),
}

ATTENTION_VARIANTS = {
    "plain": AttentionVariant(name="plain", suffix="", enabled=False),
    "attn": AttentionVariant(name="attn", suffix="_attn", enabled=True),
}

DIMS = [32]
CODEBOOK_SIZES = [16384, 8192, 4096]

# ── Generated registry ────────────────────────────────────────────

MODEL_CONFIGS: list[ModelConfig] = [
    # Vanilla (residual + compress_space + compress_time)
    *(
        ModelConfig(
            name=f"dim{dim}_cb{cb}_vanilla",
            init_dim=dim,
            codebook_size=cb,
            layers=build_vanilla_layers(dim),
            scale_name="vanilla",
            attention_name="plain",
        )
        for dim in DIMS
        for cb in CODEBOOK_SIZES
    ),
    # Open-Genie variants
    *(
        ModelConfig(
            name=f"dim{dim}_cb{cb}_{scale.name}{variant.suffix}",
            init_dim=dim,
            codebook_size=cb,
            layers=build_open_genie_layers(
                dim,
                scale.residual_blocks,
                use_attention=variant.enabled,
            ),
            scale_name=scale.name,
            attention_name=variant.name,
        )
        for dim in DIMS
        for cb in CODEBOOK_SIZES
        for scale in SCALE_CONFIGS.values()
        for variant in ATTENTION_VARIANTS.values()
    ),
    # Deep vanilla (5× spatial, 3× temporal compression)
    *(
        ModelConfig(
            name=f"dim{dim}_cb{cb}_vanilla_deep",
            init_dim=dim,
            codebook_size=cb,
            layers=build_vanilla_layers_deep(dim),
            scale_name="vanilla_deep",
            attention_name="plain",
        )
        for dim in DIMS
        for cb in CODEBOOK_SIZES
    ),
    # Deep Open-Genie variants (5× spatial, 3× temporal compression)
    *(
        ModelConfig(
            name=f"dim{dim}_cb{cb}_{scale.name}_deep{variant.suffix}",
            init_dim=dim,
            codebook_size=cb,
            layers=build_open_genie_layers_deep(
                dim,
                scale.residual_blocks,
                use_attention=variant.enabled,
            ),
            scale_name=f"{scale.name}_deep",
            attention_name=variant.name,
        )
        for dim in DIMS
        for cb in CODEBOOK_SIZES
        for scale in SCALE_CONFIGS.values()
        for variant in ATTENTION_VARIANTS.values()
    ),
]

MODEL_CONFIGS_BY_NAME: dict[str, ModelConfig] = {m.name: m for m in MODEL_CONFIGS}


def get_model_config(name: str) -> ModelConfig:
    """Look up a model config by name, raising ``KeyError`` if not found."""
    try:
        return MODEL_CONFIGS_BY_NAME[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_CONFIGS_BY_NAME))
        raise KeyError(f"Unknown model config {name!r}. Available: {available}")
