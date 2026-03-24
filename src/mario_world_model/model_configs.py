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
    num_codebooks: int = 1
    sequence_length: int = 16
    context_frames: int = 8
    model_type: str = "magvit2"  # "magvit2" or "genie2"


def build_open_genie_layers(
    init_dim: int,
    residual_blocks: tuple[int, int, int, int, int],
    *,
    use_attention: bool = False,
    temporal_compressions: int = 2,
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
        *([] if temporal_compressions < 2 else [f"compress_time:{stage2_dim}"]),
        f"compress_space:{stage2_dim}",
        *attention_layers,
        f"consecutive_residual:{blocks_3}",
        *([] if temporal_compressions < 1 else [f"compress_time:{stage3_dim}"]),
        f"compress_space:{stage3_dim}",
        *attention_layers,
        f"consecutive_residual:{blocks_4}",
        f"compress_space:{stage4_dim}",
        f"consecutive_residual:{blocks_5}",
    ]
    return ",".join(layers)


def build_vanilla_layers(init_dim: int, *, temporal_compressions: int = 2) -> str:
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
        *([] if temporal_compressions < 1 else [f"compress_time:{dim2}"]),
        f"compress_space:{dim2}",
        f"residual:{dim4}",
        *([] if temporal_compressions < 2 else [f"compress_time:{dim4}"]),
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

# Controlling model width.
DIMS = [32, 16]

# (codebook_size, num_codebooks) — name uses "cb{size}" for 1 codebook,
# "cb{size}x{n}" for multiple.
CODEBOOK_VARIANTS = [
    (16384, 1),
    (256, 4),
    (512, 2),
    (1024, 4),
]

# ── Generated registry ────────────────────────────────────────────

# (suffix, temporal_compressions)
TEMPORAL_VARIANTS = [("_1t", 1), ("_0t", 0)]
DEFAULT_SEQ_LEN = 16
HALF_SEQ_LEN = 8

MODEL_CONFIGS: list[ModelConfig] = [
    # Vanilla (residual + compress_space + compress_time)
    *(
        ModelConfig(
            name=f"dim{dim}_cb{cb}_ncb{ncb}_vanilla{tsuf}{seq_suffix}",
            init_dim=dim,
            codebook_size=cb,
            layers=build_vanilla_layers(dim, temporal_compressions=tc),
            scale_name="vanilla",
            attention_name="plain",
            num_codebooks=ncb,
            sequence_length=seq_len,
        )
        for dim in DIMS
        for cb, ncb in CODEBOOK_VARIANTS
        for tsuf, tc in TEMPORAL_VARIANTS
        for seq_len, seq_suffix in [
            (DEFAULT_SEQ_LEN, ""),
            (HALF_SEQ_LEN, f"_s{HALF_SEQ_LEN}"),
        ]
    ),
    # Open-Genie variants
    *(
        ModelConfig(
            name=f"dim{dim}_cb{cb}_ncb{ncb}_{scale.name}{variant.suffix}{tsuf}{seq_suffix}",
            init_dim=dim,
            codebook_size=cb,
            layers=build_open_genie_layers(
                dim,
                scale.residual_blocks,
                use_attention=variant.enabled,
                temporal_compressions=tc,
            ),
            scale_name=scale.name,
            attention_name=variant.name,
            num_codebooks=ncb,
            sequence_length=seq_len,
        )
        for dim in DIMS
        for cb, ncb in CODEBOOK_VARIANTS
        for scale in SCALE_CONFIGS.values()
        for variant in ATTENTION_VARIANTS.values()
        for tsuf, tc in TEMPORAL_VARIANTS
        for seq_len, seq_suffix in [
            (DEFAULT_SEQ_LEN, ""),
            (HALF_SEQ_LEN, f"_s{HALF_SEQ_LEN}"),
        ]
    ),
]

MODEL_CONFIGS_BY_NAME: dict[str, ModelConfig] = {m.name: m for m in MODEL_CONFIGS}


# ── Genie 2 VAE config ────────────────────────────────────────────

GENIE2_CONFIG = ModelConfig(
    name="genie2_vae",
    init_dim=32,
    codebook_size=0,     # continuous latents, no codebook
    layers="",           # not used — FrameVAE has its own architecture
    scale_name="genie2",
    attention_name="plain",
    num_codebooks=0,
    sequence_length=1,   # frame-independent
    model_type="genie2",
)

MODEL_CONFIGS.append(GENIE2_CONFIG)
MODEL_CONFIGS_BY_NAME[GENIE2_CONFIG.name] = GENIE2_CONFIG


if __name__ == "__main__":
    import sys, os, math

    magvit_configs = [c for c in MODEL_CONFIGS if c.model_type == "magvit2"]
    profile = "--profile" in sys.argv

    IMAGE_SIZE = 224
    NUM_PALETTE_COLORS = 23

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from magvit2_pytorch import VideoTokenizer

    if profile:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
        from tqdm import tqdm

    rows = []
    iterator = tqdm(magvit_configs, desc="Profiling") if profile else magvit_configs
    for mc in iterator:
        layer_tokens = [t.strip() for t in mc.layers.split(",") if t.strip()]
        layers = tuple(
            (name, int(val)) if ":" in tok else tok
            for tok in layer_tokens
            for name, _, val in [tok.partition(":")]
        )
        model = VideoTokenizer(
            image_size=IMAGE_SIZE,
            init_dim=mc.init_dim,
            channels=NUM_PALETTE_COLORS,
            codebook_size=mc.codebook_size,
            layers=layers,
            use_gan=False,
            perceptual_loss_weight=0.0,
        )
        n_params = sum(p.numel() for p in model.parameters())

        flops = None
        if profile:
            from mario_world_model.tokenizer_compat import resolve_video_contains_first_frame
            model.eval()
            total_frames = mc.sequence_length + mc.context_frames
            x = torch.randn(1, NUM_PALETTE_COLORS, total_frames, IMAGE_SIZE, IMAGE_SIZE)
            vcff = resolve_video_contains_first_frame(model, total_frames)
            flop_counter = FlopCounterMode(display=False)
            with torch.no_grad(), flop_counter:
                model(x, return_recon_loss_only=True, video_contains_first_frame=vcff)
            flops = flop_counter.get_total_flops()

        n_spatial = sum(1 for t in layer_tokens if t.startswith("compress_space"))
        n_temporal = sum(1 for t in layer_tokens if t.startswith("compress_time"))
        spatial_positions = (IMAGE_SIZE // (2 ** n_spatial)) ** 2
        latent_frames = mc.sequence_length / (2 ** n_temporal)
        codes_per_frame = int(spatial_positions * mc.num_codebooks * latent_frames / mc.sequence_length)
        bits_per_frame = codes_per_frame * math.log2(mc.codebook_size)
        total_bits = bits_per_frame * mc.sequence_length
        vocab_size = mc.num_codebooks * mc.codebook_size

        rows.append((mc.name, n_params, flops, bits_per_frame, total_bits, vocab_size, mc.sequence_length, mc.context_frames))

    rows.sort(key=lambda r: (r[1] or 0, r[0]))

    def _fmt_flops(f: float) -> str:
        if f >= 1e12:
            return f"{f / 1e12:.1f}T"
        if f >= 1e9:
            return f"{f / 1e9:.1f}G"
        return f"{f / 1e6:.0f}M"

    print(f"\n{len(magvit_configs)} magvit2 models, {len(MODEL_CONFIGS)} total\n")
    if profile:
        print(f"{'Name':<55} {'Params':>12}  {'FLOPs':>7}  {'Bits/Img':>8}  {'TotalBits':>9}  {'Vocab':>9}  {'Imgs':>4}  {'Ctx':>3}")
        print("-" * 118)
        for name, n_params, flops, bits, total_bits, vocab, seq_len, ctx in rows:
            print(f"{name:<55} {n_params:>12,}  {_fmt_flops(flops):>7}  {bits:>8.0f}  {total_bits:>9.0f}  {vocab:>9,}  {seq_len:>4}  {ctx:>3}")
    else:
        print(f"{'Name':<55} {'Params':>12}  {'Bits/Img':>8}  {'TotalBits':>9}  {'Vocab':>9}  {'Imgs':>4}  {'Ctx':>3}")
        print("-" * 110)
        for name, n_params, _, bits, total_bits, vocab, seq_len, ctx in rows:
            print(f"{name:<55} {n_params:>12,}  {bits:>8.0f}  {total_bits:>9.0f}  {vocab:>9,}  {seq_len:>4}  {ctx:>3}")
