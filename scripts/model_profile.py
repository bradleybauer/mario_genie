#!/usr/bin/env python3
"""Profile MAGVIT-2 tokenizer architecture, parameters, and GFLOPs.

Example:
    python scripts/model_profile.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from torchinfo import summary
from magvit2_pytorch import VideoTokenizer


# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.config import (
    CODEBOOK_SIZE,
    IMAGE_SIZE,
    SEQUENCE_LENGTH,
    TOKENIZER_LAYERS,
)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    seen = set()
    total = 0
    trainable = 0
    for param in model.parameters():
        key = id(param)
        if key in seen:
            continue
        seen.add(key)
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def count_named_parameters(model: torch.nn.Module) -> tuple[int, int]:
    seen = set()
    total = 0
    trainable = 0
    for _, param in model.named_parameters():
        key = id(param)
        if key in seen:
            continue
        seen.add(key)
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def count_parameters_in_modules(*modules: torch.nn.Module) -> tuple[int, int]:
    seen = set()
    total = 0
    trainable = 0
    for module in modules:
        for param in module.parameters():
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n
    return total, trainable


def estimate_forward_gflops(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: str,
    warmup_steps: int,
    profile_steps: int,
) -> float:
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(sample, return_loss=True)

        if device == "cuda":
            torch.cuda.synchronize()

        with profile(activities=activities, with_flops=True) as prof:
            for _ in range(profile_steps):
                _ = model(sample, return_loss=True)

        if device == "cuda":
            torch.cuda.synchronize()

    total_flops = 0
    for event in prof.key_averages():
        total_flops += getattr(event, "flops", 0) or 0

    flops_per_forward = total_flops / max(profile_steps, 1)
    return flops_per_forward / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    parser.add_argument("--init-dim", type=int, default=32)
    parser.add_argument("--summary-depth", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available.")
    else:
        device = args.device

    layers = tuple(TOKENIZER_LAYERS)

    model = VideoTokenizer(
        image_size=args.image_size,
        init_dim=args.init_dim,
        codebook_size=args.codebook_size,
        layers=layers,
        use_gan=False,
        perceptual_loss_weight=0.0,
    ).to(device)
    model.eval()

    sample = torch.randn(
        args.batch_size,
        3,
        args.sequence_length,
        args.image_size,
        args.image_size,
        device=device,
    )

    optimizer_total_params, optimizer_trainable_params = count_parameters(model)
    registered_total_params, registered_trainable_params = count_named_parameters(model)
    encoder_total, encoder_trainable = count_parameters_in_modules(model.conv_in, model.encoder_layers)
    decoder_total, decoder_trainable = count_parameters_in_modules(model.decoder_layers, model.conv_out)
    quantizer_total, quantizer_trainable = count_parameters_in_modules(model.quantizers)
    discr_total, discr_trainable = count_parameters_in_modules(model.discr)

    print("=== Effective Configuration ===")
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"image_size: {args.image_size}")
    print(f"sequence_length: {args.sequence_length}")
    print(f"codebook_size: {args.codebook_size}")
    print(f"init_dim: {args.init_dim}")
    print(f"layers: {layers}")
    print()

    print("=== Parameter Counts ===")
    print(f"tokenizer_parameters (model.parameters): {optimizer_total_params:,}")
    print(f"tokenizer_trainable_parameters: {optimizer_trainable_params:,}")
    print(f"all_registered_parameters (named_parameters): {registered_total_params:,}")
    print(f"all_registered_trainable_parameters: {registered_trainable_params:,}")
    print(f"discriminator_parameters: {discr_total:,} (trainable: {discr_trainable:,})")
    print(f"encoder_parameters: {encoder_total:,} (trainable: {encoder_trainable:,})")
    print(f"decoder_parameters: {decoder_total:,} (trainable: {decoder_trainable:,})")
    print(f"quantizer_parameters: {quantizer_total:,} (trainable: {quantizer_trainable:,})")
    print()

    print("=== Architecture Summary (torchinfo) ===")
    try:
        summary(
            model,
            input_data=[sample],
            depth=args.summary_depth,
            col_names=("input_size", "output_size", "num_params", "kernel_size"),
            verbose=1,
            return_loss=True,
        )
    except Exception as exc:
        print(f"[warning] torchinfo summary failed: {exc}")
    print("[note] torchinfo aligns with all_registered_parameters; training uses model.parameters().")
    print()

    print("=== FLOPs Profile ===")
    gflops = estimate_forward_gflops(
        model=model,
        sample=sample,
        device=device,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
    )
    print(f"forward_GFLOPs: {gflops:.4f}")

    results = {
        "device": device,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "sequence_length": args.sequence_length,
        "codebook_size": args.codebook_size,
        "init_dim": args.init_dim,
        "layers": list(layers),
        "tokenizer_parameters": optimizer_total_params,
        "tokenizer_trainable_parameters": optimizer_trainable_params,
        "all_registered_parameters": registered_total_params,
        "all_registered_trainable_parameters": registered_trainable_params,
        "discriminator_parameters": discr_total,
        "discriminator_trainable_parameters": discr_trainable,
        "encoder_parameters": encoder_total,
        "encoder_trainable_parameters": encoder_trainable,
        "decoder_parameters": decoder_total,
        "decoder_trainable_parameters": decoder_trainable,
        "quantizer_parameters": quantizer_total,
        "quantizer_trainable_parameters": quantizer_trainable,
        "forward_gflops": gflops,
        "warmup_steps": args.warmup_steps,
        "profile_steps": args.profile_steps,
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to: {out_path}")


if __name__ == "__main__":
    main()
