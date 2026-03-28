#!/usr/bin/env python3
"""Rank all dataset samples by reconstruction loss (best → worst).

Loads a trained checkpoint, runs inference on every sample in the dataset,
and writes a JSON report + optional worst/best reconstruction PNGs.

Usage:
    python scripts/rank_samples.py checkpoints/magvit2/training_state_latest.pt
    python scripts/rank_samples.py checkpoints/magvit2/training_state_latest.pt --data-dir data/ --top-k 20
    python scripts/rank_samples.py checkpoints/magvit2/training_state_latest.pt --save-images 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.config import IMAGE_SIZE, SEQUENCE_LENGTH
from mario_world_model.model_configs import MODEL_CONFIGS_BY_NAME
from mario_world_model.tokenizer_compat import resolve_video_contains_first_frame
from mario_world_model.palette_tokenizer import PaletteVideoTokenizer

# Re-use dataset and crop constants from the training script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_magvit import MarioVideoDataset, OVERSCAN_CROP_SIZE, CROP_224_SIZE, _crop_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank dataset samples by reconstruction loss")
    parser.add_argument("checkpoint", type=str,
                        help="Path to training_state*.pt or magvit2*.pt file")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: read from config.json next to checkpoint)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: sample_rankings.json next to checkpoint)")
    parser.add_argument("--save-images", type=int, default=10,
                        help="Save reconstruction PNGs for the N best and N worst samples (0 to disable)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Print the top-K best and worst samples to stdout")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-fraction", type=float, default=1.0,
                        help="Evaluate a random fraction of the dataset (e.g. 0.1 for 10%%)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    # ── Load config ──────────────────────────────────────────────────
    config_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(config_path):
        raise SystemExit(f"No config.json found at {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model"]
    mc = MODEL_CONFIGS_BY_NAME[model_name]
    image_size = config.get("image_size", IMAGE_SIZE)
    seq_len = config.get("sequence_length", SEQUENCE_LENGTH)

    crop_size = None
    if config.get("crop_240"):
        crop_size = OVERSCAN_CROP_SIZE
    elif config.get("crop_224"):
        crop_size = CROP_224_SIZE

    data_dir = args.data_dir or config.get("data_dir", "data")

    # ── Load weights ─────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # training_state has a 'model' key; raw .pt is the state_dict directly
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    step = ckpt.get("global_step", "?") if isinstance(ckpt, dict) else "?"
    del ckpt

    # Infer palette size from checkpoint (conv_out.conv.bias has num_palette_colors elements)
    num_palette_colors = state_dict["conv_out.conv.bias"].shape[0]

    # ── Load palette ─────────────────────────────────────────────────
    palette_path = os.path.join(data_dir, "palette.json")
    if not os.path.isfile(palette_path):
        raise FileNotFoundError(f"No palette.json found at {palette_path}")
    with open(palette_path) as f:
        palette_rgb = json.load(f)
    palette = torch.tensor(palette_rgb[:num_palette_colors], dtype=torch.float32).to(device) / 255.0

    # ── Build tokenizer layers ───────────────────────────────────────
    tokenizer_layers = tuple(
        (name, int(val)) if ":" in tok else tok
        for tok in mc.layers.split(",")
        for name, _, val in [tok.partition(":")]
    )

    # ── Create model ─────────────────────────────────────────────────
    tokenizer = PaletteVideoTokenizer(
        num_palette_colors=num_palette_colors,
        image_size=image_size,
        init_dim=mc.init_dim,
        codebook_size=mc.codebook_size,
        layers=tokenizer_layers,
    ).to(device)
    tokenizer.discr = None
    tokenizer.multiscale_discrs = None

    tokenizer.load_state_dict(state_dict)
    print(f"Loaded checkpoint from step {step} (palette size: {num_palette_colors})")
    del state_dict

    video_contains_first_frame = resolve_video_contains_first_frame(tokenizer, seq_len)

    # ── Load dataset ─────────────────────────────────────────────────
    dataset = MarioVideoDataset(
        data_dir,
        seq_len=seq_len,
        crop_size=crop_size,
        num_workers=args.num_workers,
    )
    print(f"Dataset: {len(dataset)} samples")

    if args.sample_fraction < 1.0:
        n_total = len(dataset)
        n_subset = max(1, int(n_total * args.sample_fraction))
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(n_total, generator=generator)[:n_subset].tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        print(f"Sampling {n_subset}/{n_total} ({args.sample_fraction:.0%})")
    else:
        subset = dataset
        indices = None

    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ── Run inference on all samples ─────────────────────────────────
    tokenizer.eval()
    all_results = []
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device, non_blocking=True)
            bs = batch.shape[0]
            targets = batch.long().clamp(max=num_palette_colors - 1)
            inp = PaletteVideoTokenizer.indices_to_onehot(targets, num_palette_colors)

            codes = tokenizer(inp, return_codes=True,
                              video_contains_first_frame=video_contains_first_frame)
            recon_video = tokenizer.decode_from_code_indices(
                codes, video_contains_first_frame=video_contains_first_frame,
            )

            # Per-sample cross-entropy loss
            for i in range(bs):
                sample_loss = F.cross_entropy(
                    recon_video[i:i+1], targets[i:i+1]
                ).item()
                recon_idx = recon_video[i].argmax(dim=0)  # (T, H, W)
                correct = (recon_idx == targets[i]).sum().item()
                total = targets[i].numel()
                pixel_acc = correct / total

                if indices is not None:
                    global_idx = indices[sample_offset + i]
                else:
                    global_idx = sample_offset + i
                file_idx, t_start = dataset.samples[global_idx]
                file_path = dataset.data_files[file_idx]

                all_results.append({
                    "sample_idx": global_idx,
                    "loss": round(sample_loss, 6),
                    "pixel_accuracy": round(pixel_acc, 6),
                    "file": file_path,
                    "t_start": t_start,
                })

            sample_offset += bs

    # ── Sort by loss ─────────────────────────────────────────────────
    all_results.sort(key=lambda r: r["loss"])

    losses = [r["loss"] for r in all_results]
    accs = [r["pixel_accuracy"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"Total samples evaluated: {len(all_results)}")
    print(f"Loss  — mean: {np.mean(losses):.6f}  median: {np.median(losses):.6f}  "
          f"std: {np.std(losses):.6f}")
    print(f"Pixel acc — mean: {np.mean(accs):.4f}  median: {np.median(accs):.4f}")
    print(f"{'='*60}")

    k = min(args.top_k, len(all_results))
    print(f"\n--- TOP {k} BEST (lowest loss) ---")
    for r in all_results[:k]:
        print(f"  loss={r['loss']:.6f}  acc={r['pixel_accuracy']:.4f}  "
              f"file={os.path.basename(r['file'])}  t={r['t_start']}")

    print(f"\n--- TOP {k} WORST (highest loss) ---")
    for r in all_results[-k:][::-1]:
        print(f"  loss={r['loss']:.6f}  acc={r['pixel_accuracy']:.4f}  "
              f"file={os.path.basename(r['file'])}  t={r['t_start']}")

    # ── Save JSON ────────────────────────────────────────────────────
    output_path = args.output or os.path.join(ckpt_dir, "sample_rankings.json")
    with open(output_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "step": step,
            "num_samples": len(all_results),
            "mean_loss": round(float(np.mean(losses)), 6),
            "median_loss": round(float(np.median(losses)), 6),
            "mean_pixel_accuracy": round(float(np.mean(accs)), 6),
            "samples": all_results,
        }, f, indent=2)
    print(f"\nFull rankings saved to {output_path}")

    # ── Save reconstruction images for best/worst ────────────────────
    if args.save_images > 0:
        img_dir = os.path.join(ckpt_dir, "sample_analysis")
        os.makedirs(img_dir, exist_ok=True)
        n_save = min(args.save_images, len(all_results))

        file_idx_lookup = {f: i for i, f in enumerate(dataset.data_files)}

        def save_sample_reconstruction(sample_info, label):
            fi = file_idx_lookup[sample_info["file"]]
            t = sample_info["t_start"]

            if dataset.frames_by_file is not None:
                frames = dataset.frames_by_file[fi]
            else:
                frames = dataset._get_frames_mmap(fi)

            chunk = frames[t:t + seq_len, 0]  # (T, H, W) uint8
            if crop_size is not None:
                chunk = _crop_chunk(chunk, crop_size)

            sample = torch.from_numpy(chunk.copy()).unsqueeze(0).to(device)  # (1, T, H, W)
            targets = sample.long().clamp(max=num_palette_colors - 1)
            inp = PaletteVideoTokenizer.indices_to_onehot(targets, num_palette_colors)

            with torch.no_grad():
                codes = tokenizer(inp, return_codes=True,
                                  video_contains_first_frame=video_contains_first_frame)
                recon_video = tokenizer.decode_from_code_indices(
                    codes, video_contains_first_frame=video_contains_first_frame,
                )

            original_rgb = palette[targets[0]]           # (T, H, W, 3)
            original_frames = original_rgb.permute(0, 3, 1, 2)  # (T, 3, H, W)
            recon_idx = recon_video[0].argmax(dim=0)     # (T, H, W)
            recon_rgb = palette[recon_idx]
            recon_frames = recon_rgb.permute(0, 3, 1, 2)
            comparison = torch.cat([original_frames, recon_frames], dim=3)

            fname = (f"{label}_loss{sample_info['loss']:.4f}"
                     f"_acc{sample_info['pixel_accuracy']:.4f}"
                     f"_t{sample_info['t_start']}.png")
            save_image(comparison, os.path.join(img_dir, fname), nrow=1, padding=0)

        print(f"\nSaving {n_save} best and {n_save} worst reconstructions to {img_dir}/")
        for i, r in enumerate(tqdm(all_results[:n_save], desc="Saving best")):
            save_sample_reconstruction(r, f"best_{i:03d}")
        for i, r in enumerate(tqdm(all_results[-n_save:][::-1], desc="Saving worst")):
            save_sample_reconstruction(r, f"worst_{i:03d}")

        print(f"Images saved to {img_dir}/")


if __name__ == "__main__":
    main()
