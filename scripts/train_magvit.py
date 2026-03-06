import argparse
import concurrent.futures
import glob
import json
import math
import os
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from magvit2_pytorch import VideoTokenizer
from tqdm import tqdm

from mario_world_model.config import IMAGE_SIZE, CODEBOOK_SIZE, TOKENIZER_LAYERS, SEQUENCE_LENGTH
from mario_world_model.auto_batch_sizer import find_max_batch_size

def _index_chunk(chunk_idx, filepath, seq_len):
    """Index a single chunk file. Returns list of (chunk_idx, seq_idx, t_start) tuples."""
    try:
        meta_path = filepath.replace(".npz", ".meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as mf:
                meta = json.load(mf)
            num_seqs = meta["num_sequences"]
            total_t = meta["sequence_length"]
        else:
            npz = np.load(filepath, mmap_mode='r')
            num_seqs, total_t = npz['frames'].shape[0], npz['frames'].shape[1]

        if total_t < seq_len:
            return []

        npz = np.load(filepath, mmap_mode='r')
        dones = np.array(npz['dones'])

        samples = []
        for i in range(num_seqs):
            for t in range(0, total_t - seq_len + 1, seq_len):
                window_dones = dones[i, t:t+seq_len]
                if np.any(window_dones[:-1]):
                    continue
                samples.append((chunk_idx, i, t))
        return samples
    except Exception as e:
        print(f"Skipping {filepath} due to error: {e}")
        return []

class MarioVideoDataset(Dataset):
    def __init__(self, data_dir, seq_len=4):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.seq_len = seq_len
        self.samples = []
        
        print(f"Indexing {len(self.chunk_files)} chunks (lazy loading)...")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(_index_chunk, idx, f, seq_len): idx
                for idx, f in enumerate(self.chunk_files)
            }
            for future in concurrent.futures.as_completed(futures):
                self.samples.extend(future.result())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_idx, seq_idx, t_start = self.samples[idx]
        
        # Load only the slice we need via memory-mapped access
        npz = np.load(self.chunk_files[chunk_idx], mmap_mode='r')
        frames = np.array(npz['frames'][seq_idx, t_start:t_start+self.seq_len])
        
        # Convert to tensor: [T, C, H, W] -> [C, T, H, W] required by VideoTokenizer
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(1, 0, 2, 3) 
        
        return frames

class NSampleDataset(Dataset):
    """Wraps a dataset to uniformly sample from N fixed indices. Useful for overfit sanity checks."""
    def __init__(self, dataset, indices, length=1000):
        self.dataset = dataset
        self.indices = list(indices)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx % len(self.indices)]]

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to chunk files")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--auto-batch-size", action="store_true",
        help="Probe GPU to find the largest batch size that fits in VRAM. "
             "Overrides --batch-size.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints/magvit2")
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--overfit-n", type=int, default=0, help="Train on N random samples (overfit sanity check)")
    parser.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    parser.add_argument("--init-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of the dataset")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of DataLoader workers")
    parser.add_argument("--run-name", type=str, default=None, help="Subfolder under output-dir for this run")
    parser.add_argument(
        "--layers", type=str, default=None,
        help=(
            "Comma-separated encoder layer spec, overrides config.TOKENIZER_LAYERS. "
            "e.g. 'residual,compress_space,residual,compress_space,residual,compress_space'. "
            "Use 'consecutive_residual:N' for parameterised layers."
        ),
    )
    parser.add_argument(
        "--max-patience", type=int, default=None,
        help=(
            "Early stopping: halt training after this many consecutive "
            "validation checks with no recon_loss improvement (>=1e-6). "
            "Disabled by default (train for full --epochs)."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help=(
            "Stop training once eval recon_loss drops below this value "
            "(0 = disabled). Checkpoint is saved before exit."
        ),
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help=(
            "Stop training after this many gradient steps (0 = no limit, "
            "use --epochs only). The first epoch always completes fully; "
            "the step budget is checked between epochs. "
            "Also used as the total duration of the cosine-annealing LR schedule."
        ),
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=50,
        help=(
            "Number of linear warmup steps at the start of training. "
            "LR ramps from 0 to 2x --lr over this many steps, then cosine-"
            "anneals to 1e-6 over the remaining --max-steps minus warmup. "
            "Requires --max-steps > 0. (default=100; 0 = no warmup)"
        ),
    )
    args = parser.parse_args()

    # Parse --layers into a tuple
    if args.layers is not None:
        parsed = []
        for tok in args.layers.split(","):
            tok = tok.strip()
            if ":" in tok:
                name, val = tok.split(":", 1)
                parsed.append((name, int(val)))
            else:
                parsed.append(tok)
        tokenizer_layers = tuple(parsed)
    else:
        tokenizer_layers = TOKENIZER_LAYERS

    # ── Normal training ──────────────────────────────────────────────────

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # Run directory
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MarioVideoDataset(args.data_dir, seq_len=SEQUENCE_LENGTH)
    print(f"Found {len(dataset)} sequence segments of length {SEQUENCE_LENGTH} frames.")

    num_unique_samples = len(dataset)
    if args.overfit_n > 0:
        indices = np.random.choice(len(dataset), size=args.overfit_n, replace=False)
        num_unique_samples = len(indices)
        steps_per_epoch = max(1000, len(indices) * 100, args.batch_size * 100)
        dataset = NSampleDataset(dataset, indices=indices, length=steps_per_epoch)
        print(f">> Overfit mode: training on {args.overfit_n} samples ({steps_per_epoch} virtual samples/epoch).")

    num_w = args.num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=num_w,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_w > 0,
        prefetch_factor=2 if num_w > 0 else None,
    )

    tokenizer = VideoTokenizer(
        image_size=IMAGE_SIZE,
        init_dim=args.init_dim,
        codebook_size=args.codebook_size,
        layers=tokenizer_layers,
        use_gan=False,
        perceptual_loss_weight=0.0
    ).to(device)

    # ── Auto batch-size ──────────────────────────────────────────────
    if args.auto_batch_size and device.type == "cuda":
        print("\n[auto-batch] Probing GPU for maximum batch size …")
        args.batch_size = find_max_batch_size(
            tokenizer,
            image_size=IMAGE_SIZE,
            seq_len=SEQUENCE_LENGTH,
            device=device,
            floor=1,
            ceiling=num_unique_samples,
        )
        # Rebuild DataLoader with the discovered batch size
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle,
            num_workers=num_w,
            pin_memory=True,
            persistent_workers=num_w > 0,
            prefetch_factor=2 if num_w > 0 else None,
        )
        print(f"[auto-batch] ✓ Selected batch size: {args.batch_size}\n")
    elif args.auto_batch_size:
        print("[auto-batch] Skipped — no CUDA device. Using --batch-size", args.batch_size)
    else:
      # Cap batch size to unique samples (duplicates in a batch waste compute)
      if args.batch_size > num_unique_samples:
          print(f"Capped batch_size {args.batch_size} -> {num_unique_samples} (unique samples)")
          args.batch_size = num_unique_samples

    optimizer = AdamW(tokenizer.parameters(), lr=args.lr)

    # ── LR schedule: linear warmup + cosine annealing ────────────────
    if args.max_steps > 0:
        warmup = args.warmup_steps
        cosine_steps = max((args.max_steps//2) - warmup, 1)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=args.lr / 4,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup],
        )
    else:
        scheduler = None

    # Save config for reproducibility
    config = vars(args).copy()
    config["image_size"] = IMAGE_SIZE
    config["sequence_length"] = SEQUENCE_LENGTH
    config["layers"] = [list(l) if isinstance(l, tuple) else l for l in tokenizer_layers]
    config["git_hash"] = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or None
    config["timestamp"] = datetime.now().isoformat()
    config["num_parameters"] = sum(p.numel() for p in tokenizer.parameters())
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    print(json.dumps(config, indent=2))

    metrics_log = []
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    train_start = time.time()

    # Patience-based early stopping state
    best_recon_loss = float('inf')
    patience_counter = 0
    patience_exhausted = False

    global_step = 0
    for epoch in range(args.epochs):
        if patience_exhausted:
            break
        if epoch > 0 and args.max_steps > 0 and global_step >= args.max_steps:
            print(f"\n[max-steps] Reached {global_step} steps (limit {args.max_steps}). Stopping.")
            break
        tokenizer.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss, loss_breakdown = tokenizer(batch, return_loss=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'recon': f"{loss_breakdown.recon_loss.item():.4f}", 'lr': f"{current_lr:.2e}"})
            
            if global_step % args.val_interval == 0:
                tokenizer.eval()
                with torch.no_grad():
                    # Generate a reconstruction sample
                    codes = tokenizer.tokenize(batch)        # discrete codebook indices
                    recon_video = tokenizer.decode_from_code_indices(codes)
                    
                    codebook_usage = codes.unique().numel()
                    step_metrics = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "recon_loss": loss_breakdown.recon_loss.item(),
                        "lr": current_lr,
                        "elapsed_s": round(time.time() - train_start, 2),
                    }
                    metrics_log.append(step_metrics)
                    step_metrics["codebook_usage"] = codebook_usage
                    
                    # Take first sequence from batch, slice along time
                    # batch[0] shape: [C, T, H, W]
                    original_frames = batch[0].permute(1, 0, 2, 3) # [T, C, H, W]
                    recon_frames = recon_video[0].permute(1, 0, 2, 3).clamp(0, 1) # [T, C, H, W]
                    
                    # Cat side by side
                    comparison = torch.cat([original_frames, recon_frames], dim=3) # [T, C, H, W*2]
                    save_image(comparison, os.path.join(args.output_dir, f"step_{global_step:06d}.png"), nrow=1)

                    # Eval-mode recon loss for checkpoint & early stopping decisions
                    _, eval_breakdown = tokenizer(batch, return_loss=True)
                    eval_recon = eval_breakdown.recon_loss.item()
                    step_metrics["eval_recon_loss"] = eval_recon

                # Save checkpoint only when eval recon loss improves
                if eval_recon < best_recon_loss - 1e-6:
                    best_recon_loss = eval_recon
                    patience_counter = 0
                    torch.save(tokenizer.state_dict(),
                               os.path.join(args.output_dir, "magvit2_latest.pt"))
                else:
                    patience_counter += 1

                # Threshold early exit
                if args.threshold > 0 and eval_recon < args.threshold:
                    print(f"\n[threshold] eval_recon={eval_recon:.6f} < "
                          f"threshold={args.threshold:.6f}. Stopping.")
                    torch.save(tokenizer.state_dict(),
                               os.path.join(args.output_dir, "magvit2_latest.pt"))
                    patience_exhausted = True

                # Patience-based early stopping
                elif args.max_patience is not None and patience_counter >= args.max_patience:
                    print(f"\n[early-stop] Patience exhausted after {args.max_patience} "
                          f"val checks with no improvement. "
                          f"Best recon_loss={best_recon_loss:.6f}")
                    patience_exhausted = True

                # Write metrics to disk at each validation step
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_log, f, indent=2)
                tokenizer.train()

            if patience_exhausted:
                break
            
            global_step += 1
            
        # Write metrics at end of each epoch
        with open(metrics_path, 'w') as f:
            json.dump(metrics_log, f, indent=2)

if __name__ == "__main__":
    train()