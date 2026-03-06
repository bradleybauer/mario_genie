import argparse
import glob
import json
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
from magvit2_pytorch import VideoTokenizer
from tqdm import tqdm

from mario_world_model.config import IMAGE_SIZE, CODEBOOK_SIZE, TOKENIZER_LAYERS, SEQUENCE_LENGTH
from mario_world_model.auto_batch_sizer import find_max_batch_size

class MarioVideoDataset(Dataset):
    def __init__(self, data_dir, seq_len=4):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.seq_len = seq_len
        self.samples = []
        
        print(f"Indexing {len(self.chunk_files)} chunks (lazy loading)...")
        for chunk_idx, f in enumerate(self.chunk_files):
            try:
                # Memory-map to read shapes/dones without loading frames into RAM
                npz = np.load(f, mmap_mode='r')
                frames_shape = npz['frames'].shape  # [num_seqs, T, C, H, W]
                dones = np.array(npz['dones'])       # small bool array, fine to load
                num_seqs = frames_shape[0]
                total_t = frames_shape[1]
                
                # If total_t >= seq_len, we can sample sub-sequences
                if total_t >= seq_len:
                    for i in range(num_seqs):
                        for t in range(0, total_t - seq_len + 1, seq_len):
                            # Check if a boundary break occurs inside this sequence
                            # A 'done=True' means the NEXT frame will be a scene reset.
                            # So it's safe if 'done=True' is exactly on the VERY LAST frame
                            # of our window (index -1), but NOT anywhere before that.
                            window_dones = dones[i, t:t+seq_len]
                            if np.any(window_dones[:-1]):
                                continue # Skip sequences that contain an internal boundary break
                            self.samples.append((chunk_idx, i, t))
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

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
        self.samples = [dataset[i] for i in indices]
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.samples[idx % len(self.samples)]

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
    parser.add_argument("--val-interval", type=int, default=50)
    parser.add_argument("--overfit-one", action="store_true", help="Train on a single sample (overfit sanity check)")
    parser.add_argument("--overfit-n", type=int, default=0, help="Train on N random samples (overfit sanity check)")
    parser.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    parser.add_argument("--init-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of the dataset")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of DataLoader workers")
    parser.add_argument("--run-name", type=str, default=None, help="Subfolder under output-dir for this run")
    parser.add_argument(
        "--sanity-check", action="store_true",
        help=(
            "Build the model, run one forward+backward pass with synthetic data, "
            "verify tokenize+decode, report param count, then exit. "
            "Skips dataset loading entirely."
        ),
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help=(
            "Comma-separated encoder layer spec, overrides config.TOKENIZER_LAYERS. "
            "e.g. 'residual,compress_space,residual,compress_space,residual,compress_space'. "
            "Use 'consecutive_residual:N' for parameterised layers."
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

    # ── Sanity-check mode ────────────────────────────────────────────────
    if args.sanity_check:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        run_label = args.run_name or "default"
        print(f"[sanity-check] {run_label}  layers={list(tokenizer_layers)}  "
              f"init_dim={args.init_dim}  codebook_size={args.codebook_size}  device={device}")

        # 1. Construct model
        try:
            tokenizer = VideoTokenizer(
                image_size=IMAGE_SIZE,
                init_dim=args.init_dim,
                codebook_size=args.codebook_size,
                layers=tokenizer_layers,
                use_gan=False,
                perceptual_loss_weight=0.0,
            ).to(device)
        except Exception as e:
            print(f"[sanity-check] FAIL  {run_label}  model construction: {e}")
            sys.exit(1)

        num_params = sum(p.numel() for p in tokenizer.parameters())
        print(f"[sanity-check] {run_label}  params={num_params:,}")

        # 2. Forward + backward with synthetic data
        dummy = torch.randn(args.batch_size, 3, SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, device=device)
        try:
            tokenizer.train()
            loss, loss_breakdown = tokenizer(dummy, return_loss=True)
            loss.backward()
            print(f"[sanity-check] {run_label}  fwd+bwd OK  loss={loss.item():.4f}  "
                  f"recon={loss_breakdown.recon_loss.item():.4f}")
        except Exception as e:
            print(f"[sanity-check] FAIL  {run_label}  fwd+bwd: {e}")
            sys.exit(1)

        # 3. Tokenize + decode (validation path)
        try:
            tokenizer.eval()
            with torch.no_grad():
                codes = tokenizer.tokenize(dummy)
                recon = tokenizer.decode_from_code_indices(codes)
            print(f"[sanity-check] {run_label}  tokenize+decode OK  "
                  f"codes={list(codes.shape)}  recon={list(recon.shape)}  "
                  f"unique_codes={codes.unique().numel()}")
        except Exception as e:
            print(f"[sanity-check] FAIL  {run_label}  tokenize+decode: {e}")
            sys.exit(1)

        print(f"[sanity-check] PASS  {run_label}")
        sys.exit(0)

    # ── Normal training ──────────────────────────────────────────────────

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Run directory
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MarioVideoDataset(args.data_dir, seq_len=SEQUENCE_LENGTH)
    print(f"Found {len(dataset)} sequence segments of length {SEQUENCE_LENGTH} frames.")

    if args.overfit_one:
        dataset = NSampleDataset(dataset, indices=[0], length=1000)
        print(">> Overfit mode: training on a single sample.")
    elif args.overfit_n > 0:
        indices = np.random.choice(len(dataset), size=args.overfit_n, replace=False)
        dataset = NSampleDataset(dataset, indices=indices, length=1000 * args.overfit_n)
        print(f">> Overfit mode: training on {args.overfit_n} multiple samples.")

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
        args.batch_size = find_max_batch_size(
            tokenizer,
            image_size=IMAGE_SIZE,
            seq_len=SEQUENCE_LENGTH,
            device=device,
            floor=1,
            ceiling=256,
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
        print(f"Auto batch size: {args.batch_size}")
    elif args.auto_batch_size:
        print("[auto-batch] Skipped — no CUDA device. Using --batch-size", args.batch_size)

    optimizer = AdamW(tokenizer.parameters(), lr=args.lr)

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

    metrics_log = []
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    train_start = time.time()

    metrics_interval = max(1, len(dataloader) // 10)

    global_step = 0
    for epoch in range(args.epochs):
        tokenizer.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step_in_epoch, batch in enumerate(pbar):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss, loss_breakdown = tokenizer(batch, return_loss=True)
            loss.backward()
            optimizer.step()
            
            log_this_step = (step_in_epoch % metrics_interval == 0)
            
            if log_this_step:
                step_metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "recon_loss": loss_breakdown.recon_loss.item(),
                    "elapsed_s": round(time.time() - train_start, 2),
                }
                metrics_log.append(step_metrics)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'recon': f"{loss_breakdown.recon_loss.item():.4f}"})
            
            if global_step % args.val_interval == 0:
                tokenizer.eval()
                with torch.no_grad():
                    # Generate a reconstruction sample
                    codes = tokenizer.tokenize(batch)        # discrete codebook indices
                    recon_video = tokenizer.decode_from_code_indices(codes)
                    
                    codebook_usage = codes.unique().numel()
                    if not log_this_step:
                        step_metrics = {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "recon_loss": loss_breakdown.recon_loss.item(),
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
                
                # Write metrics to disk at each validation step
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_log, f, indent=2)
                tokenizer.train()
            
            global_step += 1
            
        # Write metrics + save checkpoint at end of each epoch
        with open(metrics_path, 'w') as f:
            json.dump(metrics_log, f, indent=2)
        torch.save(tokenizer.state_dict(), os.path.join(args.output_dir, "magvit2_latest.pt"))

if __name__ == "__main__":
    train()