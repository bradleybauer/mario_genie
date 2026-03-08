import argparse
import concurrent.futures
from collections import defaultdict
import glob
import json
import math
import os
import queue as queue_mod
import threading
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
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
from mario_world_model.tokenizer_compat import resolve_video_contains_first_frame

def _count_scene_cuts(npz, seq_idx, t_start, seq_len):
    if "world" in npz.files and "stage" in npz.files:
        world_window = np.asarray(npz["world"][seq_idx, t_start:t_start+seq_len])
        stage_window = np.asarray(npz["stage"][seq_idx, t_start:t_start+seq_len])
        if len(world_window) <= 1:
            return 0
        transitions = (world_window[1:] != world_window[:-1]) | (stage_window[1:] != stage_window[:-1])
        return int(np.count_nonzero(transitions))

    if "dones" in npz.files:
        done_window = np.asarray(npz["dones"][seq_idx, t_start:t_start+seq_len], dtype=bool)
        return int(np.count_nonzero(done_window[:-1]))

    return 0


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
        samples = []
        for i in range(num_seqs):
            for t in range(0, total_t - seq_len + 1, seq_len):
                if _count_scene_cuts(npz, i, t, seq_len) >= 2:
                    continue
                samples.append((chunk_idx, i, t))
        return samples
    except Exception as e:
        print(f"Skipping {filepath} due to error: {e}")
        return []

class MarioVideoDataset(Dataset):
    def __init__(self, data_dir, seq_len=4, preload=True, subset_n=0, seed=42):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.seq_len = seq_len
        self.samples = []
        
        print(f"Indexing {len(self.chunk_files)} chunks (lazy loading)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_index_chunk, idx, f, seq_len): idx
                for idx, f in enumerate(self.chunk_files)
            }
            for future in concurrent.futures.as_completed(futures):
                self.samples.extend(future.result())

        if subset_n > 0 and subset_n < len(self.samples):
            total = len(self.samples)
            rng = np.random.RandomState(seed)
            indices = rng.choice(total, size=subset_n, replace=False)
            self.samples = [self.samples[i] for i in indices]
            print(f"Subset: kept {subset_n} of {total} samples")

        self.preloaded = False
        if preload and len(self.samples) > 0:
            self._preload_all()

    def _preload_all(self):
        """Load all samples into a single pinned CPU tensor (parallel I/O)."""
        # Probe shape from first sample
        chunk_idx, seq_idx, t_start = self.samples[0]
        npz = np.load(self.chunk_files[chunk_idx], mmap_mode='r')
        probe = npz['frames'][seq_idx, t_start:t_start+self.seq_len]
        T, C, H, W = probe.shape

        n = len(self.samples)
        total_mb = n * C * T * H * W * 4 / 2**20  # float32
        print(f"Pre-loading {n} samples into CPU tensor ({total_mb:.0f} MB)...")
        self.data = torch.empty(n, C, T, H, W, dtype=torch.float32)

        print(f"Grouping samples by chunk for efficient loading...")
        # Group samples by chunk to open each file only once
        chunk_groups = defaultdict(list)
        for sample_idx, (chunk_idx, seq_idx, t_start) in enumerate(self.samples):
            chunk_groups[chunk_idx].append((sample_idx, seq_idx, t_start))

        def _load_chunk(chunk_idx, entries):
            """Load one chunk file and write samples into the shared tensor."""
            npz = np.load(self.chunk_files[chunk_idx], mmap_mode='r')
            frames_arr = npz['frames']
            for sample_idx, seq_idx, t_start in entries:
                frames = np.array(frames_arr[seq_idx, t_start:t_start+self.seq_len])
                # [T, C, H, W] -> [C, T, H, W], float32, /255
                t = torch.from_numpy(frames).float().div_(255.0).permute(1, 0, 2, 3)
                self.data[sample_idx] = t
            return len(entries)

        num_threads = min(4, len(chunk_groups))
        print(f"Parallel loading with {num_threads} threads...")
        loaded = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {
                pool.submit(_load_chunk, ci, entries): ci
                for ci, entries in chunk_groups.items()
            }
            with tqdm(total=n, desc="Pre-loading samples", unit="samp") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    count = future.result()
                    loaded += count
                    pbar.update(count)

        self.preloaded = True
        print(f"Pre-load complete. Tensor shape: {list(self.data.shape)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preloaded:
            return self.data[idx]

        chunk_idx, seq_idx, t_start = self.samples[idx]
        
        # Load only the slice we need via memory-mapped access
        npz = np.load(self.chunk_files[chunk_idx], mmap_mode='r')
        frames = np.array(npz['frames'][seq_idx, t_start:t_start+self.seq_len])
        
        # Convert to tensor: [T, C, H, W] -> [C, T, H, W] required by VideoTokenizer
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(1, 0, 2, 3) 
        
        return frames

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to chunk files")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--auto-batch-size", action="store_true",
        help="Probe GPU to find the largest batch size that fits in VRAM. "
             "Overrides --batch-size.",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=0,
        help="Cap batch size (0 = no cap). Use with --auto-batch-size to "
             "limit large batches for faster convergence.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints/magvit2")
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--image-interval-secs", type=float, default=300,
                        help="Minimum seconds between saving reconstruction images (default: 300)")
    parser.add_argument("--overfit-n", type=int, default=0, help="Train on N random samples (overfit sanity check)")
    parser.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE)
    parser.add_argument("--init-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of the dataset")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of DataLoader workers")
    parser.add_argument("--no-preload", action="store_true",
                        help="Disable pre-loading all data into RAM (use lazy per-sample loading)")
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
        "--max-patience", type=float, default=None,
        help=(
            "Early stopping: halt training after this many seconds "
            "with no recon_loss improvement (>=1e-6). "
            "Disabled by default."
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
        "--max-minutes", type=float, default=0,
        help=(
            "Stop training after this many minutes of wall-clock time "
            "(0 = no limit, rely on --max-patience / --threshold to stop). "
            "Also sets the cosine-annealing LR schedule duration."
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

    dataset = MarioVideoDataset(
        args.data_dir, seq_len=SEQUENCE_LENGTH,
        preload=not args.no_preload,
        subset_n=args.overfit_n,
        seed=args.seed,
    )
    print(f"Found {len(dataset)} sequence segments of length {SEQUENCE_LENGTH} frames.")

    num_unique_samples = len(dataset)
    gpu_cache = None
    if args.overfit_n > 0:
        num_unique_samples = len(dataset)
        # Try to cache on GPU; fall back to DataLoader
        sample = dataset[0]
        bytes_per_sample = sample.nelement() * sample.element_size()
        total_bytes = bytes_per_sample * num_unique_samples
        if device.type == "cuda":
            gpu_free = torch.cuda.mem_get_info(device)[0]
            budget = gpu_free // 4
        else:
            budget = 0
        if total_bytes <= budget:
            print(f">> Overfit mode: caching {num_unique_samples} samples on GPU ({total_bytes / 2**20:.0f} MB).")
            gpu_cache = torch.stack([dataset[i] for i in range(num_unique_samples)]).to(device)
        else:
            print(f">> Overfit mode: {num_unique_samples} samples ({total_bytes / 2**20:.0f} MB) too large for GPU cache, using DataLoader.")

    # When data is preloaded into pinned RAM, workers add IPC overhead with no benefit
    preloaded = getattr(dataset, 'preloaded', False) or (
        isinstance(dataset, Subset) and getattr(dataset.dataset, 'preloaded', False)
    )
    num_w = 0 if preloaded else args.num_workers
    if gpu_cache is None:
        sampler_len = 10**7
        sampler = RandomSampler(dataset, replacement=True, num_samples=sampler_len) if not args.no_shuffle else None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_w,
            pin_memory=not preloaded and torch.cuda.is_available(),
            persistent_workers=num_w > 0,
            prefetch_factor=2 if num_w > 0 else None,
        )
    else:
        dataloader = None

    tokenizer = VideoTokenizer(
        image_size=IMAGE_SIZE,
        init_dim=args.init_dim,
        codebook_size=args.codebook_size,
        layers=tokenizer_layers,
        use_gan=False,
        perceptual_loss_weight=0.0
    ).to(device)
    video_contains_first_frame = resolve_video_contains_first_frame(tokenizer, SEQUENCE_LENGTH)

    # ── Auto batch-size ──────────────────────────────────────────────
    if args.auto_batch_size and device.type == "cuda":
        print("\n[auto-batch] Probing GPU for maximum batch size …")
        ceiling = num_unique_samples
        if args.max_batch_size > 0:
            ceiling = min(ceiling, args.max_batch_size)
        args.batch_size = find_max_batch_size(
            tokenizer,
            image_size=IMAGE_SIZE,
            seq_len=SEQUENCE_LENGTH,
            device=device,
            floor=1,
            ceiling=ceiling,
            video_contains_first_frame=video_contains_first_frame,
        )
        # Rebuild DataLoader with the discovered batch size
        if gpu_cache is None:
            sampler_len = 10**7
            sampler = RandomSampler(dataset, replacement=True, num_samples=sampler_len) if not args.no_shuffle else None
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_w,
                pin_memory=not preloaded,
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

    # ── LR schedule: time-based cosine annealing ─────────────────────
    max_seconds = args.max_minutes * 60 if args.max_minutes > 0 else 0
    eta_min = args.lr / 2

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

    smooth_alpha = 0.95
    smoothed_recon = None
    best_smoothed_recon = float('inf')
    last_improvement_time = time.time()
    last_image_time = 0.0

    tokenizer.train()

    # Get the preloaded CPU tensor if available (bypass DataLoader for speed)
    cpu_cache = None
    if gpu_cache is None and preloaded:
        raw = dataset.data if hasattr(dataset, 'data') else None
        if raw is None and isinstance(dataset, Subset) and hasattr(dataset.dataset, 'data'):
            raw = dataset.dataset.data
        cpu_cache = raw

    def batch_iter():
        """Yields batches: GPU cache > pinned CPU prefetch > DataLoader."""
        bs = args.batch_size
        if gpu_cache is not None:
            n = gpu_cache.shape[0]
            while True:
                idx = torch.randint(n, (bs,), device=device)
                yield gpu_cache[idx]
        elif cpu_cache is not None:
            # Shuffle-then-stream: sequential RAM reads are 10-50x faster
            # than random gathers across a large tensor.
            n = cpu_cache.shape[0]
            sample_shape = cpu_cache.shape[1:]
            q = queue_mod.Queue(maxsize=3)

            def _prefetch():
                buf = torch.empty(bs, *sample_shape, dtype=torch.float32, pin_memory=True)
                while True:
                    perm = torch.randperm(n)
                    for start in range(0, n - bs + 1, bs):
                        idx = perm[start:start + bs]
                        # Sort indices within batch for sequential memory access
                        idx, _ = idx.sort()
                        torch.index_select(cpu_cache, 0, idx, out=buf)
                        q.put(buf.clone())

            t = threading.Thread(target=_prefetch, daemon=True)
            t.start()
            while True:
                yield q.get()
        else:
            yield from dataloader

    pbar = tqdm(batch_iter(), desc="Training")

    for global_step, batch in enumerate(pbar):
        elapsed = time.time() - train_start
        if max_seconds > 0 and elapsed >= max_seconds:
            print(f"\n[max-time] Reached {elapsed/60:.1f} min. Stopping.")
            break

        # Time-based cosine LR
        if max_seconds > 0:
            progress = min(elapsed / max_seconds, 1.0)
            new_lr = eta_min + 0.5 * (args.lr - eta_min) * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

        if gpu_cache is None:
            batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, loss_breakdown = tokenizer(
            batch,
            return_loss=True,
            video_contains_first_frame=video_contains_first_frame,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        recon = loss_breakdown.recon_loss.item()

        # Update EMA of recon loss
        if smoothed_recon is None:
            smoothed_recon = recon
        else:
            smoothed_recon = smooth_alpha * smoothed_recon + (1 - smooth_alpha) * recon

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'recon': f"{recon:.4f}", 'smooth': f"{smoothed_recon:.4f}", 'lr': f"{current_lr:.2e}"})

        # ── Checkpoint: smoothed trend improved AND raw batch confirms ──
        if smoothed_recon < best_smoothed_recon - 1e-6 and recon < best_smoothed_recon:
            best_smoothed_recon = smoothed_recon
            last_improvement_time = time.time()

        if args.threshold > 0 and smoothed_recon < args.threshold:
            print(f"\n[threshold] smoothed_recon={smoothed_recon:.6f} < "
                  f"threshold={args.threshold:.6f}. Stopping.")
            break

        if args.max_patience is not None and (time.time() - last_improvement_time) >= args.max_patience:
            print(f"\n[early-stop] No improvement for {time.time() - last_improvement_time:.0f}s "
                  f"(patience={args.max_patience:.0f}s). "
                  f"Best smoothed_recon={best_smoothed_recon:.6f}")
            break

        # ── Periodic logging & image saves ───────────────────────────
        if global_step % args.val_interval == 0:
            step_metrics = {
                "step": global_step,
                "loss": loss.item(),
                "recon_loss": recon,
                "smoothed_recon_loss": smoothed_recon,
                "best_smoothed_recon": best_smoothed_recon,
                "lr": current_lr,
                "elapsed_s": round(time.time() - train_start, 2),
            }
            metrics_log.append(step_metrics)

            # Image generation on a wall-clock schedule
            now = time.time()
            if now - last_image_time >= args.image_interval_secs:
                last_image_time = now
                tokenizer.eval()
                with torch.no_grad():
                    codes = tokenizer(batch, return_codes=True, video_contains_first_frame=video_contains_first_frame)
                    recon_video = tokenizer.decode_from_code_indices(
                        codes,
                        video_contains_first_frame=video_contains_first_frame,
                    )

                    step_metrics["codebook_usage"] = codes.unique().numel()

                    original_frames = batch[0].permute(1, 0, 2, 3)
                    recon_frames = recon_video[0].permute(1, 0, 2, 3).clamp(0, 1)
                    comparison = torch.cat([original_frames, recon_frames], dim=3)
                    save_image(comparison, os.path.join(args.output_dir, f"step_{global_step:06d}.png"), nrow=1)
                tokenizer.train()

            with open(metrics_path, 'w') as f:
                json.dump(metrics_log, f, indent=2)

    # Always record the final state so sweep readers see the terminal smoothed_recon
    final_metrics = {
        "step": global_step,
        "loss": loss.item(),
        "recon_loss": recon,
        "smoothed_recon_loss": smoothed_recon,
        "best_smoothed_recon": best_smoothed_recon,
        "lr": optimizer.param_groups[0]['lr'],
        "elapsed_s": round(time.time() - train_start, 2),
    }
    if not metrics_log or metrics_log[-1]["step"] != global_step:
        metrics_log.append(final_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)

    # ── Save checkpoint & final reconstruction ───────────────────────
    ckpt_path = os.path.join(args.output_dir, "magvit2_best.pt")
    torch.save(tokenizer.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    tokenizer.eval()
    with torch.no_grad():
        codes = tokenizer(batch, return_codes=True, video_contains_first_frame=video_contains_first_frame)
        recon_video = tokenizer.decode_from_code_indices(
            codes,
            video_contains_first_frame=video_contains_first_frame,
        )
        original_frames = batch[0].permute(1, 0, 2, 3)
        recon_frames = recon_video[0].permute(1, 0, 2, 3).clamp(0, 1)
        comparison = torch.cat([original_frames, recon_frames], dim=3)
        save_image(comparison, os.path.join(args.output_dir, "final_reconstruction.png"), nrow=1)
    print(f"Final reconstruction saved to {args.output_dir}/final_reconstruction.png")

if __name__ == "__main__":
    train()