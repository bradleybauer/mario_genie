import argparse
import concurrent.futures
import gc
import json
import math
import os
import random
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
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

from tqdm import tqdm

from mario_world_model.config import IMAGE_SIZE, SEQUENCE_LENGTH
from mario_world_model.auto_batch_sizer import find_max_batch_size
from mario_world_model.dataset_paths import find_session_files
from mario_world_model.model_configs import MODEL_CONFIGS, MODEL_CONFIGS_BY_NAME
from mario_world_model.tokenizer_compat import resolve_video_contains_first_frame
from mario_world_model.palette_tokenizer import PaletteVideoTokenizer
from mario_world_model.system_info import collect_system_info, print_system_info, get_available_memory, get_effective_cpu_count


def _format_elapsed(seconds):
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"




def _index_file(file_idx, filepath, seq_len):
    """Index a session file.

    Returns (file_idx, frame_count, samples) where samples is a list of
    (file_idx, t_start) tuples.
    """
    try:
        npz = np.load(filepath, mmap_mode='r')
        frames = npz['frames']

        if frames.ndim != 4:
            print(f"Skipping {filepath}: expected session format (N,C,H,W), got ndim={frames.ndim}")
            return file_idx, 0, []

        total_t = frames.shape[0]
        if total_t < seq_len:
            return file_idx, 0, []
        samples = [(file_idx, t) for t in range(total_t - seq_len + 1)]
        return file_idx, total_t, samples
    except Exception as e:
        print(f"Skipping {filepath} due to error: {e}")
        return file_idx, 0, []

class MarioVideoDataset(Dataset):
    def __init__(self, data_dir, seq_len=4, subset_n=0, seed=42, num_workers=None, system_info=None):
        if num_workers is None:
            num_workers = get_effective_cpu_count(system_info)
        self.data_files = find_session_files(data_dir)
        self.seq_len = seq_len
        self.samples = []
        
        n_files = len(self.data_files)
        print(f"Indexing {n_files} data files (stride=1)...")
        frame_counts = [0] * n_files
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_index_file, idx, f, seq_len): idx
                for idx, f in enumerate(self.data_files)
            }
            with tqdm(total=n_files, desc="Indexing files", unit="file") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    file_idx, total_t, samples = future.result()
                    frame_counts[file_idx] = total_t
                    self.samples.extend(samples)
                    pbar.update(1)

        # Remove files with no usable frames and remap indices
        valid = [i for i in range(n_files) if frame_counts[i] > 0]
        if len(valid) < n_files:
            skipped = n_files - len(valid)
            print(f"Dropped {skipped} file(s) with no usable frames")
            old_to_new = {old: new for new, old in enumerate(valid)}
            self.data_files = [self.data_files[i] for i in valid]
            frame_counts = [frame_counts[i] for i in valid]
            self.samples = [(old_to_new[fi], t) for fi, t in self.samples]
            n_files = len(self.data_files)

        if subset_n > 0 and subset_n < len(self.samples):
            total = len(self.samples)
            rng = np.random.RandomState(seed)
            indices = rng.choice(total, size=subset_n, replace=False)
            self.samples = [self.samples[i] for i in indices]
            print(f"Subset: kept {subset_n} of {total} samples")

        # Decide whether to decompress all frames into RAM or use mmap
        total_frames = sum(frame_counts)
        self.frames_by_file = None
        if total_frames > 0:
            available = get_available_memory(system_info)
            probe = np.load(self.data_files[0], mmap_mode='r')['frames']
            bytes_per_frame = math.prod(probe.shape[1:])  # C*H*W, uint8
            total_bytes = total_frames * bytes_per_frame
            headroom = 2 * 2**30  # keep 2 GB free

            if available > 0 and total_bytes < available - headroom:
                print(f"Loading all frames into RAM ({total_bytes / 2**30:.0f} GB, "
                      f"{available / 2**30:.0f} GB available)...")
                self.frames_by_file = [None] * n_files

                def _load_file(idx):
                    return idx, np.load(self.data_files[idx])['frames']

                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
                    futures = [pool.submit(_load_file, i) for i in range(n_files)]
                    with tqdm(total=n_files, desc="Loading into RAM", unit="file") as pbar:
                        for future in concurrent.futures.as_completed(futures):
                            idx, frames = future.result()
                            self.frames_by_file[idx] = frames
                            pbar.update(1)
                actual = sum(f.nbytes for f in self.frames_by_file)
                print(f"Loaded {actual / 2**30:.0f} GB into RAM (COW-shared with workers)")
            else:
                print(f"Dataset too large for RAM ({total_bytes / 2**30:.0f} GB, "
                      f"{available / 2**30:.0f} GB available). Using mmap.")

    def __len__(self):
        return len(self.samples)

    def _get_frames_mmap(self, file_idx):
        """Return the mmap'd frames array for a file, caching the handle."""
        if not hasattr(self, '_mmap_cache'):
            self._mmap_cache = {}
        if file_idx not in self._mmap_cache:
            self._mmap_cache[file_idx] = np.load(
                self.data_files[file_idx], mmap_mode='r'
            )['frames']
        return self._mmap_cache[file_idx]

    def __getitem__(self, idx):
        file_idx, t_start = self.samples[idx]
        if self.frames_by_file is not None:
            frames = self.frames_by_file[file_idx]
        else:
            frames = self._get_frames_mmap(file_idx)
        chunk = frames[t_start:t_start+self.seq_len, 0]  # (T, H, W) uint8
        return torch.from_numpy(chunk.copy())


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to session files or a parent directory containing nested session folders",
    )
    parser.add_argument("--batch-size", type=int, default=4)
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
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="Linear warmup from 0 to --lr over this many steps (default: 200)")
    parser.add_argument("--output-dir", type=str, default="checkpoints/magvit2")
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--image-interval-secs", type=float, default=300,
                        help="Minimum seconds between saving reconstruction images (default: 300)")
    parser.add_argument("--overfit-n", type=int, default=0, help="Train on N random samples (overfit sanity check)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Named model config (random if omitted). Use --list-models to see available configs.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model configs and exit.",
    )
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (enabled by default)")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 tensor cores for matmul (Ampere+ GPUs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of the dataset")
    _default_workers = min(get_effective_cpu_count(), 64)
    parser.add_argument("--num-workers", type=int, default=_default_workers,
                        help=f"Number of DataLoader workers (default: {_default_workers})")
    parser.add_argument("--run-name", type=str, default=None, help="Subfolder under output-dir for this run")
    parser.add_argument(
        "--max-patience", type=float, default=None,
        help=(
            "Early stopping: halt training after this many seconds "
            "with no recon_loss improvement (>=1e-6). "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=(
            "Stop training once eval recon_loss drops below this value "
            "Checkpoint is saved before exit."
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
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Stop after this many gradient steps (0 = no limit).",
    )
    parser.add_argument(
        "--total-steps", type=int, default=0,
        help=(
            "Total steps for the cosine LR schedule. "
            "Defaults to --max-steps when set, otherwise estimated from --max-minutes."
        ),
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to training_state.pt to resume training from.",
    )
    args = parser.parse_args()

    if args.list_models:
        for name in sorted(MODEL_CONFIGS_BY_NAME):
            print(name)
        return

    if args.model is None:
        mc = random.choice(MODEL_CONFIGS)
        args.model = mc.name
        print(f"[random] Selected model: {args.model}")
    elif args.model not in MODEL_CONFIGS_BY_NAME:
        parser.error(f"Unknown model {args.model!r}. Use --list-models to see available configs.")
    else:
        mc = MODEL_CONFIGS_BY_NAME[args.model]
    tokenizer_layers = tuple(
        (name, int(val)) if ":" in tok else tok
        for tok in mc.layers.split(",")
        for name, _, val in [tok.partition(":")]
    )

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

    # ── System info ──────────────────────────────────────────────────
    system_info = collect_system_info()
    print_system_info(system_info)

    dataset = MarioVideoDataset(
        args.data_dir, seq_len=SEQUENCE_LENGTH,
        subset_n=args.overfit_n,
        seed=args.seed,
        num_workers=args.num_workers,
        system_info=system_info,
    )
    print(f"Found {len(dataset)} sequence segments of length {SEQUENCE_LENGTH} frames.")

    # ── Load palette from palette.json in data dir ──
    palette_path = os.path.join(args.data_dir, "palette.json")
    if not os.path.isfile(palette_path):
        raise FileNotFoundError(f"No palette.json found in {args.data_dir}")
    with open(palette_path) as f:
        palette_rgb = json.load(f)
    palette = torch.tensor(palette_rgb, dtype=torch.float32) / 255.0  # (K, 3)
    num_palette_colors = palette.shape[0]
    print(f"[palette] Loaded {num_palette_colors} colours from {palette_path}")

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
            print(f">> Overfit mode: caching {num_unique_samples} samples on GPU ({total_bytes / 2**30:.0f} GB).")
            gpu_cache = torch.stack([dataset[i] for i in range(num_unique_samples)]).to(device)
        else:
            print(f">> Overfit mode: {num_unique_samples} samples ({total_bytes / 2**30:.0f} GB) too large for GPU cache, using DataLoader.")

    max_workers = get_effective_cpu_count(system_info)
    num_w = min(args.num_workers, max_workers)

    def make_dataloader(batch_size):
        sampler = RandomSampler(dataset, replacement=True, num_samples=10**7) if not args.no_shuffle else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_w,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_w > 0,
            prefetch_factor=2 if num_w > 0 else None,
        )

    dataloader = make_dataloader(args.batch_size) if gpu_cache is None else None

    tokenizer = PaletteVideoTokenizer(
        num_palette_colors=num_palette_colors,
        image_size=IMAGE_SIZE,
        init_dim=mc.init_dim,
        codebook_size=mc.codebook_size,
        layers=tokenizer_layers,
    ).to(device)
    # Drop unused discriminator (use_gan=False, but upstream still creates 52M+ params)
    tokenizer.discr = None
    tokenizer.multiscale_discrs = None
    palette = palette.to(device)
    video_contains_first_frame = resolve_video_contains_first_frame(tokenizer, SEQUENCE_LENGTH)

    # ── Auto batch-size (before compile — compiled graphs cache shapes) ──
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
            dataloader = make_dataloader(args.batch_size)
        print(f"[auto-batch] ✓ Selected batch size: {args.batch_size}\n")
    elif args.auto_batch_size:
        print("[auto-batch] Skipped — no CUDA device. Using --batch-size", args.batch_size)
    else:
      # Cap batch size to unique samples (duplicates in a batch waste compute)
      if args.batch_size > num_unique_samples:
          print(f"Capped batch_size {args.batch_size} -> {num_unique_samples} (unique samples)")
          args.batch_size = num_unique_samples

    # ── TF32 ─────────────────────────────────────────────────────────
    if args.tf32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("[tf32] TF32 matmul precision enabled")

    # ── Count parameters (before compile — wrapper can change counts) ──
    num_parameters = sum(p.numel() for p in tokenizer.parameters())

    # ── Compile (after auto-batch so shape probing isn't cached) ──
    if not args.no_compile:
        tokenizer = torch.compile(tokenizer)
        print("[compile] torch.compile enabled")

    optimizer = AdamW(tokenizer.parameters(), lr=args.lr)

    # ── LR schedule: linear warmup → cosine decay to 10 % of peak ────
    max_seconds = args.max_minutes * 60 if args.max_minutes > 0 else 0
    min_lr = args.lr * 0.1
    warmup_steps = args.warmup_steps

    # Estimate total training steps for the cosine T_max
    total_steps_for_schedule = args.total_steps or args.max_steps
    if total_steps_for_schedule > 0:
        estimated_total = max(total_steps_for_schedule, warmup_steps + 100)
    elif max_seconds > 0:
        estimated_total = max(int(max_seconds * 5), warmup_steps + 100)
    else:
        estimated_total = 100_000
    decay_steps = max(estimated_total - warmup_steps, 1)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    decay_scheduler = CosineAnnealingLR(
        optimizer, T_max=decay_steps, eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )

    # Save config for reproducibility
    config = vars(args).copy()
    config["image_size"] = IMAGE_SIZE
    config["sequence_length"] = SEQUENCE_LENGTH
    config["layers"] = [list(l) if isinstance(l, tuple) else l for l in tokenizer_layers]
    config["git_hash"] = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or None
    config["timestamp"] = datetime.now().isoformat()
    config["num_parameters"] = num_parameters
    config["system_info"] = system_info
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    print(json.dumps(config, indent=2))

    metrics_log = []
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    train_start = time.time()

    # ── Resume from checkpoint ───────────────────────────────────────
    start_step = 0
    ckpt = None
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        unwrapped = getattr(tokenizer, '_orig_mod', tokenizer)
        unwrapped.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['global_step']
        print(f"[resume] Loaded training state from {args.resume_from} at step {start_step}")

    # ── Perf tracking ────────────────────────────────────────────────
    _perf_step_count = start_step
    _perf_sample_count = 0
    _perf_window_start = time.time()
    _perf_window_steps = 0
    _perf_window_samples = 0
    _PERF_WINDOW = 20  # rolling window size in steps
    # Bytes per sample for throughput: one-hot input (B, K, T, H, W) float32
    _bytes_per_sample = num_palette_colors * SEQUENCE_LENGTH * IMAGE_SIZE * IMAGE_SIZE * 4

    def _gpu_stats():
        """Return dict of GPU stats, or empty dict if unavailable."""
        if not torch.cuda.is_available():
            return {}
        idx = device.index or 0
        try:
            mem_used = torch.cuda.memory_allocated(idx) / 2**30
            mem_total = torch.cuda.get_device_properties(idx).total_memory / 2**30
            stats = {
                "gpu_mem_used_gb": round(mem_used, 2),
                "gpu_mem_total_gb": round(mem_total, 2),
                "gpu_mem_pct": round(100 * mem_used / mem_total, 1),
            }
        except Exception:
            stats = {}
        # nvidia-smi fields via pynvml-free approach
        try:
            handle = torch.cuda.current_device()
            util = torch.cuda.utilization(handle)
            stats["gpu_util_pct"] = util
        except Exception:
            pass
        try:
            temp = torch.cuda.temperature(idx)
            stats["gpu_temp_c"] = temp
        except Exception:
            pass
        return stats

    smooth_alpha = 0.95
    smoothed_recon = None
    best_smoothed_recon = float('inf')
    last_improvement_time = time.time()
    last_image_time = 0.0

    if args.resume_from and ckpt is not None:
        smoothed_recon = ckpt.get('smoothed_recon')
        best_smoothed_recon = ckpt.get('best_smoothed_recon', float('inf'))
        metrics_log = ckpt.get('metrics_log', [])
        _perf_sample_count = ckpt.get('total_samples', 0)

    def save_reconstruction(batch, path):
        """Encode → decode a batch and save side-by-side comparison image.

        Returns (codes, codebook_usage).
        """
        tokenizer.eval()
        with torch.no_grad():
            inp = PaletteVideoTokenizer.indices_to_onehot(
                batch.long().to(device), num_palette_colors,
            )
            codes = tokenizer(inp, return_codes=True, video_contains_first_frame=video_contains_first_frame)
            recon_video = tokenizer.decode_from_code_indices(
                codes,
                video_contains_first_frame=video_contains_first_frame,
            )
            original_rgb = palette[batch[0].long()]
            original_frames = original_rgb.permute(0, 3, 1, 2)
            recon_idx = recon_video[0].argmax(dim=0)
            recon_rgb = palette[recon_idx]
            recon_frames = recon_rgb.permute(0, 3, 1, 2)
            comparison = torch.cat([original_frames, recon_frames], dim=3)
            save_image(comparison, path, nrow=1)
            codebook_usage = codes.unique().numel()
        tokenizer.train()
        return codes, codebook_usage

    tokenizer.train()

    def batch_iter():
        """Yields batches: GPU cache > DataLoader."""
        bs = args.batch_size
        if gpu_cache is not None:
            n = gpu_cache.shape[0]
            while True:
                idx = torch.randint(n, (bs,), device=device)
                yield gpu_cache[idx]
        else:
            yield from dataloader

    pbar_total = args.max_steps if args.max_steps > 0 else None
    pbar = tqdm(batch_iter(), desc="Training", initial=start_step, total=pbar_total)

    global_step = start_step
    consecutive_ooms = 0
    for batch in pbar:
        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"\n[max-steps] Reached {global_step} steps. Stopping.")
            break
        elapsed = time.time() - train_start
        if max_seconds > 0 and elapsed >= max_seconds:
            print(f"\n[max-time] Reached {elapsed/60:.1f} min. Stopping.")
            break

        if gpu_cache is None:
            batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        targets = batch.long()
        inp = PaletteVideoTokenizer.indices_to_onehot(
            targets, num_palette_colors,
        )
        try:
            loss, loss_breakdown = tokenizer(
                inp,
                targets=targets,
                return_loss=True,
                video_contains_first_frame=video_contains_first_frame,
            )
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            # Clean up GPU memory
            optimizer.zero_grad(set_to_none=True)
            del inp, targets, batch
            gc.collect()
            torch.cuda.empty_cache()
            if global_step == start_step:
                print(f"\n[OOM] Out of memory on first step with batch_size={args.batch_size}. Exiting.")
                sys.exit(1)
            consecutive_ooms += 1
            print(f"\n[OOM] Out of memory at step {global_step}, skipping batch. ({consecutive_ooms} consecutive)")
            if consecutive_ooms >= 20:
                print(f"\n[OOM] {consecutive_ooms} consecutive OOMs — model cannot fit. Aborting.")
                sys.exit(1)
            global_step += 1
            continue
        consecutive_ooms = 0
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        recon = loss_breakdown.recon_loss.item()

        # ── Perf tracking ────────────────────────────────────────────
        _perf_step_count += 1
        _perf_sample_count += batch.shape[0]
        _perf_window_steps += 1
        _perf_window_samples += batch.shape[0]
        now_perf = time.time()
        if _perf_window_steps >= _PERF_WINDOW:
            _window_elapsed = now_perf - _perf_window_start
            _samples_per_sec = _perf_window_samples / max(_window_elapsed, 1e-9)
            _steps_per_sec = _perf_window_steps / max(_window_elapsed, 1e-9)
            _perf_window_start = now_perf
            _perf_window_steps = 0
            _perf_window_samples = 0
        else:
            _window_elapsed = now_perf - _perf_window_start
            _samples_per_sec = _perf_window_samples / max(_window_elapsed, 1e-9)
            _steps_per_sec = _perf_window_steps / max(_window_elapsed, 1e-9)

        # Update EMA of recon loss
        if smoothed_recon is None:
            smoothed_recon = recon
        else:
            smoothed_recon = smooth_alpha * smoothed_recon + (1 - smooth_alpha) * recon

        # ── Checkpoint: smoothed trend improved AND raw batch confirms ──
        if smoothed_recon < best_smoothed_recon - 1e-6 and recon < best_smoothed_recon:
            best_smoothed_recon = smoothed_recon
            last_improvement_time = time.time()

        postfix = {
            'loss': f"{loss.item():.4f}",
            'recon': f"{recon:.4f}",
            'smooth': f"{smoothed_recon:.4f}",
            'best': f"{best_smoothed_recon:.4f}",
            'since_best': _format_elapsed(time.time() - last_improvement_time),
            'lr': f"{current_lr:.2e}",
            'samp/s': f"{_samples_per_sec:.0f}",
            'MB/s': f"{_samples_per_sec * _bytes_per_sample / 2**20:.0f}",
        }
        gs = _gpu_stats()
        if 'gpu_util_pct' in gs:
            postfix['gpu'] = f"{gs['gpu_util_pct']}%"
        if 'gpu_mem_pct' in gs:
            postfix['mem'] = f"{gs['gpu_mem_pct']:.0f}%"
        if 'gpu_temp_c' in gs:
            postfix['temp'] = f"{gs['gpu_temp_c']}C"
        pbar.set_postfix(postfix)

        if args.threshold is not None and smoothed_recon < args.threshold:
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
            gs = _gpu_stats()
            step_metrics = {
                "step": global_step,
                "loss": loss.item(),
                "recon_loss": recon,
                "smoothed_recon_loss": smoothed_recon,
                "best_smoothed_recon": best_smoothed_recon,
                "lr": current_lr,
                "elapsed_s": round(time.time() - train_start, 2),
                "samples_per_sec": round(_samples_per_sec, 1),
                "data_throughput_mbps": round(_samples_per_sec * _bytes_per_sample / 2**20, 1),
                "steps_per_sec": round(_steps_per_sec, 2),
                "total_samples": _perf_sample_count,
                **gs,
            }
            metrics_log.append(step_metrics)

            # Image generation on a wall-clock schedule
            now = time.time()
            if now - last_image_time >= args.image_interval_secs:
                last_image_time = now
                codes, codebook_usage = save_reconstruction(
                    batch, os.path.join(args.output_dir, f"step_{global_step:06d}.png"),
                )
                step_metrics["codebook_usage"] = codebook_usage

            with open(metrics_path, 'w') as f:
                json.dump(metrics_log, f, indent=2)

        global_step += 1

    # Always record the final state so sweep readers see the terminal smoothed_recon
    total_elapsed = time.time() - train_start
    gs = _gpu_stats()
    final_metrics = {
        "step": global_step,
        "loss": loss.item(),
        "recon_loss": recon,
        "smoothed_recon_loss": smoothed_recon,
        "best_smoothed_recon": best_smoothed_recon,
        "lr": optimizer.param_groups[0]['lr'],
        "elapsed_s": round(total_elapsed, 2),
        "samples_per_sec": round(_perf_sample_count / max(total_elapsed, 1e-9), 1),
        "total_samples": _perf_sample_count,
        "total_steps": _perf_step_count,
        **gs,
    }
    if not metrics_log or metrics_log[-1]["step"] != global_step:
        metrics_log.append(final_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)

    # ── Save checkpoint & final reconstruction ───────────────────────
    ckpt_path = os.path.join(args.output_dir, "magvit2_best.pt")
    unwrapped = getattr(tokenizer, '_orig_mod', tokenizer)
    torch.save(unwrapped.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path} ({len(unwrapped.state_dict())} keys)")

    # Full training state for ASHA / resume
    training_state = {
        'model': unwrapped.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'smoothed_recon': smoothed_recon,
        'best_smoothed_recon': best_smoothed_recon,
        'metrics_log': metrics_log,
        'total_samples': _perf_sample_count,
    }
    state_path = os.path.join(args.output_dir, "training_state.pt")
    torch.save(training_state, state_path)
    print(f"Training state saved to {state_path}")

    save_reconstruction(batch, os.path.join(args.output_dir, "final_reconstruction.png"))
    print(f"Final reconstruction saved to {args.output_dir}/final_reconstruction.png")

if __name__ == "__main__":
    train()