import argparse
import concurrent.futures
import copy
import gc
import json
import math
import os
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F
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


DEFAULT_IMAGE_SIZE = IMAGE_SIZE
OVERSCAN_CROP_SIZE = 240
CROP_224_SIZE = 224


def _format_elapsed(seconds):
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"




def _crop_chunk(chunk: np.ndarray, crop_size: int) -> np.ndarray:
    """Crop padded 256x256 frames to a centered square size.

    Assumes the preprocessing pipeline padded 240x256 gameplay frames to 256x256,
    so reducing the square size removes equal borders on all sides.
    """
    height, width = chunk.shape[-2:]
    if (height, width) == (crop_size, crop_size):
        return chunk
    if (height, width) != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
        raise ValueError(
            f"crop expects frames of shape 256x256 or {crop_size}x{crop_size}, got {height}x{width}"
        )
    border = (DEFAULT_IMAGE_SIZE - crop_size) // 2
    return chunk[..., border:-border, border:-border]


def _index_file(file_idx, filepath, seq_len):
    """Index a session file.

    Returns (file_idx, frame_count, samples) where samples is a list of
    (file_idx, t_start) tuples.
    """
    try:
        # Try reading frame count from the lightweight .meta.json sidecar
        meta_path = filepath.replace('.npz', '.meta.json')
        total_t = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if 'num_frames' in meta:
                total_t = meta['num_frames']

        if total_t is None:
            # Fallback: open the npz to get the frame count
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
    def __init__(
        self,
        data_dir,
        seq_len=4,
        subset_n=0,
        seed=42,
        num_workers=None,
        system_info=None,
        crop_size=None,
    ):
        if num_workers is None:
            num_workers = get_effective_cpu_count(system_info)
        self.data_files = find_session_files(data_dir)
        self.seq_len = seq_len
        self.samples = []
        self.crop_size = crop_size
        
        n_files = len(self.data_files)
        print(f"Indexing {n_files} data files (stride=1)...")
        frame_counts = [0] * n_files
        index_workers = max(num_workers, 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
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

        self.samples.sort()  # deterministic order regardless of thread completion

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

                with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
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
        if self.crop_size is not None:
            chunk = _crop_chunk(chunk, self.crop_size)
        return torch.from_numpy(chunk.copy())


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
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
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear warmup from 0 to --lr over this many steps (default: 1000)")
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
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 tensor cores for matmul (Ampere+ GPUs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of the dataset")
    parser.add_argument("--eval-samples", type=int, default=256,
                        help="Number of held-out samples for end-of-training evaluation (0 to disable)")
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
    parser.add_argument(
        "--crop-240",
        action="store_true",
        help=(
            "Center-crop preprocessed 256x256 frames to 240x240 by removing 8px "
            "on each edge. Useful for SMB1/SMB3-style overscan cleanup while "
            "preserving the original 240px gameplay height."
        ),
    )
    parser.add_argument(
        "--crop-224",
        action="store_true",
        help=(
            "Center-crop preprocessed 256x256 frames to 224x224 by removing 16px "
            "on each edge. Useful when you want a 14x14 latent grid instead of 15x15."
        ),
    )
    args = parser.parse_args()

    if args.crop_240 and args.crop_224:
        parser.error("Use at most one of --crop-240 and --crop-224.")

    crop_size = None
    if args.crop_240:
        crop_size = OVERSCAN_CROP_SIZE
    elif args.crop_224:
        crop_size = CROP_224_SIZE

    image_size = crop_size or DEFAULT_IMAGE_SIZE

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

    # ── Early exit: already at max_steps (avoid loading data/model) ────
    if args.resume_from and args.max_steps > 0:
        ckpt_meta = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if ckpt_meta.get("global_step", 0) >= args.max_steps:
            print(f"[max-steps] Already at step {ckpt_meta['global_step']} >= {args.max_steps}. Nothing to do.")
            return
        del ckpt_meta

    # Restore batch_size from previous run's config on resume
    if args.resume_from:
        prev_config_path = os.path.join(os.path.dirname(args.resume_from), "config.json")
        if os.path.exists(prev_config_path):
            with open(prev_config_path) as f:
                prev_config = json.load(f)
            if "batch_size" in prev_config:
                args.batch_size = prev_config["batch_size"]
                print(f"[resume] Restored batch_size={args.batch_size} from {prev_config_path}")

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
        crop_size=crop_size,
    )
    print(
        f"Found {len(dataset)} sequence segments of length {SEQUENCE_LENGTH} frames "
        f"(image_size={image_size})."
    )

    # ── Split eval set (deterministic via torch seed) ────────────────
    eval_dataset = None
    if args.eval_samples > 0 and args.overfit_n == 0 and len(dataset) > args.eval_samples:
        eval_rng = torch.Generator()
        eval_rng.manual_seed(args.seed)
        perm = torch.randperm(len(dataset), generator=eval_rng)
        eval_indices = perm[:args.eval_samples].tolist()
        train_indices = perm[args.eval_samples:].tolist()
        eval_dataset = copy.copy(dataset)
        eval_dataset.samples = [dataset.samples[i] for i in eval_indices]
        dataset.samples = [dataset.samples[i] for i in train_indices]
        print(f"Eval split: {len(eval_dataset)} eval, {len(dataset)} train samples")

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
        image_size=image_size,
        init_dim=mc.init_dim,
        codebook_size=mc.codebook_size,
        layers=tokenizer_layers,
    ).to(device)
    # Drop unused discriminator (use_gan=False, but upstream still creates 52M+ params)
    tokenizer.discr = None
    tokenizer.multiscale_discrs = None
    palette = palette.to(device)
    video_contains_first_frame = resolve_video_contains_first_frame(tokenizer, SEQUENCE_LENGTH)

    # ── Auto batch-size ──
    if args.auto_batch_size and device.type == "cuda" and not args.resume_from:
        print("\n[auto-batch] Probing GPU for maximum batch size …")
        ceiling = num_unique_samples
        if args.max_batch_size > 0:
            ceiling = min(ceiling, args.max_batch_size)
        args.batch_size = find_max_batch_size(
            tokenizer,
            image_size=image_size,
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

    # ── Count parameters ──
    num_parameters = sum(p.numel() for p in tokenizer.parameters())

    optimizer = AdamW(tokenizer.parameters(), lr=args.lr)

    # ── LR schedule: linear warmup → cosine decay to 10 % of peak ────
    max_seconds = args.max_minutes * 60 if args.max_minutes > 0 else 0
    min_lr = args.lr * 0.25
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
    config["image_size"] = image_size
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

        if args.max_steps > 0 and start_step >= args.max_steps:
            print(f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do.")
            return

    # ── Perf tracking ────────────────────────────────────────────────
    _perf_step_count = start_step
    _perf_sample_count = 0
    _perf_window_start = time.time()
    _perf_window_steps = 0
    _perf_window_samples = 0
    _PERF_WINDOW = 20  # rolling window size in steps
    # Bytes per sample for throughput: one-hot input (B, K, T, H, W) float32
    _bytes_per_sample = num_palette_colors * SEQUENCE_LENGTH * image_size * image_size * 4

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
            save_image(comparison, path, nrow=1, padding=0)
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
        try:
            targets = batch.long()
            inp = PaletteVideoTokenizer.indices_to_onehot(
                targets, num_palette_colors,
            )
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
            for _v in ('inp', 'targets', 'batch'):
                locals().pop(_v, None)
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
        aux_loss = loss_breakdown.lfq_aux_loss.item()

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
                "lfq_aux_loss": aux_loss,
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

    # If no training steps ran (e.g. resumed at max_steps), skip final bookkeeping
    if _perf_step_count == 0:
        print("No training steps executed. Exiting.")
        return

    gs = _gpu_stats()
    final_metrics = {
        "step": global_step,
        "loss": loss.item(),
        "recon_loss": recon,
        "lfq_aux_loss": aux_loss,
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

    # Keep a reference for final reconstruction before cleanup
    final_recon_batch = batch

    # ── Save checkpoint & final reconstruction ───────────────────────
    ckpt_path = os.path.join(args.output_dir, "magvit2_best.pt")
    unwrapped = getattr(tokenizer, '_orig_mod', tokenizer)
    model_sd = unwrapped.state_dict()
    torch.save(model_sd, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path} ({len(model_sd)} keys)")

    # Full training state for ASHA / resume
    training_state = {
        'model': model_sd,
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
    del training_state, model_sd

    # Free training-only objects before eval to reduce RSS
    del optimizer, scheduler, dataloader
    # Drop training dataset and its mmap handles to release resident pages
    if hasattr(dataset, '_mmap_cache'):
        dataset._mmap_cache.clear()
    if eval_dataset is not None and hasattr(eval_dataset, '_mmap_cache'):
        eval_dataset._mmap_cache.clear()
    del dataset, batch, pbar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Eval on held-out set ──────────────────────────────────────────
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
        )
        tokenizer.eval()
        eval_losses = []
        code_counts = torch.zeros(mc.codebook_size, dtype=torch.long)
        total_pixels = 0
        correct_pixels = 0
        with torch.no_grad():
            for eval_batch in tqdm(eval_loader, desc="Evaluating"):
                eval_batch = eval_batch.to(device, non_blocking=True)
                targets = eval_batch.long()
                inp = PaletteVideoTokenizer.indices_to_onehot(targets, num_palette_colors)
                # Single encode → codes
                codes = tokenizer(inp, return_codes=True, video_contains_first_frame=video_contains_first_frame)
                code_counts += torch.bincount(codes.cpu().flatten(), minlength=mc.codebook_size)
                # Single decode → logits
                recon_video = tokenizer.decode_from_code_indices(
                    codes, video_contains_first_frame=video_contains_first_frame,
                )
                del codes, inp
                # Recon loss (same as PaletteVideoTokenizer's cross-entropy)
                eval_losses.append(F.cross_entropy(recon_video, targets).item())
                # Pixel accuracy
                recon_idx = recon_video.argmax(dim=1)  # (B, T, H, W)
                total_pixels += targets.numel()
                correct_pixels += (recon_idx == targets).sum().item()
                del recon_video, recon_idx, targets

        eval_recon_loss = sum(eval_losses) / len(eval_losses)
        pixel_accuracy = correct_pixels / max(total_pixels, 1)

        # Codebook analysis (from running bincount)
        unique_codes_n = (code_counts > 0).sum().item()
        codebook_utilization = unique_codes_n / mc.codebook_size
        # Entropy of code distribution (bits)
        code_probs = code_counts[code_counts > 0].float() / code_counts.sum()
        code_entropy = -(code_probs * code_probs.log2()).sum().item()
        max_entropy = math.log2(mc.codebook_size)

        print(f"Eval recon_loss: {eval_recon_loss:.6f} ({len(eval_dataset)} samples, {len(eval_losses)} batches)")
        print(f"Eval pixel_accuracy: {pixel_accuracy:.4f}")
        print(f"Eval codebook: {unique_codes_n}/{mc.codebook_size} codes used "
              f"({codebook_utilization:.1%}), entropy={code_entropy:.2f}/{max_entropy:.2f} bits")

        eval_stats = {
            "eval_recon_loss": eval_recon_loss,
            "eval_pixel_accuracy": round(pixel_accuracy, 6),
            "eval_codebook_used": unique_codes_n,
            "eval_codebook_size": mc.codebook_size,
            "eval_codebook_utilization": round(codebook_utilization, 4),
            "eval_code_entropy_bits": round(code_entropy, 4),
            "eval_max_entropy_bits": round(max_entropy, 4),
        }
        final_metrics.update(eval_stats)
        if metrics_log and metrics_log[-1].get("step") == global_step:
            metrics_log[-1].update(eval_stats)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_log, f, indent=2)

        # Save codebook histogram
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            # Full histogram (all code indices)
            ax = axes[0]
            ax.bar(range(mc.codebook_size), code_counts.numpy(), width=1.0, linewidth=0)
            ax.set_xlabel("Code index")
            ax.set_ylabel("Count")
            ax.set_title(f"Code usage ({unique_codes_n}/{mc.codebook_size} used, "
                         f"entropy={code_entropy:.2f} bits)")
            # Log-scale sorted frequency (rank-frequency plot)
            ax = axes[1]
            sorted_counts = code_counts[code_counts > 0].sort(descending=True).values.numpy()
            ax.bar(range(len(sorted_counts)), sorted_counts, width=1.0, linewidth=0)
            ax.set_yscale('log')
            ax.set_xlabel("Rank")
            ax.set_ylabel("Count (log)")
            ax.set_title(f"Rank-frequency (pixel acc={pixel_accuracy:.3f}, "
                         f"recon={eval_recon_loss:.4f})")
            fig.tight_layout()
            hist_path = os.path.join(args.output_dir, "eval_codebook_histogram.png")
            fig.savefig(hist_path, dpi=150)
            plt.close(fig)
            print(f"Codebook histogram saved to {hist_path}")
        except ImportError:
            print("[eval] matplotlib not available, skipping histogram")

        save_reconstruction(
            eval_batch,
            os.path.join(args.output_dir, "eval_reconstruction.png"),
        )
        print(f"Eval reconstruction saved to {args.output_dir}/eval_reconstruction.png")
        tokenizer.train()

        save_reconstruction(eval_batch, os.path.join(args.output_dir, "final_reconstruction.png"))
        print(f"Final reconstruction saved to {args.output_dir}/final_reconstruction.png")
    else:
        save_reconstruction(final_recon_batch, os.path.join(args.output_dir, "final_reconstruction.png"))
        print(f"Final reconstruction saved to {args.output_dir}/final_reconstruction.png")


if __name__ == "__main__":
    train()