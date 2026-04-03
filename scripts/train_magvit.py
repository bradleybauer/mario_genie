import argparse
import concurrent.futures
import copy
import gc
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from torchvision.utils import save_image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR

from rich.progress import (
    Progress, BarColumn, MofNCompleteColumn, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from mario_world_model.config import IMAGE_SIZE
from mario_world_model.losses import focal_cross_entropy, softened_inverse_frequency_weights
from mario_world_model.model_configs import MODEL_CONFIGS, MODEL_CONFIGS_BY_NAME
from mario_world_model.tokenizer_compat import resolve_video_contains_first_frame
from mario_world_model.lfq import swap_tokenizer_lfq
from mario_world_model.gan_discriminator import build_palette_discriminator, count_trainable_parameters
from mario_world_model.palette_tokenizer import PaletteVideoTokenizer
from mario_world_model.system_info import collect_system_info, print_system_info, get_available_memory, get_effective_cpu_count


CHECKPOINTS_DIR = "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/normalized",
        help="Directory containing normalized .npz files and palette.json",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear warmup from 0 to --lr over this many steps (default: 1000)")
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
    parser.add_argument("--tf16", action="store_true",
                        help="Enable TF16")
    parser.add_argument(
        "--onehot-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype for one-hot input tensors (reduces memory vs float32 when using float16/bfloat16).",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Enable CUDA autocast for model forward/decode paths.",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Compute dtype for --autocast.",
    )
    parser.add_argument("--compile", action="store_true",
                        help="Enable graph compilation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-samples", type=int, default=5000,
                        help="Number of held-out samples for end-of-training evaluation (0 to disable)")
    _default_workers = min(get_effective_cpu_count(), 16)
    parser.add_argument("--num-workers", type=int, default=_default_workers,
                        help=f"Number of DataLoader workers (default: {_default_workers})")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name under checkpoints/ (defaults to the selected model name)",
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
        "--threshold", type=float, default=None,
        help=(
            "Stop training once smoothed training recon_loss drops below this value. "
            "Checkpoint is saved before exit."
        ),
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Stop after this many gradient steps. If omitted, defaults to --total-steps.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=0,
        help=(
            "Total steps for the cosine LR schedule. "
            "If omitted, defaults to --max-steps."
        ),
    )
    parser.add_argument(
        "--constant-lr", action="store_true",
        help="Use a constant learning rate (no warmup or cosine decay). "
             "On resume, the scheduler state is not restored.",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=0,
        help=(
            "Save weights and run eval every this many steps (0 = disabled). "
            "Checkpoints are saved as training_state_latest.pt and periodic eval "
            "metrics are appended to metrics.json."
        ),
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to a training_state*.pt file to resume training from.",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0,
                        help="Focal loss gamma (0 = standard cross-entropy, 2 = typical focal)")
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Enable class-weighted reconstruction loss using distribution JSON from --data-dir.",
    )
    parser.add_argument(
        "--class-weights-file",
        type=str,
        default="palette_distribution.json",
        help="Distribution JSON filename in --data-dir containing class counts/probabilities.",
    )
    parser.add_argument(
        "--class-weight-soften",
        type=float,
        default=0.5,
        help="Softening exponent for inverse-frequency weights (0=uniform, 1=full inverse-frequency).",
    )
    parser.add_argument(
        "--lfq-entropy-mode",
        type=str,
        default="legacy",
        choices=["legacy", "factorized"],
        help=(
            "Entropy estimator for LFQ. 'factorized' avoids materializing the "
            "full (tokens x codebook_size) softmax tensor and is much cheaper "
            "for large codebooks."
        ),
    )
    parser.add_argument(
        "--use-gan",
        action="store_true",
        help="Enable adversarial training with a compact 3D discriminator.",
    )
    parser.add_argument(
        "--gan-weight",
        type=float,
        default=0.1,
        help="Generator adversarial loss weight.",
    )
    parser.add_argument(
        "--gan-lr",
        type=float,
        default=2e-4,
        help="Discriminator learning rate.",
    )
    parser.add_argument(
        "--gan-start-step",
        type=int,
        default=0,
        help="Global step to start GAN updates.",
    )
    parser.add_argument(
        "--gan-target-size",
        type=str,
        default="~5m",
        choices=["~10m", "~5m"],
        help="Discriminator size preset.",
    )
    parser.add_argument(
        "--use-lecam",
        action="store_true",
        help="Enable LeCAM regularization for discriminator stabilization.",
    )
    parser.add_argument(
        "--lecam-weight",
        type=float,
        default=0.1,
        help="LeCAM regularization weight added to discriminator loss.",
    )
    parser.add_argument(
        "--lecam-decay",
        type=float,
        default=0.999,
        help="EMA decay for LeCAM running real/fake logits.",
    )
    args = parser.parse_args()

    # Listing configs should work as a lightweight introspection command.
    if args.list_models:
        return args

    if args.max_steps <= 0 and args.total_steps <= 0:
        parser.error("either --max-steps or --total-steps must be set to a positive integer")

    if args.max_steps <= 0 and args.total_steps > 0:
        args.max_steps = args.total_steps
    elif args.total_steps <= 0 and args.max_steps > 0:
        args.total_steps = args.max_steps

    return args


def _format_elapsed(seconds):
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_bytes(num_bytes: int) -> str:
    num_bytes = max(int(num_bytes), 0)
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def indices_to_onehot_reuse(
    indices: torch.Tensor,
    num_classes: int,
    *,
    dtype: torch.dtype = torch.float32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if indices.ndim != 4:
        raise ValueError(f"Expected indices with shape (B, T, H, W), got {tuple(indices.shape)}")
    if indices.dtype != torch.long:
        indices = indices.long()

    expected_shape = (indices.shape[0], num_classes, indices.shape[1], indices.shape[2], indices.shape[3])
    if (
        out is None
        or out.shape != expected_shape
        or out.dtype != dtype
        or out.device != indices.device
    ):
        out = torch.empty(expected_shape, dtype=dtype, device=indices.device)
    out.zero_()
    out.scatter_(1, indices.unsqueeze(1), 1)
    return out


def load_class_weights(
    data_dir: str,
    *,
    num_classes: int,
    filename: str,
    soften: float,
    device: torch.device,
) -> torch.Tensor:
    dist_path = Path(data_dir) / filename
    if not dist_path.is_file():
        raise FileNotFoundError(
            f"Class weight distribution file not found: {dist_path}. "
            "Run scripts/normalize.py first or disable --use-class-weights."
        )

    with dist_path.open() as handle:
        dist = json.load(handle)

    counts_data = dist.get("counts")
    probs_data = dist.get("probabilities")
    if counts_data is not None:
        counts = torch.tensor(counts_data, dtype=torch.float32)
    elif probs_data is not None:
        counts = torch.tensor(probs_data, dtype=torch.float32)
    else:
        raise ValueError(
            f"{dist_path} must contain either 'counts' or 'probabilities'"
        )

    if counts.ndim != 1 or counts.numel() != num_classes:
        raise ValueError(
            f"{dist_path} has {counts.numel()} classes but model expects {num_classes}"
        )

    weights = softened_inverse_frequency_weights(counts, soften=soften)
    return weights.to(device)


def _index_file(file_idx, filepath, seq_len, *, load_frames=False):
    """Index a normalized .npz file.

    Returns (file_idx, frame_count, samples, frame_shape, frames_opt) where
    samples is a list of (file_idx, t_start) tuples and frames_opt is loaded
    frame data when load_frames=True.
    """
    try:
        mmap_mode = None if load_frames else 'r'
        with np.load(filepath, mmap_mode=mmap_mode) as npz:
            frames = np.asarray(npz['frames']) if load_frames else npz['frames']
            if frames.ndim != 3:
                print(f"Skipping {filepath}: expected (N, H, W), got ndim={frames.ndim}")
                return file_idx, 0, [], None, None
            total_t = frames.shape[0]
            frame_shape = tuple(int(dim) for dim in frames.shape[1:])

            if total_t < seq_len:
                return file_idx, 0, [], frame_shape, None
            samples = [(file_idx, t) for t in range(total_t - seq_len + 1)]
            loaded_frames = frames if load_frames else None
            return file_idx, total_t, samples, frame_shape, loaded_frames
    except Exception as e:
        print(f"Skipping {filepath} due to error: {e}")
        return file_idx, 0, [], None, None

class MarioVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        seq_len=4,
        subset_n=0,
        seed=42,
        num_workers=None,
        system_info=None,
    ):
        if num_workers is None:
            num_workers = get_effective_cpu_count(system_info)
        self.data_files = sorted(str(p) for p in Path(data_dir).glob('*.npz'))
        self.seq_len = seq_len
        self.samples = []
        self.num_files = len(self.data_files)
        self.total_frames = 0
        self.dataset_bytes = sum(Path(path).stat().st_size for path in self.data_files)

        available_memory = get_available_memory(system_info)
        headroom = 2 * 2**30  # keep 2 GB free
        # Conservative gate: only pre-load while indexing when disk size is well below RAM budget.
        preload_during_index = (
            available_memory > 0
            and self.dataset_bytes < max(available_memory - headroom, 0) * 0.5
        )
        
        n_files = len(self.data_files)
        print(f"Indexing {n_files} data files (stride=1)...")
        frame_counts = [0] * n_files
        frame_shapes = [None] * n_files
        self.frames_by_file = [None] * n_files if preload_during_index else None
        index_workers = max(num_workers, 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
            futures = {
                pool.submit(_index_file, idx, f, seq_len, load_frames=preload_during_index): idx
                for idx, f in enumerate(self.data_files)
            }
            with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                          BarColumn(), MofNCompleteColumn(),
                          TimeElapsedColumn(), TimeRemainingColumn()) as progress:
                task = progress.add_task("Indexing files", total=n_files)
                for future in concurrent.futures.as_completed(futures):
                    file_idx, total_t, samples, frame_shape, loaded_frames = future.result()
                    frame_counts[file_idx] = total_t
                    frame_shapes[file_idx] = frame_shape
                    if preload_during_index and self.frames_by_file is not None and loaded_frames is not None:
                        self.frames_by_file[file_idx] = loaded_frames
                    self.samples.extend(samples)
                    progress.advance(task)

        # Remove files with no usable frames and remap indices
        valid = [i for i in range(n_files) if frame_counts[i] > 0]
        if len(valid) < n_files:
            skipped = n_files - len(valid)
            print(f"Dropped {skipped} file(s) with no usable frames")
            old_to_new = {old: new for new, old in enumerate(valid)}
            self.data_files = [self.data_files[i] for i in valid]
            frame_counts = [frame_counts[i] for i in valid]
            frame_shapes = [frame_shapes[i] for i in valid]
            self.samples = [(old_to_new[fi], t) for fi, t in self.samples]
            if self.frames_by_file is not None:
                self.frames_by_file = [self.frames_by_file[i] for i in valid]
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
        self.total_frames = total_frames
        if total_frames > 0:
            available = available_memory
            shape_sample = next((shape for shape in frame_shapes if shape is not None), None)
            if shape_sample is None:
                raise RuntimeError("Could not determine frame shape during indexing.")
            bytes_per_frame = math.prod(shape_sample)  # H*W, uint8
            total_bytes = total_frames * bytes_per_frame
            ram_budget = available - headroom

            if self.frames_by_file is not None:
                if available > 0 and total_bytes < ram_budget:
                    actual = sum(f.nbytes for f in self.frames_by_file if f is not None)
                    print(f"Loaded {actual / 2**30:.0f} GB into RAM during indexing (single pass)")
                else:
                    print(
                        f"[index] RAM preload exceeded safe budget ({total_bytes / 2**30:.0f} GB, "
                        f"{available / 2**30:.0f} GB available). Falling back to mmap."
                    )
                    self.frames_by_file = None

            if self.frames_by_file is None and available > 0 and total_bytes < ram_budget:
                print(f"Loading all frames into RAM ({total_bytes / 2**30:.0f} GB, "
                      f"{available / 2**30:.0f} GB available)...")
                self.frames_by_file = [None] * n_files

                def _load_file(idx):
                    return idx, np.load(self.data_files[idx])['frames']

                with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
                    futures = [pool.submit(_load_file, i) for i in range(n_files)]
                    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                                  BarColumn(), MofNCompleteColumn(),
                                  TimeElapsedColumn(), TimeRemainingColumn()) as progress:
                        task = progress.add_task("Loading into RAM", total=n_files)
                        for future in concurrent.futures.as_completed(futures):
                            idx, frames = future.result()
                            self.frames_by_file[idx] = frames
                            progress.advance(task)
                actual = sum(f.nbytes for f in self.frames_by_file)
                print(f"Loaded {actual / 2**30:.0f} GB into RAM (COW-shared with workers)")
            elif self.frames_by_file is None:
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
        chunk = frames[t_start:t_start+self.seq_len]  # (T, H, W) uint8
        return torch.from_numpy(chunk.copy())


def train():
    args = parse_args()

    if args.use_lecam and not args.use_gan:
        raise SystemExit("--use-lecam requires --use-gan")

    if not (0.0 < args.lecam_decay < 1.0):
        raise SystemExit("--lecam-decay must be in (0, 1)")

    def hinge_discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return (torch.relu(1.0 - real_logits) + torch.relu(1.0 + fake_logits)).mean()

    def hinge_generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()

    def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    class LeCAMEMA:
        def __init__(self, decay: float) -> None:
            self.decay = decay
            self.logits_real_ema: torch.Tensor | None = None
            self.logits_fake_ema: torch.Tensor | None = None

        def update(self, real_logit_mean: torch.Tensor, fake_logit_mean: torch.Tensor) -> None:
            real_value = real_logit_mean.detach()
            fake_value = fake_logit_mean.detach()
            if self.logits_real_ema is None or self.logits_fake_ema is None:
                self.logits_real_ema = real_value
                self.logits_fake_ema = fake_value
                return

            self.logits_real_ema = self.decay * self.logits_real_ema + (1.0 - self.decay) * real_value
            self.logits_fake_ema = self.decay * self.logits_fake_ema + (1.0 - self.decay) * fake_value

        def regularizer(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
            if self.logits_real_ema is None or self.logits_fake_ema is None:
                return real_logits.new_zeros(())

            reg_real = torch.relu(real_logits - self.logits_fake_ema).pow(2).mean()
            reg_fake = torch.relu(self.logits_real_ema - fake_logits).pow(2).mean()
            return reg_real + reg_fake

        def state_dict(self) -> dict[str, torch.Tensor | float | None]:
            return {
                "decay": self.decay,
                "logits_real_ema": self.logits_real_ema,
                "logits_fake_ema": self.logits_fake_ema,
            }

        def load_state_dict(self, state: dict[str, torch.Tensor | float | None]) -> None:
            self.decay = float(state.get("decay", self.decay))
            self.logits_real_ema = state.get("logits_real_ema")
            self.logits_fake_ema = state.get("logits_fake_ema")

    magvit_configs = [c for c in MODEL_CONFIGS if c.model_type == "magvit2"]
    magvit_configs_by_name = {c.name: c for c in magvit_configs}

    if args.list_models:
        for name in sorted(magvit_configs_by_name):
            print(name)
        return

    if args.model is None:
        mc = random.choice(magvit_configs)
        args.model = mc.name
        print(f"[random] Selected model: {args.model}")
    elif args.model not in magvit_configs_by_name:
        if args.model in MODEL_CONFIGS_BY_NAME:
            raise SystemExit(f"Model {args.model!r} is not a magvit2 model. Use scripts/train_genie2.py instead.")
        raise SystemExit(f"Unknown model {args.model!r}. Use --list-models to see available configs.")
    else:
        mc = magvit_configs_by_name[args.model]

    if args.run_name is None:
        args.run_name = args.model

    args.output_dir = os.path.join(CHECKPOINTS_DIR, args.run_name)

    seq_len = mc.sequence_length
    ctx_frames = mc.context_frames
    total_frames = seq_len + ctx_frames

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

    # Restore batch_size from previous run's config on resume (only if not
    # explicitly provided on the command line).
    if args.resume_from:
        prev_config_path = os.path.join(os.path.dirname(args.resume_from), "config.json")
        if os.path.exists(prev_config_path):
            with open(prev_config_path) as f:
                prev_config = json.load(f)
            if "batch_size" in prev_config and args.batch_size is None:
                args.batch_size = prev_config["batch_size"]
                print(f"[resume] Restored batch_size={args.batch_size} from {prev_config_path}")

    # Apply default batch size if not set by CLI or resume
    if args.batch_size is None:
        args.batch_size = 4

    # ── Normal training ──────────────────────────────────────────────────

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # Run directory
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── System info ──────────────────────────────────────────────────
    system_info = collect_system_info()
    print_system_info(system_info)

    dataset = MarioVideoDataset(
        args.data_dir, seq_len=total_frames,
        subset_n=args.overfit_n,
        seed=args.seed,
        num_workers=args.num_workers,
        system_info=system_info,
    )
    print(
        f"Found {len(dataset)} sequence segments of {total_frames} frames "
        f"({ctx_frames} context + {seq_len} target, image_size={IMAGE_SIZE})."
    )
    print(
        f"Dataset: {dataset.num_files} files, {len(dataset):,} samples, "
        f"{dataset.total_frames:,} frames, {_format_bytes(dataset.dataset_bytes)} on disk."
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
        palette_info = json.load(f)
    palette = torch.tensor(palette_info["colors_rgb"], dtype=torch.float32) / 255.0  # (K, 3)
    num_palette_colors = palette.shape[0]
    print(f"[palette] Loaded {num_palette_colors} colours from {palette_path}")
    class_weight = None
    if args.use_class_weights:
        class_weight = load_class_weights(
            args.data_dir,
            num_classes=num_palette_colors,
            filename=args.class_weights_file,
            soften=args.class_weight_soften,
            device=device,
        )
        print(
            "[class-weights] Enabled from "
            f"{Path(args.data_dir) / args.class_weights_file} "
            f"(soften={args.class_weight_soften:.2f}, "
            f"min={class_weight.min().item():.3f}, max={class_weight.max().item():.3f})"
        )

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
        if args.overfit_n > 0:
            sampler = None
        else:
            sampler = RandomSampler(dataset, replacement=True, num_samples=10**7)
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

    tokenizer = PaletteVideoTokenizer(
        num_palette_colors=num_palette_colors,
        image_size=IMAGE_SIZE,
        init_dim=mc.init_dim,
        codebook_size=mc.codebook_size,
        num_codebooks=mc.num_codebooks,
        layers=tokenizer_layers,
        # quantizer_aux_loss_weight=.2
    ).to(device)
    # Drop unused discriminator (use_gan=False, but upstream still creates 52M+ params)
    tokenizer.discr = None
    tokenizer.multiscale_discrs = None
    if args.lfq_entropy_mode != "legacy":
        swap_tokenizer_lfq(tokenizer, entropy_mode=args.lfq_entropy_mode)
        print(f"[lfq] Using {args.lfq_entropy_mode} entropy estimator")

    discriminator = None
    discriminator_optimizer = None
    lecam_ema = None
    discriminator_num_parameters = 0
    if args.use_gan:
        discriminator = build_palette_discriminator(
            num_palette_colors,
            target_size=args.gan_target_size,
        ).to(device)
        discriminator_num_parameters = count_trainable_parameters(discriminator)
        discriminator_optimizer = AdamW(discriminator.parameters(), lr=args.gan_lr)
        if args.use_lecam:
            lecam_ema = LeCAMEMA(decay=args.lecam_decay)
        print(
            f"[gan] Enabled discriminator {args.gan_target_size} "
            f"({discriminator_num_parameters:,} params), "
            f"gan_weight={args.gan_weight}, gan_lr={args.gan_lr:.2e}, "
            f"gan_start_step={args.gan_start_step}, "
            f"lecam={'on' if args.use_lecam else 'off'}"
        )

    palette = palette.to(device)
    video_contains_first_frame = resolve_video_contains_first_frame(tokenizer, total_frames)

    # Cap batch size to unique samples (duplicates in a batch waste compute)
    if args.batch_size > num_unique_samples:
        print(f"Capped batch_size {args.batch_size} -> {num_unique_samples} (unique samples)")
        args.batch_size = num_unique_samples

    dataloader = make_dataloader(args.batch_size) if gpu_cache is None else None

    # ── Float Precision ─────────────────────────────────────────────────────────
    if args.tf16:
        torch.set_float32_matmul_precision('medium')
        print("[tf16] TF16 matmul precision enabled")
    elif torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("[tf32] TF32 matmul precision enabled")

    onehot_dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    onehot_dtype = onehot_dtype_map[args.onehot_dtype]
    onehot_element_size = torch.tensor([], dtype=onehot_dtype).element_size()

    autocast_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    autocast_enabled = bool(args.autocast and device.type == "cuda")
    autocast_dtype = autocast_dtype_map[args.autocast_dtype]
    if args.autocast and device.type != "cuda":
        print("[autocast] Requested but CUDA is unavailable; disabling autocast.")
    if autocast_enabled and autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("[autocast] bfloat16 unsupported on this GPU; falling back to float16.")
        autocast_dtype = torch.float16
    if autocast_enabled:
        label = "bf16" if autocast_dtype == torch.bfloat16 else "fp16"
        print(f"[autocast] Enabled ({label})")

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and autocast_dtype == torch.float16,
    )

    def autocast_context():
        if not autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    # ── Count parameters ──
    num_parameters = sum(p.numel() for p in tokenizer.parameters())

    optimizer = AdamW(tokenizer.parameters(), lr=args.lr)

    # ── LR schedule ──────────────────────────────────────────────────
    if args.constant_lr:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        print(f"[lr] Constant LR = {args.lr}")
    else:
        # Linear warmup → cosine decay to 25 % of peak
        min_lr = args.lr * 0.25
        warmup_steps = args.warmup_steps

        # total_steps is guaranteed positive by argument validation.
        estimated_total = max(args.total_steps, warmup_steps + 100)
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
    config["sequence_length"] = seq_len
    config["context_frames"] = ctx_frames
    config["total_frames"] = total_frames
    config["layers"] = [list(l) if isinstance(l, tuple) else l for l in tokenizer_layers]
    config["git_hash"] = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or None
    config["timestamp"] = datetime.now().isoformat()
    config["num_parameters"] = num_parameters
    config["discriminator_parameters"] = discriminator_num_parameters
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
        if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
            if "discriminator" in ckpt and "discriminator_optimizer" in ckpt:
                discriminator.load_state_dict(ckpt["discriminator"])
                discriminator_optimizer.load_state_dict(ckpt["discriminator_optimizer"])
            else:
                print("[resume] Checkpoint has no discriminator state; GAN starts from scratch.")
            if args.use_lecam and lecam_ema is not None:
                if "lecam_ema" in ckpt:
                    lecam_ema.load_state_dict(ckpt["lecam_ema"])
                else:
                    print("[resume] Checkpoint has no LeCAM EMA state; LeCAM EMA resets.")
        if not args.constant_lr:
            try:
                scheduler.load_state_dict(ckpt['scheduler'])
            except (KeyError, TypeError):
                print("[resume] Scheduler state incompatible (likely changed schedule type), rebuilding from current step")
        else:
            # Rebuild the constant scheduler *after* optimizer restore so it
            # locks in the checkpoint's last LR as its base_lr.
            # If --lr differs from the checkpoint's LR, honour --lr as an
            # explicit override; otherwise keep the checkpoint value.
            resumed_lr = optimizer.param_groups[0]['lr']
            target_lr = args.lr if args.lr != resumed_lr else resumed_lr
            for pg in optimizer.param_groups:
                pg['lr'] = target_lr
                pg['initial_lr'] = target_lr
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            print(f"[resume] Constant LR = {target_lr:.6e} (checkpoint was {resumed_lr:.6e})")
        start_step = ckpt['global_step']
        print(f"[resume] Loaded training state from {args.resume_from} at step {start_step}")

        if args.max_steps > 0 and start_step >= args.max_steps:
            print(f"[max-steps] Already at step {start_step} >= {args.max_steps}. Nothing to do.")
            return

    # ── Torch Compile ─────────────────────────────────────────────────────────
    if args.compile:
        print("[compile] Compiling the model with torch.compile()…")
        tokenizer = torch.compile(tokenizer)
        if discriminator is not None:
            print("[compile] Compiling discriminator with torch.compile()...")
            discriminator = torch.compile(discriminator)
        print("[compile] Compilation complete.")

    # ── Perf tracking ────────────────────────────────────────────────
    _perf_step_count = 0
    _perf_sample_count = 0
    _total_sample_count = 0
    _perf_window_start = time.time()
    _perf_window_steps = 0
    _perf_window_samples = 0
    _PERF_WINDOW = 20  # rolling window size in steps
    # Bytes per sample for throughput: one-hot input (B, K, T, H, W) in selected dtype
    _bytes_per_sample = num_palette_colors * total_frames * IMAGE_SIZE * IMAGE_SIZE * onehot_element_size

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
    best_eval_recon = float('inf')
    last_improvement_time = time.time()
    last_image_time = 0.0

    if args.resume_from and ckpt is not None:
        smoothed_recon = ckpt.get('smoothed_recon')
        best_smoothed_recon = ckpt.get('best_smoothed_recon', float('inf'))
        best_eval_recon = ckpt.get('best_eval_recon', float('inf'))
        metrics_log = ckpt.get('metrics_log', [])
        _total_sample_count = ckpt.get('total_samples', 0)

    def save_reconstruction(batch, path):
        """Encode → decode a batch and save side-by-side comparison image.

        Returns (codes, codebook_usage).
        """
        tokenizer.eval()
        with torch.no_grad():
            inp = PaletteVideoTokenizer.indices_to_onehot(
                batch.long().to(device),
                num_palette_colors,
                dtype=onehot_dtype,
            )
            with autocast_context():
                codes = tokenizer(inp, return_codes=True, video_contains_first_frame=video_contains_first_frame)
                recon_video = tokenizer.decode_from_code_indices(
                    codes,
                    video_contains_first_frame=video_contains_first_frame,
                )
            # recon has fewer temporal frames (conv_in eats conv_in_time_pad)
            unwrapped_model = getattr(tokenizer, '_orig_mod', tokenizer)
            time_pad = getattr(unwrapped_model, 'conv_in_time_pad', 0)
            original_rgb = palette[batch[0, time_pad:].long()]
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

    def run_eval(eval_ds, step_label):
        """Run eval pass on eval_ds. Returns dict of eval metrics."""
        eval_loader = DataLoader(
            eval_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False,
        )
        tokenizer.eval()
        eval_losses = []
        code_counts = torch.zeros(mc.codebook_size, dtype=torch.long)
        total_pixels = 0
        correct_pixels = 0
        last_eval_batch = None
        eval_progress = Progress(
            SpinnerColumn(), TextColumn("{task.description}"),
            BarColumn(), MofNCompleteColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(),
            transient=True,
        )
        eval_progress.start()
        eval_task = eval_progress.add_task(f"Eval@{step_label}", total=len(eval_loader))
        onehot_buffer = None
        with torch.no_grad():
            for eval_batch in eval_loader:
                eval_batch = eval_batch.to(device, non_blocking=True)
                last_eval_batch = eval_batch
                targets = eval_batch.long()
                inp = indices_to_onehot_reuse(
                    targets,
                    num_palette_colors,
                    dtype=onehot_dtype,
                    out=onehot_buffer,
                )
                onehot_buffer = inp
                with autocast_context():
                    codes = tokenizer(inp, return_codes=True, video_contains_first_frame=video_contains_first_frame)
                code_counts += torch.bincount(codes.cpu().flatten(), minlength=mc.codebook_size)
                with autocast_context():
                    recon_video = tokenizer.decode_from_code_indices(
                        codes, video_contains_first_frame=video_contains_first_frame,
                    )
                del codes, inp
                # Mask context frames from eval metrics.
                # conv_in eats conv_in_time_pad frames (no temporal F.pad),
                # so recon has that many fewer frames than the input.
                unwrapped = getattr(tokenizer, '_orig_mod', tokenizer)
                time_pad = getattr(unwrapped, 'conv_in_time_pad', 0)
                skip_recon = max(ctx_frames - time_pad, 0) if ctx_frames > 0 else 0
                eval_recon = recon_video[:, :, skip_recon:].float()
                eval_tgt = targets[:, ctx_frames:]
                eval_losses.append(
                    focal_cross_entropy(
                        eval_recon,
                        eval_tgt,
                        gamma=args.focal_gamma,
                        class_weight=class_weight,
                    ).item()
                )
                recon_idx = eval_recon.argmax(dim=1)
                total_pixels += eval_tgt.numel()
                correct_pixels += (recon_idx == eval_tgt).sum().item()
                del recon_video, recon_idx, targets
                eval_progress.advance(eval_task)
        eval_progress.stop()

        eval_recon_loss = sum(eval_losses) / len(eval_losses)
        pixel_accuracy = correct_pixels / max(total_pixels, 1)
        unique_codes_n = (code_counts > 0).sum().item()
        codebook_utilization = unique_codes_n / mc.codebook_size
        code_probs = code_counts[code_counts > 0].float() / code_counts.sum()
        code_entropy = -(code_probs * code_probs.log2()).sum().item()
        max_entropy = math.log2(mc.codebook_size)

        print(f"Eval recon_loss: {eval_recon_loss:.6f} ({len(eval_ds)} samples, {len(eval_losses)} batches)")
        print(f"Eval pixel_accuracy: {pixel_accuracy:.4f}")
        print(f"Eval codebook: {unique_codes_n}/{mc.codebook_size} codes used "
              f"({codebook_utilization:.1%}), entropy={code_entropy:.2f}/{max_entropy:.2f} bits")

        tokenizer.train()
        return {
            "eval_recon_loss": eval_recon_loss,
            "eval_pixel_accuracy": round(pixel_accuracy, 6),
            "eval_codebook_used": unique_codes_n,
            "eval_codebook_size": mc.codebook_size,
            "eval_codebook_utilization": round(codebook_utilization, 4),
            "eval_code_entropy_bits": round(code_entropy, 4),
            "eval_max_entropy_bits": round(max_entropy, 4),
        }, last_eval_batch, code_counts

    def batch_iter():
        """Yields batches: GPU cache > DataLoader."""
        bs = args.batch_size
        if gpu_cache is not None:
            n = gpu_cache.shape[0]
            while True:
                for i in range(0, n, bs):
                    yield gpu_cache[i:min(i + bs, n)]
        else:
            while True:
                yield from dataloader

    pbar_total = args.max_steps if args.max_steps > 0 else None
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
    )
    progress.start()
    train_task = progress.add_task("Training", total=pbar_total, completed=start_step, status="")

    global_step = start_step
    consecutive_ooms = 0
    gan_discr_loss_value = 0.0
    gan_gen_loss_value = 0.0
    gan_real_logit_value = 0.0
    gan_fake_logit_value = 0.0
    gan_lecam_reg_value = 0.0
    gan_lecam_ema_real_value = 0.0
    gan_lecam_ema_fake_value = 0.0
    onehot_buffer = None
    for batch in batch_iter():
        if args.max_steps > 0 and global_step >= args.max_steps:
            print(f"\n[max-steps] Reached {global_step} steps. Stopping.")
            break

        if gpu_cache is None:
            batch = batch.to(device, non_blocking=True)

        gan_active = args.use_gan and global_step >= args.gan_start_step
        gan_discr_loss_value = 0.0
        gan_gen_loss_value = 0.0
        gan_real_logit_value = 0.0
        gan_fake_logit_value = 0.0
        gan_lecam_reg_value = 0.0
        gan_lecam_ema_real_value = 0.0
        gan_lecam_ema_fake_value = 0.0

        optimizer.zero_grad()
        try:
            targets = batch.long()
            inp = indices_to_onehot_reuse(
                targets, num_palette_colors,
                dtype=onehot_dtype,
                out=onehot_buffer,
            )
            onehot_buffer = inp

            with autocast_context():
                loss, loss_breakdown, recon_logits = tokenizer(
                    inp,
                    targets=targets,
                    return_loss=True,
                    video_contains_first_frame=video_contains_first_frame,
                    context_frames=ctx_frames,
                    focal_gamma=args.focal_gamma,
                    class_weight=class_weight,
                )

            # Save detached fakes for discriminator update (before graph is freed).
            fake_video_detached = None
            if gan_active and discriminator is not None:
                # --- Generator adversarial loss (live graph through discriminator) ---
                set_requires_grad(discriminator, False)
                with autocast_context():
                    fake_video = recon_logits.softmax(dim=1)
                    gen_adv_loss = hinge_generator_loss(discriminator(fake_video))
                fake_video_detached = fake_video.detach()
                gan_gen_loss_value = gen_adv_loss.item()
                loss = loss + args.gan_weight * gen_adv_loss
                del fake_video, gen_adv_loss

            # Generator backward (frees computation graph before discriminator update).
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            inp = targets = batch = loss = loss_breakdown = fake_video_detached = None
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
            progress.advance(train_task)
            global_step += 1
            continue
        consecutive_ooms = 0
        if grad_scaler.is_enabled():
            grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
        if grad_scaler.is_enabled():
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        # --- Discriminator update (after generator graph is freed) ---
        if gan_active and discriminator is not None and discriminator_optimizer is not None and fake_video_detached is not None:
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                real_scores = discriminator(inp)
                fake_scores = discriminator(fake_video_detached)
                discr_loss = hinge_discriminator_loss(real_scores, fake_scores)
                if args.use_lecam and lecam_ema is not None:
                    lecam_reg = lecam_ema.regularizer(real_scores, fake_scores)
                    gan_lecam_reg_value = lecam_reg.item()
                    discr_loss = discr_loss + args.lecam_weight * lecam_reg
            discr_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discriminator_optimizer.step()

            if args.use_lecam and lecam_ema is not None:
                lecam_ema.update(real_scores.mean(), fake_scores.mean())
                if lecam_ema.logits_real_ema is not None:
                    gan_lecam_ema_real_value = float(lecam_ema.logits_real_ema.item())
                if lecam_ema.logits_fake_ema is not None:
                    gan_lecam_ema_fake_value = float(lecam_ema.logits_fake_ema.item())

            gan_discr_loss_value = discr_loss.item()
            gan_real_logit_value = real_scores.mean().item()
            gan_fake_logit_value = fake_scores.mean().item()
            del real_scores, fake_scores, discr_loss, fake_video_detached

        current_lr = optimizer.param_groups[0]['lr']
        recon = loss_breakdown.recon_loss.item()
        aux_loss = loss_breakdown.lfq_aux_loss.item()

        # ── Perf tracking ────────────────────────────────────────────
        _perf_step_count += 1
        _perf_sample_count += batch.shape[0]
        _total_sample_count += batch.shape[0]
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
        if gan_active:
            postfix['g_adv'] = f"{gan_gen_loss_value:.4f}"
            postfix['d'] = f"{gan_discr_loss_value:.4f}"
            if args.use_lecam:
                postfix['lecam'] = f"{gan_lecam_reg_value:.4f}"
        gs = _gpu_stats()
        if 'gpu_util_pct' in gs:
            postfix['gpu'] = f"{gs['gpu_util_pct']}%"
        if 'gpu_mem_pct' in gs:
            postfix['mem'] = f"{gs['gpu_mem_pct']:.0f}%"
        if 'gpu_temp_c' in gs:
            postfix['temp'] = f"{gs['gpu_temp_c']}C"
        progress.update(train_task, advance=1,
                       status=" ".join(f"{k}={v}" for k, v in postfix.items()))

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
            qlb = loss_breakdown.quantizer_loss_breakdown
            step_metrics = {
                "step": global_step,
                "loss": loss.item(),
                "recon_loss": recon,
                "lfq_aux_loss": aux_loss,
                "lfq_per_sample_entropy": qlb.per_sample_entropy.item() if qlb is not None else 0.0,
                "lfq_batch_entropy": qlb.batch_entropy.item() if qlb is not None else 0.0,
                "lfq_commitment": qlb.commitment.item() if qlb is not None else 0.0,
                "smoothed_recon_loss": smoothed_recon,
                "best_smoothed_recon": best_smoothed_recon,
                "lr": current_lr,
                "elapsed_s": round(time.time() - train_start, 2),
                "samples_per_sec": round(_samples_per_sec, 1),
                "data_throughput_mbps": round(_samples_per_sec * _bytes_per_sample / 2**20, 1),
                "steps_per_sec": round(_steps_per_sec, 2),
                "total_samples": _total_sample_count,
                "gan_enabled": bool(gan_active),
                "gan_generator_loss": float(gan_gen_loss_value),
                "gan_discriminator_loss": float(gan_discr_loss_value),
                "gan_real_logit": float(gan_real_logit_value),
                "gan_fake_logit": float(gan_fake_logit_value),
                "gan_lecam_reg": float(gan_lecam_reg_value),
                "gan_lecam_ema_real": float(gan_lecam_ema_real_value),
                "gan_lecam_ema_fake": float(gan_lecam_ema_fake_value),
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

        # ── Periodic checkpoint & eval ───────────────────────────────
        if (args.checkpoint_interval > 0
                and global_step > 0
                and global_step % args.checkpoint_interval == 0):
            print(f"\n[checkpoint] Saving latest weights at step {global_step}")
            unwrapped = getattr(tokenizer, '_orig_mod', tokenizer)
            _ckpt_sd = unwrapped.state_dict()
            torch.save(_ckpt_sd, os.path.join(args.output_dir, "magvit2_latest.pt"))
            _training_state = {
                'model': _ckpt_sd,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'global_step': global_step,
                'smoothed_recon': smoothed_recon,
                'best_smoothed_recon': best_smoothed_recon,
                'best_eval_recon': best_eval_recon,
                'metrics_log': metrics_log,
                'total_samples': _total_sample_count,
            }
            if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
                _training_state['discriminator'] = discriminator.state_dict()
                _training_state['discriminator_optimizer'] = discriminator_optimizer.state_dict()
                if args.use_lecam and lecam_ema is not None:
                    _training_state['lecam_ema'] = lecam_ema.state_dict()
            torch.save(_training_state, os.path.join(args.output_dir, "training_state_latest.pt"))

            if eval_dataset is not None:
                eval_stats, _, _ = run_eval(eval_dataset, global_step)
                # Append eval stats to the most recent metrics entry
                if metrics_log and metrics_log[-1].get("step") == global_step:
                    metrics_log[-1].update(eval_stats)
                else:
                    eval_entry = {"step": global_step, "elapsed_s": round(time.time() - train_start, 2)}
                    eval_entry.update(eval_stats)
                    metrics_log.append(eval_entry)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_log, f, indent=2)

                # Save best checkpoint if eval loss improved
                if eval_stats["eval_recon_loss"] < best_eval_recon:
                    best_eval_recon = eval_stats["eval_recon_loss"]
                    torch.save(_ckpt_sd, os.path.join(args.output_dir, "magvit2_best.pt"))
                    _training_state['best_eval_recon'] = best_eval_recon
                    torch.save(_training_state, os.path.join(args.output_dir, "training_state_best.pt"))
                    print(f"[checkpoint] New best eval_recon_loss={best_eval_recon:.6f} → saved *_best.pt")

            del _ckpt_sd, _training_state

        global_step += 1

    progress.stop()

    # Always record the final state so sweep readers see the terminal smoothed_recon
    total_elapsed = time.time() - train_start

    # If no training steps ran (e.g. resumed at max_steps), skip final bookkeeping
    if _perf_step_count == 0:
        print("No training steps executed. Exiting.")
        return

    gs = _gpu_stats()
    qlb = loss_breakdown.quantizer_loss_breakdown
    final_metrics = {
        "step": global_step,
        "loss": loss.item(),
        "recon_loss": recon,
        "lfq_aux_loss": aux_loss,
        "lfq_per_sample_entropy": qlb.per_sample_entropy.item() if qlb is not None else 0.0,
        "lfq_batch_entropy": qlb.batch_entropy.item() if qlb is not None else 0.0,
        "lfq_commitment": qlb.commitment.item() if qlb is not None else 0.0,
        "smoothed_recon_loss": smoothed_recon,
        "best_smoothed_recon": best_smoothed_recon,
        "lr": optimizer.param_groups[0]['lr'],
        "elapsed_s": round(total_elapsed, 2),
        "samples_per_sec": round(_perf_sample_count / max(total_elapsed, 1e-9), 1),
        "data_throughput_mbps": round(_perf_sample_count / max(total_elapsed, 1e-9) * _bytes_per_sample / 2**20, 1),
        "steps_per_sec": round(_perf_step_count / max(total_elapsed, 1e-9), 2),
        "total_samples": _total_sample_count,
        "total_steps": global_step,
        "gan_enabled": bool(args.use_gan and global_step >= args.gan_start_step),
        "gan_generator_loss": float(gan_gen_loss_value),
        "gan_discriminator_loss": float(gan_discr_loss_value),
        "gan_real_logit": float(gan_real_logit_value),
        "gan_fake_logit": float(gan_fake_logit_value),
        "gan_lecam_reg": float(gan_lecam_reg_value),
        "gan_lecam_ema_real": float(gan_lecam_ema_real_value),
        "gan_lecam_ema_fake": float(gan_lecam_ema_fake_value),
        **gs,
    }
    if not metrics_log or metrics_log[-1]["step"] != global_step:
        metrics_log.append(final_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)

    # Keep a reference for final reconstruction before cleanup
    final_recon_batch = batch

    # ── Save latest checkpoint & final reconstruction ────────────────
    ckpt_path = os.path.join(args.output_dir, "magvit2_latest.pt")
    unwrapped = getattr(tokenizer, '_orig_mod', tokenizer)
    model_sd = unwrapped.state_dict()
    torch.save(model_sd, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path} ({len(model_sd)} keys)")

    # Full training state for resume
    training_state = {
        'model': model_sd,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'smoothed_recon': smoothed_recon,
        'best_smoothed_recon': best_smoothed_recon,
        'best_eval_recon': best_eval_recon,
        'metrics_log': metrics_log,
        'total_samples': _total_sample_count,
    }
    if args.use_gan and discriminator is not None and discriminator_optimizer is not None:
        training_state['discriminator'] = discriminator.state_dict()
        training_state['discriminator_optimizer'] = discriminator_optimizer.state_dict()
        if args.use_lecam and lecam_ema is not None:
            training_state['lecam_ema'] = lecam_ema.state_dict()
    state_path = os.path.join(args.output_dir, "training_state_latest.pt")
    torch.save(training_state, state_path)
    print(f"Training state saved to {state_path}")
    _optimizer_sd = training_state['optimizer']
    _scheduler_sd = training_state['scheduler']
    _discriminator_sd = training_state.get('discriminator')
    _discriminator_optimizer_sd = training_state.get('discriminator_optimizer')
    _lecam_ema_sd = training_state.get('lecam_ema')
    del training_state, model_sd

    # Free training-only objects before eval to reduce RSS
    del optimizer, scheduler, dataloader
    if args.use_gan and discriminator_optimizer is not None:
        del discriminator_optimizer
    if args.use_gan and discriminator is not None:
        del discriminator
    # Drop training dataset and its mmap handles to release resident pages
    if hasattr(dataset, '_mmap_cache'):
        dataset._mmap_cache.clear()
    if eval_dataset is not None and hasattr(eval_dataset, '_mmap_cache'):
        eval_dataset._mmap_cache.clear()
    del dataset, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Eval on held-out set ──────────────────────────────────────────
    if eval_dataset is not None:
        eval_stats, eval_batch, code_counts = run_eval(eval_dataset, "final")
        final_metrics.update(eval_stats)
        if metrics_log and metrics_log[-1].get("step") == global_step:
            metrics_log[-1].update(eval_stats)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_log, f, indent=2)

        # Save best checkpoint if final eval loss improved
        if eval_stats["eval_recon_loss"] < best_eval_recon:
            best_eval_recon = eval_stats["eval_recon_loss"]
            unwrapped = getattr(tokenizer, '_orig_mod', tokenizer)
            _best_sd = unwrapped.state_dict()
            torch.save(_best_sd, os.path.join(args.output_dir, "magvit2_best.pt"))
            torch.save({
                'model': _best_sd,
                'optimizer': _optimizer_sd,
                'scheduler': _scheduler_sd,
                'global_step': global_step,
                'smoothed_recon': smoothed_recon,
                'best_smoothed_recon': best_smoothed_recon,
                'best_eval_recon': best_eval_recon,
                'metrics_log': metrics_log,
                'total_samples': _total_sample_count,
                **(
                    {
                        'discriminator': _discriminator_sd,
                        'discriminator_optimizer': _discriminator_optimizer_sd,
                        'lecam_ema': _lecam_ema_sd,
                    }
                    if _discriminator_sd is not None and _discriminator_optimizer_sd is not None
                    else {}
                ),
            }, os.path.join(args.output_dir, "training_state_best.pt"))
            del _best_sd
            print(f"[final] New best eval_recon_loss={best_eval_recon:.6f} → saved *_best.pt")

        unique_codes_n = eval_stats["eval_codebook_used"]
        codebook_utilization = eval_stats["eval_codebook_utilization"]
        code_entropy = eval_stats["eval_code_entropy_bits"]
        pixel_accuracy = eval_stats["eval_pixel_accuracy"]
        eval_recon_loss = eval_stats["eval_recon_loss"]

        # Save codebook histogram
        try:
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