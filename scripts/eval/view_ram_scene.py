#!/usr/bin/env python3
"""Run a RAM-video VAE over a random normalized scene and show the latest frame.

The script picks one normalized recording, chooses a random valid sliding-window
offset, runs the RAM->video model on every window from that offset to the end of
the scene, and displays the final reconstructed frame from the final window.

Usage examples:
    python scripts/eval/view_ram_scene.py checkpoints/ram_video_vae_v2_run/latest.pt
    python scripts/eval/view_ram_scene.py checkpoints/ram_video_vae_v2_run --recording "some_scene.npz"
    python scripts/eval/view_ram_scene.py checkpoints/ram_video_vae_v2_run/latest.pt --offset 1000 --output /tmp/latest.png
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.normalized_dataset import load_palette_tensor
from src.data.video_frames import resize_palette_frames
from src.models.ram_video_vae import RAMVideoVAE
from src.models.ram_video_vae_v2 import RAMVideoVAEv2
from src.training.training_utils import load_model_state_dict


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a RAM-video checkpoint (.pt) or a run directory containing latest.pt/best.pt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.json path. Defaults to sibling of the resolved checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing normalized .npz scenes. Defaults to config training.data_dir or data/normalized.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Specific .npz filename to use. Default: choose one at random.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Starting sliding-window offset. Default: choose one at random.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of sliding windows to evaluate per forward pass (default: 64).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for recording/offset selection.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Enable CUDA autocast during inference.",
    )
    parser.add_argument(
        "--sample-posterior",
        action="store_true",
        help="Sample latent noise instead of using posterior mean.",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="side-by-side",
        choices=["recon", "side-by-side"],
        help="Display reconstructed frame only or target+reconstruction (default: side-by-side).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the rendered latest-frame image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window; useful with --output.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.offset is not None and args.offset < 0:
        parser.error("--offset must be >= 0")
    return args


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {resolved}")

    for candidate in (resolved / "latest.pt", resolved / "best.pt"):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No latest.pt or best.pt found in run directory: {resolved}"
    )


def _load_config(checkpoint_path: Path, explicit_config: Path | None) -> dict:
    config_path = explicit_config.resolve() if explicit_config is not None else checkpoint_path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as handle:
        return json.load(handle)


def _choose_recording(data_dir: Path, recording_name: str | None, rng: np.random.Generator) -> Path:
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No normalized .npz files found in {data_dir}")
    if recording_name is None:
        return npz_files[int(rng.integers(len(npz_files)))]

    direct = data_dir / recording_name
    if direct.is_file():
        return direct
    for path in npz_files:
        if path.name == recording_name:
            return path
    raise FileNotFoundError(f"Recording not found: {recording_name}")


def _build_model_from_config(config: dict, *, device: torch.device, num_colors: int):
    model_name = str(config.get("model_name", ""))
    training_cfg = dict(config.get("training", {}))
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))

    n_bytes = int(data_cfg.get("n_bytes", 1365))
    frame_height = int(data_cfg["frame_height"])
    frame_width = int(data_cfg["frame_width"])

    common_kwargs = {
        "n_bytes": n_bytes,
        "num_colors": num_colors,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "hidden_dim": int(model_cfg.get("hidden_dim", training_cfg.get("hidden_dim", 256))),
        "latent_dim": int(model_cfg.get("latent_dim", training_cfg.get("latent_dim", 128))),
        "n_fc_blocks": int(model_cfg.get("n_fc_blocks", training_cfg.get("n_fc_blocks", 2))),
        "n_temporal_blocks": int(model_cfg.get("n_temporal_blocks", training_cfg.get("n_temporal_blocks", 2))),
        "temporal_kernel_size": int(model_cfg.get("temporal_kernel_size", training_cfg.get("temporal_kernel_size", 3))),
        "video_base_channels": int(model_cfg.get("video_base_channels", training_cfg.get("video_base_channels", 24))),
        "video_latent_channels": int(model_cfg.get("video_latent_channels", training_cfg.get("video_latent_channels", 16))),
        "temporal_downsample": int(model_cfg.get("temporal_downsample", training_cfg.get("temporal_downsample", 1))),
    }

    if model_name == "ram_video_vae_v2" or training_cfg.get("model_version") == "v2":
        model = RAMVideoVAEv2(
            **common_kwargs,
            video_adapter_dim=int(model_cfg.get("video_adapter_dim", training_cfg.get("video_adapter_dim", 256))),
            video_adapter_heads=int(model_cfg.get("video_adapter_heads", training_cfg.get("video_adapter_heads", 8))),
            n_ram_groups=int(model_cfg.get("n_ram_groups", training_cfg.get("n_ram_groups", 128))),
            n_video_temporal_blocks=int(model_cfg.get("n_video_temporal_blocks", training_cfg.get("n_video_temporal_blocks", 2))),
            n_video_renderer_blocks=int(model_cfg.get("n_video_renderer_blocks", training_cfg.get("n_video_renderer_blocks", 2))),
        )
    elif model_name in {"ram_video_vae", ""}:
        model = RAMVideoVAE(**common_kwargs)
    else:
        raise ValueError(f"Unsupported model_name in config: {model_name!r}")

    return model.to(device).eval()


def _latest_frame_rgb(
    *,
    model: torch.nn.Module,
    ram: np.ndarray,
    frames: np.ndarray,
    clip_frames: int,
    frame_height: int,
    frame_width: int,
    start_offset: int,
    device: torch.device,
    autocast_enabled: bool,
    sample_posterior: bool,
    palette_rgb: np.ndarray,
    on_step: callable | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    max_start = int(frames.shape[0]) - clip_frames
    if max_start < 0:
        raise ValueError(
            f"Recording has {frames.shape[0]} frames, fewer than clip length {clip_frames}"
        )
    if start_offset > max_start:
        raise IndexError(f"offset must be in [0, {max_start}], got {start_offset}")

    final_start = start_offset
    final_pred_frame: np.ndarray | None = None
    final_target_frame: np.ndarray | None = None

    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if autocast_enabled and device.type == "cuda"
        else nullcontext
    )

    for start_idx in tqdm(
        range(start_offset, max_start + 1),
        desc="Running scene",
        unit="step",
    ):
        ram_window = np.asarray(ram[start_idx:start_idx + clip_frames])
        ram_batch = torch.from_numpy(ram_window[None, ...]).to(device=device, non_blocking=True).float() / 255.0

        with torch.no_grad():
            with autocast_context():
                outputs = model(
                    ram_batch,
                    output_video_shape=(clip_frames, frame_height, frame_width),
                    sample_posterior=sample_posterior,
                )

        pred_indices = outputs.video_logits[0].argmax(dim=0).detach().cpu().numpy()
        final_start = int(start_idx)
        final_pred_frame = palette_rgb[pred_indices[-1]]
        target_indices = resize_palette_frames(
            np.asarray(frames[final_start + clip_frames - 1]),
            target_height=frame_height,
            target_width=frame_width,
        )
        final_target_frame = palette_rgb[target_indices]

        if on_step is not None:
            on_step(
                target_rgb=final_target_frame,
                recon_rgb=final_pred_frame,
                window_start=final_start,
                latest_frame_idx=final_start + clip_frames - 1,
            )

        del outputs, pred_indices, ram_batch

    if final_pred_frame is None or final_target_frame is None:
        raise RuntimeError("No windows were evaluated")

    latest_frame_idx = final_start + clip_frames - 1
    return final_target_frame, final_pred_frame, final_start, latest_frame_idx


def _build_live_view(*, plt, view: str, recording_name: str, frame_height: int, frame_width: int):
    if view == "recon":
        fig, ax = plt.subplots(figsize=(6, 6))
        initial = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        im = ax.imshow(initial, interpolation="nearest")
        ax.axis("off")

        def update(*, target_rgb: np.ndarray, recon_rgb: np.ndarray, window_start: int, latest_frame_idx: int) -> None:
            del target_rgb, window_start
            im.set_data(recon_rgb)
            ax.set_title(f"Reconstruction | {recording_name} | frame {latest_frame_idx}")
            fig.canvas.draw_idle()
            plt.pause(0.001)

        return fig, update

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    initial = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    target_im = axes[0].imshow(initial, interpolation="nearest")
    axes[0].set_title("Target")
    axes[0].axis("off")
    recon_im = axes[1].imshow(initial, interpolation="nearest")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    def update(*, target_rgb: np.ndarray, recon_rgb: np.ndarray, window_start: int, latest_frame_idx: int) -> None:
        target_im.set_data(target_rgb)
        recon_im.set_data(recon_rgb)
        fig.suptitle(f"{recording_name} | latest frame {latest_frame_idx} | window {window_start}")
        fig.canvas.draw_idle()
        plt.pause(0.001)

    return fig, update


def main() -> None:
    args = parse_args()

    backend = "Agg" if args.no_show else "TkAgg"
    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    config = _load_config(checkpoint_path, args.config)

    data_dir = Path(
        args.data_dir
        or config.get("training", {}).get("data_dir")
        or DEFAULT_DATA_DIR
    ).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    palette = load_palette_tensor(data_dir)
    palette_rgb = (palette.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    model = _build_model_from_config(config, device=device, num_colors=int(palette.shape[0]))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise TypeError(f"Checkpoint format is unsupported: {checkpoint_path}")
    load_model_state_dict(model, checkpoint["model"])

    clip_frames = int(config.get("data", {}).get("clip_frames", config.get("training", {}).get("clip_frames", 16)))
    frame_height = int(config.get("data", {}).get("frame_height", 224))
    frame_width = int(config.get("data", {}).get("frame_width", 224))

    recording_path = _choose_recording(data_dir, args.recording, rng)
    fig = None
    on_step = None
    if not args.no_show:
        plt.ion()
        fig, on_step = _build_live_view(
            plt=plt,
            view=args.view,
            recording_name=recording_path.name,
            frame_height=frame_height,
            frame_width=frame_width,
        )
        plt.show(block=False)

    with np.load(recording_path, mmap_mode="r") as npz:
        if "frames" not in npz.files or "ram" not in npz.files:
            raise KeyError(f"{recording_path.name} must contain both 'frames' and 'ram' arrays")
        frames = npz["frames"]
        ram = npz["ram"]

        if frames.shape[0] != ram.shape[0]:
            raise ValueError(
                f"frames and ram length mismatch in {recording_path.name}: {frames.shape[0]} vs {ram.shape[0]}"
            )

        max_start = int(frames.shape[0]) - clip_frames
        if max_start < 0:
            raise ValueError(
                f"{recording_path.name} has only {frames.shape[0]} frames, fewer than clip length {clip_frames}"
            )
        start_offset = args.offset if args.offset is not None else int(rng.integers(max_start + 1))

        target_rgb, recon_rgb, final_start, latest_frame_idx = _latest_frame_rgb(
            model=model,
            ram=ram,
            frames=frames,
            clip_frames=clip_frames,
            frame_height=frame_height,
            frame_width=frame_width,
            start_offset=start_offset,
            device=device,
            autocast_enabled=args.autocast,
            sample_posterior=args.sample_posterior,
            palette_rgb=palette_rgb,
            on_step=on_step,
        )

        scene_frames = int(frames.shape[0])

    if args.no_show:
        if args.view == "recon":
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(recon_rgb, interpolation="nearest")
            ax.axis("off")
            ax.set_title(f"Reconstruction | {recording_path.name} | frame {latest_frame_idx}")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            axes[0].imshow(target_rgb, interpolation="nearest")
            axes[0].set_title("Target")
            axes[0].axis("off")
            axes[1].imshow(recon_rgb, interpolation="nearest")
            axes[1].set_title("Reconstruction")
            axes[1].axis("off")
            fig.suptitle(f"{recording_path.name} | latest frame {latest_frame_idx}")
    else:
        plt.ioff()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Recording: {recording_path.name}")
    print(f"Scene frames: {scene_frames}")
    print(f"Clip frames: {clip_frames}")
    print(f"Random start offset: {start_offset}")
    print(f"Final window start: {final_start}")
    print(f"Latest displayed frame: {latest_frame_idx}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved image to {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()