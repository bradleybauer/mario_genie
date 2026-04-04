#!/usr/bin/env python3
"""Play an action-conditioned video-latent DiT in real time.

This script:
1) loads a trained video-latent DiT checkpoint,
2) loads the frozen video VAE used by that DiT,
3) seeds latent context from a short emulator warmup,
4) denoises one-step future latents from live controller actions,
5) decodes predicted latents to video frames.

Usage examples:
    python scripts/play_video_latent_dit.py \
      --dit-checkpoint checkpoints/world1/video_latent_dit_best.pt

    python scripts/play_video_latent_dit.py \
      --dit-checkpoint checkpoints/world1/video_latent_dit_best.pt \
      --rom mario --view compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mario_world_model.ltx_video_vae_v2 import LTXVideoVAEv2 as LTXVideoVAE
from mario_world_model.normalized_dataset import load_palette_info
from mario_world_model.video_latent_dit import VideoLatentDiT

import play_nes as nes_play


DEFAULT_SCALE = 3
DEFAULT_FPS = 60
CROP_H = 224
CROP_W = 224


def _nes_to_recorded_action_byte(nes_byte: int) -> int:
    """Convert nes_py button layout to the recorded DataCollector action layout.

    nes_py layout:
        bit 0=A, 1=B, 2=Select, 3=Start, 4=Up, 5=Down, 6=Left, 7=Right
    recorded layout (from .input.npy / actions.json):
        bit 0=Up, 1=Down, 2=Left, 3=Right, 4=Start, 5=Select, 6=B, 7=A
    """
    recorded = 0
    recorded |= ((nes_byte >> 4) & 1) << 0  # Up
    recorded |= ((nes_byte >> 5) & 1) << 1  # Down
    recorded |= ((nes_byte >> 6) & 1) << 2  # Left
    recorded |= ((nes_byte >> 7) & 1) << 3  # Right
    recorded |= ((nes_byte >> 3) & 1) << 4  # Start
    recorded |= ((nes_byte >> 2) & 1) << 5  # Select
    recorded |= ((nes_byte >> 1) & 1) << 6  # B
    recorded |= ((nes_byte >> 0) & 1) << 7  # A
    return recorded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained video-latent DiT.")
    parser.add_argument(
        "--dit-checkpoint",
        type=str,
        required=True,
        help="Path to DiT checkpoint (.pt) from train_video_latent_dit.py",
    )
    parser.add_argument(
        "--dit-config",
        type=str,
        default=None,
        help="Optional DiT config.json path (defaults to sibling of --dit-checkpoint)",
    )

    parser.add_argument("--video-vae-checkpoint", type=str, default=None)
    parser.add_argument("--video-vae-config", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--rom", type=str, default=None)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--view",
        type=str,
        default="imagined",
        choices=["imagined", "compare"],
        help="imagined: model output only | compare: emulator vs model side-by-side",
    )

    parser.add_argument(
        "--window-frames",
        type=int,
        default=None,
        help="History length used for DiT conditioning (defaults to clip_frames from DiT config)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=None,
        help="Number of real emulator frames used to seed latent context (defaults to context_frames from DiT config)",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=8,
        help="Euler denoising steps used per predicted frame",
    )

    parser.add_argument(
        "--warmup-policy",
        type=str,
        default="noop",
        choices=["noop", "live"],
        help="noop: use no-op actions during warmup | live: use controller actions during warmup",
    )

    parser.add_argument(
        "--onehot-dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="One-hot dtype for VAE encode path (defaults to DiT config or bfloat16)",
    )
    parser.add_argument("--autocast", action="store_true", help="Enable CUDA autocast")
    parser.add_argument("--compile", action="store_true", help="Compile DiT with torch.compile")
    parser.add_argument("--tf16", action="store_true", help="Enable lower precision float32 matmul on CUDA")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _resolve_path(value: str | None, *, config_dir: Path | None = None) -> Path | None:
    if value is None:
        return None

    raw = Path(value)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(PROJECT_ROOT / raw)
        if config_dir is not None:
            candidates.append(config_dir / raw)
        if len(raw.parts) > 1 and raw.parts[0] == "checkpoints":
            candidates.append(PROJECT_ROOT / "results" / Path(*raw.parts[1:]))

    # Common transfer case: configs created on a remote host under /root/mario
    # and later copied into a different local workspace root.
    if raw.is_absolute() and raw.parts[:3] == ("/", "root", "mario"):
        rel = raw.relative_to("/root/mario")
        remapped = PROJECT_ROOT / rel
        candidates.append(remapped)
        if len(rel.parts) > 1 and rel.parts[0] == "checkpoints":
            candidates.append(PROJECT_ROOT / "results" / Path(*rel.parts[1:]))

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except PermissionError:
            # Ignore unreadable locations and continue searching other candidates.
            continue
        except OSError:
            continue

    # Fall back to cwd resolution for clearer error messages later.
    return (Path.cwd() / raw).resolve() if not raw.is_absolute() else raw


def _is_readable_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (PermissionError, OSError):
        return False


def _infer_dit_config_path(args: argparse.Namespace) -> Path:
    if args.dit_config is not None:
        return Path(args.dit_config).resolve()

    sibling = Path(args.dit_checkpoint).resolve().parent / "config.json"
    if sibling.is_file():
        return sibling
    raise FileNotFoundError(
        "Could not infer --dit-config from --dit-checkpoint directory; pass --dit-config explicitly."
    )


def _infer_video_vae_paths(
    *,
    args: argparse.Namespace,
    model_cfg: dict[str, Any],
    model_cfg_dir: Path,
) -> tuple[Path, Path]:
    vae_ckpt_val = args.video_vae_checkpoint or model_cfg.get("video_vae_checkpoint")
    vae_cfg_val = args.video_vae_config or model_cfg.get("video_vae_config")

    vae_ckpt = _resolve_path(vae_ckpt_val, config_dir=model_cfg_dir)
    vae_cfg = _resolve_path(vae_cfg_val, config_dir=model_cfg_dir)

    if vae_ckpt is None:
        raise FileNotFoundError(
            "Video VAE checkpoint path is missing. Pass --video-vae-checkpoint or include it in the DiT config."
        )
    # If config is missing or points to an unreadable machine-specific absolute path,
    # fall back to config.json beside the resolved VAE checkpoint.
    if vae_cfg is None or not _is_readable_file(vae_cfg):
        inferred = vae_ckpt.parent / "config.json"
        if _is_readable_file(inferred):
            vae_cfg = inferred
        elif vae_cfg is None:
            raise FileNotFoundError(
                "Video VAE config path is missing. Pass --video-vae-config or ensure DiT config contains video_vae_config."
            )

    if not _is_readable_file(vae_ckpt):
        raise FileNotFoundError(f"Video VAE checkpoint not found: {vae_ckpt}")
    if not _is_readable_file(vae_cfg):
        raise FileNotFoundError(f"Video VAE config not found: {vae_cfg}")

    return vae_ckpt, vae_cfg


def _load_action_remap(data_dir: str | Path, *, num_actions: int) -> tuple[dict[int, int], int]:
    info_path = Path(data_dir) / "actions.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing actions metadata: {info_path}")

    info = _load_json(info_path)
    reduced_to_original = [int(v) for v in info.get("reduced_to_original_value", [])]
    if not reduced_to_original:
        raise ValueError(f"actions.json has empty reduced_to_original_value: {info_path}")

    action_to_index = {orig: idx for idx, orig in enumerate(reduced_to_original)}
    if any(idx >= num_actions for idx in action_to_index.values()):
        raise ValueError(
            f"actions.json contains indices outside model action range [0, {num_actions - 1}]"
        )

    default_index = action_to_index.get(0, 0)
    return action_to_index, default_index


def _build_rgb_lut(palette_rgb: np.ndarray) -> np.ndarray:
    lut = np.full((256, 256, 256), -1, dtype=np.int16)
    for idx, (r, g, b) in enumerate(palette_rgb):
        if lut[r, g, b] == -1:
            lut[r, g, b] = idx
    return lut


def _rgb_to_palette_indices(frame_rgb: np.ndarray, lut: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    idx = lut[frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]]
    missing = idx < 0
    if missing.any():
        unique = np.unique(frame_rgb[missing], axis=0)
        palette_i16 = palette_rgb.astype(np.int16)
        for color in unique:
            diff = palette_i16 - color.astype(np.int16)
            nearest = int(np.argmin(np.sum(diff * diff, axis=1)))
            lut[color[0], color[1], color[2]] = nearest
        idx = lut[frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]]
    return idx.astype(np.int64, copy=False)


def _center_crop_224(frame_rgb: np.ndarray) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if h < CROP_H or w < CROP_W:
        raise ValueError(f"Frame too small for 224x224 crop: {frame_rgb.shape}")
    y0 = (h - CROP_H) // 2
    x0 = (w - CROP_W) // 2
    return frame_rgb[y0:y0 + CROP_H, x0:x0 + CROP_W]


def _frames_to_one_hot(
    frames: torch.Tensor,
    num_colors: int,
    *,
    dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (B, T, H, W), got {tuple(frames.shape)}")
    if frames.dtype != torch.long:
        frames = frames.long()

    expected_shape = (frames.shape[0], num_colors, frames.shape[1], frames.shape[2], frames.shape[3])
    if (
        out is None
        or out.shape != expected_shape
        or out.dtype != dtype
        or out.device != frames.device
    ):
        out = torch.empty(expected_shape, dtype=dtype, device=frames.device)
    out.zero_()
    out.scatter_(1, frames.unsqueeze(1), 1)
    return out


def _load_video_vae(
    *,
    checkpoint_path: Path,
    config_path: Path,
    num_colors: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = _load_json(config_path)

    patch_size = int(cfg.get("patch_size", 4))
    base_channels = int(cfg.get("base_channels", 64))
    latent_channels = int(cfg.get("latent_channels", 64))
    vae_num_colors = int(cfg.get("num_colors", num_colors))

    if vae_num_colors != num_colors:
        raise ValueError(
            f"VAE config expects num_colors={vae_num_colors} but palette has {num_colors}."
        )

    vae: torch.nn.Module = LTXVideoVAE(
        num_colors=vae_num_colors,
        patch_size=patch_size,
        base_channels=base_channels,
        latent_channels=latent_channels,
    )

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    summary = {
        "patch_size": patch_size,
        "base_channels": base_channels,
        "latent_channels": latent_channels,
        "num_colors": vae_num_colors,
    }
    return vae, summary


def _load_dit_model(
    *,
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
    compile_model: bool,
) -> VideoLatentDiT:
    model = VideoLatentDiT(
        latent_channels=int(config["latent_channels"]),
        num_actions=int(config["num_actions"]),
        d_model=int(config.get("d_model", 512)),
        num_layers=int(config.get("num_layers", 12)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        dropout=float(config.get("dropout", 0.0)),
        max_frames=int(config.get("max_frames", 64)),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    if compile_model:
        model = torch.compile(model)

    return model


class LiveDiTWorldModel:
    def __init__(
        self,
        *,
        dit_model: VideoLatentDiT,
        video_vae: torch.nn.Module,
        device: torch.device,
        palette_rgb: np.ndarray,
        onehot_dtype: torch.dtype,
        autocast_enabled: bool,
        autocast_dtype: torch.dtype,
        window_frames: int,
        ode_steps: int,
        action_to_index: dict[int, int],
        default_action_index: int,
    ) -> None:
        self.dit_model = dit_model
        self.video_vae = video_vae
        self.device = device
        self.palette_rgb = palette_rgb
        self.onehot_dtype = onehot_dtype
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = autocast_dtype
        self.window_frames = window_frames
        self.ode_steps = int(ode_steps)
        self.action_to_index = action_to_index
        self.default_action_index = int(default_action_index)
        self.num_actions = int(self.dit_model.action_embed.num_embeddings)

        self.num_colors = int(palette_rgb.shape[0])
        self.rgb_lut = _build_rgb_lut(palette_rgb)
        self.onehot_buffer: torch.Tensor | None = None
        self._unknown_actions_seen: set[int] = set()

        self.latent_history: deque[torch.Tensor] = deque(maxlen=window_frames)
        self.action_history: deque[int] = deque(maxlen=max(window_frames - 1, 1))

    def _encode_action(self, action_byte: int) -> int:
        encoded = self.action_to_index.get(int(action_byte), self.default_action_index)
        if int(action_byte) not in self.action_to_index and int(action_byte) not in self._unknown_actions_seen:
            self._unknown_actions_seen.add(int(action_byte))
            print(
                f"[action-remap] unseen raw action {int(action_byte)}; "
                f"falling back to reduced action index {self.default_action_index}"
            )
        if not (0 <= encoded < self.num_actions):
            raise ValueError(
                f"Reduced action index {encoded} out of range [0, {self.num_actions - 1}]"
            )
        return encoded

    def _denoise_future_segment(self, history_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch, channels, context_frames, height, width = history_latents.shape
        future = torch.randn(
            batch,
            channels,
            1,
            height,
            width,
            device=history_latents.device,
            dtype=history_latents.dtype,
        )

        dt = 1.0 / float(self.ode_steps)
        for step in range(self.ode_steps, 0, -1):
            t_val = (step - 0.5) / float(self.ode_steps)
            t = torch.full((batch,), t_val, device=history_latents.device, dtype=history_latents.dtype)
            model_input = torch.cat((history_latents, future), dim=2)
            with self._autocast_context():
                velocity = self.dit_model(model_input, actions, t)
            velocity_future = velocity[:, :, context_frames:]
            future = future - dt * velocity_future

        return future

    def _autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def reset_with_frame_indices(self, frame_idx: np.ndarray) -> np.ndarray:
        self.latent_history.clear()
        self.action_history.clear()

        latent = self.encode_frame_indices(frame_idx)
        self.latent_history.append(latent)
        return frame_idx

    def preprocess_frame(self, obs_rgb: np.ndarray) -> np.ndarray:
        cropped = _center_crop_224(obs_rgb)
        return _rgb_to_palette_indices(cropped, self.rgb_lut, self.palette_rgb)

    def encode_frame_indices(self, frame_idx: np.ndarray) -> torch.Tensor:
        frames_t = torch.from_numpy(frame_idx).unsqueeze(0).unsqueeze(0).to(self.device, non_blocking=True)
        inputs = _frames_to_one_hot(
            frames_t,
            self.num_colors,
            dtype=self.onehot_dtype,
            out=self.onehot_buffer,
        )
        self.onehot_buffer = inputs

        with torch.no_grad(), self._autocast_context():
            mean, _ = self.video_vae.encode(inputs)
        return mean.float()

    def reset_with_observation(self, obs_rgb: np.ndarray) -> np.ndarray:
        frame_idx = self.preprocess_frame(obs_rgb)
        return self.reset_with_frame_indices(frame_idx)

    def recontext_from_history(self, frame_history: list[np.ndarray], action_history_raw: list[int]) -> None:
        if not frame_history:
            raise ValueError("frame_history must be non-empty")
        if len(action_history_raw) != len(frame_history) - 1:
            raise ValueError(
                f"Expected len(action_history_raw) == len(frame_history)-1, got "
                f"{len(action_history_raw)} and {len(frame_history)}"
            )

        self.reset_with_frame_indices(frame_history[0])
        for raw_action, frame_idx in zip(action_history_raw, frame_history[1:]):
            latent = self.encode_frame_indices(frame_idx)
            self.action_history.append(self._encode_action(int(raw_action)))
            self.latent_history.append(latent)

    def ingest_transition(self, action_byte: int, obs_rgb: np.ndarray) -> np.ndarray:
        frame_idx = self.preprocess_frame(obs_rgb)
        latent = self.encode_frame_indices(frame_idx)
        self.action_history.append(self._encode_action(action_byte))
        self.latent_history.append(latent)
        return frame_idx

    def ready(self, required_frames: int) -> bool:
        return len(self.latent_history) >= required_frames and len(self.action_history) >= required_frames - 1

    def predict_next_frame(self, action_byte: int) -> np.ndarray:
        if not self.latent_history:
            raise RuntimeError("DiT context is empty")

        history_latents = torch.cat(list(self.latent_history), dim=2)
        history_steps = int(history_latents.shape[2])

        history_actions = list(self.action_history)
        if len(history_actions) != history_steps - 1:
            raise RuntimeError(
                f"Action/latent history mismatch: actions={len(history_actions)} latents={history_steps}"
            )

        next_action = self._encode_action(action_byte)
        actions_seq = [self.default_action_index] + history_actions + [next_action]
        actions = torch.tensor(actions_seq, device=self.device, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            next_latent = self._denoise_future_segment(history_latents, actions).float()
            with self._autocast_context():
                logits = self.video_vae.decode(next_latent)

        frame_idx = logits.argmax(dim=1)[0, 0].detach().cpu().numpy()

        self.action_history.append(next_action)
        self.latent_history.append(next_latent)

        return self.palette_rgb[frame_idx]


def _make_display_sizes(scale: int, view: str) -> tuple[int, int, int, int]:
    panel_w = CROP_W * scale
    panel_h = CROP_H * scale
    hud_h = 22
    if view == "compare":
        win_w = panel_w * 2
    else:
        win_w = panel_w
    win_h = panel_h + hud_h
    return win_w, win_h, panel_w, panel_h


def _frame_to_surface(frame_rgb: np.ndarray, out_size: tuple[int, int]) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(np.swapaxes(frame_rgb, 0, 1))
    if surface.get_size() != out_size:
        surface = pygame.transform.scale(surface, out_size)
    return surface


def _warmup_nes_py(
    env: Any,
    controller: nes_play.GamepadController,
    world: LiveDiTWorldModel,
    *,
    initial_obs: np.ndarray,
    warmup_frames: int,
    warmup_policy: str,
) -> dict[str, Any]:
    real_cropped_idx = world.reset_with_observation(initial_obs)
    current_obs = initial_obs
    frame_history: list[np.ndarray] = [real_cropped_idx.copy()]
    action_history: list[int] = []

    while not world.ready(warmup_frames):
        if warmup_policy == "live":
            nes_action = controller.get_action()
        else:
            nes_action = 0
        recorded_action = _nes_to_recorded_action_byte(nes_action)

        obs, _reward, done, _info = env.step(nes_action)
        if done:
            obs = env.reset()
            current_obs = obs
            real_cropped_idx = world.reset_with_observation(obs)
            frame_history = [real_cropped_idx.copy()]
            action_history = []
            continue

        current_obs = obs
        action_history.append(int(recorded_action))
        real_cropped_idx = world.ingest_transition(recorded_action, obs)
        frame_history.append(real_cropped_idx.copy())

    return {
        "real_rgb": world.palette_rgb[real_cropped_idx],
        "obs": current_obs,
        "frame_history": frame_history,
        "action_history": action_history,
    }


def _warmup_retro(
    env: Any,
    controller: nes_play.GamepadController,
    world: LiveDiTWorldModel,
    *,
    initial_obs: np.ndarray,
    warmup_frames: int,
    warmup_policy: str,
) -> dict[str, Any]:
    real_cropped_idx = world.reset_with_observation(initial_obs)
    current_obs = initial_obs
    frame_history: list[np.ndarray] = [real_cropped_idx.copy()]
    action_history: list[int] = []

    while not world.ready(warmup_frames):
        if warmup_policy == "live":
            nes_action = controller.get_action()
        else:
            nes_action = 0
        recorded_action = _nes_to_recorded_action_byte(nes_action)

        obs, _reward, terminated, truncated, _info = env.step(
            nes_play.nes_byte_to_retro_action(nes_action)
        )
        if terminated or truncated:
            obs, _info = env.reset()
            current_obs = obs
            real_cropped_idx = world.reset_with_observation(obs)
            frame_history = [real_cropped_idx.copy()]
            action_history = []
            continue

        current_obs = obs
        action_history.append(int(recorded_action))
        real_cropped_idx = world.ingest_transition(recorded_action, obs)
        frame_history.append(real_cropped_idx.copy())

    return {
        "real_rgb": world.palette_rgb[real_cropped_idx],
        "obs": current_obs,
        "frame_history": frame_history,
        "action_history": action_history,
    }


def _run_nes_py(
    name: str,
    rom_path: str,
    *,
    world: LiveDiTWorldModel,
    scale: int,
    fps: int,
    view: str,
    warmup_frames: int,
    warmup_policy: str,
) -> None:
    from nes_py.nes_env import NESEnv

    env = NESEnv(rom_path)
    obs = env.reset()

    pygame.init()
    win_w, win_h, panel_w, panel_h = _make_display_sizes(scale, view)
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{name} (video-latent DiT)")
    clock = pygame.time.Clock()
    controller = nes_play.GamepadController()
    hud_font = pygame.font.SysFont("monospace", 14, bold=True)

    print(f"Warming up DiT context with {warmup_frames} frame(s)...")
    warmup_state = _warmup_nes_py(
        env,
        controller,
        world,
        initial_obs=obs,
        warmup_frames=warmup_frames,
        warmup_policy=warmup_policy,
    )
    obs = warmup_state["obs"]
    real_rgb = warmup_state["real_rgb"]
    frame_history: deque[np.ndarray] = deque(warmup_state["frame_history"], maxlen=world.window_frames)
    action_history: deque[int] = deque(warmup_state["action_history"], maxlen=max(world.window_frames - 1, 1))

    fps_surface = hud_font.render("-- FPS", True, (100, 255, 100))
    fps_frame = 0

    running = True
    while running:
        recontext_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                recontext_requested = True
            controller.process_event(event)

        if recontext_requested and len(frame_history) >= 2 and len(action_history) == len(frame_history) - 1:
            world.recontext_from_history(list(frame_history), list(action_history))
            print(f"[recontext] rebuilt context from {len(frame_history)} real frame(s)")

        nes_action = controller.get_action()
        recorded_action = _nes_to_recorded_action_byte(nes_action)
        predicted_rgb = world.predict_next_frame(recorded_action)

        obs, _reward, done, _info = env.step(nes_action)
        if done:
            obs = env.reset()
            warmup_state = _warmup_nes_py(
                env,
                controller,
                world,
                initial_obs=obs,
                warmup_frames=warmup_frames,
                warmup_policy=warmup_policy,
            )
            obs = warmup_state["obs"]
            real_rgb = warmup_state["real_rgb"]
            frame_history = deque(warmup_state["frame_history"], maxlen=world.window_frames)
            action_history = deque(warmup_state["action_history"], maxlen=max(world.window_frames - 1, 1))
            predicted_rgb = world.predict_next_frame(0)
        else:
            real_idx = world.preprocess_frame(obs)
            real_rgb = world.palette_rgb[real_idx]
            action_history.append(int(recorded_action))
            frame_history.append(real_idx.copy())

        screen.fill((8, 8, 12))
        if view == "compare":
            real_surface = _frame_to_surface(real_rgb, (panel_w, panel_h))
            pred_surface = _frame_to_surface(predicted_rgb, (panel_w, panel_h))
            screen.blit(real_surface, (0, 0))
            screen.blit(pred_surface, (panel_w, 0))
            label = "REAL | DIT"
        else:
            pred_surface = _frame_to_surface(predicted_rgb, (panel_w, panel_h))
            screen.blit(pred_surface, (0, 0))
            label = "DIT"

        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = hud_font.render(f"{fps_val:.0f} FPS", True, (100, 255, 100))
            fps_frame = 0

        label_surface = hud_font.render(label, True, (220, 220, 220))
        hud_y = panel_h
        screen.blit(label_surface, (4, hud_y + 2))
        screen.blit(fps_surface, (win_w - fps_surface.get_width() - 4, hud_y + 2))

        pygame.display.flip()
        clock.tick(fps)

    env.close()
    pygame.quit()


def _run_retro(
    name: str,
    rom_path: str,
    *,
    world: LiveDiTWorldModel,
    scale: int,
    fps: int,
    view: str,
    warmup_frames: int,
    warmup_policy: str,
) -> None:
    if nes_play.retro is None:
        raise RuntimeError("stable-retro is not installed")

    game_id = nes_play.ensure_retro_game(rom_path)
    env = nes_play.retro.make(
        game_id,
        state=nes_play.retro.State.NONE,
        render_mode=None,
        use_restricted_actions=nes_play.retro.Actions.ALL,
    )
    obs, _info = env.reset()

    pygame.init()
    win_w, win_h, panel_w, panel_h = _make_display_sizes(scale, view)
    screen = pygame.display.set_mode((win_w, win_h), pygame.DOUBLEBUF)
    pygame.display.set_caption(f"{name} (video-latent DiT)")
    clock = pygame.time.Clock()
    controller = nes_play.GamepadController()
    hud_font = pygame.font.SysFont("monospace", 14, bold=True)

    print(f"Warming up DiT context with {warmup_frames} frame(s)...")
    warmup_state = _warmup_retro(
        env,
        controller,
        world,
        initial_obs=obs,
        warmup_frames=warmup_frames,
        warmup_policy=warmup_policy,
    )
    obs = warmup_state["obs"]
    real_rgb = warmup_state["real_rgb"]
    frame_history: deque[np.ndarray] = deque(warmup_state["frame_history"], maxlen=world.window_frames)
    action_history: deque[int] = deque(warmup_state["action_history"], maxlen=max(world.window_frames - 1, 1))

    fps_surface = hud_font.render("-- FPS", True, (100, 255, 100))
    fps_frame = 0

    running = True
    while running:
        recontext_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                recontext_requested = True
            controller.process_event(event)

        if recontext_requested and len(frame_history) >= 2 and len(action_history) == len(frame_history) - 1:
            world.recontext_from_history(list(frame_history), list(action_history))
            print(f"[recontext] rebuilt context from {len(frame_history)} real frame(s)")

        nes_action = controller.get_action()
        recorded_action = _nes_to_recorded_action_byte(nes_action)
        predicted_rgb = world.predict_next_frame(recorded_action)

        obs, _reward, terminated, truncated, _info = env.step(
            nes_play.nes_byte_to_retro_action(nes_action)
        )
        if terminated or truncated:
            obs, _info = env.reset()
            warmup_state = _warmup_retro(
                env,
                controller,
                world,
                initial_obs=obs,
                warmup_frames=warmup_frames,
                warmup_policy=warmup_policy,
            )
            obs = warmup_state["obs"]
            real_rgb = warmup_state["real_rgb"]
            frame_history = deque(warmup_state["frame_history"], maxlen=world.window_frames)
            action_history = deque(warmup_state["action_history"], maxlen=max(world.window_frames - 1, 1))
            predicted_rgb = world.predict_next_frame(0)
        else:
            real_idx = world.preprocess_frame(obs)
            real_rgb = world.palette_rgb[real_idx]
            action_history.append(int(recorded_action))
            frame_history.append(real_idx.copy())

        screen.fill((8, 8, 12))
        if view == "compare":
            real_surface = _frame_to_surface(real_rgb, (panel_w, panel_h))
            pred_surface = _frame_to_surface(predicted_rgb, (panel_w, panel_h))
            screen.blit(real_surface, (0, 0))
            screen.blit(pred_surface, (panel_w, 0))
            label = "REAL | DIT"
        else:
            pred_surface = _frame_to_surface(predicted_rgb, (panel_w, panel_h))
            screen.blit(pred_surface, (0, 0))
            label = "DIT"

        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = hud_font.render(f"{fps_val:.0f} FPS", True, (100, 255, 100))
            fps_frame = 0

        label_surface = hud_font.render(label, True, (220, 220, 220))
        hud_y = panel_h
        screen.blit(label_surface, (4, hud_y + 2))
        screen.blit(fps_surface, (win_w - fps_surface.get_width() - 4, hud_y + 2))

        pygame.display.flip()
        clock.tick(fps)

    env.close()
    pygame.quit()


def main() -> None:
    args = parse_args()

    if args.scale < 1:
        raise SystemExit("--scale must be >= 1")
    if args.fps < 1:
        raise SystemExit("--fps must be >= 1")
    if args.ode_steps < 1:
        raise SystemExit("--ode-steps must be >= 1")

    dit_checkpoint = Path(args.dit_checkpoint).resolve()
    if not dit_checkpoint.is_file():
        raise FileNotFoundError(f"DiT checkpoint not found: {dit_checkpoint}")

    device = _resolve_device(args.device)
    if device.type == "cuda":
        if args.tf16:
            torch.set_float32_matmul_precision("medium")
        else:
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dit_config_path = _infer_dit_config_path(args)
    if not _is_readable_file(dit_config_path):
        raise FileNotFoundError(f"DiT config not found: {dit_config_path}")
    dit_config = _load_json(dit_config_path)

    palette_info = load_palette_info(args.data_dir)
    palette_rgb = np.asarray(palette_info["colors_rgb"], dtype=np.uint8)
    if palette_rgb.ndim != 2 or palette_rgb.shape[1] != 3:
        raise ValueError("palette.json must contain colors_rgb as (N, 3)")

    vae_ckpt, vae_cfg = _infer_video_vae_paths(
        args=args,
        model_cfg=dit_config,
        model_cfg_dir=dit_config_path.parent,
    )

    onehot_name = args.onehot_dtype or str(dit_config.get("onehot_dtype", "bfloat16"))
    onehot_dtype = _dtype_from_name(onehot_name)

    autocast_enabled = bool(args.autocast and device.type == "cuda")
    autocast_dtype = torch.bfloat16 if onehot_dtype == torch.bfloat16 else torch.float16
    if autocast_enabled and autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        autocast_dtype = torch.float16

    num_colors = int(palette_rgb.shape[0])
    video_vae, vae_summary = _load_video_vae(
        checkpoint_path=vae_ckpt,
        config_path=vae_cfg,
        num_colors=num_colors,
        device=device,
    )

    dit_model = _load_dit_model(
        checkpoint_path=dit_checkpoint,
        config=dit_config,
        device=device,
        compile_model=bool(args.compile),
    )

    action_to_index, default_action_index = _load_action_remap(
        args.data_dir,
        num_actions=int(dit_config["num_actions"]),
    )

    model_max_frames = int(dit_config.get("max_frames", 64))
    default_window = int(dit_config.get("clip_frames", max(2, min(16, model_max_frames - 1))))
    window_frames = int(args.window_frames or default_window)
    warmup_frames = int(args.warmup_frames or dit_config.get("context_frames", 4))

    if window_frames < 2:
        raise SystemExit("--window-frames must be >= 2")
    if warmup_frames < 1:
        raise SystemExit("--warmup-frames must be >= 1")
    if warmup_frames > window_frames:
        raise SystemExit("--warmup-frames must be <= --window-frames")
    if window_frames + 1 > model_max_frames:
        raise SystemExit(
            f"--window-frames ({window_frames}) + 1 exceeds DiT max_frames ({model_max_frames})"
        )

    world = LiveDiTWorldModel(
        dit_model=dit_model,
        video_vae=video_vae,
        device=device,
        palette_rgb=palette_rgb,
        onehot_dtype=onehot_dtype,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        window_frames=window_frames,
        ode_steps=args.ode_steps,
        action_to_index=action_to_index,
        default_action_index=default_action_index,
    )

    roms = nes_play.discover_roms()
    if not roms:
        raise SystemExit(f"No .nes files found in {nes_play.ROM_DIR}")

    name, path, mapper, backend = nes_play.choose_rom(roms, args.rom)
    if backend is None:
        print(f"\n'{name}' uses mapper {mapper} which nes_py doesn't support.")
        print("Install stable-retro (pip install stable-retro) for broader mapper support.")
        return

    print(f"Using device: {device}")
    print(
        f"DiT: d_model={dit_config.get('d_model')} layers={dit_config.get('num_layers')} "
        f"heads={dit_config.get('num_heads')} max_frames={model_max_frames}"
    )
    print(
        f"VAE: latent_channels={vae_summary['latent_channels']} "
        f"onehot_dtype={onehot_name}"
    )
    print(
        f"Rollout: window_frames={window_frames}, warmup_frames={warmup_frames}, "
        f"ode_steps={args.ode_steps}, warmup_policy={args.warmup_policy}, view={args.view}"
    )
    print(
        f"Actions: {len(action_to_index)} mapped -> reduced space of "
        f"{dit_config.get('num_actions')} (fallback index={default_action_index})"
    )
    print(f"Loading {name} (backend: {backend})")
    print(
        "Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  "
        "RShift=Select  R=Recontext  Esc/Q=Quit"
    )

    if backend == "retro":
        _run_retro(
            name,
            path,
            world=world,
            scale=args.scale,
            fps=args.fps,
            view=args.view,
            warmup_frames=warmup_frames,
            warmup_policy=args.warmup_policy,
        )
    else:
        _run_nes_py(
            name,
            path,
            world=world,
            scale=args.scale,
            fps=args.fps,
            view=args.view,
            warmup_frames=warmup_frames,
            warmup_policy=args.warmup_policy,
        )


if __name__ == "__main__":
    main()
