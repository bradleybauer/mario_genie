#!/usr/bin/env python3
"""Play NES through the DiT world model, seeded from the latent dataset.

A random episode (or a specific one) is loaded from the latent dataset.
The first ``context_latents`` frames are used as context, then the world model
takes over — predicting the next frame autoregressively from your controller
input.  No emulator or ROM required.

Usage examples:
    python scripts/eval/play_world_model.py --dit-checkpoint checkpoints/video_latent_dit_20260412_201206
    python scripts/eval/play_world_model.py --dit-checkpoint checkpoints/video_latent_dit_20260412_201206 --ode-steps 4
    python scripts/eval/play_world_model.py --dit-checkpoint checkpoints/video_latent_dit_20260412_201206 --episode 0 --start 100
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pygame
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.latent_utils import (
    LatentNormalization,
    denoise_future_segment,
    is_readable as _is_readable,
    load_json as _load_json,
    load_latent_normalization,
    load_latent_stats_path,
    load_video_vae,
)
from src.models.video_latent_dit_diffusers import VideoLatentDiTDiffusers
from src.models.video_vae import VideoVAE
from src.path_utils import resolve_workspace_path
from src.data.video_frames import CANONICAL_FRAME_HEIGHT, CANONICAL_FRAME_WIDTH

from scripts.eval.play_nes import GamepadController

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SCALE = 3
DEFAULT_ODE_STEPS = 8


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play NES through the DiT world model.")
    parser.add_argument("--dit-checkpoint", type=str, required=True,
                        help="Path to a DiT checkpoint directory (or best/latest sub-dir)")
    parser.add_argument("--best", action="store_true",
                        help="Prefer best/ over latest/ when resolving checkpoint directory")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Latent data directory (default: data/latents)")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help=f"Display scale factor (default: {DEFAULT_SCALE})")
    parser.add_argument("--ode-steps", type=int, default=DEFAULT_ODE_STEPS,
                        help=f"ODE integration steps for denoising (default: {DEFAULT_ODE_STEPS})")
    parser.add_argument("--context-latents", type=int, default=None,
                        help="Context window size (default: model max_latents - 1)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode index to seed from (default: random)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting latent index within the episode (default: 0)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Execution device (default: auto)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_path(value: str | None, *, config_dir: Path | None = None) -> Path | None:
    return resolve_workspace_path(value, project_root=PROJECT_ROOT, config_dir=config_dir)


def _resolve_dit_checkpoint(checkpoint: str, *, prefer_best: bool = False) -> Path:
    path = Path(checkpoint).resolve()
    # Direct .pt file
    if path.is_file() and path.suffix == ".pt":
        return path
    # Directory — look for best.pt / latest.pt
    names = ("best.pt", "latest.pt") if prefer_best else ("latest.pt", "best.pt")
    for name in names:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No .pt checkpoint found in {path} (looked for best.pt, latest.pt)"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dit_model(checkpoint_path: Path, *, device: torch.device) -> tuple[VideoLatentDiTDiffusers, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = VideoLatentDiTDiffusers(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model_config


# ---------------------------------------------------------------------------
# Dataset seed loading
# ---------------------------------------------------------------------------

def discover_episodes(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.npz"))


def load_seed_from_episode(
    path: Path,
    *,
    seed_latents: int,
    start: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Load a chunk of pre-encoded latents and actions from a .npz episode.

    Returns (latents, actions, episode_name) where:
      latents: (1, C, seed_latents, H, W) float32
      actions: (seed_latents,) int64
    """
    data = np.load(path)
    all_latents = data["latents"]   # (C, T, H, W)
    all_actions = data["actions"]   # (T,)
    T = all_latents.shape[1]

    if start + seed_latents > T:
        start = max(0, T - seed_latents)
    end = start + seed_latents
    if end > T:
        raise ValueError(f"Episode {path.name} has {T} latents, need {seed_latents} from index {start}")

    latents = torch.from_numpy(all_latents[:, start:end].astype(np.float32)).unsqueeze(0).to(device)
    actions = torch.from_numpy(all_actions[start:end].astype(np.int64)).to(device)
    name = path.stem
    return latents, actions, name


# ---------------------------------------------------------------------------
# World model player
# ---------------------------------------------------------------------------

class WorldModelPlayer:
    """Drives the world model from controller input, seeded from the dataset."""

    def __init__(
        self,
        *,
        dit_model: VideoLatentDiTDiffusers,
        vae: VideoVAE,
        palette_rgb: np.ndarray,
        latent_normalization: LatentNormalization | None,
        action_values: list[int],
        default_action_index: int,
        context_latents: int,
        ode_steps: int,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> None:
        self.dit = dit_model
        self.vae = vae
        self.palette_rgb = palette_rgb
        self.latent_norm = latent_normalization
        self.action_values = action_values
        self.default_action_index = default_action_index
        self.context_latents = context_latents
        self.ode_steps = ode_steps
        self.device = device
        self.autocast_dtype = autocast_dtype

        # Reverse map: NES byte -> action index
        self._byte_to_action_idx: dict[int, int] = {v: i for i, v in enumerate(action_values)}

        # Latent history (normalized, on device)
        self._latent_history: torch.Tensor | None = None
        # Action history (action indices)
        self._action_history: list[int] = []
        # Step counter
        self.dream_steps = 0

    def seed(self, latents: torch.Tensor, actions: torch.Tensor) -> None:
        """Initialize from pre-encoded latents (1, C, T, H, W) and actions (T,)."""
        if self.latent_norm is not None:
            latents = self.latent_norm.normalize(latents)
        T = latents.shape[2]
        if T > self.context_latents:
            latents = latents[:, :, -self.context_latents:]
            actions = actions[-self.context_latents:]
        self._latent_history = latents
        self._action_history = actions.cpu().tolist()
        self.dream_steps = 0

    def nes_byte_to_action_index(self, byte: int) -> int:
        return self._byte_to_action_idx.get(byte, self.default_action_index)

    def decode_current_frame(self) -> np.ndarray:
        """Decode the last latent in history to an RGB frame for display."""
        last = self._latent_history[:, :, -1:]  # (1, C, 1, H', W')
        denormed = self.latent_norm.denormalize(last) if self.latent_norm is not None else last
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            logits = self.vae.decode(denormed)
        indices = logits[:, :, -1].argmax(dim=1)[0].cpu().numpy()
        return self.palette_rgb[indices]

    def step(self, action_idx: int) -> np.ndarray:
        """Predict the next frame given an action index. Returns RGB frame."""
        ctx = self._latent_history.shape[2]
        hist_actions = list(self._action_history)

        # Causal shift: shifted[0]=default, shifted[i]=original[i-1]
        # original = hist_actions + [action_idx], length = ctx + 1
        # shifted = [default] + hist_actions, length = ctx + 1
        shifted_actions = [self.default_action_index] + hist_actions
        actions_t = torch.tensor([shifted_actions], dtype=torch.long, device=self.device)

        pred = denoise_future_segment(
            self.dit,
            history_latents=self._latent_history,
            actions=actions_t,
            future_latents=1,
            ode_steps=self.ode_steps,
            autocast_ctx=lambda: torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype),
        )  # (1, C, 1, H', W')

        # Decode predicted latent
        denormed = self.latent_norm.denormalize(pred) if self.latent_norm is not None else pred
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            logits = self.vae.decode(denormed)
        indices = logits[:, :, -1].argmax(dim=1)[0].cpu().numpy()
        dream_rgb = self.palette_rgb[indices]

        # Slide context window
        self._latent_history = torch.cat([self._latent_history[:, :, 1:], pred], dim=2)
        self._action_history.pop(0)
        self._action_history.append(action_idx)
        self.dream_steps += 1

        return dream_rgb


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _frame_to_surface(frame_rgb: np.ndarray, out_size: tuple[int, int]) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(np.swapaxes(frame_rgb, 0, 1))
    if surface.get_size() != out_size:
        surface = pygame.transform.scale(surface, out_size)
    return surface


def run_game_loop(
    *,
    player: WorldModelPlayer,
    episode_name: str,
    frame_height: int,
    frame_width: int,
    scale: int,
) -> None:
    panel_w = frame_width * scale
    panel_h = frame_height * scale
    hud_h = 22
    win_w = panel_w
    win_h = panel_h + hud_h

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"World Model \u2014 {episode_name}")
    clock = pygame.time.Clock()
    controller = GamepadController()
    hud_font = pygame.font.SysFont("monospace", 14, bold=True)

    # Show the seed frame first
    current_rgb = player.decode_current_frame()

    fps_surface = hud_font.render("-- FPS", True, (100, 255, 100))
    fps_frame = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            controller.process_event(event)

        action_byte = controller.get_action()
        action_idx = player.nes_byte_to_action_index(action_byte)
        current_rgb = player.step(action_idx)

        screen.fill((8, 8, 12))
        dream_surface = _frame_to_surface(current_rgb, (panel_w, panel_h))
        screen.blit(dream_surface, (0, 0))

        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = hud_font.render(f"{fps_val:.0f} FPS", True, (100, 255, 100))
            fps_frame = 0

        label = f"DREAM  step {player.dream_steps}"
        label_surface = hud_font.render(label, True, (220, 220, 220))
        hud_y = panel_h
        screen.blit(label_surface, (4, hud_y + 2))
        screen.blit(fps_surface, (win_w - fps_surface.get_width() - 4, hud_y + 2))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.scale < 1:
        raise SystemExit("--scale must be at least 1")

    device = _resolve_device(args.device)
    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    # ------------------------------------------------------------------
    # Resolve data directory
    # ------------------------------------------------------------------
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = (PROJECT_ROOT / "data" / "latents").resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    latent_meta: dict | None = None
    latent_meta_path = data_dir / "latent_config.json"
    if latent_meta_path.is_file():
        latent_meta = _load_json(latent_meta_path)

    # ------------------------------------------------------------------
    # Load DiT model
    # ------------------------------------------------------------------
    dit_ckpt_path = _resolve_dit_checkpoint(args.dit_checkpoint, prefer_best=args.best)
    print(f"Loading DiT from {dit_ckpt_path}")
    dit_model, model_config = load_dit_model(dit_ckpt_path, device=device)
    print(f"  d_model={model_config['d_model']}, "
          f"layers={model_config['num_encoder_layers']}+{model_config['num_decoder_layers']}, "
          f"params={sum(p.numel() for p in dit_model.parameters()):,}")

    action_values = model_config.get("action_values", list(range(model_config["num_actions"])))
    default_action_index = int({v: i for i, v in enumerate(action_values)}.get(0, 0))

    context_latents = args.context_latents
    if context_latents is None:
        context_latents = model_config.get("max_latents", 256) - 1
    print(f"  context_latents={context_latents}")

    # ------------------------------------------------------------------
    # Latent normalization
    # ------------------------------------------------------------------
    latent_channels = model_config["latent_channels"]
    latent_height = int(latent_meta.get("latent_height", 7)) if latent_meta is not None else 7
    latent_width = int(latent_meta.get("latent_width", 7)) if latent_meta is not None else 7

    latent_stats_path = load_latent_stats_path(data_dir=data_dir, latent_meta=latent_meta, project_root=PROJECT_ROOT)
    latent_normalization: LatentNormalization | None = None
    if latent_stats_path is not None:
        latent_normalization = load_latent_normalization(
            stats_path=latent_stats_path,
            latent_channels=latent_channels,
            latent_height=latent_height,
            latent_width=latent_width,
            device=device,
        )
        print(f"  Latent normalization loaded from {latent_stats_path}")
    else:
        print("  WARNING: No latent normalization found")

    # ------------------------------------------------------------------
    # Load palette
    # ------------------------------------------------------------------
    palette_path = data_dir / "palette.json"
    if not palette_path.is_file():
        raise SystemExit(f"Palette not found: {palette_path}")
    palette_info = _load_json(palette_path)
    palette_rgb = np.asarray(palette_info["colors_rgb"], dtype=np.uint8)
    num_colors = int(palette_rgb.shape[0])
    print(f"  Palette: {num_colors} colors")

    # ------------------------------------------------------------------
    # Load VAE (from latent_config.json)
    # ------------------------------------------------------------------
    vae_ckpt = latent_meta.get("video_vae_checkpoint") if latent_meta is not None else None
    vae_cfg = latent_meta.get("video_vae_config_path") if latent_meta is not None else None
    vae_ckpt_path = _resolve_path(vae_ckpt, config_dir=data_dir)
    vae_cfg_path = _resolve_path(vae_cfg, config_dir=data_dir)
    if not _is_readable(vae_cfg_path) and _is_readable(vae_ckpt_path):
        sibling = vae_ckpt_path.parent / "config.json"
        if sibling.is_file():
            vae_cfg_path = sibling

    if not _is_readable(vae_ckpt_path) or not _is_readable(vae_cfg_path):
        raise SystemExit(
            f"Cannot find VAE checkpoint/config. "
            f"Resolved: ckpt={vae_ckpt_path}, cfg={vae_cfg_path}"
        )

    print(f"Loading VAE from {vae_ckpt_path}")
    vae, vae_info = load_video_vae(
        checkpoint_path=vae_ckpt_path,
        config_path=vae_cfg_path,
        num_colors=num_colors,
        device=device,
    )
    frame_height = int(latent_meta.get("frame_height", CANONICAL_FRAME_HEIGHT)) if latent_meta else CANONICAL_FRAME_HEIGHT
    frame_width = int(latent_meta.get("frame_width", CANONICAL_FRAME_WIDTH)) if latent_meta else CANONICAL_FRAME_WIDTH
    print(f"  VAE: latent_channels={vae_info['latent_channels']}, "
          f"temporal_downsample={vae_info['temporal_downsample']}, "
          f"frame={frame_height}x{frame_width}")

    # ------------------------------------------------------------------
    # Load seed episode from dataset
    # ------------------------------------------------------------------
    episodes = discover_episodes(data_dir)
    if not episodes:
        raise SystemExit(f"No .npz files found in {data_dir}")

    if args.episode is not None:
        if args.episode < 0 or args.episode >= len(episodes):
            raise SystemExit(f"--episode {args.episode} out of range (0-{len(episodes)-1})")
        episode_path = episodes[args.episode]
    else:
        episode_path = random.choice(episodes)

    print(f"Seeding from: {episode_path.name} (start={args.start}, latents={context_latents})")
    seed_latents_t, seed_actions_t, episode_name = load_seed_from_episode(
        episode_path,
        seed_latents=context_latents,
        start=args.start,
        device=device,
    )
    print(f"  Loaded {seed_latents_t.shape[2]} latents, shape {tuple(seed_latents_t.shape)}")

    # ------------------------------------------------------------------
    # Create player and seed it
    # ------------------------------------------------------------------
    player = WorldModelPlayer(
        dit_model=dit_model,
        vae=vae,
        palette_rgb=palette_rgb,
        latent_normalization=latent_normalization,
        action_values=action_values,
        default_action_index=default_action_index,
        context_latents=context_latents,
        ode_steps=args.ode_steps,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    player.seed(seed_latents_t, seed_actions_t)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    print(f"\nUsing device: {device}")
    print(f"ODE steps: {args.ode_steps}, context: {context_latents}")
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc/Q=Quit")

    run_game_loop(
        player=player,
        episode_name=episode_name,
        frame_height=frame_height,
        frame_width=frame_width,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
