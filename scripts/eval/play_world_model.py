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
    python scripts/eval/play_world_model.py --dit-checkpoint checkpoints/video_latent_dit_20260412_201206 --record-gif run.gif
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.cm as _cm
import numpy as np
import pygame
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.latent_utils import (
    FLOW_SAMPLERS,
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

from scripts.eval.play_nes import GamepadController, BUTTON_BITS
from scripts.eval.play_raw_recording import draw_controller

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# play_nes.py byte layout:   A=0, B=1, SELECT=2, START=3, UP=4, DOWN=5, LEFT=6, RIGHT=7
# Mesen DataCollector layout: UP=0, DOWN=1, LEFT=2, RIGHT=3, START=4, SELECT=5, B=6, A=7
# The model was trained on Mesen-format bytes; GamepadController returns play_nes bytes.
_MESEN_BITS = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3, 'START': 4, 'SELECT': 5, 'B': 6, 'A': 7}

# Build a 256-entry lookup table: play_nes byte -> Mesen byte
_PLAYNES_TO_MESEN = np.zeros(256, dtype=np.uint8)
for _i in range(256):
    _m = 0
    for _name, _pbit in BUTTON_BITS.items():
        if _i & (1 << _pbit):
            _m |= (1 << _MESEN_BITS[_name])
    _PLAYNES_TO_MESEN[_i] = _m
DEFAULT_SCALE = 3
DEFAULT_ODE_STEPS = 8
DEFAULT_ODE_SAMPLER = "heun"
DEFAULT_RECORD_FPS = 20
GENERATIVE_FPS = 60


# ---------------------------------------------------------------------------
# Latent visualization helpers
# ---------------------------------------------------------------------------

def _latent_channel_grid(latent: np.ndarray, sep: int = 1) -> np.ndarray:
    """Arrange (C, H, W) latent channels into a rectangular grid image.

    *sep* pixels of NaN are inserted between channels so that the colormap
    renders them as a visible separator line.
    """
    C, H, W = latent.shape
    cols = 8
    rows = (C + cols - 1) // cols
    pad = rows * cols - C
    if pad > 0:
        latent = np.concatenate([latent, np.full((pad, H, W), np.nan, dtype=latent.dtype)], axis=0)
    latent = latent.reshape(rows, cols, H, W)
    cell_h = H + sep
    cell_w = W + sep
    grid = np.full((rows * cell_h - sep, cols * cell_w - sep), np.nan, dtype=latent.dtype)
    for r in range(rows):
        for c in range(cols):
            y = r * cell_h
            x = c * cell_w
            grid[y:y + H, x:x + W] = latent[r, c]
    return grid


def _latent_grid_to_surface(latent: np.ndarray, target_height: int) -> pygame.Surface:
    """Render a (C, H, W) latent as a colormapped pygame surface scaled to *target_height*."""
    grid = _latent_channel_grid(latent)
    normalized = np.clip((grid + 3.0) / 6.0, 0.0, 1.0)
    cmap = _cm.get_cmap("coolwarm")
    rgba = cmap(normalized)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    nan_mask = np.isnan(grid)
    rgb[nan_mask] = [8, 8, 12]
    surface = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
    gh, gw = grid.shape
    scaled_w = max(1, int(target_height * gw / gh))
    return pygame.transform.scale(surface, (scaled_w, target_height))


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
    parser.add_argument(
        "--ode-sampler",
        type=str,
        choices=FLOW_SAMPLERS,
        default=DEFAULT_ODE_SAMPLER,
        help=f"Flow ODE sampler used during denoising (default: {DEFAULT_ODE_SAMPLER})",
    )
    parser.add_argument(
        "--action-cfg-scale",
        type=float,
        default=1.0,
        help="Action classifier-free guidance scale used during denoising.",
    )
    parser.add_argument("--context-latents", type=int, default=None,
                        help="Context window size (default: model max_latents - 1)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode index to seed from (default: random)")
    parser.add_argument("--start", type=int, default=0,
                        help="Starting latent index within the episode (default: 0)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Execution device (default: auto)")
    parser.add_argument("--record-gif", type=str, default=None,
                        help="Optional output GIF path. Records the rendered window until quit.")
    parser.add_argument("--record-fps", type=int, default=DEFAULT_RECORD_FPS,
                        help=(
                            f"GIF playback FPS in generative time (default: {DEFAULT_RECORD_FPS}). "
                            "Recording is based on 60 generated frames/sec, not wall-clock runtime."
                        ))
    args = parser.parse_args()
    if args.action_cfg_scale < 0.0:
        parser.error("--action-cfg-scale must be >= 0")
    return args


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
            actions: (seed_latents, action_frame_count) int64
    """
    data = np.load(path)
    all_latents = data["latents"]   # (C, T, H, W)
    all_actions = data["actions"]   # (T, action_frame_count)
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
        latent_frame_stride: int,
        ode_steps: int,
        ode_sampler: str,
        action_cfg_scale: float,
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
        self.latent_frame_stride = max(1, int(latent_frame_stride))
        self.ode_steps = ode_steps
        self.ode_sampler = ode_sampler
        self.action_cfg_scale = float(action_cfg_scale)
        self.device = device
        self.autocast_dtype = autocast_dtype

        # Reverse map: NES byte -> action index
        self._byte_to_action_idx: dict[int, int] = {v: i for i, v in enumerate(action_values)}

        # Latent history (normalized, on device)
        self._latent_history: torch.Tensor | None = None
        # Action history (grouped frame action indices)
        self._action_history: list[list[int]] = []
        self.action_frame_count = 0
        # Decoded generated frames for the most recent predicted latent.
        self._decoded_future_frames: list[np.ndarray] = []
        # Step counter
        self.dream_steps = 0
        # Generated frame counter (can be > latent steps when temporal stride > 1).
        self.generated_frames = 0

    def seed(self, latents: torch.Tensor, actions: torch.Tensor) -> None:
        """Initialize from pre-encoded latents (1, C, T, H, W) and actions (T, action_frame_count)."""
        if actions.ndim != 2:
            raise ValueError(f"Expected seed actions (T, action_frame_count), got {tuple(actions.shape)}")
        if self.latent_norm is not None:
            latents = self.latent_norm.normalize(latents)
        T = latents.shape[2]
        if T > self.context_latents:
            latents = latents[:, :, -self.context_latents:]
            actions = actions[-self.context_latents:]
        self._latent_history = latents
        self.action_frame_count = int(actions.shape[1])
        self._action_history = actions.cpu().tolist()
        self._decoded_future_frames = []
        self.dream_steps = 0
        self.generated_frames = 0

    def get_current_latent(self) -> np.ndarray:
        """Return the last latent in history as (C, H, W) numpy array."""
        return self._latent_history[0, :, -1].cpu().float().numpy()

    def nes_byte_to_action_index(self, byte: int) -> int:
        return self._byte_to_action_idx.get(byte, self.default_action_index)

    def decode_current_frame(self) -> np.ndarray:
        """Decode the last latent in history to an RGB frame for display."""
        last = self._latent_history[:, :, -1:]  # (1, C, 1, H', W')
        decoded = self._decode_latent_to_rgb(last)
        return decoded[-1]

    def _decode_latent_to_rgb(self, latents: torch.Tensor) -> list[np.ndarray]:
        """Decode latent tensor (1, C, T, H', W') into a list of RGB frames."""
        denormed = self.latent_norm.denormalize(latents) if self.latent_norm is not None else latents
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            logits = self.vae.decode(denormed)
        frame_indices = logits[0].argmax(dim=0).cpu().numpy()  # (T_out, H, W)
        return [self.palette_rgb[frame_indices[t]] for t in range(frame_indices.shape[0])]

    def _decode_new_latent_frames(self, pred: torch.Tensor) -> list[np.ndarray]:
        """Decode a newly predicted latent with temporal context from history.

        Passes the previous latent alongside the new one so the VAE temporal
        upsampler has proper context, then returns only the frames that
        correspond to the new latent (the last ``latent_frame_stride`` frames).
        """
        stride = self.latent_frame_stride
        if stride <= 1:
            # No temporal upsampling — decode the single latent directly.
            return self._decode_latent_to_rgb(pred)

        # Decode [prev_latent, new_latent] together so both output frames
        # for the new latent have temporal context from the previous one.
        prev = self._latent_history[:, :, -1:]  # already updated; -1 is pred
        if self._latent_history.shape[2] >= 2:
            prev = self._latent_history[:, :, -2:-1]
        pair = torch.cat([prev, pred], dim=2)  # (1, C, 2, H', W')
        all_frames = self._decode_latent_to_rgb(pair)
        # Take only the last `stride` frames (those belonging to the new latent).
        return all_frames[-stride:]

    def step(self, action_idx: int) -> np.ndarray:
        """Predict (when needed) and return the next generated RGB frame."""
        if not self._decoded_future_frames:
            # The player is pressing the action aligned with the next latent
            # to be predicted, so overwrite the current future slot directly
            # before denoising.
            action_group = [int(action_idx)] * self.action_frame_count
            self._action_history[-1] = action_group
            actions_t = torch.tensor([self._action_history], dtype=torch.long, device=self.device)

            pred = denoise_future_segment(
                self.dit,
                history_latents=self._latent_history,
                actions=actions_t,
                future_latents=1,
                ode_steps=self.ode_steps,
                ode_sampler=self.ode_sampler,
                action_cfg_scale=self.action_cfg_scale,
                autocast_ctx=lambda: torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype),
            )  # (1, C, 1, H', W')

            # Slide context window once per latent prediction (before decode
            # so _decode_new_latent_frames can access the previous latent).
            self._latent_history = torch.cat([self._latent_history[:, :, 1:], pred], dim=2)
            self._action_history.pop(0)
            # Append a placeholder for the predicted latent's action slot.
            # It will be overwritten with the player's real input on the
            # next call to step().
            self._action_history.append([self.default_action_index] * self.action_frame_count)
            self.dream_steps += 1

            decoded_frames = self._decode_new_latent_frames(pred)
            if not decoded_frames:
                raise RuntimeError("VAE decode returned zero frames for predicted latent")
            self._decoded_future_frames = decoded_frames

        dream_rgb = self._decoded_future_frames.pop(0)
        self.generated_frames += 1
        return dream_rgb


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _frame_to_surface(frame_rgb: np.ndarray, out_size: tuple[int, int]) -> pygame.Surface:
    surface = pygame.surfarray.make_surface(np.swapaxes(frame_rgb, 0, 1))
    if surface.get_size() != out_size:
        surface = pygame.transform.scale(surface, out_size)
    return surface


def _capture_surface_rgb(surface: pygame.Surface) -> np.ndarray:
    return np.swapaxes(pygame.surfarray.array3d(surface), 0, 1).copy()


def _save_recording(frames: list[np.ndarray], durations_ms: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def run_game_loop(
    *,
    player: WorldModelPlayer,
    episode_name: str,
    frame_height: int,
    frame_width: int,
    scale: int,
    record_gif: Path | None = None,
    record_fps: int = DEFAULT_RECORD_FPS,
) -> None:
    panel_w = frame_width * scale
    panel_h = frame_height * scale
    hud_h = 22
    ctrl_h = 86  # height for NES controller display
    gap = 4

    # Compute latent panel width from initial latent shape
    init_lat = player.get_current_latent()
    lat_grid_shape = _latent_channel_grid(init_lat).shape
    lat_panel_w = max(1, int(panel_h * lat_grid_shape[1] / lat_grid_shape[0]))

    win_w = panel_w + gap + lat_panel_w
    win_h = panel_h + hud_h + ctrl_h

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
    recorded_frames: list[np.ndarray] = []
    recorded_durations_ms: list[int] = []
    # Record against generated-frame progression, not real-time inference speed.
    record_every_n = max(1, int(round(GENERATIVE_FPS / max(record_fps, 1))))
    effective_record_fps = GENERATIVE_FPS / record_every_n
    frame_duration_ms = max(1, int(round(1000.0 / effective_record_fps)))

    ctrl_font = pygame.font.SysFont("monospace", 12, bold=True)
    current_action_byte = 0

    def _present_frame(rgb: np.ndarray) -> None:
        """Render one generated frame to the screen and flip."""
        screen.fill((8, 8, 12))
        dream_surface = _frame_to_surface(rgb, (panel_w, panel_h))
        screen.blit(dream_surface, (0, 0))

        # Latent visualization panel
        lat_surface = _latent_grid_to_surface(player.get_current_latent(), panel_h)
        screen.blit(lat_surface, (panel_w + gap, 0))

        nonlocal fps_frame, fps_surface
        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = hud_font.render(f"{fps_val:.0f} FPS", True, (100, 255, 100))
            fps_frame = 0

        label = f"DREAM frame {player.generated_frames}  latent {player.dream_steps}"
        label_surface = hud_font.render(label, True, (220, 220, 220))
        lat_label = hud_font.render("LATENT \u03bc", True, (180, 180, 220))
        hud_y = panel_h
        screen.blit(label_surface, (4, hud_y + 2))
        screen.blit(lat_label, (panel_w + gap, hud_y + 2))
        screen.blit(fps_surface, (win_w - fps_surface.get_width() - 4, hud_y + 2))

        # NES controller input display
        draw_controller(screen, ctrl_font, current_action_byte, 4, hud_y + hud_h)

        pygame.display.flip()
        # Enforce minimum visibility per frame.  clock.tick() alone won't
        # sleep after a long inference because it only caps *average* FPS.
        pygame.time.delay(1000 // GENERATIVE_FPS)
        clock.tick()  # update clock for FPS measurement

        if record_gif is not None:
            if ((player.generated_frames - 1) % record_every_n) == 0:
                recorded_frames.append(_capture_surface_rgb(screen))
                recorded_durations_ms.append(frame_duration_ms)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            controller.process_event(event)

        action_byte = controller.get_action()
        mesen_byte = int(_PLAYNES_TO_MESEN[action_byte])
        current_action_byte = mesen_byte  # draw_controller expects Mesen layout
        action_idx = player.nes_byte_to_action_index(mesen_byte)

        # step() may decode multiple frames from one latent (e.g. 2 when
        # temporal_downsample=1).  Present each frame individually so none
        # are skipped due to slow inference dominating the timing.
        current_rgb = player.step(action_idx)
        _present_frame(current_rgb)

        while player._decoded_future_frames and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                controller.process_event(event)
            if not running:
                break
            current_rgb = player.step(action_idx)
            _present_frame(current_rgb)

    pygame.quit()

    if record_gif is not None:
        if not recorded_frames:
            print(f"No GIF frames captured; nothing saved to {record_gif}")
            return
        _save_recording(recorded_frames, recorded_durations_ms, record_gif)
        print(f"Saved GIF to {record_gif}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.scale < 1:
        raise SystemExit("--scale must be at least 1")
    if args.record_fps < 1:
        raise SystemExit("--record-fps must be at least 1")

    device = _resolve_device(args.device)
    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    record_gif_path = Path(args.record_gif).resolve() if args.record_gif is not None else None

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
    latent_frame_stride = int(
        latent_meta.get("latent_temporal_stride", 2 ** int(vae_info.get("temporal_downsample", 0)))
    ) if latent_meta is not None else 2 ** int(vae_info.get("temporal_downsample", 0))
    latent_frame_stride = max(1, latent_frame_stride)
    print(f"  latent_frame_stride={latent_frame_stride} generated frames / latent")

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
        latent_frame_stride=latent_frame_stride,
        ode_steps=args.ode_steps,
        ode_sampler=args.ode_sampler,
        action_cfg_scale=args.action_cfg_scale,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    player.seed(seed_latents_t, seed_actions_t)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    print(f"\nUsing device: {device}")
    print(
        f"ODE steps: {args.ode_steps}, sampler: {args.ode_sampler}, "
        f"context: {context_latents}, action_cfg_scale: {args.action_cfg_scale}"
    )
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc/Q=Quit")
    if record_gif_path is not None:
        print(
            f"Recording GIF to {record_gif_path} "
            f"({args.record_fps} FPS generative-time, {GENERATIVE_FPS} generated frames/sec)"
        )

    run_game_loop(
        player=player,
        episode_name=episode_name,
        frame_height=frame_height,
        frame_width=frame_width,
        scale=args.scale,
        record_gif=record_gif_path,
        record_fps=args.record_fps,
    )


if __name__ == "__main__":
    main()
