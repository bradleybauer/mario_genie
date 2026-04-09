#!/usr/bin/env python3
"""Play NES through a live Video VAE reconstruction.

Usage examples:
    python scripts/play_video_vae.py --checkpoint checkpoints/run/video_vae_best.pt
    python scripts/play_video_vae.py --checkpoint checkpoints/run/video_vae_latest.pt --rom mario
    python scripts/play_video_vae.py --checkpoint checkpoints/run/video_vae_best.pt --view side-by-side

The script:
1) reads NES observations from the emulator,
2) maps RGB pixels -> palette indices,
3) encodes + decodes with VideoVAE,
4) displays the reconstructed frame in real time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pygame
import torch
from nes_py.nes_env import NESEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.video_vae import VideoVAE
from src.data.normalized_dataset import load_palette_info
from src.data.video_frames import (
    CANONICAL_FRAME_HEIGHT,
    CANONICAL_FRAME_WIDTH,
    SUPPORTED_FRAME_SIZES,
    preprocess_live_nes_frame,
)

from scripts.eval import play_nes as nes_play

DEFAULT_SCALE = 3
DEFAULT_WINDOW_FRAMES = 4
DEFAULT_FPS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play NES through Video VAE reconstruction.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a checkpoint (.pt) from train_video_vae.py")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config.json path. Defaults to sibling of --checkpoint if found.")
    parser.add_argument("--data-dir", type=str, default="data/normalized",
                        help="Directory containing palette.json (default: data/normalized)")
    parser.add_argument("--rom", type=str, default=None,
                        help="ROM number or partial name match (same behavior as play_nes.py)")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help=f"Display scale factor (default: {DEFAULT_SCALE})")
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES,
                        help=f"Temporal window fed to the model (default: {DEFAULT_WINDOW_FRAMES})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help=f"Viewer target FPS (default: {DEFAULT_FPS})")
    parser.add_argument("--view", type=str, default="recon", choices=["recon", "side-by-side"],
                        help="Show reconstructed frame only or input+reconstruction (default: recon)")
    parser.add_argument(
        "--frame-size",
        type=int,
        default=None,
        choices=SUPPORTED_FRAME_SIZES,
        help="Override checkpoint frame size for live preprocessing.",
    )

    parser.add_argument("--base-channels", type=int, default=None,
                        help="Override model base channels (defaults to config.json or 64)")
    parser.add_argument("--latent-channels", type=int, default=None,
                        help="Override model latent channels (defaults to config.json or 64)")
    parser.add_argument("--onehot-dtype", type=str, default=None,
                        choices=["float32", "float16", "bfloat16"],
                        help="Input one-hot dtype (defaults to config.json or float32)")
    parser.add_argument("--autocast", action="store_true",
                        help="Enable CUDA autocast during model forward")
    parser.add_argument("--sample-posterior", action="store_true",
                        help="Sample latents instead of using posterior mean (more stochastic)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Execution device (default: auto)")
    return parser.parse_args()


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(args: argparse.Namespace) -> dict:
    config_path: Path | None = None
    if args.config is not None:
        config_path = Path(args.config)
    else:
        sibling = Path(args.checkpoint).resolve().parent / "config.json"
        if sibling.is_file():
            config_path = sibling

    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as handle:
        return json.load(handle)


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


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


class LiveAutoencoder:
    def __init__(
        self,
        model: VideoVAE,
        *,
        device: torch.device,
        palette_rgb: np.ndarray,
        onehot_dtype: torch.dtype,
        window_frames: int,
        frame_height: int,
        frame_width: int,
        autocast_enabled: bool,
        autocast_dtype: torch.dtype,
        sample_posterior: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.palette_rgb = palette_rgb
        self.onehot_dtype = onehot_dtype
        self.window_frames = window_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = autocast_dtype
        self.sample_posterior = sample_posterior

        self.num_colors = int(palette_rgb.shape[0])
        self.rgb_lut = _build_rgb_lut(palette_rgb)
        self.history: deque[np.ndarray] = deque(maxlen=window_frames)
        self.onehot_buffer: torch.Tensor | None = None

    def _autocast_context(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def reconstruct(self, obs_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        processed = preprocess_live_nes_frame(
            obs_rgb,
            target_height=self.frame_height,
            target_width=self.frame_width,
        )
        indices = _rgb_to_palette_indices(processed, self.rgb_lut, self.palette_rgb)
        self.history.append(indices)

        frames = list(self.history)
        if len(frames) < self.window_frames:
            pad = [frames[0]] * (self.window_frames - len(frames))
            frames = pad + frames

        clip = np.stack(frames, axis=0)
        frames_t = torch.from_numpy(clip).unsqueeze(0).to(self.device, non_blocking=True)
        inputs = _frames_to_one_hot(
            frames_t,
            self.num_colors,
            dtype=self.onehot_dtype,
            out=self.onehot_buffer,
        )
        self.onehot_buffer = inputs

        with torch.inference_mode():
            with self._autocast_context():
                outputs = self.model(inputs, sample_posterior=self.sample_posterior)

        pred_idx = outputs.logits[:, :, -1].argmax(dim=1)[0].detach().cpu().numpy()
        recon_rgb = self.palette_rgb[pred_idx]
        return processed, recon_rgb


def _make_display_sizes(scale: int, view: str, *, frame_height: int, frame_width: int) -> tuple[int, int, int, int]:
    panel_w = frame_width * scale
    panel_h = frame_height * scale
    hud_h = 22
    if view == "side-by-side":
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


def _run_nes_py(
    name: str,
    rom_path: str,
    *,
    viewer: LiveAutoencoder,
    scale: int,
    fps: int,
    view: str,
) -> None:
    env = NESEnv(rom_path)
    obs = env.reset()

    win_w, win_h, panel_w, panel_h = _make_display_sizes(
        scale,
        view,
        frame_height=viewer.frame_height,
        frame_width=viewer.frame_width,
    )

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{name} (VAE reconstruction)")
    clock = pygame.time.Clock()
    controller = nes_play.GamepadController()
    hud_font = pygame.font.SysFont("monospace", 14, bold=True)

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

        action = controller.get_action()
        obs, _reward, done, _info = env.step(action)
        if done:
            obs = env.reset()

        source_rgb, recon_rgb = viewer.reconstruct(obs)

        screen.fill((8, 8, 12))
        if view == "side-by-side":
            src_surface = _frame_to_surface(source_rgb, (panel_w, panel_h))
            recon_surface = _frame_to_surface(recon_rgb, (panel_w, panel_h))
            screen.blit(src_surface, (0, 0))
            screen.blit(recon_surface, (panel_w, 0))
        else:
            recon_surface = _frame_to_surface(recon_rgb, (panel_w, panel_h))
            screen.blit(recon_surface, (0, 0))

        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = hud_font.render(f"{fps_val:.0f} FPS", True, (100, 255, 100))
            fps_frame = 0

        label = "INPUT | RECON" if view == "side-by-side" else "RECON"
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
    viewer: LiveAutoencoder,
    scale: int,
    fps: int,
    view: str,
) -> None:
    game_id = nes_play.ensure_retro_game(rom_path)
    env = nes_play.retro.make(
        game_id,
        state=nes_play.retro.State.NONE,
        render_mode=None,
        use_restricted_actions=nes_play.retro.Actions.ALL,
    )
    obs, _info = env.reset()

    win_w, win_h, panel_w, panel_h = _make_display_sizes(
        scale,
        view,
        frame_height=viewer.frame_height,
        frame_width=viewer.frame_width,
    )

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h), pygame.DOUBLEBUF)
    pygame.display.set_caption(f"{name} (VAE reconstruction)")
    clock = pygame.time.Clock()
    controller = nes_play.GamepadController()
    hud_font = pygame.font.SysFont("monospace", 14, bold=True)

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
        obs, _reward, terminated, truncated, _info = env.step(
            nes_play.nes_byte_to_retro_action(action_byte)
        )
        if terminated or truncated:
            obs, _info = env.reset()

        source_rgb, recon_rgb = viewer.reconstruct(obs)

        screen.fill((8, 8, 12))
        if view == "side-by-side":
            src_surface = _frame_to_surface(source_rgb, (panel_w, panel_h))
            recon_surface = _frame_to_surface(recon_rgb, (panel_w, panel_h))
            screen.blit(src_surface, (0, 0))
            screen.blit(recon_surface, (panel_w, 0))
        else:
            recon_surface = _frame_to_surface(recon_rgb, (panel_w, panel_h))
            screen.blit(recon_surface, (0, 0))

        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = hud_font.render(f"{fps_val:.0f} FPS", True, (100, 255, 100))
            fps_frame = 0

        label = "INPUT | RECON" if view == "side-by-side" else "RECON"
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
        raise SystemExit("--scale must be at least 1")
    if args.window_frames < 1:
        raise SystemExit("--window-frames must be at least 1")
    if args.fps < 1:
        raise SystemExit("--fps must be at least 1")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    palette_info = load_palette_info(args.data_dir)
    palette_rgb = np.asarray(palette_info["colors_rgb"], dtype=np.uint8)
    if palette_rgb.ndim != 2 or palette_rgb.shape[1] != 3:
        raise ValueError("palette.json must contain colors_rgb as (N, 3)")

    device = _resolve_device(args.device)
    config = _load_config(args)

    config_model = dict(config.get("model", {}))
    config_data = dict(config.get("data", {}))
    config_training = dict(config.get("training", {}))

    base_channels = args.base_channels or int(config_model.get("base_channels", config.get("base_channels", 64)))
    latent_channels = args.latent_channels or int(config_model.get("latent_channels", config.get("latent_channels", 64)))
    temporal_downsample = int(config_model.get("temporal_downsample", config.get("temporal_downsample", 0)))
    global_bottleneck_attn = bool(
        config_model.get("global_bottleneck_attn", config.get("global_bottleneck_attn", False))
    )
    global_bottleneck_attn_heads = int(
        config_model.get("global_bottleneck_attn_heads", config.get("global_bottleneck_attn_heads", 8))
    )
    frame_height = int(args.frame_size or config_data.get("frame_height", CANONICAL_FRAME_HEIGHT))
    frame_width = int(args.frame_size or config_data.get("frame_width", CANONICAL_FRAME_WIDTH))

    onehot_name = args.onehot_dtype or str(config_training.get("onehot_dtype", config.get("onehot_dtype", "float32")))
    onehot_dtype = _dtype_from_name(onehot_name)

    autocast_enabled = bool(args.autocast and device.type == "cuda")
    autocast_dtype = torch.bfloat16 if onehot_dtype == torch.bfloat16 else torch.float16
    if autocast_enabled and autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        autocast_dtype = torch.float16

    model = VideoVAE(
        num_colors=int(palette_rgb.shape[0]),
        base_channels=base_channels,
        latent_channels=latent_channels,
        temporal_downsample=temporal_downsample,
        global_bottleneck_attn=global_bottleneck_attn,
        global_bottleneck_attn_heads=global_bottleneck_attn_heads,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    viewer = LiveAutoencoder(
        model,
        device=device,
        palette_rgb=palette_rgb,
        onehot_dtype=onehot_dtype,
        window_frames=args.window_frames,
        frame_height=frame_height,
        frame_width=frame_width,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
        sample_posterior=args.sample_posterior,
    )

    roms = nes_play.discover_roms()
    if not roms:
        raise SystemExit(f"No .nes files found in {nes_play.ROM_DIR}")

    name, path, mapper, backend = nes_play.choose_rom(roms, args.rom)
    if backend is None:
        print(f"\\n'{name}' uses mapper {mapper} which nes_py doesn't support.")
        print("Install stable-retro (pip install stable-retro) for broader mapper support.")
        return

    print(f"Using device: {device}")
    print(
        f"Model: base_channels={base_channels}, latent_channels={latent_channels}, "
        f"temporal_downsample={temporal_downsample}, global_bottleneck_attn={global_bottleneck_attn}, "
        f"frame={frame_height}x{frame_width}, onehot_dtype={onehot_name}"
    )
    print(f"Loading {name} (backend: {backend})")
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc/Q=Quit")

    if backend == "retro":
        _run_retro(name, path, viewer=viewer, scale=args.scale, fps=args.fps, view=args.view)
    else:
        _run_nes_py(name, path, viewer=viewer, scale=args.scale, fps=args.fps, view=args.view)


if __name__ == "__main__":
    main()
