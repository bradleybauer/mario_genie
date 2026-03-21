"""Play NES ROMs through an autoencoder's lens.

Encodes each frame through the MAGVIT-2 tokenizer and decodes it back,
showing the original and reconstructed views side-by-side in real time.

Usage:
    python scripts/play_autoencoder.py --checkpoint results/asha_sweep/dim16_cb4096_genie_base_1t
    python scripts/play_autoencoder.py --checkpoint results/asha_sweep/dim16_cb4096_genie_base_1t --rom mario
    python scripts/play_autoencoder.py --checkpoint results/asha_sweep/dim16_cb4096_genie_base_1t --scale 2
    python scripts/play_autoencoder.py --checkpoint results/asha_sweep/dim16_cb4096_genie_base_1t --recon-only
"""
import sys
import os
import json
import argparse
from collections import deque

import numpy as np
import torch
import pygame

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mario_world_model.config import IMAGE_SIZE, SEQUENCE_LENGTH
from mario_world_model.model_configs import MODEL_CONFIGS_BY_NAME
from mario_world_model.palette_tokenizer import PaletteVideoTokenizer
from mario_world_model.palette_mapper import PaletteMapper
from mario_world_model.preprocess import preprocess_frame
from mario_world_model.tokenizer_compat import resolve_video_contains_first_frame

# Reuse ROM discovery / selection / controller from play_nes
from play_nes import (
    discover_roms, choose_rom, GamepadController,
    ensure_retro_game, nes_byte_to_retro_action,
    SUPPORTED_MAPPERS, FPS,
)

try:
    import stable_retro as retro
except ImportError:
    try:
        import retro
    except ImportError:
        retro = None

PALETTE_PATH = os.path.join(PROJECT_ROOT, 'data', 'palette.json')


def load_model(checkpoint_dir: str, device: torch.device):
    """Load a PaletteVideoTokenizer from a checkpoint directory.

    Expects the directory to contain config.json and magvit2_best.pt.
    """
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json in {checkpoint_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model_name = config['model']
    image_size = config.get('image_size', IMAGE_SIZE)
    layers = tuple(tuple(l) if isinstance(l, list) else l for l in config['layers'])
    num_palette_colors = 23  # NES palette

    # Resolve init_dim and codebook_size from the named config
    mc = MODEL_CONFIGS_BY_NAME[model_name]

    model = PaletteVideoTokenizer(
        num_palette_colors=num_palette_colors,
        image_size=image_size,
        init_dim=mc.init_dim,
        codebook_size=mc.codebook_size,
        layers=layers,
    ).to(device)

    weights_path = os.path.join(checkpoint_dir, 'magvit2_best.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No magvit2_best.pt in {checkpoint_dir}")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Half precision for faster inference on GPU
    if device.type == 'cuda':
        model = model.half()

    crop_size = None
    if config.get('crop_224'):
        crop_size = 224
    elif config.get('crop_240'):
        crop_size = 240

    print(f"Loaded model '{model_name}' from {checkpoint_dir}")
    print(f"  image_size={image_size}, codebook_size={mc.codebook_size}, "
          f"init_dim={mc.init_dim}, crop={crop_size}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    return model, image_size, crop_size


def crop_frame(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Center-crop a 256x256 frame to crop_size x crop_size."""
    h, w = frame.shape[:2]
    if (h, w) == (crop_size, crop_size):
        return frame
    border = (256 - crop_size) // 2
    return frame[border:256 - border, border:256 - border]


def uncrop_frame(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Pad a crop_size x crop_size frame back to 256x256 with black borders."""
    if frame.shape[0] == 256 and frame.shape[1] == 256:
        return frame
    border = (256 - crop_size) // 2
    if frame.ndim == 3:
        out = np.zeros((256, 256, frame.shape[2]), dtype=frame.dtype)
        out[border:border + crop_size, border:border + crop_size] = frame
    else:
        out = np.zeros((256, 256), dtype=frame.dtype)
        out[border:border + crop_size, border:border + crop_size] = frame
    return out


class FrameBuffer:
    """Accumulates palette-index frames and reconstructs via the full sequence."""

    def __init__(
        self,
        model: PaletteVideoTokenizer,
        palette_rgb: np.ndarray,
        device: torch.device,
        crop_size: int | None,
        seq_len: int = SEQUENCE_LENGTH,
        skip: int = 1,
    ):
        self.model = model
        self.palette_rgb = palette_rgb
        self.device = device
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.skip = max(1, skip)
        self.vcff = resolve_video_contains_first_frame(model, seq_len)
        self._buf: deque[np.ndarray] = deque(maxlen=seq_len)
        self._last_recon: np.ndarray | None = None
        self._frame_count = 0
        self._use_half = next(iter(model.parameters())).dtype == torch.float16
        # Pre-allocate GPU palette tensor for fast index→RGB
        self._palette_gpu = torch.from_numpy(palette_rgb).to(device)

    @torch.no_grad()
    def push(self, palette_indices: np.ndarray) -> np.ndarray:
        """Add a frame and return the reconstructed RGB for the latest frame."""
        # Clamp to valid palette range inside the buffer to be safe
        clamped = np.clip(palette_indices, 0, self.model.num_palette_colors - 1)
        self._buf.append(clamped)
        self._frame_count += 1

        # Before buffer is full, show palette-mapped original
        if len(self._buf) < self.seq_len:
            if self._last_recon is None:
                self._last_recon = self.palette_rgb[clamped]
            return self._last_recon

        # Only run inference every `skip` frames
        if self._frame_count % self.skip != 0 and self._last_recon is not None:
            return self._last_recon

        # Stack into (1, T, H, W) directly on GPU
        indices = torch.from_numpy(np.stack(self._buf)).long().unsqueeze(0)
        if self.crop_size is not None:
            border = (256 - self.crop_size) // 2
            indices = indices[:, :, border:256 - border, border:256 - border]

        inp = PaletteVideoTokenizer.indices_to_onehot(
            indices, self.model.num_palette_colors,
        ).to(self.device)
        if self._use_half:
            inp = inp.half()

        codes = self.model(inp, return_codes=True, video_contains_first_frame=self.vcff)
        recon_logits = self.model.decode_from_code_indices(
            codes, video_contains_first_frame=self.vcff,
        )

        # (1, K, T, H, W) → take the last frame
        recon_idx = recon_logits[0, :, -1].argmax(dim=0)

        if self.crop_size is not None:
            recon_idx_np = recon_idx.cpu().numpy().astype(np.uint8)
            recon_idx_np = uncrop_frame(recon_idx_np, self.crop_size)
            self._last_recon = self.palette_rgb[recon_idx_np]
        else:
            # Palette lookup on GPU, transfer final RGB to CPU
            self._last_recon = self._palette_gpu[recon_idx.long()].cpu().numpy()

        return self._last_recon


def run_game_loop(
    env,
    backend: str,
    model: PaletteVideoTokenizer,
    palette_mapper: PaletteMapper,
    palette_rgb: np.ndarray,
    device: torch.device,
    scale: int,
    crop_size: int | None,
    recon_only: bool,
    name: str,
    skip: int = 1,
):
    """Main game loop: play + reconstruct every frame."""
    if backend == 'retro':
        obs, _ = env.reset()
    else:
        obs = env.reset()
    h, w, _ = obs.shape

    frame_buf = FrameBuffer(model, palette_rgb, device, crop_size, skip=skip)

    frame_h = 256  # padded height
    frame_w = 256
    scaled_w = frame_w * scale
    scaled_h = frame_h * scale

    if recon_only:
        win_w = scaled_w
    else:
        gap = 4 * scale
        win_w = scaled_w * 2 + gap
    win_h = scaled_h + 24  # room for status bar

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    title = f"{name} — Autoencoder View"
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    controller = GamepadController()

    fps_font = pygame.font.SysFont('monospace', 14, bold=True)
    label_font = pygame.font.SysFont('monospace', 12, bold=True)
    fps_surface = fps_font.render('-- FPS', True, (0, 255, 0))
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

        if backend == 'retro':
            obs, reward, terminated, truncated, info = env.step(
                nes_byte_to_retro_action(action)
            )
            if terminated or truncated:
                obs, _ = env.reset()
        else:
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()

        # Preprocess: pad 240x256 → 256x256
        padded = preprocess_frame(obs)

        # Convert RGB → palette indices, clamping to valid range
        palette_idx = np.clip(palette_mapper.map_frame(padded), 0, len(palette_rgb) - 1)

        # Original RGB from palette indices
        original_rgb = palette_rgb[palette_idx]  # (256, 256, 3)

        # Reconstruct through autoencoder
        recon_rgb = frame_buf.push(palette_idx)

        # Render
        screen.fill((10, 10, 15))

        if recon_only:
            surf = pygame.surfarray.make_surface(np.swapaxes(recon_rgb, 0, 1))
            screen.blit(pygame.transform.scale(surf, (scaled_w, scaled_h)), (0, 0))
        else:
            # Original on left
            orig_surf = pygame.surfarray.make_surface(np.swapaxes(original_rgb, 0, 1))
            screen.blit(pygame.transform.scale(orig_surf, (scaled_w, scaled_h)), (0, 0))

            # Reconstructed on right
            recon_surf = pygame.surfarray.make_surface(np.swapaxes(recon_rgb, 0, 1))
            screen.blit(
                pygame.transform.scale(recon_surf, (scaled_w, scaled_h)),
                (scaled_w + gap, 0),
            )

            # Labels
            orig_label = label_font.render('ORIGINAL', True, (200, 200, 200))
            recon_label = label_font.render('RECONSTRUCTED', True, (200, 200, 200))
            screen.blit(orig_label, (4, scaled_h + 2))
            screen.blit(recon_label, (scaled_w + gap + 4, scaled_h + 2))

        # FPS counter
        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = fps_font.render(f'{fps_val:.0f} FPS', True, (0, 255, 0))
            fps_frame = 0
        screen.blit(fps_surface, (win_w - fps_surface.get_width() - 4, scaled_h + 4))

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Play NES through autoencoder lens")
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to checkpoint directory (containing config.json + magvit2_best.pt)',
    )
    parser.add_argument('--rom', default=None, help='ROM name or number (interactive if omitted)')
    parser.add_argument('--scale', type=int, default=2, help='Display scale factor (default: 2)')
    parser.add_argument(
        '--recon-only', action='store_true',
        help='Show only the reconstructed view (no side-by-side)',
    )
    parser.add_argument(
        '--skip', type=int, default=4,
        help='Run inference every N frames, reuse last reconstruction in between (default: 4)',
    )
    parser.add_argument(
        '--no-compile', action='store_true',
        help='Disable torch.compile (use if compilation fails or for debugging)',
    )
    parser.add_argument(
        '--palette', default=PALETTE_PATH,
        help=f'Path to palette JSON (default: {PALETTE_PATH})',
    )
    args = parser.parse_args()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, image_size, crop_size = load_model(args.checkpoint, device)

    if not args.no_compile and device.type == 'cuda':
        try:
            model = torch.compile(model)
            print('  torch.compile enabled')
        except Exception as e:
            print(f'  torch.compile failed ({e}), continuing without it')

    # Load palette
    palette_mapper = PaletteMapper(args.palette)
    with open(args.palette) as f:
        palette_list = json.load(f)
    palette_rgb = np.array(palette_list, dtype=np.uint8)

    # Choose ROM
    roms = discover_roms()
    if not roms:
        print("No .nes files found in nes/ directory")
        return
    name, path, mapper, backend = choose_rom(roms, args.rom)
    if backend is None:
        print(f"'{name}' uses mapper {mapper} — no compatible backend available.")
        return

    print(f"Loading {name} (backend: {backend}) ...")
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc=Quit")

    # Create environment
    if backend == 'retro':
        game_id = ensure_retro_game(path)
        env = retro.make(game_id, state=retro.State.NONE, render_mode=None,
                         use_restricted_actions=retro.Actions.ALL)
    else:
        from nes_py.nes_env import NESEnv
        env = NESEnv(path)

    run_game_loop(
        env, backend, model, palette_mapper, palette_rgb,
        device, args.scale, crop_size, args.recon_only, name,
        skip=args.skip,
    )


if __name__ == '__main__':
    main()
