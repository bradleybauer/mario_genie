"""Play NES ROMs through the Genie 2 world model.

Two modes:
  --mode ae       Encode+decode each frame through the VAE (like play_autoencoder
                  but for the Genie 2 autoencoder).
  --mode dynamics Run the full world model: after an initial context window of
                  real frames, the model autoregressively predicts future frames
                  conditioned on your controller input.

Usage:
    python scripts/play_world_model.py --rom mario --mode ae
    python scripts/play_world_model.py --rom mario --mode dynamics
    python scripts/play_world_model.py --rom mario --mode dynamics --ctx-frames 16
    python scripts/play_world_model.py --rom mario --mode ae --scale 3
"""
import sys
import os
import json
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import pygame

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mario_world_model.preprocess import preprocess_frame
from mario_world_model.palette_mapper import PaletteMapper
from mario_world_model.actions import COMPLEX_MOVEMENT, NUM_ACTIONS

from play_nes import (
    discover_roms, choose_rom, GamepadController,
    ensure_retro_game, nes_byte_to_retro_action,
    BUTTON_BITS, FPS,
)

try:
    import stable_retro as retro
except ImportError:
    try:
        import retro
    except ImportError:
        retro = None

# Import model classes from train_genie2
from train_genie2 import (
    FrameVAE, DynamicsTransformer, LatentDenoiser, DiffusionSchedule,
    LATENT_CHANNELS, LATENT_SIZE, CROP_224_SIZE,
    TRANSFORMER_DIM, TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    DIFFUSION_STEPS,
)

PALETTE_PATH = os.path.join(PROJECT_ROOT, 'data', 'palette.json')


# ---------------------------------------------------------------------------
# NES controller byte → COMPLEX_MOVEMENT action index
# ---------------------------------------------------------------------------

def _build_byte_to_action_map() -> dict[int, int]:
    """Build a mapping from NES controller byte → nearest COMPLEX_MOVEMENT index.

    COMPLEX_MOVEMENT uses button names like 'A', 'B', 'left', 'right', 'down', 'up'.
    The NES byte uses bits for 'A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT'.
    We ignore SELECT/START for action mapping (they don't affect gameplay dynamics).
    """
    # Map button name in COMPLEX_MOVEMENT → bit position in NES byte
    name_to_bit = {
        'A': BUTTON_BITS['A'],
        'B': BUTTON_BITS['B'],
        'left': BUTTON_BITS['LEFT'],
        'right': BUTTON_BITS['RIGHT'],
        'down': BUTTON_BITS['DOWN'],
        'up': BUTTON_BITS['UP'],
    }

    # Build the set of buttons for each action index
    action_button_sets = []
    for combo in COMPLEX_MOVEMENT:
        if combo == ["NOOP"]:
            action_button_sets.append(frozenset())
        else:
            action_button_sets.append(frozenset(combo))

    # For every possible byte (0-255), find the best matching action
    byte_to_action = {}
    for byte_val in range(256):
        # Extract which gameplay buttons are pressed
        pressed = set()
        for btn_name, bit_pos in name_to_bit.items():
            if byte_val & (1 << bit_pos):
                pressed.add(btn_name)
        pressed = frozenset(pressed)

        # Find exact match first
        best_idx = 0  # NOOP fallback
        for idx, action_set in enumerate(action_button_sets):
            if action_set == pressed:
                best_idx = idx
                break
        else:
            # No exact match — find closest by Jaccard similarity
            best_score = -1
            for idx, action_set in enumerate(action_button_sets):
                if not action_set and not pressed:
                    score = 1.0
                elif not action_set or not pressed:
                    score = 0.0
                else:
                    score = len(action_set & pressed) / len(action_set | pressed)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        byte_to_action[byte_val] = best_idx

    return byte_to_action


BYTE_TO_ACTION = _build_byte_to_action_map()


def crop_frame(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Center-crop a 256x256 frame to crop_size x crop_size."""
    h, w = frame.shape[:2]
    if (h, w) == (crop_size, crop_size):
        return frame
    border = (256 - crop_size) // 2
    return frame[border:256 - border, border:256 - border]


def uncrop_frame(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Pad a crop_size x crop_size frame back to 256x256 with black borders."""
    if frame.shape[0] == 256:
        return frame
    border = (256 - crop_size) // 2
    if frame.ndim == 3:
        out = np.zeros((256, 256, frame.shape[2]), dtype=frame.dtype)
    else:
        out = np.zeros((256, 256), dtype=frame.dtype)
    out[border:border + crop_size, border:border + crop_size] = frame
    return out


def palette_indices_to_rgb(indices: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """(H, W) uint8 indices → (H, W, 3) uint8 RGB."""
    return palette_rgb[indices]


def load_models(args, device, num_colors):
    """Load AE and optionally dynamics models."""
    ae = FrameVAE(num_palette_colors=num_colors, latent_channels=LATENT_CHANNELS).to(device)
    ae_ckpt = args.ae_checkpoint or os.path.join(args.checkpoint_dir, 'ae_best.pt')
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=True))
    ae.eval()
    print(f"Loaded AE from {ae_ckpt}")

    dynamics = None
    denoiser = None
    schedule = None

    if args.mode == 'dynamics':
        dyn_ckpt = args.dyn_checkpoint or os.path.join(args.checkpoint_dir, 'dynamics_best.pt')
        dynamics = DynamicsTransformer(
            latent_channels=LATENT_CHANNELS, latent_size=LATENT_SIZE,
            dim=TRANSFORMER_DIM, num_heads=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS, num_actions=NUM_ACTIONS,
            max_frames=args.ctx_frames + 1,
        ).to(device)
        denoiser = LatentDenoiser(
            latent_channels=LATENT_CHANNELS, cond_dim=TRANSFORMER_DIM,
        ).to(device)
        state = torch.load(dyn_ckpt, map_location=device, weights_only=True)
        dynamics.load_state_dict(state['dynamics'])
        denoiser.load_state_dict(state['denoiser'])
        dynamics.eval()
        denoiser.eval()
        schedule = DiffusionSchedule(DIFFUSION_STEPS, device)
        print(f"Loaded dynamics from {dyn_ckpt}")

    return ae, dynamics, denoiser, schedule


@torch.no_grad()
def encode_frame(ae: FrameVAE, palette_idx: np.ndarray, num_colors: int,
                 device: torch.device, crop_size: int) -> torch.Tensor:
    """Encode a single 256x256 palette-index frame → latent mean (1, C, 14, 14)."""
    frame = crop_frame(palette_idx, crop_size)
    t = torch.from_numpy(frame).long().unsqueeze(0).to(device)  # (1, H, W)
    x_oh = F.one_hot(t, num_colors).float().permute(0, 3, 1, 2)
    h = ae.encoder(x_oh)
    mu = ae.enc_mu(h)
    return mu


@torch.no_grad()
def decode_latent_to_rgb(ae: FrameVAE, latent: torch.Tensor,
                         palette_rgb: np.ndarray, crop_size: int) -> np.ndarray:
    """Decode latent (1, C, 14, 14) → 256x256 RGB numpy array."""
    logits = ae.decode(latent)  # (1, K, H, W)
    pred_idx = logits[0].argmax(0).cpu().numpy().astype(np.uint8)  # (crop, crop)
    pred_idx = uncrop_frame(pred_idx, crop_size)
    return palette_rgb[pred_idx]


# ═══════════════════════════════════════════════════════════════════════════
# Game loop
# ═══════════════════════════════════════════════════════════════════════════

def run_ae_mode(env, backend, ae, palette_mapper, palette_rgb, num_colors,
                device, scale, name):
    """AE mode: encode+decode every frame, show original vs reconstruction."""
    if backend == 'retro':
        obs, _ = env.reset()
    else:
        obs = env.reset()

    crop_size = CROP_224_SIZE

    pygame.init()
    frame_h, frame_w = 256, 256
    scaled = scale * frame_w
    gap = 4 * scale
    win_w = scaled * 2 + gap
    win_h = scaled + 24
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{name} — Genie2 AE")
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

        action_byte = controller.get_action()

        if backend == 'retro':
            obs, _, terminated, truncated, info = env.step(nes_byte_to_retro_action(action_byte))
            if terminated or truncated:
                obs, _ = env.reset()
        else:
            obs, _, done, info = env.step(action_byte)
            if done:
                obs = env.reset()

        padded = preprocess_frame(obs)
        palette_idx = np.clip(palette_mapper.map_frame(padded), 0, num_colors - 1)
        original_rgb = palette_rgb[palette_idx]

        # Encode + decode
        latent = encode_frame(ae, palette_idx, num_colors, device, crop_size)
        recon_rgb = decode_latent_to_rgb(ae, latent, palette_rgb, crop_size)

        # Render
        screen.fill((10, 10, 15))
        orig_surf = pygame.surfarray.make_surface(np.swapaxes(original_rgb, 0, 1))
        screen.blit(pygame.transform.scale(orig_surf, (scaled, scaled)), (0, 0))

        recon_surf = pygame.surfarray.make_surface(np.swapaxes(recon_rgb, 0, 1))
        screen.blit(pygame.transform.scale(recon_surf, (scaled, scaled)), (scaled + gap, 0))

        orig_label = label_font.render('ORIGINAL', True, (200, 200, 200))
        recon_label = label_font.render('RECONSTRUCTION', True, (200, 200, 200))
        screen.blit(orig_label, (4, scaled + 2))
        screen.blit(recon_label, (scaled + gap + 4, scaled + 2))

        fps_frame += 1
        if fps_frame >= 30:
            fps_surface = fps_font.render(f'{clock.get_fps():.0f} FPS', True, (0, 255, 0))
            fps_frame = 0
        screen.blit(fps_surface, (win_w - fps_surface.get_width() - 4, scaled + 4))

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


def run_dynamics_mode(env, backend, ae, dynamics, denoiser, schedule,
                      palette_mapper, palette_rgb, num_colors, device,
                      scale, name, ctx_frames):
    """Dynamics mode: collect real context frames, then predict future frames.

    Shows three columns: original | last-real-context | model prediction.
    After the context window fills, the model runs autoregressively:
    it uses its own predicted latents (plus your actions) to keep generating.
    """
    if backend == 'retro':
        obs, _ = env.reset()
    else:
        obs = env.reset()

    crop_size = CROP_224_SIZE

    pygame.init()
    frame_size = 256
    scaled = scale * frame_size
    gap = 4 * scale
    win_w = scaled * 3 + gap * 2
    win_h = scaled + 40
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{name} — Genie2 World Model")
    clock = pygame.time.Clock()
    controller = GamepadController()

    fps_font = pygame.font.SysFont('monospace', 14, bold=True)
    label_font = pygame.font.SysFont('monospace', 12, bold=True)
    status_font = pygame.font.SysFont('monospace', 14, bold=True)
    fps_surface = fps_font.render('-- FPS', True, (0, 255, 0))
    fps_frame = 0

    # State for context/autoregressive generation
    latent_history = deque(maxlen=ctx_frames)
    action_history = deque(maxlen=ctx_frames)
    phase = 'context'  # 'context' or 'dreaming'
    frames_collected = 0
    last_real_rgb = None  # last real frame before dreaming starts
    predicted_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    # Reset: go back to context collection and reset env
                    phase = 'context'
                    latent_history.clear()
                    action_history.clear()
                    frames_collected = 0
                    if backend == 'retro':
                        obs, _ = env.reset()
                    else:
                        obs = env.reset()
                elif event.key == pygame.K_t:
                    # Re-collect context from current gameplay (no env reset)
                    phase = 'context'
                    latent_history.clear()
                    action_history.clear()
                    frames_collected = 0
            controller.process_event(event)

        action_byte = controller.get_action()
        action_idx = BYTE_TO_ACTION[action_byte]

        if phase == 'context':
            # Step the real env
            if backend == 'retro':
                obs, _, terminated, truncated, info = env.step(nes_byte_to_retro_action(action_byte))
                if terminated or truncated:
                    obs, _ = env.reset()
            else:
                obs, _, done, info = env.step(action_byte)
                if done:
                    obs = env.reset()

            padded = preprocess_frame(obs)
            palette_idx = np.clip(palette_mapper.map_frame(padded), 0, num_colors - 1)
            original_rgb = palette_rgb[palette_idx]

            # Encode and store
            latent = encode_frame(ae, palette_idx, num_colors, device, crop_size)
            latent_history.append(latent)
            action_history.append(action_idx)
            frames_collected += 1

            # Also show AE recon as the "predicted" column during context
            recon_rgb = decode_latent_to_rgb(ae, latent, palette_rgb, crop_size)
            predicted_rgb = recon_rgb
            last_real_rgb = recon_rgb.copy()

            if frames_collected >= ctx_frames:
                phase = 'dreaming'
                print(f"Context collected ({ctx_frames} frames). Now DREAMING. Press R to reset.")

            display_original = original_rgb
        else:
            # Dreaming: autoregressive prediction
            # We still step the real env for the "ground truth" column
            if backend == 'retro':
                obs, _, terminated, truncated, info = env.step(nes_byte_to_retro_action(action_byte))
                if terminated or truncated:
                    obs, _ = env.reset()
            else:
                obs, _, done, info = env.step(action_byte)
                if done:
                    obs = env.reset()

            padded = preprocess_frame(obs)
            palette_idx = np.clip(palette_mapper.map_frame(padded), 0, num_colors - 1)
            display_original = palette_rgb[palette_idx]

            # Build context from history
            ctx_latents = torch.cat(list(latent_history), dim=0).unsqueeze(0)  # (1, T, C, H, W)
            ctx_actions = torch.tensor(
                list(action_history), dtype=torch.long, device=device,
            ).unsqueeze(0)  # (1, T)

            # Get dynamics context vector
            context = dynamics(ctx_latents, ctx_actions)

            # DDPM sample next frame latent
            latent_shape = (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
            pred_latent = schedule.ddpm_sample(denoiser, context, latent_shape, device)

            # Decode to RGB
            predicted_rgb = decode_latent_to_rgb(ae, pred_latent, palette_rgb, crop_size)

            # Push predicted latent into history for autoregressive rollout
            latent_history.append(pred_latent)
            action_history.append(action_idx)

        # --- Render ---
        screen.fill((10, 10, 15))

        # Column 1: Real game
        orig_surf = pygame.surfarray.make_surface(np.swapaxes(display_original, 0, 1))
        screen.blit(pygame.transform.scale(orig_surf, (scaled, scaled)), (0, 0))

        # Column 2: Last real context frame (frozen once dreaming)
        if last_real_rgb is not None:
            ctx_surf = pygame.surfarray.make_surface(np.swapaxes(last_real_rgb, 0, 1))
        else:
            ctx_surf = pygame.surfarray.make_surface(np.swapaxes(display_original, 0, 1))
        screen.blit(pygame.transform.scale(ctx_surf, (scaled, scaled)), (scaled + gap, 0))

        # Column 3: Model prediction
        pred_surf = pygame.surfarray.make_surface(np.swapaxes(predicted_rgb, 0, 1))
        screen.blit(pygame.transform.scale(pred_surf, (scaled, scaled)), (scaled * 2 + gap * 2, 0))

        # Labels
        screen.blit(label_font.render('REAL GAME', True, (200, 200, 200)), (4, scaled + 2))
        screen.blit(label_font.render('LAST CONTEXT', True, (200, 200, 200)),
                     (scaled + gap + 4, scaled + 2))
        screen.blit(label_font.render('MODEL PREDICTION', True, (200, 200, 200)),
                     (scaled * 2 + gap * 2 + 4, scaled + 2))

        # Status
        if phase == 'context':
            status = f'CONTEXT: {frames_collected}/{ctx_frames}'
            color = (255, 200, 0)
        else:
            status = 'DREAMING (R=reset env, T=re-context)'
            color = (0, 255, 100)
        status_surf = status_font.render(status, True, color)
        screen.blit(status_surf, (4, scaled + 20))

        # Action label
        if action_idx < len(COMPLEX_MOVEMENT):
            act_name = '+'.join(COMPLEX_MOVEMENT[action_idx]) if COMPLEX_MOVEMENT[action_idx] != ['NOOP'] else 'NOOP'
        else:
            act_name = f'?{action_idx}'
        act_surf = label_font.render(f'Action: {act_name}', True, (180, 180, 180))
        screen.blit(act_surf, (win_w // 2 - act_surf.get_width() // 2, scaled + 22))

        # FPS
        fps_frame += 1
        if fps_frame >= 30:
            fps_surface = fps_font.render(f'{clock.get_fps():.0f} FPS', True, (0, 255, 0))
            fps_frame = 0
        screen.blit(fps_surface, (win_w - fps_surface.get_width() - 4, scaled + 4))

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Play NES through Genie 2 world model")
    parser.add_argument('--mode', choices=['ae', 'dynamics'], default='ae',
                        help='ae = autoencoder view, dynamics = world model prediction')
    parser.add_argument('--rom', default=None, help='ROM name or number')
    parser.add_argument('--scale', type=int, default=2, help='Display scale (default: 2)')
    parser.add_argument('--checkpoint-dir', default='checkpoints/genie2',
                        help='Directory with ae_best.pt / dynamics_best.pt')
    parser.add_argument('--ae-checkpoint', default=None, help='Path to AE checkpoint')
    parser.add_argument('--dyn-checkpoint', default=None, help='Path to dynamics checkpoint')
    parser.add_argument('--ctx-frames', type=int, default=8,
                        help='Number of real context frames before dreaming (default: 8)')
    parser.add_argument('--palette', default=PALETTE_PATH, help='Path to palette JSON')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load palette
    palette_mapper = PaletteMapper(args.palette, freeze=True)
    with open(args.palette) as f:
        palette_list = json.load(f)
    palette_rgb = np.array(palette_list, dtype=np.uint8)
    num_colors = len(palette_rgb)

    # Load models
    ae, dynamics, denoiser, schedule = load_models(args, device, num_colors)

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
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select")
    print("          R=Reset env+context  T=Re-context from current  Esc/Q=Quit")

    # Create environment
    if backend == 'retro':
        game_id = ensure_retro_game(path)
        env = retro.make(game_id, state=retro.State.NONE, render_mode=None,
                         use_restricted_actions=retro.Actions.ALL)
    else:
        from nes_py.nes_env import NESEnv
        env = NESEnv(path)

    if args.mode == 'ae':
        run_ae_mode(env, backend, ae, palette_mapper, palette_rgb, num_colors,
                    device, args.scale, name)
    else:
        run_dynamics_mode(env, backend, ae, dynamics, denoiser, schedule,
                          palette_mapper, palette_rgb, num_colors, device,
                          args.scale, name, args.ctx_frames)


if __name__ == '__main__':
    main()
