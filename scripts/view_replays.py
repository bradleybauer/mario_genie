#!/usr/bin/env python3
"""Visualize recorded rollouts by replaying their actions in the emulator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.envs import make_shimmed_env
from mario_world_model.rollouts import Rollout


def load_rollouts(path: Path) -> list[Rollout]:
    rollouts: list[Rollout] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rollouts.append(Rollout.from_json_line(line))
    return rollouts


def play_rollout(rollout: Rollout, screen: pygame.Surface, clock: pygame.time.Clock, fps: int) -> bool:
    """Replay a single rollout. Returns False if the user quit."""
    env = make_shimmed_env(world=rollout.world, stage=rollout.stage)
    obs, _ = env.reset()

    def render(obs_hwc: np.ndarray) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    return False  # skip to next
        surface = pygame.surfarray.make_surface(np.transpose(obs_hwc, (1, 0, 2)))
        if surface.get_size() != screen.get_size():
            surface = pygame.transform.scale(surface, screen.get_size())
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(max(1, fps))
        return True

    if not render(obs):
        env.close()
        return False

    for i, action in enumerate(rollout.actions):
        obs, _, terminated, truncated, info = env.step(action)
        if not render(obs):
            env.close()
            return False
        if terminated or truncated:
            break

    # Pause briefly on the final frame
    for _ in range(fps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_SPACE):
                    env.close()
                    return True
        clock.tick(max(1, fps))

    env.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize recorded rollouts")
    parser.add_argument("data_dir", type=Path, help="Directory containing rollouts.jsonl")
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS")
    parser.add_argument("--scale", type=int, default=2, help="Window scale multiplier")
    parser.add_argument("--world", type=int, default=None, help="Filter to this world")
    parser.add_argument("--stage", type=int, default=None, help="Filter to this stage")
    parser.add_argument("--outcome", type=str, default=None, choices=["death", "flag", "timeout", "transition"],
                        help="Filter to this outcome type")
    parser.add_argument("--min-x", type=int, default=None, help="Only show rollouts with max_x >= this value")
    parser.add_argument("--limit", type=int, default=None, help="Max number of rollouts to play")
    parser.add_argument("--reverse", action="store_true", help="Play longest rollouts first")
    args = parser.parse_args()

    rollouts_path = args.data_dir / "rollouts.jsonl"
    if not rollouts_path.exists():
        print(f"No rollouts.jsonl found in {args.data_dir}")
        sys.exit(1)

    rollouts = load_rollouts(rollouts_path)
    print(f"Loaded {len(rollouts)} rollouts")

    # Filter
    if args.world is not None:
        rollouts = [r for r in rollouts if r.world == args.world]
    if args.stage is not None:
        rollouts = [r for r in rollouts if r.stage == args.stage]
    if args.outcome is not None:
        rollouts = [r for r in rollouts if r.outcome == args.outcome]
    if args.min_x is not None:
        rollouts = [r for r in rollouts if r.max_x >= args.min_x]

    if not rollouts:
        print("No rollouts match the given filters.")
        sys.exit(0)

    # Sort
    rollouts.sort(key=lambda r: r.max_x, reverse=args.reverse)

    if args.limit is not None:
        rollouts = rollouts[:args.limit]

    print(f"Playing {len(rollouts)} rollouts (Space=skip, Esc=quit)")

    pygame.init()
    # NES native resolution is 256x240
    w, h = 256 * args.scale, 240 * args.scale
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Mario Replay Viewer")
    clock = pygame.time.Clock()

    try:
        for i, rollout in enumerate(rollouts):
            label = (f"[{i + 1}/{len(rollouts)}] W{rollout.world}-{rollout.stage} "
                     f"outcome={rollout.outcome} max_x={rollout.max_x} steps={rollout.num_steps}")
            print(label)
            pygame.display.set_caption(f"Replay: {label}")
            if not play_rollout(rollout, screen, clock, args.fps):
                break
    finally:
        pygame.quit()

    print("Done.")


if __name__ == "__main__":
    main()
