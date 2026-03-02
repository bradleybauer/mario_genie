#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model_phase1.actions import get_action_meanings
from mario_world_model_phase1.config import SEQUENCE_LENGTH
from mario_world_model_phase1.envs import RandomLevelMarioEnv, make_shimmed_env
from mario_world_model_phase1.preprocess import preprocess_frame
from mario_world_model_phase1.storage import ChunkWriter


@dataclass
class VectorRunStats:
    mode: str
    level_mode: str
    world: int
    stage: int
    num_envs: int
    total_env_steps: int
    total_sequences: int
    output_dir: str
    sequence_length: int
    sequences_per_chunk: int


class VectorActionPolicy:
    def __init__(self, mode: str, num_envs: int, action_meanings: list[list[str]], seed: int = 0):
        if mode not in {"random", "heuristic"}:
            raise ValueError("Vector collector supports mode in {'random', 'heuristic'}")
        self.mode = mode
        self.rng = random.Random(seed)
        self.num_envs = num_envs
        self.num_actions = len(action_meanings)
        self._action_to_index = {frozenset(token.lower() for token in action): idx for idx, action in enumerate(action_meanings)}
        self._noop_index = self._action_to_index.get(frozenset({"noop"}), 0)
        self._sticky_actions = [0 for _ in range(num_envs)]
        self._sticky_until = [0 for _ in range(num_envs)]
        self._t = [0 for _ in range(num_envs)]

        self._heuristic_candidates = self._build_heuristic_candidates()

    def _build_heuristic_candidates(self) -> list[int]:
        preferred = [
            frozenset({"right"}),
            frozenset({"right", "a"}),
            frozenset({"right", "b"}),
            frozenset({"right", "a", "b"}),
            frozenset({"left"}),
            frozenset({"a"}),
            frozenset({"down"}),
        ]
        out: list[int] = []
        for key in preferred:
            idx = self._action_to_index.get(key)
            if idx is not None and idx not in out:
                out.append(idx)
        if not out:
            out.append(self._noop_index)
        return out

    def _sample_random(self, idx: int) -> int:
        if self._t[idx] >= self._sticky_until[idx]:
            self._sticky_actions[idx] = self.rng.randrange(self.num_actions)
            hold_for = self.rng.randint(3, 12)
            self._sticky_until[idx] = self._t[idx] + hold_for
        self._t[idx] += 1
        return self._sticky_actions[idx]

    def _sample_heuristic(self) -> int:
        p = self.rng.random()
        if p < 0.55 and len(self._heuristic_candidates) >= 1:
            return self._heuristic_candidates[0]
        if p < 0.75 and len(self._heuristic_candidates) >= 2:
            return self._heuristic_candidates[1]
        if p < 0.88 and len(self._heuristic_candidates) >= 3:
            return self._heuristic_candidates[2]
        if p < 0.95 and len(self._heuristic_candidates) >= 4:
            return self._heuristic_candidates[3]
        if p < 0.98 and len(self._heuristic_candidates) >= 5:
            return self._heuristic_candidates[4]
        if len(self._heuristic_candidates) >= 6:
            return self._heuristic_candidates[5]
        return self._heuristic_candidates[0]

    def sample(self) -> np.ndarray:
        actions = np.zeros((self.num_envs,), dtype=np.int64)
        for idx in range(self.num_envs):
            if self.mode == "random":
                actions[idx] = self._sample_random(idx)
            else:
                actions[idx] = self._sample_heuristic()
        return actions


class HumanActionPolicy:
    def __init__(self, action_meanings: list[list[str]], fps: int = 75, seed: int = 0):
        self.fps = int(fps)
        self.rng = random.Random(seed)
        pygame.init()
        self.screen = pygame.display.set_mode((512, 480))
        pygame.display.set_caption("Mario Phase1 Human Recorder")
        self.clock = pygame.time.Clock()
        self._action_to_index = {frozenset(token.lower() for token in action): idx for idx, action in enumerate(action_meanings)}
        self._noop_index = self._action_to_index.get(frozenset({"noop"}), 0)

    def draw_frame(self, frame_hwc: np.ndarray) -> None:
        surface = pygame.surfarray.make_surface(np.transpose(frame_hwc, (1, 0, 2)))
        surface = pygame.transform.scale(surface, self.screen.get_size())
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def _lookup_action(self, buttons: set[str]) -> int:
        key = frozenset(token.lower() for token in buttons)
        if key in self._action_to_index:
            return self._action_to_index[key]
        print(f"Warning: {buttons} not found in actions! Falling back to NOOP.")
        return self._noop_index

    def _keys_to_action(self, keys: pygame.key.ScancodeWrapper) -> int:
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        jump = keys[pygame.K_o]
        sprint = keys[pygame.K_p]

        if right and left:
            left = False

        buttons: set[str] = set()
        if down:
            buttons.add("down")

        if right:
            buttons.add("right")
        elif left:
            buttons.add("left")

        if jump:
            buttons.add("a")
        if sprint:
            buttons.add("b")

        if buttons:
            return self._lookup_action(buttons)

        return self._noop_index

    def sample(self) -> int:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("Human recorder window closed.")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt("Human recorder stopped by Escape key.")

        keys = pygame.key.get_pressed()
        action = self._keys_to_action(keys)
        target_fps = self.fps
        self.clock.tick(max(1, target_fps))
        return action

    def close(self):
        pygame.quit()


def to_tchw(frame_hwc: np.ndarray) -> np.ndarray:
    padded = preprocess_frame(frame_hwc)
    return np.transpose(padded, (2, 0, 1)).astype(np.uint8, copy=False)


def build_vector_env(
    world: int,
    stage: int,
    num_envs: int,
    seed: int,
    vector_mode: str,
    level_mode: str,
):
    def thunk(rank: int):
        def _make():
            if level_mode == "random":
                return RandomLevelMarioEnv(seed=seed + rank)
            return make_shimmed_env(world=world, stage=stage, seed=seed + rank)

        return _make

    env_fns = [thunk(i) for i in range(num_envs)]
    if vector_mode == "async":
        return gym.vector.AsyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)


def run_collection(
    output_dir: Path,
    mode: str,
    world: int,
    stage: int,
    seed: int,
    sequence_length: int,
    sequences_per_chunk: int,
    num_envs: int,
    total_steps: int,
    vector_mode: str,
    human_fps: int,
    level_mode: str,
    compress_chunks: bool,
    async_write: bool,
    max_pending_writes: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[config] mode={mode} level_mode={level_mode} world={world} stage={stage} num_envs={num_envs}"
    )

    action_meanings = get_action_meanings()

    if mode == "human":
        if num_envs != 1:
            raise ValueError("--mode human requires --num-envs 1")
        return run_human_collection(
            output_dir=output_dir,
            world=world,
            stage=stage,
            seed=seed,
            sequence_length=sequence_length,
            sequences_per_chunk=sequences_per_chunk,
            total_steps=total_steps,
            human_fps=human_fps,
            action_meanings=action_meanings,
            level_mode=level_mode,
            compress_chunks=compress_chunks,
            async_write=async_write,
            max_pending_writes=max_pending_writes,
        )

    env = build_vector_env(
        world=world,
        stage=stage,
        num_envs=num_envs,
        seed=seed,
        vector_mode=vector_mode,
        level_mode=level_mode,
    )
    writer = ChunkWriter(
        output_dir=output_dir,
        sequence_length=sequence_length,
        sequences_per_chunk=sequences_per_chunk,
        compress=compress_chunks,
        async_write=async_write,
        max_pending_writes=max_pending_writes,
    )
    policy = VectorActionPolicy(mode=mode, num_envs=num_envs, action_meanings=action_meanings, seed=seed)

    obs, info = env.reset(seed=seed)

    INFO_KEYS = ["coins", "flag_get", "life", "score", "stage", "time", "world", "x_pos", "y_pos"]
    STATUS_MAP = {"small": 0, "tall": 1, "fireball": 2}

    seq_frames = [[] for _ in range(num_envs)]
    seq_actions = [[] for _ in range(num_envs)]
    seq_dones = [[] for _ in range(num_envs)]
    seq_info = {k: [[] for _ in range(num_envs)] for k in INFO_KEYS + ["status"]}

    total_sequences = 0

    for step_idx in range(total_steps):
        actions = policy.sample()
        next_obs, _, terminated, truncated, next_info = env.step(actions)

        done_flags = np.logical_or(terminated, truncated)

        for env_idx in range(num_envs):
            seq_frames[env_idx].append(to_tchw(obs[env_idx]))
            seq_actions[env_idx].append(int(actions[env_idx]))
            seq_dones[env_idx].append(bool(done_flags[env_idx]))
            
            for k in INFO_KEYS:
                val = info.get(k)
                if val is not None:
                    seq_info[k][env_idx].append(int(val[env_idx]))
                else:
                    seq_info[k][env_idx].append(0)

            status_val = info.get("status")
            if status_val is not None:
                seq_info["status"][env_idx].append(STATUS_MAP.get(status_val[env_idx], 0))
            else:
                seq_info["status"][env_idx].append(0)

            if len(seq_frames[env_idx]) == sequence_length:
                frames_tchw = np.stack(seq_frames[env_idx], axis=0)
                actions_t = np.asarray(seq_actions[env_idx], dtype=np.uint8)
                dones_t = np.asarray(seq_dones[env_idx], dtype=bool)
                
                kwargs_t = {}
                for k in INFO_KEYS + ["status"]:
                    kwargs_t[k] = np.asarray(seq_info[k][env_idx], dtype=np.int32)
                    seq_info[k][env_idx].clear()

                out = writer.add_sequence(frames_tchw, actions_t, dones_t, **kwargs_t)
                total_sequences += 1
                seq_frames[env_idx].clear()
                seq_actions[env_idx].clear()
                seq_dones[env_idx].clear()
                if out is not None:
                    print(f"[chunk] wrote {out}")

        obs = next_obs
        info = next_info

        if (step_idx + 1) % 500 == 0:
            print(f"[step {step_idx + 1}/{total_steps}] sequences={total_sequences}")

    try:
        final = writer.flush()
        if final is not None:
            print(f"[chunk] wrote {final}")
    finally:
        writer.close()
        env.close()

    stats = VectorRunStats(
        mode=mode,
        level_mode=level_mode,
        world=world,
        stage=stage,
        num_envs=num_envs,
        total_env_steps=total_steps * num_envs,
        total_sequences=total_sequences,
        output_dir=str(output_dir),
        sequence_length=sequence_length,
        sequences_per_chunk=sequences_per_chunk,
    )

    with (output_dir / "run_stats.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    with (output_dir / "action_meanings.json").open("w", encoding="utf-8") as f:
        json.dump({i: meaning for i, meaning in enumerate(action_meanings)}, f, indent=2)

    print("[done]", json.dumps(asdict(stats), indent=2))


def run_human_collection(
    output_dir: Path,
    world: int,
    stage: int,
    seed: int,
    sequence_length: int,
    sequences_per_chunk: int,
    total_steps: int,
    human_fps: int,
    action_meanings: list[list[str]],
    level_mode: str,
    compress_chunks: bool,
    async_write: bool,
    max_pending_writes: int,
):
    if level_mode == "random":
        env = RandomLevelMarioEnv(seed=seed)
    else:
        env = make_shimmed_env(world=world, stage=stage, seed=seed)
    writer = ChunkWriter(
        output_dir=output_dir,
        sequence_length=sequence_length,
        sequences_per_chunk=sequences_per_chunk,
        compress=compress_chunks,
        async_write=async_write,
        max_pending_writes=max_pending_writes,
    )
    policy = HumanActionPolicy(action_meanings=action_meanings, fps=human_fps, seed=seed)

    obs, info = env.reset(seed=seed)

    INFO_KEYS = ["coins", "flag_get", "life", "score", "stage", "time", "world", "x_pos", "y_pos"]
    STATUS_MAP = {"small": 0, "tall": 1, "fireball": 2}

    seq_frames: list[np.ndarray] = []
    seq_actions: list[int] = []
    seq_dones: list[bool] = []
    seq_info = {k: [] for k in INFO_KEYS + ["status"]}

    total_sequences = 0

    try:
        for step_idx in range(total_steps):
            policy.draw_frame(obs)
            action = policy.sample()
            next_obs, _, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            seq_frames.append(to_tchw(obs))
            seq_actions.append(int(action))
            seq_dones.append(done)

            for k in INFO_KEYS:
                seq_info[k].append(int(info.get(k, 0)))
            seq_info["status"].append(STATUS_MAP.get(info.get("status", "small"), 0))

            if len(seq_frames) == sequence_length:
                frames_tchw = np.stack(seq_frames, axis=0)
                actions_t = np.asarray(seq_actions, dtype=np.uint8)
                dones_t = np.asarray(seq_dones, dtype=bool)
                
                kwargs_t = {}
                for k in INFO_KEYS + ["status"]:
                    kwargs_t[k] = np.asarray(seq_info[k], dtype=np.int32)
                    seq_info[k].clear()
                
                out = writer.add_sequence(frames_tchw, actions_t, dones_t, **kwargs_t)
                total_sequences += 1
                seq_frames.clear()
                seq_actions.clear()
                seq_dones.clear()
                if out is not None:
                    print(f"[chunk] wrote {out}")

            obs = next_obs
            info = next_info
            if done:
                obs, info = env.reset()

            if (step_idx + 1) % 500 == 0:
                print(f"[step {step_idx + 1}/{total_steps}] sequences={total_sequences}")

    finally:
        policy.close()
        writer.close()
        env.close()

    stats = VectorRunStats(
        mode="human",
        level_mode=level_mode,
        world=world,
        stage=stage,
        num_envs=1,
        total_env_steps=total_steps,
        total_sequences=total_sequences,
        output_dir=str(output_dir),
        sequence_length=sequence_length,
        sequences_per_chunk=sequences_per_chunk,
    )

    with (output_dir / "run_stats.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    with (output_dir / "action_meanings.json").open("w", encoding="utf-8") as f:
        json.dump({i: meaning for i, meaning in enumerate(action_meanings)}, f, indent=2)

    print("[done]", json.dumps(asdict(stats), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 Mario vectorized data collection")
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase1/vector"))
    parser.add_argument("--mode", type=str, default="random", choices=["random", "heuristic", "human"])
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--level-mode", type=str, default="random", choices=["random", "fixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequences-per-chunk", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--vector-mode", type=str, default="sync", choices=["sync", "async"])
    parser.add_argument("--human-fps", type=int, default=75)
    parser.add_argument("--compress-chunks", action="store_true", help="Compress chunks with np.savez_compressed")
    parser.add_argument("--async-write", action="store_true", help="Write chunk files in a background thread")
    parser.add_argument("--max-pending-writes", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()

    compress_chunks = False
    if args.compress_chunks:
        compress_chunks = True

    run_collection(
        output_dir=args.output_dir,
        mode=args.mode,
        world=args.world,
        stage=args.stage,
        seed=args.seed,
        sequence_length=SEQUENCE_LENGTH,
        sequences_per_chunk=args.sequences_per_chunk,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        vector_mode=args.vector_mode,
        human_fps=args.human_fps,
        level_mode=args.level_mode,
        compress_chunks=compress_chunks,
        async_write=args.async_write,
        max_pending_writes=args.max_pending_writes,
    )

    elapsed = time.time() - started
    print(f"[timing] elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
