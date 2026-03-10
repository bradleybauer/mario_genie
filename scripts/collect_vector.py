#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import select

import evdev
import gymnasium as gym
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.actions import get_action_meanings, get_num_actions
from mario_world_model.config import SEQUENCE_LENGTH
from mario_world_model.coverage import (
    PROGRESSION_BIN_SIZE,
    compute_action_balance,
    compute_progression_balance,
    print_progression_guidance,
    scan_action_coverage,
    scan_progression_coverage,
    validate_progression_bin_size,
)
from mario_world_model.envs import RandomLevelMarioEnv, make_shimmed_env
from mario_world_model.palette_mapper import PaletteMapper
from mario_world_model.preprocess import preprocess_frame
from mario_world_model.rollouts import (
    EpisodeTracker,
    RolloutIndex,
    RolloutWriter,
)
from mario_world_model.storage import ChunkWriter


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

        # Action balance weights (None → use default policy)
        self._action_weights: Optional[list[float]] = None

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

    def update_action_weights(self, weights: dict[int, float]) -> None:
        """Set per-action sampling weights for action-balanced collection.

        The *weights* dict maps action indices to non-negative floats.
        When active, heuristic sampling blends its built-in distribution
        with the balance weights, and random sampling uses weighted choice.
        """
        raw = [max(0.0, weights.get(a, 0.0)) for a in range(self.num_actions)]
        total = sum(raw)
        if total == 0:
            self._action_weights = None
        else:
            self._action_weights = [w / total for w in raw]

    def _sample_random(self, idx: int) -> int:
        if self._t[idx] >= self._sticky_until[idx]:
            if self._action_weights is not None:
                self._sticky_actions[idx] = self.rng.choices(
                    range(self.num_actions), weights=self._action_weights, k=1
                )[0]
            else:
                self._sticky_actions[idx] = self.rng.randrange(self.num_actions)
            hold_for = self.rng.randint(3, 12)
            self._sticky_until[idx] = self._t[idx] + hold_for
        self._t[idx] += 1
        return self._sticky_actions[idx]

    def _sample_heuristic(self) -> int:
        # If action balance weights are active, blend with heuristic
        if self._action_weights is not None:
            # 50% heuristic, 50% balance-weighted
            if self.rng.random() < 0.5:
                return self.rng.choices(
                    range(self.num_actions), weights=self._action_weights, k=1
                )[0]

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
        pygame.display.set_caption("Mario Data Human Recorder")
        self.clock = pygame.time.Clock()
        self._action_to_index = {frozenset(token.lower() for token in action): idx for idx, action in enumerate(action_meanings)}
        self._noop_index = self._action_to_index.get(frozenset({"noop"}), 0)

        # Initialize gamepad via evdev (works in WSL without joydev module)
        self._evdev_device: evdev.InputDevice | None = None
        self._evdev_axes = {"ABS_X": 128, "ABS_Y": 128}  # centered defaults
        self._evdev_buttons: set[int] = set()  # currently held button codes
        self._init_evdev_gamepad()

    def _init_evdev_gamepad(self) -> None:
        """Try to find a gamepad via evdev (reads /dev/input/event* directly)."""
        try:
            paths = evdev.list_devices()
            if not paths:
                print("No evdev input devices found. Falling back to keyboard.")
                return
            for path in paths:
                try:
                    dev = evdev.InputDevice(path)
                except PermissionError:
                    print(f"Permission denied: {path}")
                    print(f"  Fix: sudo chmod 666 {path}")
                    print(f"  Permanent: sudo usermod -aG input $(whoami)  (then restart WSL)")
                    continue
                caps = dev.capabilities(verbose=False)
                # A gamepad should have EV_KEY and EV_ABS
                has_abs = evdev.ecodes.EV_ABS in caps
                has_key = evdev.ecodes.EV_KEY in caps
                if has_abs and has_key:
                    self._evdev_device = dev
                    self._evdev_axis_ranges: dict[str, tuple[int, int]] = {}
                    # Read axis ranges for normalization
                    for abs_code, abs_info in caps.get(evdev.ecodes.EV_ABS, []):
                        name = evdev.ecodes.ABS.get(abs_code, f"ABS_{abs_code}")
                        if isinstance(name, list):
                            name = name[0]
                        # Store the midpoint as the default/centered value
                        mid = (abs_info.min + abs_info.max) // 2
                        self._evdev_axes[name] = mid
                        self._evdev_axis_ranges[name] = (abs_info.min, abs_info.max)
                        # Keep legacy single-range for ABS_X/ABS_Y
                        if name in ("ABS_X", "ABS_Y"):
                            self._evdev_axis_range = (abs_info.min, abs_info.max)
                    print(f"Initialized evdev gamepad: {dev.name} ({dev.path})")
                    print(f"  Axes: {list(self._evdev_axis_ranges.keys())}")
                    return
        except Exception as exc:
            print(f"evdev gamepad probe failed: {exc}")
        print("No gamepad found via evdev. Falling back to keyboard.")
        print("  Hint: ensure /dev/input/event* is readable (sudo chmod 666 /dev/input/event*)")

    def _poll_evdev(self) -> None:
        """Non-blocking read of all pending evdev events to update axis/button state."""
        dev = self._evdev_device
        if dev is None:
            return
        try:
            while select.select([dev.fd], [], [], 0)[0]:
                for event in dev.read():
                    if event.type == evdev.ecodes.EV_ABS:
                        name = evdev.ecodes.ABS.get(event.code, f"ABS_{event.code}")
                        if isinstance(name, list):
                            name = name[0]
                        self._evdev_axes[name] = event.value
                    elif event.type == evdev.ecodes.EV_KEY:
                        if event.value >= 1:  # pressed or held
                            self._evdev_buttons.add(event.code)
                        else:  # released
                            self._evdev_buttons.discard(event.code)
        except (OSError, IOError):
            # Device disconnected
            print("Gamepad disconnected!")
            self._evdev_device = None

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

    def _compose_action(self, *, right: bool, left: bool, up: bool, down: bool, jump: bool, sprint: bool) -> int:
        if right and left:
            left = False

        # The action space only includes a standalone "up" action, so give it
        # priority over other held inputs when climbing vines or entering pipes.
        if up and not down:
            return self._lookup_action({"up"})

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

    def _keys_to_action(self, keys: pygame.key.ScancodeWrapper) -> int:
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        jump = keys[pygame.K_o]
        sprint = keys[pygame.K_p]

        # Poll evdev gamepad (non-blocking)
        self._poll_evdev()
        if self._evdev_device is not None:
            axis_ranges = getattr(self, '_evdev_axis_ranges', {})

            def _axis_norm(axis_name: str) -> float:
                """Normalize an axis value to [-1, 1]."""
                ax_min, ax_max = axis_ranges.get(axis_name, (0, 255))
                ax_mid = (ax_min + ax_max) / 2.0
                ax_half = max((ax_max - ax_min) / 2.0, 1)
                return (self._evdev_axes.get(axis_name, ax_mid) - ax_mid) / ax_half

            # Analog stick (ABS_X / ABS_Y)
            x_val = _axis_norm("ABS_X")
            y_val = _axis_norm("ABS_Y")
            right = right or x_val > 0.5
            left = left or x_val < -0.5
            up = up or y_val < -0.5
            down = down or y_val > 0.5

            # HAT / d-pad axes (ABS_HAT0X / ABS_HAT0Y) — common on budget USB gamepads
            hat_x = _axis_norm("ABS_HAT0X")
            hat_y = _axis_norm("ABS_HAT0Y")
            right = right or hat_x > 0.5
            left = left or hat_x < -0.5
            up = up or hat_y < -0.5
            down = down or hat_y > 0.5

            # Button input: BTN_TRIGGER(0x120)=btn0, BTN_THUMB(0x121)=btn1, etc.
            # Map: button 1 (BTN_THUMB) -> jump, button 0 (BTN_TRIGGER) -> sprint
            # If these are backwards on your controller, swap the codes!
            jump = jump or (evdev.ecodes.BTN_THUMB in self._evdev_buttons)
            sprint = sprint or (evdev.ecodes.BTN_TRIGGER in self._evdev_buttons)
        return self._compose_action(
            right=right,
            left=left,
            up=up,
            down=down,
            jump=jump,
            sprint=sprint,
        )

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


class VectorCollectionVisualizer:
    def __init__(
        self,
        num_envs: int,
        fps: int = 30,
        max_window_size: tuple[int, int] = (1280, 960),
    ):
        self.num_envs = int(num_envs)
        self.fps = int(fps)
        self.max_window_size = max_window_size
        self.screen: pygame.Surface | None = None
        self.tile_size: tuple[int, int] | None = None
        self.grid_size: tuple[int, int] | None = None
        self.clock = pygame.time.Clock()
        pygame.init()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("Vector recorder window closed.")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt("Vector recorder stopped by Escape key.")

    def _ensure_screen(self, frame_hwc: np.ndarray) -> None:
        if self.screen is not None:
            return

        frame_h, frame_w = frame_hwc.shape[:2]
        cols = max(1, math.ceil(math.sqrt(self.num_envs)))
        rows = max(1, math.ceil(self.num_envs / cols))

        max_w, max_h = self.max_window_size
        scale = min(max_w / max(1, cols * frame_w), max_h / max(1, rows * frame_h))
        scale = max(scale, 0.25)

        tile_w = max(1, int(frame_w * scale))
        tile_h = max(1, int(frame_h * scale))

        self.tile_size = (tile_w, tile_h)
        self.grid_size = (cols, rows)
        self.screen = pygame.display.set_mode((cols * tile_w, rows * tile_h))
        pygame.display.set_caption("Mario Vector Data Recorder")

    def draw_frames(self, frames_bhwc: np.ndarray) -> None:
        self._handle_events()
        if frames_bhwc.ndim != 4:
            raise ValueError(f"Expected batched frames with shape (B, H, W, C), got {frames_bhwc.shape}")
        if frames_bhwc.shape[0] == 0:
            return

        self._ensure_screen(frames_bhwc[0])
        assert self.screen is not None
        assert self.tile_size is not None
        assert self.grid_size is not None

        tile_w, tile_h = self.tile_size
        cols, _rows = self.grid_size
        self.screen.fill((0, 0, 0))

        for env_idx, frame_hwc in enumerate(frames_bhwc):
            surface = pygame.surfarray.make_surface(np.transpose(frame_hwc, (1, 0, 2)))
            if surface.get_size() != self.tile_size:
                surface = pygame.transform.scale(surface, self.tile_size)

            col = env_idx % cols
            row = env_idx // cols
            self.screen.blit(surface, (col * tile_w, row * tile_h))

        pygame.display.flip()
        self.clock.tick(max(1, self.fps))

    def close(self) -> None:
        pygame.quit()


def to_tchw(frame_hwc: np.ndarray, palette_mapper: PaletteMapper | None = None) -> np.ndarray:
    """Preprocess a raw NES frame → (C, H, W) uint8.

    If *palette_mapper* is provided the output has shape ``(1, H, W)``
    containing palette indices.  Otherwise ``(3, H, W)`` RGB.
    """
    padded = preprocess_frame(frame_hwc)
    if palette_mapper is not None:
        idx_hw = palette_mapper.map_frame(padded)  # (H, W) uint8
        return idx_hw[np.newaxis, :, :]  # (1, H, W)
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


# ---------------------------------------------------------------------------
# Rebalance helpers (shared between vector and human modes)
# ---------------------------------------------------------------------------

def _build_replay_pool(
    output_dir: Path,
    progression_bin_size: int,
) -> list[tuple[tuple[int, int], list[int], int, float, int]]:
    """Scan progression coverage + rollouts, return weighted replay pool.

    Returns a list of ``(level, actions, target_step, weight, x_bin)`` tuples that
    can be fed directly to ``RandomLevelMarioEnv.update_progression_balance()``.
    """
    prog_cov = scan_progression_coverage(str(output_dir), bin_size=progression_bin_size)
    ri = RolloutIndex(output_dir)
    reachable = ri.reachable_bins(bin_size=progression_bin_size)
    prog_rpt = compute_progression_balance(prog_cov, reachable, bin_size=progression_bin_size)
    print_progression_guidance(prog_rpt, n=5)

    pool: list[tuple[tuple[int, int], list[int], int, float, int]] = []
    for (w, s, b), weight in prog_rpt.weights.items():
        if weight <= 0:
            continue
        if b == 0:
            # Bin 0 = level start — zero-step replay (just selects the level)
            pool.append(((w, s), [], 0, weight, b))
            continue
        all_replays = ri.find_all_replay_actions(w, s, b, bin_size=progression_bin_size)
        if not all_replays:
            continue
        # Split the bin's weight equally across candidate rollouts
        per_rollout_weight = weight / len(all_replays)
        for actions_list, target_step in all_replays:
            pool.append(((w, s), actions_list, target_step, per_rollout_weight, b))

    return pool


def _do_progression_rebalance_vector(
    env: gym.vector.VectorEnv,
    output_dir: Path,
    num_envs: int,
    progression_bin_size: int,
) -> None:
    """Build a replay pool and push it to every sub-env."""
    pool = _build_replay_pool(output_dir, progression_bin_size)
    if not pool:
        print("[progression] No reachable mid-level bins yet — skipping replay pool update.")
        return
    for sub_env in env.envs:  # type: ignore[attr-defined]
        if hasattr(sub_env, "update_progression_balance"):
            sub_env.update_progression_balance(pool, replay_probability=1.0)
    print(f"[progression] Updated replay pool ({len(pool)} candidates) on {num_envs} envs")


def _do_progression_rebalance_human(
    env: RandomLevelMarioEnv,
    output_dir: Path,
    progression_bin_size: int,
) -> None:
    """Build a replay pool and push it to the single env."""
    pool = _build_replay_pool(output_dir, progression_bin_size)
    if not pool:
        print("[progression] No reachable mid-level bins yet — skipping replay pool update.")
        return
    env.update_progression_balance(pool, replay_probability=1.0)
    print(f"[progression] Updated replay pool ({len(pool)} candidates)")


def _do_action_rebalance(
    policy: VectorActionPolicy,
    output_dir: Path,
    num_actions: int,
) -> None:
    """Scan action coverage and update the policy's action weights."""
    act_cov = scan_action_coverage(str(output_dir))
    act_rpt = compute_action_balance(act_cov, num_actions)
    policy.update_action_weights(act_rpt.weights)
    print(f"[actions] Rebalanced action weights ({act_rpt.total_frames} total frames)")


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
    max_pending_writes: int,
    visualize: bool = False,
    visualize_fps: int = 30,
    balance: bool = False,
    rebalance_interval: int = 5,
    balance_actions: bool = False,
    progression_bin_size: int = PROGRESSION_BIN_SIZE,
):
    progression_bin_size = validate_progression_bin_size(progression_bin_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Palette mapper (always active) ──────────────────────────────
    palette_mapper = PaletteMapper(output_dir / "palette.json")

    # --balance-actions implies --balance
    if balance_actions:
        balance = True

    if balance and level_mode != "random":
        print("[balance] Warning: --balance requires --level-mode random. Disabling.")
        balance = False
        balance_actions = False

    print(
        f"[config] mode={mode} level_mode={level_mode} world={world} stage={stage} "
        f"num_envs={num_envs} balance={balance} "
        f"balance_actions={balance_actions}"
    )

    action_meanings = get_action_meanings()
    num_actions = get_num_actions()

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
            max_pending_writes=max_pending_writes,
            balance=balance,
            rebalance_interval=rebalance_interval,
            balance_actions=balance_actions,
            progression_bin_size=progression_bin_size,
            palette_mapper=palette_mapper,
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
        compress=True,
        async_write=True,
        max_pending_writes=max_pending_writes,
    )
    policy = VectorActionPolicy(mode=mode, num_envs=num_envs, action_meanings=action_meanings, seed=seed)
    visualizer = (
        VectorCollectionVisualizer(num_envs=num_envs, fps=visualize_fps)
        if visualize
        else None
    )

    # --- Episode tracking (for progression balance) ---
    tracker: EpisodeTracker | None = None
    rollout_writer: RolloutWriter | None = None
    if balance:
        tracker = EpisodeTracker(num_envs)
        rollout_writer = RolloutWriter(output_dir)

    # --- Initial rebalance at startup ---
    # If rollouts/data exist from a previous session, configure the replay pool
    # and action weights BEFORE the first reset so they take effect immediately.
    if balance:
        _do_progression_rebalance_vector(env, output_dir, num_envs, progression_bin_size)
    if balance_actions:
        _do_action_rebalance(policy, output_dir, num_actions)

    obs, info = env.reset(seed=seed)

    # Seed the episode tracker with initial info
    if tracker is not None:
        for env_idx in range(num_envs):
            w_val = info.get("world")
            s_val = info.get("stage")
            l_val = info.get("life")
            tracker.set_initial_info(
                env_idx,
                world=int(w_val[env_idx]) if w_val is not None else 1,
                stage=int(s_val[env_idx]) if s_val is not None else 1,
                life=int(l_val[env_idx]) if l_val is not None else 2,
            )

    INFO_KEYS = ["coins", "flag_get", "life", "score", "stage", "time", "world", "x_pos", "y_pos"]
    STATUS_MAP = {"small": 0, "tall": 1, "fireball": 2}

    seq_frames = [[] for _ in range(num_envs)]
    seq_actions = [[] for _ in range(num_envs)]
    seq_dones = [[] for _ in range(num_envs)]
    seq_info = {k: [[] for _ in range(num_envs)] for k in INFO_KEYS + ["status"]}

    total_sequences = 0
    chunks_since_rebalance = 0

    try:
        for step_idx in range(total_steps):
            if visualizer is not None:
                visualizer.draw_frames(obs)

            actions = policy.sample()
            next_obs, _, terminated, truncated, next_info = env.step(actions)

            done_flags = np.logical_or(terminated, truncated)

            for env_idx in range(num_envs):
                seq_frames[env_idx].append(to_tchw(obs[env_idx], palette_mapper))
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

                # --- Episode tracking ---
                if tracker is not None:
                    x_val = info.get("x_pos")
                    w_val = info.get("world")
                    s_val = info.get("stage")
                    l_val = info.get("life")
                    tracker.record_step(
                        env_idx,
                        action=int(actions[env_idx]),
                        x_pos=int(x_val[env_idx]) if x_val is not None else 0,
                        world=int(w_val[env_idx]) if w_val is not None else 1,
                        stage=int(s_val[env_idx]) if s_val is not None else 1,
                        life=int(l_val[env_idx]) if l_val is not None else 2,
                    )
                    if done_flags[env_idx]:
                        fg_val = next_info.get("flag_get")
                        nl_val = next_info.get("life")
                        rollout = tracker.finish_episode(
                            env_idx,
                            cur_life=int(nl_val[env_idx]) if nl_val is not None else 2,
                            flag_get=int(fg_val[env_idx]) if fg_val is not None else 0,
                            terminated=bool(terminated[env_idx]),
                            truncated=bool(truncated[env_idx]),
                        )
                        if rollout is not None and rollout_writer is not None:
                            rollout_writer.write(rollout)
                        # Re-seed tracker with new episode info (auto-reset provides new info)
                        nw_val = next_info.get("world")
                        ns_val = next_info.get("stage")
                        nnl_val = next_info.get("life")
                        tracker.set_initial_info(
                            env_idx,
                            world=int(nw_val[env_idx]) if nw_val is not None else 1,
                            stage=int(ns_val[env_idx]) if ns_val is not None else 1,
                            life=int(nnl_val[env_idx]) if nnl_val is not None else 2,
                        )

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
                        chunks_since_rebalance += 1

                        # Periodic rebalancing
                        if balance and chunks_since_rebalance >= rebalance_interval:
                            chunks_since_rebalance = 0
                            _do_progression_rebalance_vector(
                                env, output_dir, num_envs, progression_bin_size,
                            )

                            # Action rebalancing
                            if balance_actions:
                                _do_action_rebalance(
                                    policy, output_dir, num_actions,
                                )

            obs = next_obs
            info = next_info

            if (step_idx + 1) % 500 == 0:
                print(f"[step {step_idx + 1}/{total_steps}] sequences={total_sequences}")

        final = writer.flush()
        if final is not None:
            print(f"[chunk] wrote {final}")
    finally:
        if visualizer is not None:
            visualizer.close()
        writer.close()
        if rollout_writer is not None:
            rollout_writer.close()
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
    max_pending_writes: int,
    balance: bool = False,
    rebalance_interval: int = 5,
    balance_actions: bool = False,
    progression_bin_size: int = PROGRESSION_BIN_SIZE,
    palette_mapper: PaletteMapper | None = None,
):
    progression_bin_size = validate_progression_bin_size(progression_bin_size)
    num_actions = get_num_actions()

    if level_mode == "random":
        env = RandomLevelMarioEnv(seed=seed)
    else:
        env = make_shimmed_env(world=world, stage=stage, seed=seed)
    writer = ChunkWriter(
        output_dir=output_dir,
        sequence_length=sequence_length,
        sequences_per_chunk=sequences_per_chunk,
        compress=True,
        async_write=True,
        max_pending_writes=max_pending_writes,
    )
    policy = HumanActionPolicy(action_meanings=action_meanings, fps=human_fps, seed=seed)

    # --- Episode tracking ---
    tracker: EpisodeTracker | None = None
    rollout_writer: RolloutWriter | None = None
    if balance:
        tracker = EpisodeTracker(1)
        rollout_writer = RolloutWriter(output_dir)

    # --- Initial progression rebalance at startup ---
    # If rollouts already exist from a previous session, configure the replay
    # pool BEFORE the first reset so replays work from the very first episode.
    if balance and hasattr(env, 'update_progression_balance'):
        _do_progression_rebalance_human(env, output_dir, progression_bin_size)  # type: ignore[arg-type]

    obs, info = env.reset(seed=seed)

    # Seed the episode tracker
    if tracker is not None:
        tracker.set_initial_info(
            0,
            world=int(info.get("world", 1)),
            stage=int(info.get("stage", 1)),
            life=int(info.get("life", 2)),
        )

    INFO_KEYS = ["coins", "flag_get", "life", "score", "stage", "time", "world", "x_pos", "y_pos"]
    STATUS_MAP = {"small": 0, "tall": 1, "fireball": 2}

    seq_frames: list[np.ndarray] = []
    seq_actions: list[int] = []
    seq_dones: list[bool] = []
    seq_info = {k: [] for k in INFO_KEYS + ["status"]}

    total_sequences = 0
    chunks_since_rebalance = 0

    try:
        for step_idx in range(total_steps):
            policy.draw_frame(obs)
            action = policy.sample()
            next_obs, _, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            seq_frames.append(to_tchw(obs, palette_mapper))
            seq_actions.append(int(action))
            seq_dones.append(done)

            for k in INFO_KEYS:
                seq_info[k].append(int(info.get(k, 0)))
            seq_info["status"].append(STATUS_MAP.get(info.get("status", "small"), 0))

            # --- Episode tracking ---
            if tracker is not None:
                tracker.record_step(
                    0,
                    action=int(action),
                    x_pos=int(info.get("x_pos", 0)),
                    world=int(info.get("world", 1)),
                    stage=int(info.get("stage", 1)),
                    life=int(info.get("life", 2)),
                )
                if done:
                    rollout = tracker.finish_episode(
                        0,
                        cur_life=int(next_info.get("life", 2)),
                        flag_get=int(next_info.get("flag_get", 0)),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                    if rollout is not None and rollout_writer is not None:
                        rollout_writer.write(rollout)

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
                    chunks_since_rebalance += 1

                    # Periodic rebalancing for human mode
                    if balance and chunks_since_rebalance >= rebalance_interval:
                        chunks_since_rebalance = 0
                        if hasattr(env, 'update_progression_balance'):
                            _do_progression_rebalance_human(env, output_dir, progression_bin_size)  # type: ignore[arg-type]

                        if balance_actions:
                            # For human mode we just print guidance (can't override human input)
                            act_cov = scan_action_coverage(str(output_dir))
                            act_rpt = compute_action_balance(act_cov, num_actions)
                            print(f"[actions] Action balance ({act_rpt.total_frames} frames):")
                            needed = sorted(act_rpt.actions, key=lambda a: a.deficit, reverse=True)[:5]
                            for ai in needed:
                                if ai.deficit == 0:
                                    break
                                name = "+".join(action_meanings[ai.action_index]) if ai.action_index < len(action_meanings) else str(ai.action_index)
                                print(f"  Try more: {name} (deficit {ai.deficit})")

            obs = next_obs
            info = next_info
            if done:
                obs, info = env.reset()
                if tracker is not None:
                    tracker.set_initial_info(
                        0,
                        world=int(info.get("world", 1)),
                        stage=int(info.get("stage", 1)),
                        life=int(info.get("life", 2)),
                    )
                if balance and hasattr(env, '_current_level') and env._current_level:
                    w, s = env._current_level
                    replayed = info.get("replayed_steps", 0)
                    if replayed:
                        print(f"[balance] World {w}-{s} (replayed {replayed} steps → x={info.get('replayed_to_x', '?')})")
                    else:
                        print(f"[balance] Next level: World {w}-{s}")

            if (step_idx + 1) % 500 == 0:
                print(f"[step {step_idx + 1}/{total_steps}] sequences={total_sequences}")

    finally:
        policy.close()
        writer.close()
        if rollout_writer is not None:
            rollout_writer.close()
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
    parser = argparse.ArgumentParser(description="Mario vectorized data collection")
    parser.add_argument("--output-dir", type=Path, default=Path("data/vector"))
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
    parser.add_argument("--max-pending-writes", type=int, default=8)
    parser.add_argument("--visualize", action="store_true", help="Render live vector collection frames in a pygame window")
    parser.add_argument("--visualize-fps", type=int, default=30, help="Target FPS for vector visualization; this also throttles collection speed when --visualize is enabled")
    parser.add_argument("--balance", action="store_true", help="Enable progression-aware balanced collection via action replay")
    parser.add_argument("--rebalance-interval", type=int, default=5, help="Re-scan data and update weights every N chunks (default: 5)")
    parser.add_argument("--balance-actions", action="store_true", help="Balance action distribution via dynamic policy weights (implies --balance)")
    parser.add_argument(
        "--progression-bin-size",
        type=int,
        default=PROGRESSION_BIN_SIZE,
        help="Progression coverage bin width in pixels; default is 64",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()

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
        max_pending_writes=args.max_pending_writes,
        visualize=args.visualize,
        visualize_fps=args.visualize_fps,
        balance=args.balance,
        rebalance_interval=args.rebalance_interval,
        balance_actions=args.balance_actions,
        progression_bin_size=args.progression_bin_size,
    )

    elapsed = time.time() - started
    print(f"[timing] elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
