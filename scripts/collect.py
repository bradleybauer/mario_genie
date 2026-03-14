#!/usr/bin/env python3
"""Session-based Mario data collection.

Each run produces a single contiguous session file:

    session_XXXXXX.npz        — flat (N, C, H, W) frames + (N,) actions/metadata
    session_XXXXXX.meta.json  — session-level metadata + coverage summaries

The dataset is a collection of many sessions.  Training dataloaders sample
windows *within* sessions, never spanning across session boundaries.  Every
frame-to-frame transition inside a session is a real gameplay transition
(including natural deaths, flag captures, and game-overs).

Usage examples:

    # Heuristic bot on World 1-1
    python scripts/collect.py --mode heuristic --world 1 --stage 1 --total-steps 20000

    # Human play
    python scripts/collect.py --mode human --world 1 --stage 1 --total-steps 10000

    # Balanced session: auto-pick an underrepresented starting position
    python scripts/collect.py --mode heuristic --balance --total-steps 20000
"""
from __future__ import annotations

import argparse
import json
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
from mario_world_model.coverage import (
    PROGRESSION_BIN_SIZE,
    compute_action_balance,
    compute_progression_balance,
    print_progression_guidance,
    scan_action_coverage,
    scan_progression_coverage,
    validate_progression_bin_size,
)
from mario_world_model.envs import make_shimmed_env
from mario_world_model.palette_mapper import PaletteMapper
from mario_world_model.preprocess import preprocess_frame
from mario_world_model.rollouts import (
    EpisodeTracker,
    RolloutIndex,
    RolloutWriter,
    remove_rollout_from_jsonl,
)
from mario_world_model.storage import _compute_summaries

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INFO_KEYS = ["coins", "flag_get", "life", "score", "stage", "time", "world", "x_pos", "y_pos"]
STATUS_MAP = {"small": 0, "tall": 1, "fireball": 2}


# ---------------------------------------------------------------------------
# Session metadata + writer
# ---------------------------------------------------------------------------

@dataclass
class SessionMeta:
    session_id: str
    mode: str
    start_world: int
    start_stage: int
    start_x_pos: int
    replay_prefix_length: int
    seed: int
    num_frames: int
    elapsed_seconds: float
    action_summary: dict[str, int] | None = None
    exact_progression_summary: dict[str, int] | None = None


class SessionWriter:
    """Accumulates frames and metadata, writes a single session ``.npz``."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(output_dir.glob("session_*.npz"))
        if existing:
            self._session_index = max(
                int(p.stem.split("_")[1]) for p in existing
            ) + 1
        else:
            self._session_index = 0
        self.session_id = f"{self._session_index:06d}"
        self._frames: list[np.ndarray] = []
        self._actions: list[int] = []
        self._dones: list[bool] = []
        self._metadata: dict[str, list[int]] = {}

    def append(
        self,
        frame_chw: np.ndarray,
        action: int,
        done: bool,
        **metadata: int,
    ) -> None:
        self._frames.append(frame_chw)
        self._actions.append(action)
        self._dones.append(done)
        for k, v in metadata.items():
            if k not in self._metadata:
                self._metadata[k] = []
            self._metadata[k].append(v)

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def write(self, meta: SessionMeta) -> Path:
        """Stack all accumulated data and write to a compressed ``.npz``."""
        npz_path = self.output_dir / f"session_{self.session_id}.npz"
        meta_path = self.output_dir / f"session_{self.session_id}.meta.json"

        if not self._frames:
            raise ValueError("No frames to write")

        frames = np.stack(self._frames, axis=0)   # (N, C, H, W)
        actions = np.asarray(self._actions, dtype=np.uint8)
        dones = np.asarray(self._dones, dtype=bool)

        arrays: dict[str, np.ndarray] = {
            "frames": frames,
            "actions": actions,
            "dones": dones,
        }
        for k, v in self._metadata.items():
            arrays[k] = np.asarray(v, dtype=np.int32)

        np.savez_compressed(npz_path, **arrays)

        # Compute summaries for fast coverage scanning
        action_summ, prog_summ = _compute_summaries(arrays)
        meta.num_frames = self.num_frames
        meta.action_summary = action_summ
        meta.exact_progression_summary = prog_summ

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        return npz_path


# ---------------------------------------------------------------------------
# Action policies
# ---------------------------------------------------------------------------

class ActionPolicy:
    """Single-env action policy for random or heuristic modes."""

    def __init__(self, mode: str, action_meanings: list[list[str]], seed: int = 0):
        if mode not in {"random", "heuristic"}:
            raise ValueError("ActionPolicy supports mode in {'random', 'heuristic'}")
        self.mode = mode
        self.rng = random.Random(seed)
        self.num_actions = len(action_meanings)
        self._action_to_index = {frozenset(token.lower() for token in action): idx for idx, action in enumerate(action_meanings)}
        self._noop_index = self._action_to_index.get(frozenset({"noop"}), 0)
        self._sticky_action = 0
        self._sticky_until = 0
        self._t = 0

        self._right_index = self._lookup_action("right")
        self._right_jump_index = self._lookup_action("right", "a")
        self._right_sprint_index = self._lookup_action("right", "b")
        self._right_jump_sprint_index = self._lookup_action("right", "a", "b")
        self._left_index = self._lookup_action("left")
        self._left_sprint_index = self._lookup_action("left", "b")
        self._jump_index = self._lookup_action("a")
        self._down_index = self._lookup_action("down")

        self._heuristic_candidates = self._build_heuristic_candidates()
        self._last_x = 0
        self._last_time: Optional[int] = None
        self._last_level: Optional[tuple[int, int]] = None
        self._stuck_steps = 0
        self._jump_until = 0
        self._sprint_until = 0
        self._backtrack_until = 0

        self._action_weights: Optional[list[float]] = None

    def _lookup_action(self, *buttons: str) -> int:
        return self._action_to_index.get(frozenset(token.lower() for token in buttons), self._noop_index)

    def _reset_heuristic_state(self) -> None:
        self._last_x = 0
        self._last_time = None
        self._last_level = None
        self._stuck_steps = 0
        self._jump_until = self._t
        self._sprint_until = self._t
        self._backtrack_until = self._t

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
        raw = [max(0.0, weights.get(a, 0.0)) for a in range(self.num_actions)]
        total = sum(raw)
        if total == 0:
            self._action_weights = None
        else:
            self._action_weights = [w / total for w in raw]

    def _get_info_value(self, info: Optional[dict], key: str, default: int) -> int:
        if info is None:
            return default
        value = info.get(key)
        if value is None:
            return default
        return int(value)

    def _plan_jump_recovery(self, *, long: bool) -> None:
        jump_frames = self.rng.randint(8, 14) if long else self.rng.randint(4, 8)
        sprint_frames = jump_frames + self.rng.randint(8, 20)
        self._jump_until = max(self._jump_until, self._t + jump_frames)
        self._sprint_until = max(self._sprint_until, self._t + sprint_frames)

    def _update_heuristic_state(self, info: Optional[dict]) -> None:
        world = self._get_info_value(info, "world", 1)
        stage = self._get_info_value(info, "stage", 1)
        x_pos = self._get_info_value(info, "x_pos", 0)
        time_left = self._get_info_value(info, "time", 0)

        current_level = (world, stage)
        prev_level = self._last_level
        prev_time = self._last_time
        prev_x = self._last_x

        level_changed = prev_level is not None and current_level != prev_level
        timer_reset = prev_time is not None and time_left > prev_time + 1
        x_reset = prev_level == current_level and x_pos <= 8 and prev_x > 96
        if level_changed or timer_reset or x_reset:
            self._reset_heuristic_state()
            prev_level = None
            prev_time = None
            prev_x = 0

        if prev_level == current_level and prev_time is not None:
            progress = x_pos - prev_x
            if progress <= 0:
                self._stuck_steps += 1
            elif progress >= 3:
                self._stuck_steps = max(0, self._stuck_steps - 3)
            else:
                self._stuck_steps = max(0, self._stuck_steps - 1)

            if self._stuck_steps >= 12 and self._t >= self._jump_until:
                self._plan_jump_recovery(long=True)
                if self._left_index != self._noop_index and self.rng.random() < 0.2:
                    self._backtrack_until = max(
                        self._backtrack_until,
                        self._t + self.rng.randint(3, 6),
                    )
            elif self._stuck_steps >= 5 and self._t >= self._jump_until and self.rng.random() < 0.2:
                self._plan_jump_recovery(long=False)
        else:
            self._stuck_steps = 0
            self._sprint_until = max(self._sprint_until, self._t + self.rng.randint(10, 24))

        self._last_x = x_pos
        self._last_time = time_left
        self._last_level = current_level

    def _base_heuristic_action(self) -> int:
        p = self.rng.random()
        if p < 0.58:
            return self._right_sprint_index
        if p < 0.76:
            return self._right_jump_sprint_index
        if p < 0.88:
            return self._right_index
        if p < 0.95:
            return self._right_jump_index
        if p < 0.98:
            return self._jump_index
        return self._down_index

    def _sample_random(self) -> int:
        if self._t >= self._sticky_until:
            if self._action_weights is not None:
                self._sticky_action = self.rng.choices(
                    range(self.num_actions), weights=self._action_weights, k=1
                )[0]
            else:
                self._sticky_action = self.rng.randrange(self.num_actions)
            hold_for = self.rng.randint(3, 12)
            self._sticky_until = self._t + hold_for
        self._t += 1
        return self._sticky_action

    def _sample_heuristic(self, info: Optional[dict]) -> int:
        self._update_heuristic_state(info)

        if self._backtrack_until > self._t:
            action = self._left_sprint_index if self._left_sprint_index != self._noop_index else self._left_index
        elif self._jump_until > self._t:
            action = (
                self._right_jump_sprint_index
                if self._sprint_until > self._t
                else self._right_jump_index
            )
        elif self._action_weights is not None and self.rng.random() < 0.3:
            action = self.rng.choices(
                range(self.num_actions), weights=self._action_weights, k=1
            )[0]
        elif self._sprint_until > self._t:
            action = self._right_sprint_index if self.rng.random() < 0.85 else self._right_index
        else:
            if self.rng.random() < 0.08:
                self._plan_jump_recovery(long=False)
                action = self._right_jump_sprint_index
            else:
                action = self._base_heuristic_action()

        self._t += 1
        return action

    def sample(self, info: Optional[dict] = None) -> int:
        if self.mode == "random":
            return self._sample_random()
        return self._sample_heuristic(info)


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

        self._evdev_device: evdev.InputDevice | None = None
        self._evdev_axes = {"ABS_X": 128, "ABS_Y": 128}
        self._evdev_buttons: set[int] = set()
        self._init_evdev_gamepad()

    def _init_evdev_gamepad(self) -> None:
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
                has_abs = evdev.ecodes.EV_ABS in caps
                has_key = evdev.ecodes.EV_KEY in caps
                if has_abs and has_key:
                    self._evdev_device = dev
                    self._evdev_axis_ranges: dict[str, tuple[int, int]] = {}
                    for abs_code, abs_info in caps.get(evdev.ecodes.EV_ABS, []):
                        name = evdev.ecodes.ABS.get(abs_code, f"ABS_{abs_code}")
                        if isinstance(name, list):
                            name = name[0]
                        mid = (abs_info.min + abs_info.max) // 2
                        self._evdev_axes[name] = mid
                        self._evdev_axis_ranges[name] = (abs_info.min, abs_info.max)
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
                        if event.value >= 1:
                            self._evdev_buttons.add(event.code)
                        else:
                            self._evdev_buttons.discard(event.code)
        except (OSError, IOError):
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

        self._poll_evdev()
        if self._evdev_device is not None:
            axis_ranges = getattr(self, '_evdev_axis_ranges', {})

            def _axis_norm(axis_name: str) -> float:
                ax_min, ax_max = axis_ranges.get(axis_name, (0, 255))
                ax_mid = (ax_min + ax_max) / 2.0
                ax_half = max((ax_max - ax_min) / 2.0, 1)
                return (self._evdev_axes.get(axis_name, ax_mid) - ax_mid) / ax_half

            x_val = _axis_norm("ABS_X")
            y_val = _axis_norm("ABS_Y")
            right = right or x_val > 0.5
            left = left or x_val < -0.5
            up = up or y_val < -0.5
            down = down or y_val > 0.5

            hat_x = _axis_norm("ABS_HAT0X")
            hat_y = _axis_norm("ABS_HAT0Y")
            right = right or hat_x > 0.5
            left = left or hat_x < -0.5
            up = up or hat_y < -0.5
            down = down or hat_y > 0.5

            jump = jump or (evdev.ecodes.BTN_THUMB in self._evdev_buttons)
            sprint = sprint or (evdev.ecodes.BTN_TRIGGER in self._evdev_buttons)
        return self._compose_action(
            right=right, left=left, up=up, down=down, jump=jump, sprint=sprint,
        )

    def sample(self) -> int:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("Human recorder window closed.")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt("Human recorder stopped by Escape key.")
        keys = pygame.key.get_pressed()
        action = self._keys_to_action(keys)
        self.clock.tick(max(1, self.fps))
        return action

    def close(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# Collection visualizer (for non-human modes)
# ---------------------------------------------------------------------------

class CollectionVisualizer:
    def __init__(self, fps: int = 30, max_window_size: tuple[int, int] = (1280, 960)):
        self.fps = int(fps)
        self.max_window_size = max_window_size
        self.screen: pygame.Surface | None = None
        self.clock = pygame.time.Clock()
        pygame.init()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("Recorder window closed.")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt("Recorder stopped by Escape key.")

    def _ensure_screen(self, frame_hwc: np.ndarray) -> None:
        if self.screen is not None:
            return
        frame_h, frame_w = frame_hwc.shape[:2]
        max_w, max_h = self.max_window_size
        scale = min(max_w / max(1, frame_w), max_h / max(1, frame_h))
        scale = max(scale, 0.25)
        win_w = max(1, int(frame_w * scale))
        win_h = max(1, int(frame_h * scale))
        self.screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Mario Data Recorder")

    def draw_frame(self, frame_hwc: np.ndarray) -> None:
        self._handle_events()
        self._ensure_screen(frame_hwc)
        assert self.screen is not None
        surface = pygame.surfarray.make_surface(np.transpose(frame_hwc, (1, 0, 2)))
        if surface.get_size() != self.screen.get_size():
            surface = pygame.transform.scale(surface, self.screen.get_size())
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(max(1, self.fps))

    def close(self) -> None:
        pygame.quit()


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def to_chw(frame_hwc: np.ndarray, palette_mapper: PaletteMapper | None = None) -> np.ndarray:
    """Preprocess a raw NES frame → (C, H, W) uint8.

    If *palette_mapper* is provided the output has shape ``(1, H, W)``
    containing palette indices.  Otherwise ``(3, H, W)`` RGB.
    """
    padded = preprocess_frame(frame_hwc)
    if palette_mapper is not None:
        idx_hw = palette_mapper.map_frame(padded)
        return idx_hw[np.newaxis, :, :]
    return np.transpose(padded, (2, 0, 1)).astype(np.uint8, copy=False)


# ---------------------------------------------------------------------------
# Balance: pick an underrepresented starting position
# ---------------------------------------------------------------------------

def find_balance_target(
    output_dir: Path,
    progression_bin_size: int,
    rng: random.Random,
) -> Optional[tuple[int, int, list[int], int]]:
    """Scan existing data, find most underrepresented region, return replay info.

    Returns ``(world, stage, replay_actions, target_step)`` or ``None``.
    """
    prog_cov = scan_progression_coverage(str(output_dir), bin_size=progression_bin_size)
    try:
        ri = RolloutIndex(output_dir)
    except Exception:
        return None
    print(f"[balance] {len(ri.rollouts)} rollouts loaded.")
    reachable = ri.reachable_bins(bin_size=progression_bin_size)
    prog_rpt = compute_progression_balance(prog_cov, reachable, bin_size=progression_bin_size)
    print_progression_guidance(prog_rpt, n=5)

    # Weighted random selection from deficit bins
    keys: list[tuple[int, int, int]] = []
    weights: list[float] = []
    for key, weight in prog_rpt.weights.items():
        if weight > 0:
            keys.append(key)
            weights.append(weight)
    if not keys:
        return None

    w, s, b = rng.choices(keys, weights=weights, k=1)[0]

    if b == 0:
        return (w, s, [], 0)

    replays = ri.find_all_replay_actions(w, s, b, bin_size=progression_bin_size)
    if not replays:
        return (w, s, [], 0)

    actions_list, target_step = rng.choice(replays)
    return (w, s, actions_list, target_step)


def _make_replay_renderer(fps: int = 30):
    """Return a callback ``(obs_hwc,) -> None`` that renders replay frames."""
    screen = [None]
    clock = pygame.time.Clock()

    def render(obs_hwc: np.ndarray) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt("Replay window closed.")
        h, w = obs_hwc.shape[:2]
        if screen[0] is None:
            scale = min(512 / max(w, 1), 480 / max(h, 1))
            screen[0] = pygame.display.set_mode((int(w * scale), int(h * scale)))
            pygame.display.set_caption("Mario Replay Visualization")
        surface = pygame.surfarray.make_surface(np.transpose(obs_hwc, (1, 0, 2)))
        if surface.get_size() != screen[0].get_size():
            surface = pygame.transform.scale(surface, screen[0].get_size())
        screen[0].blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(max(1, fps))

    return render


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def _write_session(
    writer: SessionWriter,
    meta: SessionMeta,
    action_meanings: list[list[str]],
    output_dir: Path,
) -> Optional[Path]:
    """Write a completed session to disk.  Returns the path or None."""
    if writer.num_frames == 0:
        return None
    path = writer.write(meta)
    # Write action meanings alongside sessions (idempotent)
    with (output_dir / "action_meanings.json").open("w", encoding="utf-8") as f:
        json.dump({i: meaning for i, meaning in enumerate(action_meanings)}, f, indent=2)
    return path


def _select_balance_target(
    output_dir: Path,
    progression_bin_size: int,
    rng: random.Random,
) -> tuple[Optional[int], Optional[int], list[int], int]:
    """Run balance selection.  Returns (world, stage, replay_actions, target_step)."""
    bin_size = validate_progression_bin_size(progression_bin_size)
    result = find_balance_target(output_dir, bin_size, rng)
    if result is not None:
        w, s, actions, step = result
        if step > 0:
            print(f"[balance] Target: World {w}-{s}, replay {step} steps")
        else:
            print(f"[balance] Target: World {w}-{s} from level start")
        return w, s, actions, step
    print("[balance] No balance target found, using default")
    return None, None, [], 0


def _build_env_and_replay(
    world: Optional[int],
    stage: Optional[int],
    seed: int,
    replay_actions: list[int],
    target_step: int,
    max_episode_steps: Optional[int],
    output_dir: Path,
    visualize_replays: bool,
    visualize_fps: int,
) -> tuple[gym.Env, np.ndarray, dict, int, list[int]]:
    """Build env, run replay fast-forward.

    Returns (env, obs, info, replay_prefix_length, replay_x_positions).
    """
    if (world is None) != (stage is None):
        raise ValueError("--world and --stage must both be specified, or both omitted")

    # Always use natural progression (lock_level=False) so deaths
    # decrement lives instead of resetting the entire episode.
    env = make_shimmed_env(world=world, stage=stage, seed=seed, lock_level=False)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    obs, info = env.reset(seed=seed)
    replay_prefix_length = 0
    replay_x_positions: list[int] = []

    if replay_actions and target_step > 0:
        replay_render = None
        if visualize_replays:
            pygame.init()
            replay_render = _make_replay_renderer(fps=visualize_fps)

        print(f"[replay] Fast-forwarding {target_step} steps on World {world}-{stage}...")
        for i in range(min(target_step, len(replay_actions))):
            obs, _, term, trunc, info = env.step(replay_actions[i])
            replay_prefix_length += 1
            replay_x_positions.append(int(info.get("x_pos", 0)))
            if replay_render is not None:
                replay_render(obs)
            if term or trunc:
                print(f"[replay] FAILED at step {i+1}/{target_step} — falling back to level start")
                remove_rollout_from_jsonl(
                    output_dir / "rollouts.jsonl", world, stage, replay_actions,
                )
                obs, info = env.reset(seed=seed)
                replay_prefix_length = 0
                replay_x_positions = []
                break
        else:
            print(f"[replay] OK — x={info.get('x_pos', '?')} after {replay_prefix_length} steps")

    return env, obs, info, replay_prefix_length, replay_x_positions


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collection(
    output_dir: Path,
    mode: str,
    world: Optional[int],
    stage: Optional[int],
    seed: int,
    total_steps: int,
    human_fps: int,
    visualize: bool,
    visualize_fps: int,
    visualize_replays: bool,
    balance: bool,
    balance_actions: bool,
    progression_bin_size: int,
    max_episode_steps: Optional[int],
    max_session_seconds: Optional[float],
):
    run_started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    palette_mapper = PaletteMapper(output_dir / "palette.json")
    action_meanings = get_action_meanings()
    num_actions = get_num_actions()
    rng = random.Random(seed)

    # --- Setup policy (shared across sessions) ---
    if mode == "human":
        policy = HumanActionPolicy(action_meanings=action_meanings, fps=human_fps, seed=seed)
    else:
        policy = ActionPolicy(mode=mode, action_meanings=action_meanings, seed=seed)

    # --- Visualizer (shared across sessions) ---
    visualizer = None
    if visualize and mode != "human":
        visualizer = CollectionVisualizer(fps=visualize_fps)

    # --- Rollout writer (shared across sessions) ---
    rollout_writer = RolloutWriter(output_dir)
    total_rollouts_saved = 0
    try:
        ri = RolloutIndex(output_dir)
        total_rollouts_saved = len(ri.rollouts)
    except Exception:
        pass

    remaining_steps = total_steps
    session_num = 0
    total_frames_all = 0
    env: Optional[gym.Env] = None

    try:
        while remaining_steps > 0:
            session_started = time.time()

            # --- Determine starting position ---
            cur_world, cur_stage = world, stage
            replay_actions: list[int] = []
            target_step = 0

            if balance or (session_num > 0 and max_session_seconds is not None):
                bw, bs, replay_actions, target_step = _select_balance_target(
                    output_dir, progression_bin_size, rng,
                )
                if bw is not None and bs is not None:
                    cur_world, cur_stage = bw, bs

            # --- Action balance weights ---
            if balance_actions:
                act_cov = scan_action_coverage(str(output_dir))
                act_rpt = compute_action_balance(act_cov, num_actions)
                if mode != "human":
                    policy.update_action_weights(act_rpt.weights)
                print(f"[actions] Precomputed action weights ({act_rpt.total_frames} total frames)")

            level_str = f"W{cur_world}-{cur_stage}" if cur_world is not None else "natural progression"
            session_limit_str = f", max {max_session_seconds}s" if max_session_seconds else ""
            print(
                f"[session {session_num}] mode={mode} level={level_str} "
                f"remaining_steps={remaining_steps}{session_limit_str}"
            )

            # --- Build env + replay fast-forward ---
            if env is not None:
                env.close()
            env, obs, info, replay_prefix_length, replay_x_positions = (
                _build_env_and_replay(
                    cur_world, cur_stage, seed, replay_actions, target_step,
                    max_episode_steps, output_dir, visualize_replays, visualize_fps,
                )
            )

            start_x = int(info.get("x_pos", 0))
            start_world = int(info.get("world", cur_world or 1))
            start_stage = int(info.get("stage", cur_stage or 1))
            prev_life = int(info.get("life", 2))

            # --- Episode tracker (per session) ---
            tracker = EpisodeTracker(1)
            tracker.set_initial_info(
                0,
                world=int(info.get("world", cur_world or 1)),
                stage=int(info.get("stage", cur_stage or 1)),
                life=int(info.get("life", 2)),
                prefix_actions=list(replay_actions[:replay_prefix_length]),
                prefix_x_positions=replay_x_positions,
            )

            # --- Session writer ---
            writer = SessionWriter(output_dir)
            steps_this_session = 0
            rollout: Optional[object] = None

            # --- Collection loop ---
            for _ in range(remaining_steps):
                # Display
                if mode == "human":
                    policy.draw_frame(obs)
                elif visualizer is not None:
                    visualizer.draw_frame(obs)

                # Sample action
                if mode == "human":
                    action = policy.sample()
                else:
                    action = policy.sample(info)

                # Step env
                next_obs, _, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)

                # Record frame + action + metadata
                writer.append(
                    to_chw(obs, palette_mapper),
                    int(action),
                    done,
                    world=int(info.get("world", 1)),
                    stage=int(info.get("stage", 1)),
                    x_pos=int(info.get("x_pos", 0)),
                    y_pos=int(info.get("y_pos", 0)),
                    score=int(info.get("score", 0)),
                    coins=int(info.get("coins", 0)),
                    life=int(info.get("life", 2)),
                    time=int(info.get("time", 0)),
                    flag_get=int(info.get("flag_get", 0)),
                    status=STATUS_MAP.get(info.get("status", "small"), 0),
                )
                steps_this_session += 1

                # Episode tracking
                transition_rollout = tracker.record_step(
                    0,
                    action=int(action),
                    x_pos=int(info.get("x_pos", 0)),
                    world=int(info.get("world", 1)),
                    stage=int(info.get("stage", 1)),
                    life=int(info.get("life", 2)),
                )

                # Level transition detected (record_step emits rollout)
                if transition_rollout is not None:
                    rollout_writer.write(transition_rollout)
                    total_rollouts_saved += 1
                    print(
                        f"[rollout] W{transition_rollout.world}-{transition_rollout.stage} "
                        f"max_x={transition_rollout.max_x} ({transition_rollout.outcome}) "
                        f"[total: {total_rollouts_saved}]"
                    )
                    print(
                        f"[reset] level transition — lives remaining: {next_info.get('life', 2)}, "
                        f"W{next_info.get('world', 1)}-{next_info.get('stage', 1)}"
                    )
                    tracker.set_initial_info(
                        0,
                        world=int(next_info.get("world", 1)),
                        stage=int(next_info.get("stage", 1)),
                        life=int(next_info.get("life", 2)),
                    )

                # Detect deaths via life decrease (env handles respawn internally)
                cur_life = int(next_info.get("life", 2))
                if not done and cur_life < prev_life:
                    rollout = tracker.finish_episode(
                        0,
                        cur_life=cur_life,
                        flag_get=0,
                        terminated=False,
                        truncated=False,
                    )
                    if rollout is not None:
                        rollout_writer.write(rollout)
                        total_rollouts_saved += 1
                        print(
                            f"[rollout] W{rollout.world}-{rollout.stage} "
                            f"max_x={rollout.max_x} ({rollout.outcome}) "
                            f"[total: {total_rollouts_saved}]"
                        )
                    print(
                        f"[reset] death — lives remaining: {cur_life}, "
                        f"W{next_info.get('world', 1)}-{next_info.get('stage', 1)}"
                    )
                    tracker.set_initial_info(
                        0,
                        world=int(next_info.get("world", 1)),
                        stage=int(next_info.get("stage", 1)),
                        life=cur_life,
                    )

                if done:
                    rollout = tracker.finish_episode(
                        0,
                        cur_life=int(next_info.get("life", 2)),
                        flag_get=int(next_info.get("flag_get", 0)),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                    if rollout is not None:
                        rollout_writer.write(rollout)
                        total_rollouts_saved += 1
                        print(
                            f"[rollout] W{rollout.world}-{rollout.stage} "
                            f"max_x={rollout.max_x} ({rollout.outcome}) "
                            f"[total: {total_rollouts_saved}]"
                        )

                # Advance
                prev_life = cur_life
                obs = next_obs
                info = next_info
                if done:
                    if rollout is not None:
                        reset_reason = rollout.outcome
                    elif truncated:
                        reset_reason = "timeout"
                    else:
                        reset_reason = "game over" if cur_life == 255 else "death"

                    # Game over: rebuild env for natural progression
                    # from 1-1 (mimics real NES restart).
                    is_game_over = cur_life == 255
                    if is_game_over:
                        env.close()
                        env = make_shimmed_env(seed=seed, lock_level=False)
                        if max_episode_steps is not None:
                            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
                        print("[game over] Restarting from W1-1")

                    obs, info = env.reset()
                    prev_life = int(info.get("life", 2))
                    print(
                        f"[reset] {reset_reason} — lives remaining: {prev_life}, "
                        f"W{info.get('world', 1)}-{info.get('stage', 1)}"
                    )
                    tracker.set_initial_info(
                        0,
                        world=int(info.get("world", 1)),
                        stage=int(info.get("stage", 1)),
                        life=prev_life,
                    )

                if steps_this_session % 500 == 0:
                    elapsed_total = remaining_steps - steps_this_session
                    print(
                        f"[step {total_steps - remaining_steps + steps_this_session}/{total_steps}] "
                        f"session_frames={writer.num_frames}"
                    )

                # Check session duration limit
                if max_session_seconds is not None:
                    if time.time() - session_started >= max_session_seconds:
                        print(f"[session {session_num}] Duration limit ({max_session_seconds}s) reached, rotating...")
                        break

            # --- Write session ---
            remaining_steps -= steps_this_session

            if writer.num_frames > 0:
                session_elapsed = time.time() - session_started
                meta = SessionMeta(
                    session_id=writer.session_id,
                    mode=mode,
                    start_world=start_world,
                    start_stage=start_stage,
                    start_x_pos=start_x,
                    replay_prefix_length=replay_prefix_length,
                    seed=seed,
                    num_frames=writer.num_frames,
                    elapsed_seconds=round(session_elapsed, 2),
                )
                path = _write_session(writer, meta, action_meanings, output_dir)
                total_frames_all += writer.num_frames
                print(f"[session {session_num}] wrote {path} ({writer.num_frames} frames, {session_elapsed:.1f}s)")
            else:
                print(f"[session {session_num}] No frames collected.")

            session_num += 1

            # Without a session duration limit, only one session per run
            if max_session_seconds is None:
                break

    except KeyboardInterrupt:
        print(f"\n[interrupted] Saving {writer.num_frames} frames collected so far...")
        if writer.num_frames > 0:
            session_elapsed = time.time() - session_started
            meta = SessionMeta(
                session_id=writer.session_id,
                mode=mode,
                start_world=start_world,
                start_stage=start_stage,
                start_x_pos=start_x,
                replay_prefix_length=replay_prefix_length,
                seed=seed,
                num_frames=writer.num_frames,
                elapsed_seconds=round(session_elapsed, 2),
            )
            path = _write_session(writer, meta, action_meanings, output_dir)
            total_frames_all += writer.num_frames
            session_num += 1
            print(f"[interrupted] wrote {path} ({writer.num_frames} frames)")

    finally:
        if mode == "human":
            policy.close()
        if visualizer is not None:
            visualizer.close()
        rollout_writer.close()
        if env is not None:
            env.close()

    total_elapsed = time.time() - run_started
    print(
        f"[done] {session_num} session(s), {total_frames_all} total frames, "
        f"{total_elapsed:.1f}s elapsed"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Session-based Mario data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/nes"))
    parser.add_argument(
        "--mode", type=str, default="heuristic",
        choices=["random", "heuristic", "human"],
    )
    parser.add_argument("--world", type=int, default=None, help="World (omit with --stage for natural progression)")
    parser.add_argument("--stage", type=int, default=None, help="Stage (omit with --world for natural progression)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=60*60*60, help="Total environment steps to collect (default: 1 hour at 60 FPS)")
    parser.add_argument("--human-fps", type=int, default=60)
    parser.add_argument(
        "--visualize", action="store_true",
        help="Render live collection frames in a pygame window",
    )
    parser.add_argument(
        "--visualize-fps", type=int, default=30,
        help="Target FPS for visualization (throttles collection when enabled)",
    )
    parser.add_argument(
        "--visualize-replays", action="store_true",
        help="Render replay fast-forward frames",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Auto-pick starting position based on existing data coverage",
    )
    parser.add_argument(
        "--balance-actions", action="store_true",
        help="Precompute action weights from existing data for balanced sampling",
    )
    parser.add_argument(
        "--progression-bin-size", type=int, default=PROGRESSION_BIN_SIZE,
        help="Progression coverage bin width in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=None,
        help="Truncate episodes after this many steps",
    )
    parser.add_argument(
        "--max-session-seconds", type=float, default=None,
        help="Max duration per session in seconds.  When reached, the current "
             "session is saved and a new one starts at a balance-selected position.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_collection(
        output_dir=args.output_dir,
        mode=args.mode,
        world=args.world,
        stage=args.stage,
        seed=args.seed,
        total_steps=args.total_steps,
        human_fps=args.human_fps,
        visualize=args.visualize,
        visualize_fps=args.visualize_fps,
        visualize_replays=args.visualize_replays,
        balance=args.balance,
        balance_actions=args.balance_actions,
        progression_bin_size=args.progression_bin_size,
        max_episode_steps=args.max_episode_steps,
        max_session_seconds=args.max_session_seconds,
    )


if __name__ == "__main__":
    main()
