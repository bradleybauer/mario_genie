from __future__ import annotations

import contextlib
import io
import json
import random
from typing import Optional

import gymnasium as gym
import numpy as np
from shimmy import GymV21CompatibilityV0

from mario_world_model.actions import apply_action_space


ROLLOUT_PREFIX_INFO_KEY = "rollout_prefix_json"


def _encode_rollout_prefix(actions: list[int], x_positions: list[int]) -> str:
    payload = {
        "actions": [int(action) for action in actions],
        "x_positions": [int(x_pos) for x_pos in x_positions],
    }
    return json.dumps(payload, separators=(",", ":"))


def decode_rollout_prefix(info: dict) -> tuple[list[int], list[int]]:
    """Extract ``(actions, x_positions)`` from the info dict returned by reset.

    Returns two empty lists when no replay prefix is present.
    """
    raw = info.get(ROLLOUT_PREFIX_INFO_KEY)
    if not raw:
        return [], []
    payload = json.loads(raw)
    return payload.get("actions", []), payload.get("x_positions", [])


def _load_super_mario_bros_env_class():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

    return SuperMarioBrosEnv


def make_shimmed_env(world: int, stage: int, seed: Optional[int] = None):
    """
    Build a Gymnasium-compatible Mario env from legacy gym-super-mario-bros.

    Pipeline:
      SuperMarioBrosEnv (legacy gym API) -> JoypadSpace(COMPLEX_MOVEMENT)
      -> GymV21CompatibilityV0 (modern reset/step API)
    """
    super_mario_bros_env = _load_super_mario_bros_env_class()
    legacy_env = super_mario_bros_env(target=(world, stage))
    legacy_env = apply_action_space(legacy_env)
    env = GymV21CompatibilityV0(env=legacy_env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def default_level_pool() -> list[tuple[int, int]]:
    return [(world, stage) for world in range(1, 9) for stage in range(1, 5)]


ReplayPoolEntry = tuple[tuple[int, int], list[int], int, float, int]


class RandomLevelMarioEnv(gym.Env):
    def __init__(
        self,
        *,
        level_pool: Optional[list[tuple[int, int]]] = None,
        level_weights: Optional[dict[tuple[int, int], float]] = None,
        initial_level: Optional[tuple[int, int]] = None,
        seed: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__()
        self.level_pool = level_pool or default_level_pool()
        self.rng = random.Random(seed)
        self._env = None
        self._current_level: tuple[int, int] | None = None
        self._initial_reset_level = initial_level
        self.max_episode_steps = max_episode_steps
        self._episode_steps = 0

        # Weighted level sampling (None → uniform)
        self._level_probs: Optional[list[float]] = None
        if level_weights is not None:
            self.update_weights(level_weights)

        # Pending replay schedule (consumed on next reset)
        self._pending_replay: Optional[dict] = None

        # Persistent replay pool for progression balance.
        # Each entry: (level, actions, target_step, weight, x_bin)
        self._replay_pool: list[ReplayPoolEntry] = []
        self._replay_probability: float = 0.0  # probability of doing a replay on each reset

        # Optional callback invoked with (obs_hwc,) for each frame during replay
        self.replay_render_callback: Optional[callable] = None

        # Optional callback invoked with (world, stage, actions) when a replay
        # fails so callers can remove the bad rollout from disk.
        self.on_replay_failed: Optional[callable] = None

        initial_world, initial_stage = self._sample_level()
        self._build_env_for_level(world=initial_world, stage=initial_stage)

    def _build_env_for_level(self, world: int, stage: int):
        if self._env is not None:
            self._env.close()
        self._env = make_shimmed_env(world=world, stage=stage)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._current_level = (world, stage)

    def update_weights(self, weights: dict[tuple[int, int], float]) -> None:
        """Set or replace the per-level sampling weights.

        *weights* maps ``(world, stage)`` to a non-negative float.  Entries
        for levels not in ``self.level_pool`` are ignored.  The values are
        normalised to a probability distribution internally.
        """
        raw = [max(0.0, weights.get(lvl, 0.0)) for lvl in self.level_pool]
        total = sum(raw)
        if total == 0:
            # All weights zero → fall back to uniform
            self._level_probs = None
        else:
            self._level_probs = [w / total for w in raw]

    def _sample_level(self) -> tuple[int, int]:
        """Pick a level using the current probability distribution."""
        if self._level_probs is not None:
            idx = self.rng.choices(range(len(self.level_pool)), weights=self._level_probs, k=1)[0]
            return self.level_pool[idx]
        return self.rng.choice(self.level_pool)

    _REPLAY_MAX_RETRIES: int = 1  # extra attempts after first failure

    def _remove_pool_entry(self, entry: tuple) -> None:
        """Remove a replay pool entry (bad rollout that causes death)."""
        try:
            self._replay_pool.remove(entry)
        except ValueError:
            pass

    def _try_replay(
        self, replay: dict, seed: Optional[int], pool_entry: Optional[ReplayPoolEntry] = None,
    ) -> Optional[tuple]:
        """Attempt a single replay.  Returns ``(obs, info)`` on success, ``None`` on failure.

        On failure the *pool_entry* (if given) is removed from the pool.
        """
        world, stage = replay["level"]
        actions = replay["actions"]
        target_step = replay["target_step"]
        self._build_env_for_level(world=world, stage=stage)
        obs, info = self._env.reset(seed=seed)

        # --- Fast-forward phase ---
        replayed_steps = 0
        replay_x_positions: list[int] = []
        if target_step > 0 and actions:
            print(f"[replay] Attempting replay: World {world}-{stage}, {target_step} steps")
            for i in range(min(target_step, len(actions))):
                obs, _, terminated, truncated, info = self._env.step(actions[i])
                replayed_steps += 1
                replay_x_positions.append(int(info.get("x_pos", 0)))
                if self.replay_render_callback is not None:
                    self.replay_render_callback(obs)
                if terminated or truncated:
                    print(f"[replay] FAILED at step {replayed_steps}/{target_step} "
                          f"(Mario died during fast-forward) — removing from pool")
                    if pool_entry is not None:
                        self._remove_pool_entry(pool_entry)
                    if self.on_replay_failed is not None:
                        self.on_replay_failed(world, stage, actions)
                    return None

        info = dict(info)
        info["world"] = int(world)
        info["stage"] = int(stage)
        info["replayed_to_x"] = int(info.get("x_pos", 0))
        info["replayed_steps"] = replayed_steps
        info[ROLLOUT_PREFIX_INFO_KEY] = _encode_rollout_prefix(
            actions=list(actions[:replayed_steps]),
            x_positions=replay_x_positions,
        )
        if replayed_steps > 0:
            print(f"[replay] SUCCESS: World {world}-{stage} at x={info['replayed_to_x']} "
                  f"({replayed_steps} steps replayed)")
        else:
            print(f"[replay] World {world}-{stage} from level start")
        self._episode_steps = 0
        return obs, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(seed)

        initial_level = self._initial_reset_level
        self._initial_reset_level = None

        # --- Consume pending replay if one was scheduled ---
        replay = self._pending_replay
        self._pending_replay = None
        pool_entry = None

        for attempt in range(1 + self._REPLAY_MAX_RETRIES):
            # --- Auto-select a replay from the pool (progression balance) ---
            if (
                initial_level is None
                and replay is None
                and self._replay_pool
                and self.rng.random() < self._replay_probability
            ):
                pool_weights = [entry[3] for entry in self._replay_pool]
                chosen = self.rng.choices(self._replay_pool, weights=pool_weights, k=1)[0]
                level, actions, target_step, entry_weight, x_bin = chosen
                world, stage = level
                print(
                    f"[replay] Selected choice: World {world}-{stage}, bin={x_bin}, "
                    f"target_step={target_step}, weight={entry_weight:.4f}"
                )
                replay = {"level": level, "actions": actions, "target_step": target_step}
                pool_entry = chosen

            if replay is not None:
                result = self._try_replay(replay, seed, pool_entry)
                if result is not None:
                    return result
                # Failed — clear and retry if attempts remain
                replay = None
                pool_entry = None
                if attempt < self._REPLAY_MAX_RETRIES:
                    print(f"[replay] Retrying ({attempt + 1}/{self._REPLAY_MAX_RETRIES})...")
                continue

            # No replay selected — break to normal reset
            break

        # --- Normal reset (level from options or random sampling) ---
        if options and "level" in options and options["level"]:
            world, stage = options["level"]
        elif initial_level is not None:
            world, stage = initial_level
        else:
            world, stage = self._sample_level()

        self._build_env_for_level(world=world, stage=stage)
        obs, info = self._env.reset(seed=seed)
        info = dict(info)
        info["world"] = int(world)
        info["stage"] = int(stage)
        info["replayed_to_x"] = 0
        info["replayed_steps"] = 0
        info[ROLLOUT_PREFIX_INFO_KEY] = _encode_rollout_prefix(actions=[], x_positions=[])
        self._episode_steps = 0
        return obs, info

    def update_progression_balance(
        self,
        replay_pool: list[ReplayPoolEntry],
        replay_probability: float = 1.0,
    ) -> None:
        """Configure automatic progression-based replays on future resets.

        On each ``reset()``, with probability *replay_probability*, a replay
        is sampled from *replay_pool* (weighted by the fourth element) instead
        of doing a normal level-start.  This ensures steady mid-level data
        collection between rebalance intervals.

        Parameters
        ----------
        replay_pool
            List of ``(level, actions, target_step, weight, x_bin)`` tuples.
        replay_probability
            Probability of doing a replay vs. a normal level start on each
            reset.  ``0.5`` means half the episodes start mid-level.
        """
        self._replay_pool = list(replay_pool)
        self._replay_probability = float(replay_probability)

    def schedule_replay(
        self,
        level: tuple[int, int],
        actions: list[int],
        target_step: int,
    ) -> None:
        """Schedule a replay for the *next* ``reset()`` call.

        The env will load *level*, replay *actions[:target_step]* to
        fast-forward Mario to the recorded position, then return the
        resulting observation.  If Mario dies during replay the env falls
        back to the level-start position.
        """
        self._pending_replay = {
            "level": level,
            "actions": actions,
            "target_step": target_step,
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._episode_steps += 1
        if (
            not terminated
            and not truncated
            and self.max_episode_steps is not None
            and self._episode_steps >= self.max_episode_steps
        ):
            truncated = True
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
