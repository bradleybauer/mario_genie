from __future__ import annotations

import contextlib
import io
import random
from typing import Optional

import gymnasium as gym
from shimmy import GymV21CompatibilityV0

from mario_world_model_phase1.actions import apply_action_space


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


class RandomLevelMarioEnv(gym.Env):
    def __init__(
        self,
        *,
        level_pool: Optional[list[tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.level_pool = level_pool or default_level_pool()
        self.rng = random.Random(seed)
        self._env = None
        self._current_level: tuple[int, int] | None = None
        initial_world, initial_stage = self.rng.choice(self.level_pool)
        self._build_env_for_level(world=initial_world, stage=initial_stage)

    def _build_env_for_level(self, world: int, stage: int):
        if self._env is not None:
            self._env.close()
        self._env = make_shimmed_env(world=world, stage=stage)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._current_level = (world, stage)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(seed)

        if options and "level" in options and options["level"]:
            world, stage = options["level"]
        else:
            world, stage = self.rng.choice(self.level_pool)

        self._build_env_for_level(world=world, stage=stage)
        obs, info = self._env.reset(seed=seed)
        info = dict(info)
        info["world"] = int(world)
        info["stage"] = int(stage)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
