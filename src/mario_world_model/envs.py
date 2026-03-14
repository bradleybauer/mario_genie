from __future__ import annotations

import contextlib
import io
from typing import Optional

from shimmy import GymV21CompatibilityV0

from mario_world_model.actions import apply_action_space

def _load_super_mario_bros_env_class():
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

    return SuperMarioBrosEnv


def make_shimmed_env(
    world: Optional[int] = None,
    stage: Optional[int] = None,
    seed: Optional[int] = None,
    lock_level: bool = True,
):
    """
    Build a Gymnasium-compatible Mario env from legacy gym-super-mario-bros.

    When *world* and *stage* are both ``None`` the env uses natural game
    progression (1-1 → 1-2 → … → 8-4).  Episodes only end on game-over.
    When both are given and *lock_level* is ``True`` (default), the env is
    locked to that single level.  When *lock_level* is ``False``, the level
    is used only for the initial starting position and the env behaves like
    natural progression (deaths decrement lives, game-over at 0 lives).

    Pipeline:
      SuperMarioBrosEnv (legacy gym API) -> JoypadSpace(COMPLEX_MOVEMENT)
      -> GymV21CompatibilityV0 (modern reset/step API)
    """
    super_mario_bros_env = _load_super_mario_bros_env_class()
    if world is not None and stage is not None:
        legacy_env = super_mario_bros_env(target=(world, stage))
        if not lock_level:
            # Clear target so the env uses natural progression rules:
            # deaths decrement lives instead of ending the episode.
            legacy_env._target_world = None
            legacy_env._target_area = None
    else:
        legacy_env = super_mario_bros_env()  # natural progression
    legacy_env = apply_action_space(legacy_env)
    env = GymV21CompatibilityV0(env=legacy_env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def default_level_pool() -> list[tuple[int, int]]:
    return [(world, stage) for world in range(1, 9) for stage in range(1, 5)]
