"""
Episode rollout tracking and replay-index utilities.

During data collection, completed episodes are logged to a compact JSONL file
so that the system can later *replay* recorded action sequences to fast-forward
the emulator to a particular x-position within a level.

Key classes
-----------
``EpisodeTracker``
    Maintains per-environment episode buffers during collection and emits
    ``Rollout`` objects when an episode ends.
``RolloutWriter``
    Append-only JSONL writer (one ``Rollout`` per line).
``RolloutIndex``
    Loads the JSONL file into memory and provides fast lookups for progression
    balance and replay scheduling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


DEFAULT_PROGRESSION_BIN_SIZE: int = 64


def validate_progression_bin_size(bin_size: int) -> int:
    value = int(bin_size)
    if value <= 0:
        raise ValueError("progression bin size must be a positive integer")
    return value

PROGRESSION_BIN_SIZE: int = DEFAULT_PROGRESSION_BIN_SIZE


# ---------------------------------------------------------------------------
# Rollout dataclass
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """A complete episode record (lightweight – no frames)."""

    world: int
    stage: int
    actions: list[int]
    x_positions: list[int]
    max_x: int
    outcome: str  # "death" | "flag" | "timeout"
    num_steps: int

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "Rollout":
        d = json.loads(line)
        return cls(**d)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def classify_outcome(
    prev_life: int,
    cur_life: int,
    flag_get: int,
    terminated: bool,
    truncated: bool,
) -> str:
    """Determine how an episode ended.

    Parameters
    ----------
    prev_life : int
        ``info["life"]`` from the step *before* done.
    cur_life : int
        ``info["life"]`` from the step where done=True.
    flag_get : int
        ``info["flag_get"]`` from the step where done=True.
    terminated : bool
        Whether the environment signalled termination.
    truncated : bool
        Whether the environment signalled truncation.

    Returns
    -------
    str
        One of ``"flag"``, ``"death"``, or ``"timeout"``.
    """
    if flag_get:
        return "flag"
    if cur_life < prev_life:
        return "death"
    if truncated and not terminated:
        return "timeout"
    # Fallback: if terminated but no flag and no life decrease, assume death
    return "death"


# ---------------------------------------------------------------------------
# Episode tracker (per-env buffers)
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """Maintains episode-level action+x_pos buffers for *N* environments.

    Call :meth:`record_step` every step and :meth:`finish_episode` when a
    ``done`` signal fires.  The tracker yields a :class:`Rollout` on each
    finished episode.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._actions: list[list[int]] = [[] for _ in range(num_envs)]
        self._x_positions: list[list[int]] = [[] for _ in range(num_envs)]
        self._world: list[int] = [1] * num_envs
        self._stage: list[int] = [1] * num_envs
        self._prev_life: list[int] = [2] * num_envs  # Mario starts with 2 extra lives

    def set_initial_info(self, env_idx: int, world: int, stage: int, life: int) -> None:
        """Seed the tracker with info from the very first observation."""
        self._world[env_idx] = int(world)
        self._stage[env_idx] = int(stage)
        self._prev_life[env_idx] = int(life)

    def record_step(
        self,
        env_idx: int,
        action: int,
        x_pos: int,
        world: int,
        stage: int,
        life: int,
    ) -> None:
        """Append one timestep to the episode buffer for *env_idx*."""
        self._actions[env_idx].append(int(action))
        self._x_positions[env_idx].append(int(x_pos))
        # Keep latest world/stage/life in case the env transitions mid-episode
        self._world[env_idx] = int(world)
        self._stage[env_idx] = int(stage)
        self._prev_life[env_idx] = int(life)

    def finish_episode(
        self,
        env_idx: int,
        cur_life: int,
        flag_get: int,
        terminated: bool,
        truncated: bool,
    ) -> Optional[Rollout]:
        """Finalise the current episode for *env_idx* and return a Rollout.

        Returns ``None`` if no steps were recorded (e.g. immediate reset).
        """
        actions = self._actions[env_idx]
        x_positions = self._x_positions[env_idx]
        if not actions:
            return None

        outcome = classify_outcome(
            prev_life=self._prev_life[env_idx],
            cur_life=int(cur_life),
            flag_get=int(flag_get),
            terminated=terminated,
            truncated=truncated,
        )

        rollout = Rollout(
            world=self._world[env_idx],
            stage=self._stage[env_idx],
            actions=list(actions),
            x_positions=list(x_positions),
            max_x=max(x_positions) if x_positions else 0,
            outcome=outcome,
            num_steps=len(actions),
        )

        # Reset the per-env buffer
        self._actions[env_idx] = []
        self._x_positions[env_idx] = []

        return rollout


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

class RolloutWriter:
    """Append-only JSONL writer for :class:`Rollout` objects."""

    def __init__(self, output_dir: str | Path):
        self._path = Path(output_dir) / "rollouts.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")

    def write(self, rollout: Rollout) -> None:
        self._fh.write(rollout.to_json_line() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# In-memory rollout index
# ---------------------------------------------------------------------------

class RolloutIndex:
    """Loads ``rollouts.jsonl`` and provides fast lookups.

    The index is built lazily on first access so construction is cheap even if
    the file doesn't exist yet.
    """

    def __init__(self, data_dir: str | Path):
        self._path = Path(data_dir) / "rollouts.jsonl"
        self._rollouts: list[Rollout] = []
        # (world, stage) -> list of rollout indices sorted by max_x desc
        self._by_level: dict[tuple[int, int], list[int]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = Rollout.from_json_line(line)
                    idx = len(self._rollouts)
                    self._rollouts.append(r)
                    key = (r.world, r.stage)
                    self._by_level.setdefault(key, []).append(idx)
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
        # Sort each level's rollouts by max_x descending (longest reach first)
        for key in self._by_level:
            self._by_level[key].sort(key=lambda i: self._rollouts[i].max_x, reverse=True)

    @property
    def rollouts(self) -> list[Rollout]:
        self._ensure_loaded()
        return self._rollouts

    def find_rollout(
        self,
        world: int,
        stage: int,
        target_x_bin: int,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> Optional[Rollout]:
        """Find the shortest rollout that reaches *target_x_bin* in the given level.

        Only considers rollouts that started from level-start (x_positions[0]
        within bin 0).  Rollouts recorded after a replay fast-forward have
        inflated x_positions and cannot be replayed from scratch.

        Returns ``None`` if no suitable rollout exists.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        target_x = target_x_bin * bin_size
        candidates = self._by_level.get((world, stage), [])
        best: Optional[Rollout] = None
        for ri in candidates:
            r = self._rollouts[ri]
            # Skip rollouts that started mid-level (from a replay fast-forward)
            if r.x_positions and r.x_positions[0] >= bin_size:
                continue
            if r.max_x >= target_x:
                if best is None or r.num_steps < best.num_steps:
                    best = r
            else:
                # List is sorted by max_x desc, so no more candidates can reach target
                break
        return best

    def find_replay_actions(
        self,
        world: int,
        stage: int,
        target_x_bin: int,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> Optional[tuple[list[int], int]]:
        """Return ``(actions, target_step)`` to replay up to *target_x_bin*.

        *target_step* is the index of the first action where the recorded
        x_pos reached the target bin, so replaying ``actions[:target_step]``
        positions Mario near the desired x.
        """
        bin_size = validate_progression_bin_size(bin_size)
        rollout = self.find_rollout(world, stage, target_x_bin, bin_size)
        if rollout is None:
            return None
        target_x = target_x_bin * bin_size
        # Find the step where x first reaches the target bin
        target_step = len(rollout.actions)  # fallback: replay everything
        for i, xp in enumerate(rollout.x_positions):
            if xp >= target_x:
                target_step = i + 1  # include this step
                break
        return rollout.actions, target_step

    def find_all_replay_actions(
        self,
        world: int,
        stage: int,
        target_x_bin: int,
        bin_size: int = PROGRESSION_BIN_SIZE,
        max_results: int = 10,
    ) -> list[tuple[list[int], int]]:
        """Return up to *max_results* ``(actions, target_step)`` pairs, shortest first.

        Like :meth:`find_replay_actions` but returns multiple candidates so
        the replay pool has alternatives when one rollout fails.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        target_x = target_x_bin * bin_size
        candidates = self._by_level.get((world, stage), [])
        # Collect valid rollouts that reach the target
        valid: list[Rollout] = []
        for ri in candidates:
            r = self._rollouts[ri]
            if r.x_positions and r.x_positions[0] >= bin_size:
                continue
            if r.max_x >= target_x:
                valid.append(r)
        # Sort by num_steps ascending (shortest first) and take top N
        valid.sort(key=lambda r: r.num_steps)
        valid = valid[:max_results]
        results: list[tuple[list[int], int]] = []
        for r in valid:
            target_step = len(r.actions)
            for i, xp in enumerate(r.x_positions):
                if xp >= target_x:
                    target_step = i + 1
                    break
            results.append((list(r.actions), target_step))
        return results

    def progression_coverage(
        self,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> dict[tuple[int, int, int], int]:
        """Return ``{(world, stage, x_bin): rollout_count}``.

        A rollout *covers* bin ``b`` if its ``max_x >= b * bin_size``.
        This tells us which progression bins we could potentially replay to.
        Only counts rollouts that started from level-start.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        cov: dict[tuple[int, int, int], int] = {}
        for r in self._rollouts:
            # Skip rollouts that started mid-level (from a replay fast-forward)
            if r.x_positions and r.x_positions[0] >= bin_size:
                continue
            max_bin = r.max_x // bin_size
            for b in range(max_bin + 1):
                key = (r.world, r.stage, b)
                cov[key] = cov.get(key, 0) + 1
        return cov

    def reachable_bins(
        self,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> set[tuple[int, int, int]]:
        """Return the set of ``(world, stage, x_bin)`` tuples we can replay to.

        Only considers rollouts that started from level-start.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        result: set[tuple[int, int, int]] = set()
        for r in self._rollouts:
            # Skip rollouts that started mid-level (from a replay fast-forward)
            if r.x_positions and r.x_positions[0] >= bin_size:
                continue
            max_bin = r.max_x // bin_size
            for b in range(max_bin + 1):
                result.add((r.world, r.stage, b))
        return result
