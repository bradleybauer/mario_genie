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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


def validate_progression_bin_size(bin_size: int) -> int:
    value = int(bin_size)
    if value <= 0:
        raise ValueError("progression bin size must be a positive integer")
    return value

PROGRESSION_BIN_SIZE: int = 64


# ---------------------------------------------------------------------------
# Rollout dataclass
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """A complete episode record (lightweight - no frames).

    ``actions`` and ``x_positions`` always describe the full trajectory from
    level start to episode end, including any replay prefix executed during
    reset before live collection resumed.
    """

    world: int
    stage: int
    actions: list[int]
    x_positions: list[int]
    max_x: int
    outcome: str  # "death" | "flag" | "timeout" | "transition"
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
        self._suspended_until_reset: list[bool] = [False] * num_envs

    def _clear_env_buffer(self, env_idx: int) -> None:
        self._actions[env_idx] = []
        self._x_positions[env_idx] = []

    def _build_rollout(self, env_idx: int, outcome: str) -> Optional[Rollout]:
        actions = self._actions[env_idx]
        x_positions = self._x_positions[env_idx]
        if not actions:
            return None
        return Rollout(
            world=self._world[env_idx],
            stage=self._stage[env_idx],
            actions=list(actions),
            x_positions=list(x_positions),
            max_x=max(x_positions) if x_positions else 0,
            outcome=outcome,
            num_steps=len(actions),
        )

    def set_initial_info(
        self,
        env_idx: int,
        world: int,
        stage: int,
        life: int,
        *,
        prefix_actions: Optional[list[int]] = None,
        prefix_x_positions: Optional[list[int]] = None,
    ) -> None:
        """Seed the tracker with info from the first collected observation.

        When an episode starts from a replay fast-forward, *prefix_actions* and
        *prefix_x_positions* preload the trajectory that was already executed
        during ``reset()`` so the eventual rollout remains replayable from
        level start.
        """
        seed_actions = list(prefix_actions or [])
        seed_x_positions = list(prefix_x_positions or [])
        if len(seed_actions) != len(seed_x_positions):
            raise ValueError("prefix_actions and prefix_x_positions must have the same length")

        self._world[env_idx] = int(world)
        self._stage[env_idx] = int(stage)
        self._prev_life[env_idx] = int(life)
        self._actions[env_idx] = seed_actions
        self._x_positions[env_idx] = seed_x_positions
        self._suspended_until_reset[env_idx] = False

    def record_step(
        self,
        env_idx: int,
        action: int,
        x_pos: int,
        world: int,
        stage: int,
        life: int,
    ) -> Optional[Rollout]:
        """Append one timestep to the episode buffer for *env_idx*.

        If the environment changed ``(world, stage)`` since the last observed
        frame, the previous stage-local rollout is emitted with outcome
        ``"transition"`` and tracking is suspended until the next reset.
        This prevents pipe-entered stages from being stored as replayable from
        their nominal stage start.
        """
        observed_world = int(world)
        observed_stage = int(stage)
        observed_life = int(life)

        if self._suspended_until_reset[env_idx]:
            self._world[env_idx] = observed_world
            self._stage[env_idx] = observed_stage
            self._prev_life[env_idx] = observed_life
            return None

        transition_rollout: Optional[Rollout] = None
        if self._actions[env_idx] and (
            observed_world != self._world[env_idx] or observed_stage != self._stage[env_idx]
        ):
            transition_rollout = self._build_rollout(env_idx, outcome="transition")
            self._clear_env_buffer(env_idx)
            self._world[env_idx] = observed_world
            self._stage[env_idx] = observed_stage
            self._prev_life[env_idx] = observed_life
            self._suspended_until_reset[env_idx] = True
            return transition_rollout

        self._actions[env_idx].append(int(action))
        self._x_positions[env_idx].append(int(x_pos))
        self._world[env_idx] = observed_world
        self._stage[env_idx] = observed_stage
        self._prev_life[env_idx] = observed_life
        return None

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
        if self._suspended_until_reset[env_idx]:
            return None
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

        rollout = self._build_rollout(env_idx, outcome=outcome)
        if rollout is None:
            return None

        # Reset the per-env buffer
        self._clear_env_buffer(env_idx)

        return rollout


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def remove_rollout_from_jsonl(
    jsonl_path: str | Path,
    world: int,
    stage: int,
    actions: list[int],
) -> bool:
    """Remove the first matching rollout from a JSONL file on disk.

    Matches on ``(world, stage, actions)``.  Returns ``True`` if a line was
    removed, ``False`` if no match was found.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    target_actions = json.dumps(actions, separators=(",", ":"))
    new_lines: list[str] = []
    removed = False
    for line in lines:
        if not line.strip():
            continue
        if not removed:
            try:
                r = Rollout.from_json_line(line)
            except (json.JSONDecodeError, TypeError, KeyError):
                new_lines.append(line)
                continue
            if (
                r.world == world
                and r.stage == stage
                and json.dumps(r.actions, separators=(",", ":")) == target_actions
            ):
                removed = True
                continue
        new_lines.append(line)
    if removed:
        path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    return removed


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

    @staticmethod
    def _visited_bins(rollout: Rollout, bin_size: int) -> set[int]:
        return {int(x_pos) // bin_size for x_pos in rollout.x_positions}

    @staticmethod
    def _first_step_in_bin(
        rollout: Rollout,
        target_x_bin: int,
        bin_size: int,
    ) -> Optional[int]:
        for i, x_pos in enumerate(rollout.x_positions):
            if int(x_pos) // bin_size == target_x_bin:
                return i + 1
        return None

    @staticmethod
    def _terminal_death_bin(
        rollout: Rollout,
        bin_size: int,
    ) -> Optional[int]:
        if rollout.outcome != "death" or not rollout.x_positions:
            return None
        return int(rollout.x_positions[-1]) // bin_size

    @classmethod
    def _supports_target_bin(
        cls,
        rollout: Rollout,
        target_x_bin: int,
        bin_size: int,
    ) -> Optional[int]:
        target_step = cls._first_step_in_bin(rollout, target_x_bin, bin_size)
        if target_step is None:
            return None
        if cls._terminal_death_bin(rollout, bin_size) == target_x_bin:
            return None
        return target_step

    def find_rollout(
        self,
        world: int,
        stage: int,
        target_x_bin: int,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> Optional[Rollout]:
        """Find the shortest-prefix rollout that actually visits *target_x_bin*.

        Every rollout stores a full from-level-start trajectory, so any record
        that reaches the target bin is a valid replay source, except for the
        terminal bin of a death rollout.

        Returns ``None`` if no suitable rollout exists.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        target_x = target_x_bin * bin_size
        candidates = self._by_level.get((world, stage), [])
        best: Optional[Rollout] = None
        best_target_step: Optional[int] = None
        for ri in candidates:
            r = self._rollouts[ri]
            if r.max_x >= target_x:
                target_step = self._supports_target_bin(r, target_x_bin, bin_size)
                if target_step is None:
                    continue
                if (
                    best is None
                    or best_target_step is None
                    or target_step < best_target_step
                    or (target_step == best_target_step and r.num_steps < best.num_steps)
                ):
                    best = r
                    best_target_step = target_step
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
        x_pos entered the target bin, so replaying ``actions[:target_step]``
        positions Mario near the desired x.
        """
        bin_size = validate_progression_bin_size(bin_size)
        rollout = self.find_rollout(world, stage, target_x_bin, bin_size)
        if rollout is None:
            return None
        target_step = self._first_step_in_bin(rollout, target_x_bin, bin_size)
        if target_step is None:
            return None
        return rollout.actions, target_step

    def find_all_replay_actions(
        self,
        world: int,
        stage: int,
        target_x_bin: int,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> list[tuple[list[int], int]]:
        """Return all ``(actions, target_step)`` pairs that visit *target_x_bin*.

        Like :meth:`find_replay_actions` but returns multiple candidates so
        the replay pool can sample uniformly across every known rollout that
        reaches the requested bin. Death rollouts do not support their
        terminal bin.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        target_x = target_x_bin * bin_size
        candidates = self._by_level.get((world, stage), [])
        valid: list[tuple[int, Rollout]] = []
        for ri in candidates:
            r = self._rollouts[ri]
            if r.max_x >= target_x:
                target_step = self._supports_target_bin(r, target_x_bin, bin_size)
                if target_step is not None:
                    valid.append((target_step, r))
            else:
                break
        # Sort by earliest hit first for deterministic output ordering.
        valid.sort(key=lambda item: (item[0], item[1].num_steps))
        results: list[tuple[list[int], int]] = []
        for target_step, r in valid:
            results.append((list(r.actions), target_step))
        return results

    def progression_coverage(
        self,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> dict[tuple[int, int, int], int]:
        """Return ``{(world, stage, x_bin): rollout_count}``.

        A rollout covers exactly the bins whose x-positions it actually visits.
        This avoids inferring support across discontinuous pipe jumps.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        cov: dict[tuple[int, int, int], int] = {}
        for r in self._rollouts:
            for b in self._visited_bins(r, bin_size):
                key = (r.world, r.stage, b)
                cov[key] = cov.get(key, 0) + 1
        return cov

    def reachable_bins(
        self,
        bin_size: int = PROGRESSION_BIN_SIZE,
    ) -> set[tuple[int, int, int]]:
        """Return the set of ``(world, stage, x_bin)`` tuples we can replay to.

        Death rollouts contribute support for bins they pass through, but not
        for their terminal death bin.
        """
        self._ensure_loaded()
        bin_size = validate_progression_bin_size(bin_size)
        result: set[tuple[int, int, int]] = set()
        for r in self._rollouts:
            for b in self._visited_bins(r, bin_size):
                if self._terminal_death_bin(r, bin_size) == b:
                    continue
                result.add((r.world, r.stage, b))
        return result
