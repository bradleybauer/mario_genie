"""
Balanced data-collection utilities.

Scans existing .npz / .meta.json chunks to compute per-level sequence counts,
progression (x-position bin) frame counts, and action distribution, then
reports balance status and produces sampling weights that prioritise
under-represented world-stages, progression bins, and actions.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


def default_level_pool() -> list[tuple[int, int]]:
    """All 32 Super Mario Bros levels (worlds 1-8, stages 1-4)."""
    return [(w, s) for w in range(1, 9) for s in range(1, 5)]


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _majority_level(world_t: np.ndarray, stage_t: np.ndarray) -> tuple[int, int]:
    """Assign a single (world, stage) to a sequence via majority vote over T frames."""
    codes = world_t.astype(np.int32) * 100 + stage_t.astype(np.int32)
    values, counts = np.unique(codes, return_counts=True)
    winner = values[np.argmax(counts)]
    return int(winner // 100), int(winner % 100)


def _scan_chunk_from_meta(meta_path: Path) -> Optional[dict[tuple[int, int], int]]:
    """Try to read a fast level_summary from a .meta.json file.

    Returns ``None`` if the file lacks a ``level_summary`` field (old chunks).
    """
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    summary = meta.get("level_summary")
    if summary is None:
        return None

    # Keys are strings like "1-1"; convert to (int, int)
    result: dict[tuple[int, int], int] = {}
    for key, count in summary.items():
        w, s = key.split("-")
        result[(int(w), int(s))] = int(count)
    return result


def _scan_chunk_from_npz(npz_path: Path) -> dict[tuple[int, int], int] | None:
    """Scan an .npz file and count sequences per (world, stage) via majority vote.

    Returns ``None`` if the file cannot be read (e.g. still being written).
    """
    counts: dict[tuple[int, int], int] = Counter()
    try:
        with np.load(npz_path) as data:
            world = data["world"]  # (B, T)
            stage = data["stage"]  # (B, T)
            for i in range(world.shape[0]):
                level = _majority_level(world[i], stage[i])
                counts[level] += 1
    except (EOFError, OSError, ValueError, zipfile.BadZipFile):
        return None
    return dict(counts)


def scan_coverage(data_dir: str | Path) -> dict[tuple[int, int], int]:
    """Return ``{(world, stage): num_sequences}`` for all chunks in *data_dir*.

    Uses the fast ``level_summary`` field in ``.meta.json`` when available,
    falling back to full ``.npz`` loading for older chunks.
    """
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("chunk_*.npz"))

    totals: dict[tuple[int, int], int] = Counter()

    for npz_path in npz_files:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        # Prefer fast meta path
        chunk_counts = _scan_chunk_from_meta(meta_path) if meta_path.exists() else None
        if chunk_counts is None:
            chunk_counts = _scan_chunk_from_npz(npz_path)
        if chunk_counts is None:
            continue  # skip unreadable chunks (e.g. still being written)
        for level, count in chunk_counts.items():
            totals[level] += count

    return dict(totals)


# ---------------------------------------------------------------------------
# Balance report
# ---------------------------------------------------------------------------

@dataclass
class LevelInfo:
    world: int
    stage: int
    count: int
    target: int
    deficit: int
    pct_of_total: float
    weight: float


@dataclass
class BalanceReport:
    total_sequences: int
    num_levels: int
    target_per_level: int
    mean_count: float
    min_count: int
    max_count: int
    levels: list[LevelInfo] = field(default_factory=list)
    weights: dict[tuple[int, int], float] = field(default_factory=dict)


def compute_balance_report(
    coverage: dict[tuple[int, int], int],
    level_pool: Optional[list[tuple[int, int]]] = None,
) -> BalanceReport:
    """Compute a full balance report with per-level weights.

    The *target* for each level is ``ceil(total_sequences / num_levels)`` so
    that equal representation is the goal.  Levels at or above the target get
    weight 0; levels below get weight proportional to their deficit.
    """
    pool = level_pool or default_level_pool()
    num_levels = len(pool)
    counts = {lvl: coverage.get(lvl, 0) for lvl in pool}
    total = sum(counts.values())
    target = max(1, -(-total // num_levels))  # ceil division

    # Raw deficits (clamped to >= 0)
    deficits = {lvl: max(0, target - c) for lvl, c in counts.items()}
    deficit_sum = sum(deficits.values())

    if deficit_sum == 0:
        # All levels at or above target → uniform
        weights = {lvl: 1.0 / num_levels for lvl in pool}
    else:
        weights = {lvl: d / deficit_sum for lvl, d in deficits.items()}

    count_vals = list(counts.values())
    levels = []
    for lvl in sorted(pool):
        c = counts[lvl]
        levels.append(LevelInfo(
            world=lvl[0],
            stage=lvl[1],
            count=c,
            target=target,
            deficit=max(0, target - c),
            pct_of_total=(c / total * 100) if total else 0.0,
            weight=weights[lvl],
        ))

    return BalanceReport(
        total_sequences=total,
        num_levels=num_levels,
        target_per_level=target,
        mean_count=total / num_levels if num_levels else 0,
        min_count=min(count_vals) if count_vals else 0,
        max_count=max(count_vals) if count_vals else 0,
        levels=levels,
        weights=weights,
    )


def save_report(report: BalanceReport, path: str | Path) -> None:
    """Write a machine-readable JSON report."""
    path = Path(path)
    # Convert tuple keys to strings for JSON
    out = {
        "total_sequences": report.total_sequences,
        "num_levels": report.num_levels,
        "target_per_level": report.target_per_level,
        "mean_count": report.mean_count,
        "min_count": report.min_count,
        "max_count": report.max_count,
        "levels": [asdict(lvl) for lvl in report.levels],
        "weights": {f"{w}-{s}": wt for (w, s), wt in report.weights.items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def print_report(report: BalanceReport, top_n: int = 0) -> None:
    """Pretty-print the balance report to stdout."""
    print()
    print("=" * 72)
    print("  WORLD-STAGE BALANCE REPORT")
    print("=" * 72)
    print(f"  Total sequences : {report.total_sequences:>8}")
    print(f"  Levels tracked  : {report.num_levels:>8}")
    print(f"  Target / level  : {report.target_per_level:>8}")
    print(f"  Mean count      : {report.mean_count:>8.1f}")
    print(f"  Min count       : {report.min_count:>8}")
    print(f"  Max count       : {report.max_count:>8}")
    print("-" * 72)
    print(f"  {'Level':<8} {'Count':>8} {'Target':>8} {'Deficit':>8} "
          f"{'%Total':>8} {'Weight':>8}  Bar")
    print("-" * 72)

    levels = report.levels
    if top_n > 0:
        # Sort by deficit descending, show only top_n
        levels = sorted(levels, key=lambda l: l.deficit, reverse=True)[:top_n]

    max_target = report.target_per_level or 1
    for lv in levels:
        bar_len = int(20 * lv.count / max_target) if max_target else 0
        bar = "\u2588" * min(bar_len, 20)
        marker = " \u2713" if lv.deficit == 0 else ""
        print(
            f"  {lv.world}-{lv.stage:<6} {lv.count:>8} {lv.target:>8} "
            f"{lv.deficit:>8} {lv.pct_of_total:>7.1f}% {lv.weight:>8.4f}  "
            f"{bar}{marker}"
        )
    print("=" * 72)


def print_guidance(report: BalanceReport, n: int = 5) -> None:
    """Print a short summary of the *n* most-needed levels for human guidance."""
    needed = sorted(report.levels, key=lambda l: l.deficit, reverse=True)[:n]
    if not needed or needed[0].deficit == 0:
        print("[balance] All levels are at or above the target — collecting uniformly.")
        return
    print(f"[balance] Top {n} most-needed levels:")
    for lv in needed:
        if lv.deficit == 0:
            break
        print(f"  World {lv.world}-{lv.stage}  (have {lv.count}, need {lv.deficit} more)")


def compute_level_summary(world_bt: np.ndarray, stage_bt: np.ndarray) -> dict[str, int]:
    """Compute a per-level sequence count dictionary from (B, T) world/stage arrays.

    Returns a dict like ``{"1-1": 12, "3-2": 5, ...}`` suitable for inclusion
    in ``.meta.json`` files.
    """
    counts: dict[str, int] = Counter()
    for i in range(world_bt.shape[0]):
        w, s = _majority_level(world_bt[i], stage_bt[i])
        counts[f"{w}-{s}"] += 1
    return dict(counts)


# ===================================================================
# Progression (x-position) coverage
# ===================================================================

PROGRESSION_BIN_SIZE: int = 256  # one NES screen


def _scan_progression_from_meta(meta_path: Path) -> Optional[dict[tuple[int, int, int], int]]:
    """Try the fast-path: read ``progression_summary`` from a ``.meta.json``."""
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    summary = meta.get("progression_summary")
    if summary is None:
        return None
    result: dict[tuple[int, int, int], int] = {}
    for key, count in summary.items():
        parts = key.split("-")
        if len(parts) != 3:
            continue
        w, s, b = int(parts[0]), int(parts[1]), int(parts[2])
        result[(w, s, b)] = int(count)
    return result


def _scan_progression_from_npz(
    npz_path: Path,
    bin_size: int = PROGRESSION_BIN_SIZE,
) -> dict[tuple[int, int, int], int]:
    """Scan an ``.npz`` file → ``{(world, stage, x_bin): frame_count}``.

    Returns an empty dict if the file is corrupted or still being written.
    """
    counts: dict[tuple[int, int, int], int] = Counter()
    try:
        with np.load(npz_path) as data:
            if "world" not in data or "stage" not in data or "x_pos" not in data:
                return {}
            world = data["world"].flatten()
            stage = data["stage"].flatten()
            x_pos = data["x_pos"].flatten()
            bins = (x_pos // bin_size).astype(np.int32)
            codes = world.astype(np.int64) * 1_000_000 + stage.astype(np.int64) * 1_000 + bins.astype(np.int64)
            unique, ucounts = np.unique(codes, return_counts=True)
            for code, cnt in zip(unique, ucounts):
                w = int(code // 1_000_000)
                s = int((code % 1_000_000) // 1_000)
                b = int(code % 1_000)
                counts[(w, s, b)] += int(cnt)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile):
        return {}
    return dict(counts)


def scan_progression_coverage(
    data_dir: str | Path,
    bin_size: int = PROGRESSION_BIN_SIZE,
) -> dict[tuple[int, int, int], int]:
    """Return ``{(world, stage, x_bin): frame_count}`` for all chunks."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("chunk_*.npz"))
    totals: dict[tuple[int, int, int], int] = Counter()
    for npz_path in npz_files:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        chunk_counts = _scan_progression_from_meta(meta_path) if meta_path.exists() else None
        if chunk_counts is None:
            chunk_counts = _scan_progression_from_npz(npz_path, bin_size)
        for key, count in chunk_counts.items():
            totals[key] += count
    return dict(totals)


# ===================================================================
# Action coverage
# ===================================================================

def _scan_action_from_meta(meta_path: Path) -> Optional[dict[int, int]]:
    """Try the fast-path: read ``action_summary`` from a ``.meta.json``."""
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    summary = meta.get("action_summary")
    if summary is None:
        return None
    return {int(k): int(v) for k, v in summary.items()}


def _scan_action_from_npz(npz_path: Path) -> dict[int, int]:
    """Scan an ``.npz`` file → ``{action_index: count}``.

    Returns an empty dict if the file is corrupted or still being written.
    """
    try:
        with np.load(npz_path) as data:
            if "actions" not in data:
                return {}
            actions = data["actions"].flatten()
            unique, counts = np.unique(actions, return_counts=True)
            return {int(u): int(c) for u, c in zip(unique, counts)}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile):
        return {}


def scan_action_coverage(data_dir: str | Path) -> dict[int, int]:
    """Return ``{action_index: total_frame_count}`` for all chunks."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("chunk_*.npz"))
    totals: dict[int, int] = Counter()
    for npz_path in npz_files:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        chunk_counts = _scan_action_from_meta(meta_path) if meta_path.exists() else None
        if chunk_counts is None:
            chunk_counts = _scan_action_from_npz(npz_path)
        for key, count in chunk_counts.items():
            totals[key] += count
    return dict(totals)


# ===================================================================
# Progression balance report
# ===================================================================

@dataclass
class ProgressionBinInfo:
    world: int
    stage: int
    x_bin: int
    frame_count: int
    target: int
    deficit: int
    pct_of_total: float
    weight: float
    reachable: bool  # True if a rollout exists that reaches this bin


@dataclass
class ProgressionBalanceReport:
    total_frames: int
    num_bins: int
    target_per_bin: int
    bins: list[ProgressionBinInfo] = field(default_factory=list)
    weights: dict[tuple[int, int, int], float] = field(default_factory=dict)


def compute_progression_balance(
    progression_cov: dict[tuple[int, int, int], int],
    reachable_bins: Optional[set[tuple[int, int, int]]] = None,
) -> ProgressionBalanceReport:
    """Compute per-(level, x_bin) balance with weights biased toward deficit.

    *reachable_bins* (from ``RolloutIndex.reachable_bins()``) limits the
    weight to bins that we can actually replay to.  Bins we can't reach still
    appear in the report but get weight 0.
    """
    if not progression_cov:
        return ProgressionBalanceReport(total_frames=0, num_bins=0, target_per_bin=0)

    # All observed bins
    all_bins = set(progression_cov.keys())
    if reachable_bins is not None:
        all_bins = all_bins | reachable_bins
    # Also include bin-0 for every level observed (always reachable via plain reset)
    levels_seen = {(w, s) for (w, s, _) in all_bins}
    for w, s in levels_seen:
        all_bins.add((w, s, 0))

    total = sum(progression_cov.values())
    num_bins = len(all_bins)
    target = max(1, math.ceil(total / num_bins)) if num_bins else 1

    deficits: dict[tuple[int, int, int], int] = {}
    for key in all_bins:
        count = progression_cov.get(key, 0)
        deficits[key] = max(0, target - count)

    # Only allow weight on reachable bins (bin 0 always reachable).
    # Redirect unreachable bin deficits to their level's bin-0 so that
    # levels with many unreachable deep bins (e.g. 8-4) still receive
    # collection priority proportional to their total deficit.
    effective_deficits: dict[tuple[int, int, int], float] = {}
    for key in all_bins:
        if key[2] == 0:
            effective_deficits[key] = float(deficits[key])
        elif reachable_bins is not None and key in reachable_bins:
            effective_deficits[key] = float(deficits[key])
        elif reachable_bins is None:
            effective_deficits[key] = float(deficits[key])
        else:
            # Bin is unreachable — redirect its deficit to bin-0 of the
            # same level so the level still gets selected frequently,
            # giving the player a chance to progress and unlock the bin.
            w, s, _b = key
            bin0_key = (w, s, 0)
            effective_deficits.setdefault(bin0_key, 0.0)
            effective_deficits[bin0_key] += float(deficits[key])
            effective_deficits[key] = 0.0

    deficit_sum = sum(effective_deficits.values())
    if deficit_sum == 0:
        weights = {key: 1.0 / num_bins for key in all_bins}
    else:
        weights = {key: d / deficit_sum for key, d in effective_deficits.items()}

    bins_list: list[ProgressionBinInfo] = []
    for key in sorted(all_bins):
        w, s, b = key
        count = progression_cov.get(key, 0)
        reachable = (b == 0) or (reachable_bins is not None and key in reachable_bins) or (reachable_bins is None)
        bins_list.append(ProgressionBinInfo(
            world=w, stage=s, x_bin=b,
            frame_count=count, target=target,
            deficit=deficits[key],
            pct_of_total=(count / total * 100) if total else 0.0,
            weight=weights[key],
            reachable=reachable,
        ))

    return ProgressionBalanceReport(
        total_frames=total,
        num_bins=num_bins,
        target_per_bin=target,
        bins=bins_list,
        weights=weights,
    )


# ===================================================================
# Action balance report
# ===================================================================

@dataclass
class ActionInfo:
    action_index: int
    count: int
    target: int
    deficit: int
    pct_of_total: float
    weight: float


@dataclass
class ActionBalanceReport:
    total_frames: int
    num_actions: int
    target_per_action: int
    actions: list[ActionInfo] = field(default_factory=list)
    weights: dict[int, float] = field(default_factory=dict)


def compute_action_balance(
    action_cov: dict[int, int],
    num_actions: int,
) -> ActionBalanceReport:
    """Compute per-action balance with deficit-proportional weights."""
    counts = {a: action_cov.get(a, 0) for a in range(num_actions)}
    total = sum(counts.values())
    target = max(1, math.ceil(total / num_actions)) if num_actions else 1

    deficits = {a: max(0, target - c) for a, c in counts.items()}
    deficit_sum = sum(deficits.values())

    if deficit_sum == 0:
        weights = {a: 1.0 / num_actions for a in range(num_actions)}
    else:
        weights = {a: d / deficit_sum for a, d in deficits.items()}

    actions_list: list[ActionInfo] = []
    for a in range(num_actions):
        c = counts[a]
        actions_list.append(ActionInfo(
            action_index=a,
            count=c,
            target=target,
            deficit=deficits[a],
            pct_of_total=(c / total * 100) if total else 0.0,
            weight=weights[a],
        ))

    return ActionBalanceReport(
        total_frames=total,
        num_actions=num_actions,
        target_per_action=target,
        actions=actions_list,
        weights=weights,
    )


# ===================================================================
# Printing helpers for new reports
# ===================================================================

def print_progression_report(
    report: ProgressionBalanceReport,
    top_n: int = 0,
) -> None:
    """Pretty-print the progression balance report to stdout."""
    print()
    print("=" * 80)
    print("  PROGRESSION (X-POSITION) BALANCE REPORT")
    print("=" * 80)
    print(f"  Total frames        : {report.total_frames:>10}")
    print(f"  Distinct bins       : {report.num_bins:>10}")
    print(f"  Target / bin        : {report.target_per_bin:>10}")
    print("-" * 80)
    print(f"  {'Level':<8} {'Bin':>4} {'Frames':>10} {'Target':>10} "
          f"{'Deficit':>10} {'%Total':>8} {'Weight':>8} {'Reach':>6}")
    print("-" * 80)

    bins = report.bins
    if top_n > 0:
        bins = sorted(bins, key=lambda b: b.deficit, reverse=True)[:top_n]

    for bi in bins:
        reach_str = "yes" if bi.reachable else " - "
        bar_len = int(15 * bi.frame_count / report.target_per_bin) if report.target_per_bin else 0
        bar = "\u2588" * min(bar_len, 15)
        print(
            f"  {bi.world}-{bi.stage:<6} {bi.x_bin:>4} {bi.frame_count:>10} "
            f"{bi.target:>10} {bi.deficit:>10} {bi.pct_of_total:>7.1f}% "
            f"{bi.weight:>8.4f} {reach_str:>6}  {bar}"
        )
    print("=" * 80)


def print_action_report(
    report: ActionBalanceReport,
    action_meanings: Optional[list[list[str]]] = None,
    top_n: int = 0,
) -> None:
    """Pretty-print the action balance report to stdout."""
    print()
    print("=" * 72)
    print("  ACTION DISTRIBUTION BALANCE REPORT")
    print("=" * 72)
    print(f"  Total frames        : {report.total_frames:>10}")
    print(f"  Distinct actions    : {report.num_actions:>10}")
    print(f"  Target / action     : {report.target_per_action:>10}")
    print("-" * 72)
    print(f"  {'Idx':>4} {'Action':<22} {'Count':>10} {'Target':>10} "
          f"{'Deficit':>10} {'%Total':>8} {'Weight':>8}")
    print("-" * 72)

    actions = report.actions
    if top_n > 0:
        actions = sorted(actions, key=lambda a: a.deficit, reverse=True)[:top_n]

    for ai in actions:
        name = "+".join(action_meanings[ai.action_index]) if action_meanings and ai.action_index < len(action_meanings) else str(ai.action_index)
        bar_len = int(15 * ai.count / report.target_per_action) if report.target_per_action else 0
        bar = "\u2588" * min(bar_len, 15)
        print(
            f"  {ai.action_index:>4} {name:<22} {ai.count:>10} "
            f"{ai.target:>10} {ai.deficit:>10} {ai.pct_of_total:>7.1f}% "
            f"{ai.weight:>8.4f}  {bar}"
        )
    print("=" * 72)


def print_progression_guidance(report: ProgressionBalanceReport, n: int = 5) -> None:
    """Print a short summary of the *n* most-needed progression bins."""
    needed = sorted(report.bins, key=lambda b: b.deficit, reverse=True)[:n]
    if not needed or needed[0].deficit == 0:
        print("[progression] All bins at or above target — collecting uniformly.")
        return
    print(f"[progression] Top {n} most-needed progression bins:")
    for bi in needed:
        if bi.deficit == 0:
            break
        reach = "reachable" if bi.reachable else "NOT reachable"
        print(f"  World {bi.world}-{bi.stage} bin {bi.x_bin} "
              f"(have {bi.frame_count}, need {bi.deficit} more, {reach})")

