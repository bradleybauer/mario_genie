"""
Balanced data-collection utilities.

Scans session metadata to compute progression (x-position bin) frame counts and
action distribution, then reports coverage status and produces sampling weights
from inverse frame counts over the current replay support.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional



def validate_progression_bin_size(bin_size: int) -> int:
    value = int(bin_size)
    if value <= 0:
        raise ValueError("progression bin size must be a positive integer")
    return value


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _load_meta_json(meta_path: Path) -> dict | None:
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _parse_exact_progression_summary(
    summary: dict[str, int],
) -> dict[tuple[int, int, int], int]:
    result: dict[tuple[int, int, int], int] = {}
    for key, count in summary.items():
        parts = key.split(":")
        if len(parts) != 3:
            continue
        w, s, x = int(parts[0]), int(parts[1]), int(parts[2])
        result[(w, s, x)] = int(count)
    return result


def _scan_exact_progression_from_meta(
    meta_path: Path,
) -> Optional[dict[tuple[int, int, int], int]]:
    """Try the fast-path: read exact per-(level, x) counts from ``.meta.json``."""
    meta = _load_meta_json(meta_path)
    if meta is None:
        return None
    summary = meta.get("exact_progression_summary")
    if summary is None:
        return None
    return _parse_exact_progression_summary(summary)


def _bin_exact_progression_counts(
    exact_counts: dict[tuple[int, int, int], int],
    bin_size: int,
) -> dict[tuple[int, int, int], int]:
    totals: dict[tuple[int, int, int], int] = Counter()
    for (w, s, x_pos), count in exact_counts.items():
        totals[(w, s, x_pos // bin_size)] += count
    return dict(totals)


# ===================================================================
# Progression (x-position) coverage
# ===================================================================

PROGRESSION_BIN_SIZE: int = 64


def _scan_progression_from_meta(
    meta_path: Path,
    requested_bin_size: int,
) -> Optional[dict[tuple[int, int, int], int]]:
    """Try the fast-path: read progression counts from a ``.meta.json``."""
    exact_counts = _scan_exact_progression_from_meta(meta_path)
    if exact_counts is not None:
        return _bin_exact_progression_counts(exact_counts, requested_bin_size)
    return None


def _find_data_files(data_dir: Path) -> list[Path]:
    """Find all session .npz files in a data directory, recursively."""
    return sorted(data_dir.rglob("session_*.npz"))


def scan_progression_coverage(
    data_dir: str | Path,
    bin_size: int = PROGRESSION_BIN_SIZE,
) -> dict[tuple[int, int, int], int]:
    """Return ``{(world, stage, x_bin): frame_count}`` for all data files.

    Metadata must provide exact per-position counts in
    ``exact_progression_summary``; those are rebinned on demand to the
    requested resolution. Files without usable progression metadata are
    skipped.
    """
    bin_size = validate_progression_bin_size(bin_size)
    data_dir = Path(data_dir)
    npz_files = _find_data_files(data_dir)
    totals: dict[tuple[int, int, int], int] = Counter()
    for npz_path in npz_files:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        counts = _scan_progression_from_meta(meta_path, bin_size) if meta_path.exists() else None
        if counts is None:
            continue
        for key, count in counts.items():
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


def scan_action_coverage(data_dir: str | Path) -> dict[int, int]:
    """Return ``{action_index: total_frame_count}`` for all data files.

    Only metadata is scanned. Files without an ``action_summary`` are skipped.
    """
    data_dir = Path(data_dir)
    npz_files = _find_data_files(data_dir)
    totals: dict[int, int] = Counter()
    for npz_path in npz_files:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        counts = _scan_action_from_meta(meta_path) if meta_path.exists() else None
        if counts is None:
            continue
        for key, count in counts.items():
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
    pct_of_total: float
    weight: float
    reachable: bool  # True if a rollout exists that reaches this bin


@dataclass
class ProgressionBalanceReport:
    total_frames: int
    num_bins: int
    bin_size: int
    bins: list[ProgressionBinInfo] = field(default_factory=list)
    weights: dict[tuple[int, int, int], float] = field(default_factory=dict)


def compute_progression_balance(
    progression_cov: dict[tuple[int, int, int], int],
    reachable_bins: Optional[set[tuple[int, int, int]]] = None,
    all_levels: Optional[Iterable[tuple[int, int]]] = None,
    bin_size: int = PROGRESSION_BIN_SIZE,
) -> ProgressionBalanceReport:
    """Compute inverse-count sampling weights over the current replay support.

    The balancing support is the set of bins we can currently start from via
    replay, plus bin 0 for every level in *all_levels*. Within that support,
    weights are proportional to ``1 / (frame_count + 1)`` so zero-count bins
    receive the highest probability without introducing any target/deficit
    bookkeeping.
    """
    bin_size = validate_progression_bin_size(bin_size)
    support_bins: set[tuple[int, int, int]] = set(reachable_bins or set())
    for world, stage in all_levels or []:
        support_bins.add((int(world), int(stage), 0))
    if not support_bins:
        support_bins = {(w, s, 0) for (w, s, _b) in progression_cov.keys()}
    if not support_bins:
        return ProgressionBalanceReport(total_frames=0, num_bins=0, bin_size=bin_size)

    total = sum(progression_cov.values())
    num_bins = len(support_bins)
    support_bins_per_level: dict[tuple[int, int], int] = Counter((w, s) for (w, s, _b) in support_bins)
    total_frames_per_level: dict[tuple[int, int], int] = Counter()
    for (w, s, _b), count in progression_cov.items():
        total_frames_per_level[(w, s)] += int(count)
    average_support_bins_per_level = max(
        1.0,
        float(num_bins) / float(max(1, len(support_bins_per_level))),
    )
    inverse_count_scores = {
        key: 1.0 / (float(progression_cov.get(key, 0)) + 1.0)
        for key in support_bins
    }
    for key in support_bins:
        world, stage, x_bin = key
        if x_bin != 0:
            continue
        if total_frames_per_level.get((world, stage), 0) != 0:
            continue
        # If a stage has never been seen at all, its level-start is the only
        # gateway to discovering that stage. Give it the exploration mass of a
        # whole average stage instead of treating it like just one more zero-count bin.
        inverse_count_scores[key] *= average_support_bins_per_level

    score_sum = sum(inverse_count_scores.values())
    if score_sum == 0:
        weights = {key: 1.0 / num_bins for key in support_bins}
    else:
        weights = {key: score / score_sum for key, score in inverse_count_scores.items()}

    bins_list: list[ProgressionBinInfo] = []
    for key in sorted(support_bins):
        w, s, b = key
        count = progression_cov.get(key, 0)
        bins_list.append(ProgressionBinInfo(
            world=w, stage=s, x_bin=b,
            frame_count=count,
            pct_of_total=(count / total * 100) if total else 0.0,
            weight=weights[key],
            reachable=True,
        ))

    return ProgressionBalanceReport(
        total_frames=total,
        num_bins=num_bins,
        bin_size=bin_size,
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
    print(f"  Bin size (px)       : {report.bin_size:>10}")
    print(f"  Total frames        : {report.total_frames:>10}")
    print(f"  Distinct bins       : {report.num_bins:>10}")
    print("-" * 80)
    print(f"  {'Level':<8} {'Bin':>4} {'Frames':>10} {'%Total':>8} {'Weight':>8} {'Reach':>6}")
    print("-" * 80)

    bins = report.bins
    if top_n > 0:
        bins = sorted(bins, key=lambda b: (b.frame_count, -b.weight, b.world, b.stage, b.x_bin))[:top_n]

    for bi in bins:
        reach_str = "yes" if bi.reachable else " - "
        max_count = max((entry.frame_count for entry in report.bins), default=0)
        bar_len = int(15 * bi.frame_count / max_count) if max_count else 0
        bar = "\u2588" * min(bar_len, 15)
        print(
            f"  {bi.world}-{bi.stage:<6} {bi.x_bin:>4} {bi.frame_count:>10} "
            f"{bi.pct_of_total:>7.1f}% "
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
    """Print a short summary of the lowest-coverage progression bins."""
    if report.total_frames == 0 and report.bins:
        print("[progression] No frames yet — sampling uniformly over initial replay support.")
        return
    needed = sorted(report.bins, key=lambda b: (b.frame_count, -b.weight, b.world, b.stage, b.x_bin))[:n]
    if not needed:
        print("[progression] No supported bins to sample.")
        return
    print(f"[progression] Lowest-coverage support bins:")
    for bi in needed:
        reach = "reachable" if bi.reachable else "NOT reachable"
        print(f"  World {bi.world}-{bi.stage} bin {bi.x_bin} "
              f"(frames={bi.frame_count}, weight={bi.weight:.4f}, {reach})")

