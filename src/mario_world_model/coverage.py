"""
Balanced data-collection utilities.

Scans chunk metadata to compute progression (x-position bin) frame counts and
action distribution, then reports balance status and produces sampling weights
that prioritise under-represented progression bins and actions.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_PROGRESSION_BIN_SIZE: int = 64


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

PROGRESSION_BIN_SIZE: int = DEFAULT_PROGRESSION_BIN_SIZE


def _scan_progression_from_meta(
    meta_path: Path,
    requested_bin_size: int,
) -> Optional[dict[tuple[int, int, int], int]]:
    """Try the fast-path: read progression counts from a ``.meta.json``."""
    exact_counts = _scan_exact_progression_from_meta(meta_path)
    if exact_counts is not None:
        return _bin_exact_progression_counts(exact_counts, requested_bin_size)
    return None


def scan_progression_coverage(
    data_dir: str | Path,
    bin_size: int = PROGRESSION_BIN_SIZE,
) -> dict[tuple[int, int, int], int]:
    """Return ``{(world, stage, x_bin): frame_count}`` for all chunks.

    Chunk metadata must provide exact per-position counts in
    ``exact_progression_summary``; those are rebinned on demand to the
    requested resolution. Chunks without usable progression metadata are
    skipped.
    """
    bin_size = validate_progression_bin_size(bin_size)
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("chunk_*.npz"))
    totals: dict[tuple[int, int, int], int] = Counter()
    for npz_path in npz_files:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        chunk_counts = _scan_progression_from_meta(meta_path, bin_size) if meta_path.exists() else None
        if chunk_counts is None:
            continue
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


def scan_action_coverage(data_dir: str | Path) -> dict[int, int]:
    """Return ``{action_index: total_frame_count}`` for all chunks.

    Only chunk metadata is scanned. Chunks without an ``action_summary`` are skipped.
    """
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob("chunk_*.npz"))
    totals: dict[int, int] = Counter()
    for npz_path in npz_files:
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        chunk_counts = _scan_action_from_meta(meta_path) if meta_path.exists() else None
        if chunk_counts is None:
            continue
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
    bin_size: int
    bins: list[ProgressionBinInfo] = field(default_factory=list)
    weights: dict[tuple[int, int, int], float] = field(default_factory=dict)


def compute_progression_balance(
    progression_cov: dict[tuple[int, int, int], int],
    reachable_bins: Optional[set[tuple[int, int, int]]] = None,
    bin_size: int = PROGRESSION_BIN_SIZE,
) -> ProgressionBalanceReport:
    """Compute per-(level, x_bin) balance with weights biased toward deficit.

    *reachable_bins* (from ``RolloutIndex.reachable_bins()``) limits the
    weight to bins that we can actually replay to.  Bins we can't reach still
    appear in the report but get weight 0.
    """
    bin_size = validate_progression_bin_size(bin_size)
    if not progression_cov:
        return ProgressionBalanceReport(total_frames=0, num_bins=0, target_per_bin=0, bin_size=bin_size)

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

