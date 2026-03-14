from __future__ import annotations

import numpy as np


def _compute_action_summary(arrays: dict[str, np.ndarray]) -> dict[str, int] | None:
    """Compute per-action frame counts from the ``actions`` array.

    Returns a dict like ``{"0": 140, "3": 87, ...}`` mapping the *string*
    representation of each action index to its total count, or ``None`` if
    no ``actions`` key is present.
    """
    if "actions" not in arrays:
        return None
    actions = arrays["actions"].flatten()  # (B*T,)
    unique, counts = np.unique(actions, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(unique, counts)}


def _compute_exact_progression_summary(
    arrays: dict[str, np.ndarray],
) -> dict[str, int] | None:
    """Compute exact per-(level, x-position) frame counts.

    Returns a dict like ``{"1:1:0": 50, "1:1:17": 12, ...}`` where the key
    is ``"W:S:X"`` (world, stage, exact x-position) and the value is the
    number of frames observed at that exact coordinate. Returns ``None`` if the
    necessary arrays are absent.
    """
    if "world" not in arrays or "stage" not in arrays or "x_pos" not in arrays:
        return None
    world = arrays["world"].astype(np.int64, copy=False).ravel()
    stage = arrays["stage"].astype(np.int64, copy=False).ravel()
    x_pos = arrays["x_pos"].astype(np.int64, copy=False).ravel()
    # Encode (w, s, x) into a single int64 for fast 1-D np.unique.
    # world/stage fit in 8 bits; x_pos fits in 16 bits.
    keys = (world << 32) | (stage << 16) | x_pos
    unique, counts = np.unique(keys, return_counts=True)
    result: dict[str, int] = {}
    for k, cnt in zip(unique, counts):
        k = int(k)
        w = k >> 32
        s = (k >> 16) & 0xFFFF
        x = k & 0xFFFF
        result[f"{w}:{s}:{x}"] = int(cnt)
    return result


def _compute_summaries(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    """Return ``(action_summary, exact_progression_summary)``."""
    return (
        _compute_action_summary(arrays),
        _compute_exact_progression_summary(arrays),
    )

