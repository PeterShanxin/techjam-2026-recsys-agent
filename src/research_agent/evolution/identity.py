"""Near-identity detection on score vectors. Closes the alpha-approximately-zero loophole.

Audit finding (2026-08-31): five of seven P0 offspring were of the form
``parent + alpha * residual`` with ``alpha`` grid-searched on validation and
``alpha = 0`` inside the grid. That construction cannot score below its parent, so
fitness rewarded it unconditionally and the population never left it. The winner moved
0.0044% of validation rows and gained +8.8e-6, which is roughly 2% of the paired
bootstrap standard deviation of the metric.

``_classify_validity`` only catches byte-identical source, so those candidates were
recorded as ``hypothesis_tested``. This module adds the behavioural test the textual
one cannot make: did the produced ranking actually differ from the parent's?

Ordering, not score value, is what the official metric consumes. Both GAUC and nDCG@5
are computed strictly inside one user's impression list, so any strictly monotone
per-user transform of a score vector leaves both metrics bitwise unchanged. The right
question is therefore how many rows changed their within-user rank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

# A child must move at least this share of validation rows within their own user's
# ordering to count as a distinct mechanism rather than a re-labelled parent.
DEFAULT_MIN_RANK_CHANGE = 0.001
# Below this many rows the fraction is not informative: on a handful of impressions two
# unrelated score vectors coincide in ordering by chance often enough to be useless.
# The real research split has 124,909 rows.
DEFAULT_MIN_ROWS = 1000
# ...unless it moved the metric by more than this, which is roughly one paired
# bootstrap standard deviation. A small but genuinely large-effect change stays valid.
DEFAULT_MATERIAL_DELTA = 0.0005

NEAR_IDENTITY_VALIDITY = "near_identity_noop"


@dataclass(frozen=True)
class IdentityReport:
    parent_id: str
    rank_change_fraction: float
    primary_delta: float | None
    near_identity: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "rank_change_fraction": self.rank_change_fraction,
            "primary_delta": self.primary_delta,
            "near_identity": self.near_identity,
        }


def _within_user_ranks(scores: np.ndarray, users: np.ndarray) -> np.ndarray:
    """Descending rank position of each row inside its own user's list.

    Ties resolve by original row order, matching how the official evaluator's stable
    sort on ``-score`` orders equal scores.
    """
    order = np.lexsort((np.arange(len(scores)), -scores, users))
    ranks = np.empty(len(scores), dtype=np.int64)
    ordered_users = users[order]
    boundary = np.flatnonzero(
        np.concatenate(([True], ordered_users[1:] != ordered_users[:-1]))
    )
    starts = np.repeat(boundary, np.diff(np.concatenate((boundary, [len(order)]))))
    ranks[order] = np.arange(len(order), dtype=np.int64) - starts
    return ranks


def rank_change_fraction(
    child_scores: Sequence[float] | np.ndarray,
    parent_scores: Sequence[float] | np.ndarray,
    users: Sequence[Any] | np.ndarray,
) -> float:
    """Share of rows whose within-user rank position differs between two score vectors.

    0.0 means the two vectors induce exactly the same within-user ordering everywhere,
    so they are the same model as far as GAUC and nDCG@5 can tell.
    """
    child = np.asarray(child_scores, dtype=np.float64)
    parent = np.asarray(parent_scores, dtype=np.float64)
    keys = np.asarray(list(users))
    if child.shape != parent.shape:
        raise ValueError(f"score length mismatch: {child.shape} vs {parent.shape}")
    if len(keys) != len(child):
        raise ValueError(f"users length {len(keys)} does not match scores {len(child)}")
    if child.size == 0:
        return 0.0
    codes = np.unique(keys, return_inverse=True)[1]
    changed = _within_user_ranks(child, codes) != _within_user_ranks(parent, codes)
    return float(changed.mean())


def assess_identity(
    child_scores: Sequence[float] | np.ndarray,
    parent_scores: Sequence[float] | np.ndarray,
    users: Sequence[Any] | np.ndarray,
    *,
    parent_id: str = "",
    primary_delta: float | None = None,
    min_rank_change: float = DEFAULT_MIN_RANK_CHANGE,
    material_delta: float = DEFAULT_MATERIAL_DELTA,
) -> IdentityReport:
    """Decide whether a child is a re-labelled parent rather than a new mechanism."""
    fraction = rank_change_fraction(child_scores, parent_scores, users)
    material = primary_delta is not None and abs(float(primary_delta)) >= material_delta
    near = fraction < min_rank_change and not material
    return IdentityReport(
        parent_id=parent_id,
        rank_change_fraction=fraction,
        primary_delta=None if primary_delta is None else float(primary_delta),
        near_identity=near,
    )


def load_scores(path: str | None) -> np.ndarray | None:
    """Load a published score vector, or None when it is unavailable."""
    if not path:
        return None
    try:
        loaded = np.load(path)
    except (OSError, ValueError):
        return None
    array = np.asarray(loaded, dtype=np.float64).reshape(-1)
    return array if array.size else None


def first_near_identity(
    child_scores: Sequence[float] | np.ndarray,
    parent_scores: Iterable[tuple[str, Any]],
    users: Sequence[Any] | np.ndarray,
    *,
    primary_delta_for: Any = None,
    min_rank_change: float = DEFAULT_MIN_RANK_CHANGE,
    material_delta: float = DEFAULT_MATERIAL_DELTA,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> IdentityReport | None:
    """Return the first parent this child is behaviourally identical to, if any.

    Returns None on splits smaller than ``min_rows``: agreement on a few impressions is
    coincidence, not evidence of a no-op.
    """
    if len(np.asarray(child_scores).reshape(-1)) < int(min_rows):
        return None
    for parent_id, scores in parent_scores:
        parent = None if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)
        if parent is None or parent.shape != np.asarray(child_scores).reshape(-1).shape:
            continue
        delta = None
        if callable(primary_delta_for):
            delta = primary_delta_for(parent_id)
        report = assess_identity(
            child_scores,
            parent,
            users,
            parent_id=parent_id,
            primary_delta=delta,
            min_rank_change=min_rank_change,
            material_delta=material_delta,
        )
        if report.near_identity:
            return report
    return None
