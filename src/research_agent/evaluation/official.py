"""Single official metric/data boundary.

Candidate code must not own benchmark scoring. All Phase 2 metrics come
from organizer evaluate(user_ids, labels, scores).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
STARTER = REPO_ROOT / "starter" / "kuairand"
EVALUATE_PY = STARTER / "evaluate.py"


def ensure_starter_on_path() -> Path:
    starter = str(STARTER)
    if starter not in sys.path:
        sys.path.insert(0, starter)
    return STARTER


def official_evaluate(
    user_ids: Sequence[Any],
    labels: Sequence[Any],
    scores: Sequence[Any],
    k: int = 5,
) -> dict[str, Any]:
    ensure_starter_on_path()
    from evaluate import evaluate

    return evaluate(user_ids, labels, scores, k=k)


def official_load(data_dir: Path | str) -> dict[str, list]:
    ensure_starter_on_path()
    from data import load

    return load(str(data_dir))


def split_labels(rows: Sequence[tuple]) -> tuple[list[Any], list[Any]]:
    """Official data.load() row: date, user_id, video_id, author, tab, dur, label."""
    user_ids = [row[1] for row in rows]
    labels = [row[6] for row in rows]
    return user_ids, labels


def official_metrics_from_scores(
    rows: Sequence[tuple],
    scores: Sequence[Any],
) -> Mapping[str, Any]:
    user_ids, labels = split_labels(rows)
    return official_evaluate(user_ids, labels, scores)
