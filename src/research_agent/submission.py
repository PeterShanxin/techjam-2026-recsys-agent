"""Official Track 2 CSV packing. Does not train or select models."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

HEADER = ["row_id", "user_id", "video_id", "score"]


def load_score_vector(path: Path) -> np.ndarray:
    scores = np.load(Path(path))
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
    if scores.size == 0:
        raise ValueError("scores array is empty")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf")
    return np.asarray(scores, dtype=np.float64)


def assert_aligned(rows: Sequence[tuple], scores: np.ndarray) -> None:
    if len(rows) != int(scores.size):
        raise ValueError(
            f"row count {len(rows)} does not match score count {int(scores.size)}; "
            "refusing to mix splits"
        )


def write_official_csv(path: Path, rows: Sequence[tuple], scores: np.ndarray) -> Path:
    assert_aligned(rows, scores)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for index, (row, score) in enumerate(zip(rows, scores)):
            writer.writerow([index, row[1], row[2], f"{float(score):.6g}"])
    return dest
