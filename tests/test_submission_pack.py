"""Official CSV packing and split-alignment guards. No KuaiRand training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research_agent.submission import load_score_vector, write_official_csv
from submit import read_submission


def _rows():
    return [
        (20220429, "0", "3978", "a", "0", 1000.0, 1),
        (20220429, "0", "160", "b", "0", 1000.0, 0),
        (20220429, "1", "3978", "a", "1", 2000.0, 0),
    ]


def test_write_official_csv_round_trip(tmp_path: Path):
    rows = _rows()
    scores = np.array([1.25, -0.5, 0.0], dtype=np.float64)
    path = write_official_csv(tmp_path / "submission.csv", rows, scores)
    got = read_submission(path, rows)
    assert got == pytest.approx(scores.tolist(), rel=0, abs=1e-6)
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "row_id,user_id,video_id,score"
    assert len(text) == 4


def test_load_score_vector_rejects_nan(tmp_path: Path):
    path = tmp_path / "scores.npy"
    np.save(path, np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="NaN"):
        load_score_vector(path)


def test_load_score_vector_rejects_inf(tmp_path: Path):
    path = tmp_path / "scores.npy"
    np.save(path, np.array([0.1, np.inf]))
    with pytest.raises(ValueError, match="NaN"):
        load_score_vector(path)


def test_refuses_row_count_mismatch(tmp_path: Path):
    rows = _rows()
    scores = np.array([0.1, 0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="mix splits"):
        write_official_csv(tmp_path / "bad.csv", rows, scores)
