"""Official submission writer/validator alignment without training a model."""
from __future__ import annotations

from pathlib import Path

import pytest

from submit import read_submission, write_submission


def _rows():
    # Same tuple layout as data.load(): date, user_id, video_id, author, tab, dur, label
    return [
        (20220429, "0", "3978", "a", "0", 1000.0, 1),
        (20220429, "0", "160", "b", "0", 1000.0, 0),
        (20220429, "1", "3978", "a", "1", 2000.0, 0),
    ]


def test_write_read_preserves_row_id_and_ids(tmp_path: Path):
    rows = _rows()
    scores = [1.25, -0.5, 0.0]
    path = tmp_path / "submission.csv"
    write_submission(path, rows, scores)
    got = read_submission(path, rows)
    assert got == pytest.approx(scores, rel=0, abs=1e-6)

    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "row_id,user_id,video_id,score"
    assert text[1].startswith("0,0,3978,")
    assert text[2].startswith("1,0,160,")
    assert text[3].startswith("2,1,3978,")


def test_read_rejects_row_id_gap(tmp_path: Path):
    rows = _rows()
    path = tmp_path / "bad_row_id.csv"
    write_submission(path, rows, [0.1, 0.2, 0.3])
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2].replace("1,", "3,", 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row_id"):
        read_submission(path, rows)


def test_read_rejects_id_misalign(tmp_path: Path):
    rows = _rows()
    path = tmp_path / "bad_align.csv"
    write_submission(path, rows, [0.1, 0.2, 0.3])
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[2] = "9999"
    lines[1] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="对齐"):
        read_submission(path, rows)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_read_rejects_non_finite_score(tmp_path: Path, bad: str):
    rows = _rows()
    path = tmp_path / f"bad_{bad}.csv"
    write_submission(path, rows, [0.1, 0.2, 0.3])
    lines = path.read_text(encoding="utf-8").splitlines()
    parts = lines[2].split(",")
    parts[3] = bad
    lines[2] = ",".join(parts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN/Inf"):
        read_submission(path, rows)
