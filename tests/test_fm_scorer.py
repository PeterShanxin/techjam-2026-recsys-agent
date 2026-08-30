"""FM scorer CLI contract on the mini dataset. Not the 80s KuaiRand run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_helpers import write_mini_dataset
from research_agent.recommenders import fm_scorer


def test_fm_scorer_writes_split_length_scores(tmp_path: Path, monkeypatch):
    data_dir = write_mini_dataset(tmp_path)
    out = tmp_path / "scores.npy"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"k": 4, "epochs": 1, "batch": 2, "patience": 1, "verbose": False}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = fm_scorer.main(
        [
            "--data-dir",
            str(data_dir),
            "--split",
            "valid",
            "--output-scores",
            str(out),
            "--seed",
            "0",
            "--config",
            str(config),
        ]
    )
    assert code == 0
    scores = np.load(out)
    assert scores.ndim == 1
    assert scores.size == 4
    assert np.isfinite(scores).all()
