"""Frozen final candidate lives in the repo, not gitignored generated paths."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiment_helpers import write_mini_dataset
from research_agent.final_candidate import (
    FINAL_ENTRYPOINT,
    FINAL_EXPERIMENT_ID,
    final_candidate_spec,
)
from research_agent.recommenders import fm_swa7_ensemble_scorer

ROOT = Path(__file__).resolve().parents[1]


def test_final_spec_is_validation_by_default():
    spec = final_candidate_spec()
    assert spec.experiment_id == FINAL_EXPERIMENT_ID
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert spec.parameters["num_models"] == 7
    assert spec.parameters["top_k_checkpoints"] == 2
    assert spec.implementation.entrypoint == FINAL_ENTRYPOINT
    assert not spec.implementation.entrypoint.startswith("runs/")


def test_final_spec_refuses_test_without_opt_in():
    with pytest.raises(ValueError, match="allow_test_split"):
        final_candidate_spec(evaluation_split="test", allow_test_split=False)


def test_swa7_scorer_writes_split_length_scores(tmp_path: Path, monkeypatch):
    data_dir = write_mini_dataset(tmp_path)
    out = tmp_path / "scores.npy"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "k": 4,
                "epochs": 1,
                "batch": 2,
                "patience": 1,
                "num_models": 2,
                "top_k_checkpoints": 1,
                "verbose": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    code = fm_swa7_ensemble_scorer.main(
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


def test_entrypoint_is_committed_repo_file():
    path = ROOT / FINAL_ENTRYPOINT
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "num_models" in text
    assert "top_k_checkpoints" in text
    assert "runs/generated" not in FINAL_ENTRYPOINT


def test_final_cli_refuses_test_without_allow_test():
    path = ROOT / "scripts" / "run_final_candidate.py"
    spec = importlib.util.spec_from_file_location("run_final_candidate_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    rc = module.main(["--split", "test"])
    assert rc == 2
