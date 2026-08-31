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
    FINAL_SEED,
    LEGACY_SWA7_ENTRYPOINT,
    LEGACY_SWA7_EXPERIMENT_ID,
    LIVE_ELITE_ID,
    final_candidate_spec,
    swa7_candidate_spec,
)
from research_agent.recommenders import fm_swa7_ensemble_scorer, tiered_ensemble_scorer

ROOT = Path(__file__).resolve().parents[1]


def test_final_spec_is_validation_by_default():
    spec = final_candidate_spec()
    assert spec.experiment_id == FINAL_EXPERIMENT_ID
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert spec.seed == FINAL_SEED == 42
    assert spec.implementation.entrypoint == FINAL_ENTRYPOINT
    assert not spec.implementation.entrypoint.startswith("runs/")


def test_final_spec_keeps_config_empty_so_the_live_elite_reproduces_bitwise():
    """Every tier setting is baked into the frozen entrypoint; an empty config is the
    same one the live elite ran with (config fingerprint = sha256 of "{}")."""
    assert final_candidate_spec().parameters == {}


def test_final_spec_carries_the_autonomous_lineage():
    spec = final_candidate_spec()
    assert spec.origin == "crossover"
    assert spec.parent_ids == (
        "rs-20260831T062638Z-939b7000-007",
        "rs-20260831T062638Z-939b7000-006",
    )
    assert LIVE_ELITE_ID in spec.rationale


def test_final_spec_refuses_test_without_opt_in():
    with pytest.raises(ValueError, match="allow_test_split"):
        final_candidate_spec(evaluation_split="test", allow_test_split=False)


def test_superseded_swa7_stays_reproducible_as_history():
    spec = swa7_candidate_spec()
    assert spec.experiment_id == LEGACY_SWA7_EXPERIMENT_ID == "final-swa7-ensemble"
    assert spec.experiment_id != FINAL_EXPERIMENT_ID
    assert spec.implementation.entrypoint == LEGACY_SWA7_ENTRYPOINT
    assert spec.parameters["num_models"] == 7
    assert spec.parameters["top_k_checkpoints"] == 2
    assert "superseded" in spec.tags
    assert (ROOT / LEGACY_SWA7_ENTRYPOINT).is_file()


def test_superseded_swa7_refuses_test_without_opt_in():
    with pytest.raises(ValueError, match="allow_test_split"):
        swa7_candidate_spec(evaluation_split="test", allow_test_split=False)


def _run_scorer(module, tmp_path: Path, monkeypatch, config: dict) -> np.ndarray:
    data_dir = write_mini_dataset(tmp_path)
    out = tmp_path / f"{module.__name__.rsplit('.', 1)[-1]}.npy"
    config_path = tmp_path / f"{out.stem}-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = module.main(
        [
            "--data-dir", str(data_dir),
            "--split", "valid",
            "--output-scores", str(out),
            "--seed", "0",
            "--config", str(config_path),
        ]
    )
    assert code == 0
    return np.load(out)


def test_tiered_scorer_writes_split_length_scores(tmp_path: Path, monkeypatch):
    scores = _run_scorer(
        tiered_ensemble_scorer,
        tmp_path,
        monkeypatch,
        {
            "k": 4, "epochs": 1, "batch": 2, "patience": 1,
            "num_strict": 1, "num_moderate": 1, "num_full": 1,
            "top_k_checkpoints": 1, "verbose": False,
        },
    )
    assert scores.ndim == 1
    assert scores.size == 4
    assert np.isfinite(scores).all()


def test_superseded_swa7_scorer_still_runs(tmp_path: Path, monkeypatch):
    scores = _run_scorer(
        fm_swa7_ensemble_scorer,
        tmp_path,
        monkeypatch,
        {
            "k": 4, "epochs": 1, "batch": 2, "patience": 1,
            "num_models": 2, "top_k_checkpoints": 1, "verbose": False,
        },
    )
    assert scores.ndim == 1
    assert scores.size == 4
    assert np.isfinite(scores).all()


def test_entrypoint_is_committed_repo_file():
    path = ROOT / FINAL_ENTRYPOINT
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for knob in ("num_strict", "num_moderate", "num_full", "l2_strict", "top_k_checkpoints"):
        assert knob in text, knob
    assert "runs/generated" not in FINAL_ENTRYPOINT


def _load_cli():
    path = ROOT / "scripts" / "run_final_candidate.py"
    spec = importlib.util.spec_from_file_location("run_final_candidate_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_cli_refuses_test_without_allow_test():
    assert _load_cli().main(["--split", "test"]) == 2


def test_final_cli_refuses_legacy_test_without_allow_test():
    assert _load_cli().main(["--split", "test", "--legacy-swa7"]) == 2
