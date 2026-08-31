"""Two non-FM mechanisms execute through ExperimentRunner. Zero API spend."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiment_helpers import make_spec, write_lab_dataset
from research_agent.experiments import ExperimentRunner, ImplementationRef
from research_agent.experiments.splits import RESEARCH_SPLIT

ROOT = Path(__file__).resolve().parents[1]
HISTORY = "src/research_agent/recommenders/history_recency_scorer.py"
PAIRWISE = "src/research_agent/recommenders/pairwise_bpr_scorer.py"


def _runner(tmp_path: Path) -> ExperimentRunner:
    return ExperimentRunner(
        repo_root=ROOT,
        runs_dir=tmp_path / "runs",
        data_dir=write_lab_dataset(tmp_path),
        allow_test=False,
    )


def test_history_recency_smoke_runs_on_valid(tmp_path: Path):
    runner = _runner(tmp_path)
    spec = make_spec(
        experiment_id="smoke-history-recency",
        implementation=ImplementationRef(entrypoint=HISTORY),
        hypothesis="Train history plus recency decay scores the current row.",
        origin="manual",
        evaluation_split=RESEARCH_SPLIT,
        parameters={"half_life_days": 3.0},
        timeout_seconds=30.0,
        tags=("family:history_recency", "mech:recency", "axis:history"),
    )
    result = runner.run(spec)
    assert result.status == "success"
    assert result.evaluation_split == "valid"
    assert result.metrics is not None
    assert np.isfinite(result.metrics.primary)
    scores = np.load(tmp_path / "runs" / spec.experiment_id / "scores.npy")
    assert scores.ndim == 1
    assert len(scores) == 5


def test_pairwise_bpr_smoke_runs_on_valid(tmp_path: Path):
    runner = _runner(tmp_path)
    spec = make_spec(
        experiment_id="smoke-pairwise-bpr",
        implementation=ImplementationRef(entrypoint=PAIRWISE),
        hypothesis="Train pairwise BPR on user/item ids.",
        origin="manual",
        evaluation_split=RESEARCH_SPLIT,
        parameters={"k": 4, "epochs": 2, "max_pairs": 20},
        timeout_seconds=30.0,
        tags=("family:pairwise_ranking", "mech:bpr", "axis:objective"),
    )
    result = runner.run(spec)
    assert result.status == "success"
    assert result.evaluation_split == "valid"
    assert result.metrics is not None
    assert np.isfinite(result.metrics.primary)


def test_smoke_mechanisms_are_scientifically_distinct(tmp_path: Path):
    runner = _runner(tmp_path)
    hist = runner.run(
        make_spec(
            experiment_id="smoke-hist-cmp",
            implementation=ImplementationRef(entrypoint=HISTORY),
            evaluation_split="valid",
            parameters={"half_life_days": 2.0},
            timeout_seconds=30.0,
        )
    )
    pair = runner.run(
        make_spec(
            experiment_id="smoke-bpr-cmp",
            implementation=ImplementationRef(entrypoint=PAIRWISE),
            evaluation_split="valid",
            parameters={"k": 4, "epochs": 1, "max_pairs": 20, "seed_mark": "bpr"},
            timeout_seconds=30.0,
        )
    )
    assert hist.status == pair.status == "success"
    a = np.load(tmp_path / "runs" / "smoke-hist-cmp" / "scores.npy")
    b = np.load(tmp_path / "runs" / "smoke-bpr-cmp" / "scores.npy")
    assert not np.allclose(a, b)
    assert hist.spec_hash != pair.spec_hash


def test_research_fitness_cannot_use_test_split(tmp_path: Path):
    runner = _runner(tmp_path)
    spec = make_spec(
        experiment_id="smoke-test-blocked",
        implementation=ImplementationRef(entrypoint=HISTORY),
        evaluation_split="test",
        allow_test_split=False,
        timeout_seconds=30.0,
    )
    result = runner.run(spec)
    assert result.status == "invalid"
    assert result.metrics is None
    assert result.failure is not None
    assert result.failure.kind == "test_split"
