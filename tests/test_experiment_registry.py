"""SQLite registry primitives."""
from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.experiments import (
    ExperimentRegistry,
    ExperimentResult,
    FailureInfo,
    Metrics,
    RegistryError,
)
from experiment_helpers import make_spec


def _registry(tmp_path: Path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "registry.sqlite")


def _success(experiment_id: str, primary: float, split: str = "valid") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        status="success",
        evaluation_split=split,
        seed=0,
        spec_hash="h",
        wall_seconds=1.0,
        return_code=0,
        run_dir="runs/" + experiment_id,
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        scores_path="scores.npy",
        metrics=Metrics(gauc=primary, ndcg_at_5=primary, primary=primary),
    )


def test_persist_and_reload(tmp_path: Path):
    registry = _registry(tmp_path)
    spec = make_spec(experiment_id="persist-a", origin="baseline")
    registry.insert_spec(spec)
    registry.upsert_result(_success("persist-a", 0.51))
    loaded = ExperimentRegistry(tmp_path / "registry.sqlite")
    entry = loaded.get("persist-a")
    assert entry.spec.spec_hash == spec.spec_hash
    assert entry.result is not None
    assert entry.result.metrics.primary == pytest.approx(0.51)
    assert entry.decision == "pending"
    assert loaded.count() == 1


def test_spec_hash_duplicate_lookup(tmp_path: Path):
    registry = _registry(tmp_path)
    a = make_spec(experiment_id="dup-a", parameters={"x": 1})
    b = make_spec(experiment_id="dup-b", parameters={"x": 1})
    registry.insert_spec(a)
    registry.insert_spec(b)
    found = registry.find_by_spec_hash(a.spec_hash)
    assert [e.spec.experiment_id for e in found] == ["dup-a", "dup-b"]
    assert a.spec_hash == b.spec_hash


def test_parent_lineage_and_ancestry(tmp_path: Path):
    registry = _registry(tmp_path)
    root = make_spec(experiment_id="root", origin="baseline")
    child = make_spec(experiment_id="child", origin="mutation", parent_ids=("root",))
    grand = make_spec(experiment_id="grand", origin="mutation", parent_ids=("child",))
    cross = make_spec(
        experiment_id="cross",
        origin="crossover",
        parent_ids=("child", "grand"),
        parameters={"combo": True},
    )
    for spec in (root, child, grand, cross):
        registry.insert_spec(spec)
    assert registry.parents("root") == []
    assert registry.parents("child") == ["root"]
    assert registry.children("root") == ["child"]
    assert registry.ancestry("grand") == ["child", "root"]
    assert registry.ancestry("cross") == ["child", "grand", "root"]
    assert registry.rollback_target("grand") == "child"
    assert registry.rollback_target("root") is None


def test_decision_transitions(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.insert_spec(make_spec(experiment_id="dec-a"))
    assert registry.get_decision("dec-a") == "pending"
    registry.mark_decision("dec-a", "accepted")
    assert registry.get_decision("dec-a") == "accepted"
    registry.mark_decision("dec-a", "rejected")
    assert registry.get_decision("dec-a") == "rejected"
    with pytest.raises(RegistryError, match="decision"):
        registry.mark_decision("dec-a", "maybe")


def test_elite_is_best_successful_validation(tmp_path: Path):
    registry = _registry(tmp_path)
    low = make_spec(experiment_id="low", parameters={"p": 1})
    high = make_spec(experiment_id="high", parameters={"p": 2})
    failed = make_spec(experiment_id="failed", parameters={"p": 3})
    test_win = make_spec(
        experiment_id="test-win",
        evaluation_split="test",
        allow_test_split=True,
        parameters={"p": 4},
    )
    rejected = make_spec(experiment_id="rejected", parameters={"p": 5})
    for spec in (low, high, failed, test_win, rejected):
        registry.insert_spec(spec)
    registry.upsert_result(_success("low", 0.40))
    registry.upsert_result(_success("high", 0.60))
    registry.upsert_result(
        ExperimentResult(
            experiment_id="failed",
            status="failed",
            evaluation_split="valid",
            seed=0,
            spec_hash=failed.spec_hash,
            wall_seconds=0.1,
            return_code=2,
            run_dir="runs/failed",
            stdout_path="",
            stderr_path="",
            failure=FailureInfo("subprocess", "boom"),
        )
    )
    registry.upsert_result(_success("test-win", 0.99, split="test"))
    registry.upsert_result(_success("rejected", 0.80))
    registry.mark_decision("rejected", "rejected")

    elite = registry.elite()
    assert elite is not None
    assert elite.spec.experiment_id == "high"
    ranked_ids = [e.spec.experiment_id for e in registry.rank_validation()]
    assert "test-win" not in ranked_ids
    assert "failed" not in ranked_ids
    assert "rejected" not in ranked_ids
    assert ranked_ids[0] == "high"


def test_elite_tie_is_deterministic(tmp_path: Path):
    registry = _registry(tmp_path)
    first = make_spec(experiment_id="tie-a", parameters={"p": "a"})
    second = make_spec(experiment_id="tie-b", parameters={"p": "b"})
    registry.insert_spec(first)
    registry.insert_spec(second)
    registry.upsert_result(_success("tie-a", 0.55))
    registry.upsert_result(_success("tie-b", 0.55))
    assert registry.elite().spec.experiment_id == "tie-a"


def test_failed_experiment_not_selected_as_elite(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.insert_spec(make_spec(experiment_id="only-fail"))
    registry.upsert_result(
        ExperimentResult(
            experiment_id="only-fail",
            status="failed",
            evaluation_split="valid",
            seed=0,
            spec_hash="h",
            wall_seconds=0.1,
            return_code=1,
            run_dir="runs/only-fail",
            stdout_path="",
            stderr_path="",
            failure=FailureInfo("subprocess", "fail"),
        )
    )
    assert registry.elite() is None


def test_result_requires_existing_spec(tmp_path: Path):
    registry = _registry(tmp_path)
    with pytest.raises(RegistryError, match="unknown"):
        registry.upsert_result(_success("missing", 0.1))
