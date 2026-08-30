"""ExperimentSpec / ExperimentResult contract tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.experiments import ExperimentResult, ExperimentSpec, ImplementationRef, Metrics, SpecError
from experiment_helpers import make_spec


def test_serialization_round_trip(tmp_path: Path):
    spec = make_spec(
        experiment_id="round-trip-1",
        origin="mutation",
        parent_ids=("root-a",),
        parameters={"k": 16, "nested": {"a": 1}},
        tags=("phase2", "unit"),
        notes="keep me",
    )
    path = tmp_path / "spec.json"
    spec.write_json(path)
    loaded = ExperimentSpec.from_path(path)
    assert loaded.to_dict() == spec.to_dict()
    assert loaded.spec_hash == spec.spec_hash


def test_spec_hash_ignores_identity_and_prose():
    a = make_spec(experiment_id="id-a", hypothesis="one", notes="n1", origin="manual")
    b = make_spec(experiment_id="id-b", hypothesis="two", notes="n2", origin="manual")
    assert a.spec_hash == b.spec_hash
    assert a.experiment_id != b.experiment_id


def test_spec_hash_is_deterministic_and_order_insensitive():
    a = make_spec(parameters={"b": 2, "a": 1})
    b = make_spec(parameters={"a": 1, "b": 2})
    assert a.spec_hash == b.spec_hash
    again = ExperimentSpec.from_dict(json.loads(a.to_json()))
    assert again.spec_hash == a.spec_hash


def test_spec_hash_changes_with_seed_or_split_or_params():
    base = make_spec(seed=0, evaluation_split="valid", parameters={"x": 1})
    assert make_spec(seed=1, parameters={"x": 1}).spec_hash != base.spec_hash
    assert make_spec(parameters={"x": 2}).spec_hash != base.spec_hash
    test_spec = make_spec(evaluation_split="test", allow_test_split=True, parameters={"x": 1})
    assert test_spec.spec_hash != base.spec_hash


def test_parent_validation_baseline_mutation_crossover():
    make_spec(origin="baseline", parent_ids=())
    make_spec(origin="mutation", parent_ids=("p1",))
    make_spec(origin="crossover", parent_ids=("p1", "p2"))
    make_spec(origin="manual", parent_ids=())
    make_spec(origin="manual", parent_ids=("p1", "p2"))
    with pytest.raises(SpecError, match="zero parents"):
        make_spec(origin="baseline", parent_ids=("p1",))
    with pytest.raises(SpecError, match="exactly one parent"):
        make_spec(origin="mutation", parent_ids=())
    with pytest.raises(SpecError, match="at least two parents"):
        make_spec(origin="crossover", parent_ids=("p1",))
    with pytest.raises(SpecError, match="unique"):
        make_spec(origin="crossover", parent_ids=("p1", "p1"))


def test_invalid_experiment_id_and_origin():
    with pytest.raises(SpecError, match="experiment_id"):
        make_spec(experiment_id="../escape")
    with pytest.raises(SpecError, match="origin"):
        make_spec(origin="llm")


def test_result_round_trip_and_success_requires_metrics(tmp_path: Path):
    result = ExperimentResult(
        experiment_id="r1",
        status="success",
        evaluation_split="valid",
        seed=0,
        spec_hash="abc",
        wall_seconds=1.25,
        return_code=0,
        run_dir=str(tmp_path),
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
        scores_path=str(tmp_path / "scores.npy"),
        metrics=Metrics(gauc=0.5, ndcg_at_5=0.4, primary=0.45, users=2, rows=4),
    )
    path = tmp_path / "result.json"
    result.write_json(path)
    loaded = ExperimentResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.to_dict() == result.to_dict()
    with pytest.raises(ValueError, match="metrics"):
        ExperimentResult(
            experiment_id="r2",
            status="success",
            evaluation_split="valid",
            seed=0,
            spec_hash="abc",
            wall_seconds=0.0,
            return_code=0,
            run_dir=str(tmp_path),
            stdout_path="",
            stderr_path="",
        )


def test_implementation_ref_normalizes_slashes():
    ref = ImplementationRef(entrypoint=r"src\research_agent\recommenders\random_scorer.py")
    assert ref.entrypoint == "src/research_agent/recommenders/random_scorer.py"
