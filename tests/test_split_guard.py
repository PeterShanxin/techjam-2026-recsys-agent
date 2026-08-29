"""Validation-only research policy."""
from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.experiments import (
    ExperimentRunner,
    ExperimentSpec,
    ImplementationRef,
    ForbiddenTestSplit,
)
from research_agent.experiments.splits import assert_split_allowed, is_research_split
from experiment_helpers import make_spec, write_candidate, write_mini_dataset


def test_valid_is_research_split_and_default():
    spec = make_spec()
    assert spec.evaluation_split == "valid"
    assert is_research_split("valid") is True
    assert is_research_split("test") is False
    assert_split_allowed("valid", allow_test=False)


def test_test_split_requires_explicit_opt_in():
    with pytest.raises(ForbiddenTestSplit):
        assert_split_allowed("test", allow_test=False)
    assert_split_allowed("test", allow_test=True)


def test_runner_rejects_test_without_opt_in(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    candidate = write_candidate(tmp_path / "candidate.py")
    spec = ExperimentSpec(
        experiment_id="test-blocked",
        implementation=ImplementationRef(entrypoint=str(candidate)),
        evaluation_split="test",
        allow_test_split=False,
        parameters={"action": "succeed"},
    )
    runner = ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
        allow_test=False,
    )
    result = runner.run(spec)
    assert result.status == "invalid"
    assert result.metrics is None
    assert result.failure is not None
    assert result.failure.kind == "test_split"


def test_runner_allows_test_with_spec_opt_in(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    candidate = write_candidate(tmp_path / "candidate.py")
    spec = make_spec(
        experiment_id="test-allowed",
        implementation=ImplementationRef(entrypoint=str(candidate)),
        evaluation_split="test",
        allow_test_split=True,
        parameters={"action": "succeed"},
    )
    runner = ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
        allow_test=False,
    )
    result = runner.run(spec)
    assert result.status == "success"
    assert result.evaluation_split == "test"
    assert result.metrics is not None
