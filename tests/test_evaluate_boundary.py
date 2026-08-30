"""Official evaluator stays the only metric source."""
from __future__ import annotations

import hashlib
from pathlib import Path

from conftest import EVALUATE_SHA256, EVALUATE_PY, evaluate_py_canonical_bytes
from research_agent.experiments import ExperimentRunner, ImplementationRef
from experiment_helpers import make_spec, write_candidate, write_mini_dataset


def test_organizer_evaluate_py_unchanged():
    digest = hashlib.sha256(evaluate_py_canonical_bytes()).hexdigest()
    assert digest == EVALUATE_SHA256
    assert EVALUATE_PY.is_file()


def test_runner_calls_official_evaluate(tmp_path: Path, monkeypatch):
    data_dir = write_mini_dataset(tmp_path)
    candidate = write_candidate(tmp_path / "candidate.py")
    spec = make_spec(
        experiment_id="eval-boundary",
        implementation=ImplementationRef(entrypoint=str(candidate)),
        parameters={"action": "succeed"},
    )
    calls = []

    def spy(user_ids, labels, scores, k=5):
        from evaluate import evaluate

        calls.append((list(user_ids), list(labels), k))
        return evaluate(user_ids, labels, scores, k=k)

    monkeypatch.setattr("research_agent.experiments.runner.official_evaluate", spy)
    result = ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
    ).run(spec)
    assert result.status == "success"
    assert len(calls) == 1
    assert calls[0][2] == 5
    assert len(calls[0][0]) == 4
    env = result.environment
    assert env.get("evaluate_py_sha256") == EVALUATE_SHA256
