"""ExperimentRunner success and failure modes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research_agent.experiments import ExperimentRunner, ImplementationRef
from experiment_helpers import make_spec, write_candidate, write_mini_dataset


def _runner(tmp_path: Path, data_dir: Path) -> ExperimentRunner:
    return ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
    )


def _spec(tmp_path: Path, experiment_id: str, action: str, **kwargs):
    candidate = tmp_path / "candidate.py"
    if not candidate.exists():
        write_candidate(candidate)
    return make_spec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=str(candidate)),
        parameters={"action": action, **kwargs.pop("extra_params", {})},
        **kwargs,
    )


def test_successful_candidate(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    spec = _spec(tmp_path, "ok-run", "succeed")
    result = _runner(tmp_path, data_dir).run(spec)
    assert result.status == "success"
    assert result.return_code == 0
    assert result.metrics is not None
    assert result.metrics.primary == (
        result.metrics.gauc + result.metrics.ndcg_at_5
    ) / 2.0
    run_dir = Path(result.run_dir)
    for name in ("spec.json", "result.json", "stdout.log", "stderr.log", "scores.npy", "metadata.json"):
        assert (run_dir / name).is_file()
    scores = np.load(run_dir / "scores.npy")
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    stored = _runner(tmp_path, data_dir).registry.get("ok-run")
    assert stored.result is not None
    assert stored.result.status == "success"


def test_failed_subprocess(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    result = _runner(tmp_path, data_dir).run(_spec(tmp_path, "fail-run", "fail"))
    assert result.status == "failed"
    assert result.return_code == 2
    assert result.metrics is None
    assert result.failure is not None
    assert result.failure.kind == "subprocess"


def test_timeout(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    spec = _spec(tmp_path, "slow-run", "sleep", timeout_seconds=0.3, extra_params={"sleep": 10})
    result = _runner(tmp_path, data_dir).run(spec)
    assert result.status == "timeout"
    assert result.metrics is None
    assert result.failure is not None
    assert result.failure.kind == "timeout"


def test_missing_score_artifact(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    result = _runner(tmp_path, data_dir).run(_spec(tmp_path, "miss-run", "missing"))
    assert result.status == "invalid"
    assert result.failure.kind == "missing_scores"


def test_wrong_score_length(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    result = _runner(tmp_path, data_dir).run(_spec(tmp_path, "len-run", "wrong_length"))
    assert result.status == "invalid"
    assert result.failure.kind == "wrong_length"


def test_nan_scores_rejected(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    result = _runner(tmp_path, data_dir).run(_spec(tmp_path, "nan-run", "nan"))
    assert result.status == "invalid"
    assert result.failure.kind == "non_finite"


def test_inf_scores_rejected(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    result = _runner(tmp_path, data_dir).run(_spec(tmp_path, "inf-run", "inf"))
    assert result.status == "invalid"
    assert result.failure.kind == "non_finite"


def test_wrong_dimensionality_rejected(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    result = _runner(tmp_path, data_dir).run(_spec(tmp_path, "shape-run", "wrong_shape"))
    assert result.status == "invalid"
    assert result.failure.kind == "wrong_shape"
