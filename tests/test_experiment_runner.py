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


SILENT_CANDIDATE = """\
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output-scores", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.parse_args()
    print("silent-exit-0")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def test_rerun_does_not_reuse_stale_scores(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    runner = _runner(tmp_path, data_dir)
    spec = _spec(tmp_path, "stale-id", "succeed")
    first = runner.run(spec)
    assert first.status == "success"
    assert first.metrics is not None
    first_primary = first.metrics.primary
    run_dir = Path(first.run_dir)
    first_scores = np.load(run_dir / "scores.npy").copy()
    assert first_scores.size == 4

    (tmp_path / "candidate.py").write_text(SILENT_CANDIDATE, encoding="utf-8")
    second = runner.run(spec)
    assert second.status == "invalid"
    assert second.metrics is None
    assert second.failure is not None
    assert second.failure.kind == "missing_scores"
    stored = runner.registry.get("stale-id")
    assert stored.result is not None
    assert stored.result.status == "invalid"
    assert stored.result.metrics is None
    assert stored.result.status != "success"
    published = run_dir / "scores.npy"
    assert not published.is_file()


def test_experiment_id_collision_preserves_existing(tmp_path: Path):
    data_dir = write_mini_dataset(tmp_path)
    runner = _runner(tmp_path, data_dir)
    original = _spec(tmp_path, "owned-id", "succeed", notes="original-notes")
    first = runner.run(original)
    assert first.status == "success"
    assert first.metrics is not None
    run_dir = Path(first.run_dir)
    snapshot = {
        name: (run_dir / name).read_bytes()
        for name in (
            "spec.json",
            "config.json",
            "result.json",
            "scores.npy",
            "metadata.json",
            "stdout.log",
            "stderr.log",
        )
    }
    attempts_before = sorted(p.name for p in (run_dir / "attempts").iterdir())
    before = runner.registry.get("owned-id")
    before_decision = before.decision
    before_created = before.created_at
    before_hash = before.spec.spec_hash
    before_primary = before.result.metrics.primary
    before_result = before.result.to_dict()

    colliding = _spec(tmp_path, "owned-id", "succeed", extra_params={"other": True})
    assert colliding.spec_hash != original.spec_hash
    second = runner.run(colliding)
    assert second.status == "invalid"
    assert second.metrics is None
    assert second.failure is not None
    assert second.failure.kind == "id_collision"

    after = runner.registry.get("owned-id")
    assert after.spec.spec_hash == before_hash
    assert after.spec.notes == "original-notes"
    assert after.decision == before_decision
    assert after.created_at == before_created
    assert after.result is not None
    assert after.result.status == "success"
    assert after.result.metrics.primary == pytest.approx(before_primary)
    assert after.result.to_dict() == before_result
    for name, payload in snapshot.items():
        assert (run_dir / name).read_bytes() == payload
    attempts_after = sorted(p.name for p in (run_dir / "attempts").iterdir())
    assert attempts_after == attempts_before
