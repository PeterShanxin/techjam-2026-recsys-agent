"""Optional real-data path through the Phase 2 harness."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ROOT, resolve_data_dir
from research_agent.experiments import ExperimentRunner, ExperimentSpec


@pytest.mark.integration
def test_real_random_validation_through_harness(tmp_path: Path):
    data_dir = resolve_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"KuaiRand-Pure data not present at {data_dir}")
    spec = ExperimentSpec.from_path(ROOT / "configs" / "experiments" / "random_valid.json")
    payload = spec.to_dict()
    payload["experiment_id"] = "pytest-random-valid-seed0"
    payload.pop("spec_hash", None)
    spec = ExperimentSpec.from_dict(payload)
    runner = ExperimentRunner(
        repo_root=ROOT,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
    )
    result = runner.run(spec)
    assert result.status == "success"
    assert result.evaluation_split == "valid"
    assert result.metrics is not None
    # Phase 1 recorded valid random seed 0 primary 0.4827
    assert result.metrics.primary == pytest.approx(0.4827, abs=0.002)
    elite = runner.registry.elite()
    assert elite is not None
    assert elite.spec.experiment_id == spec.experiment_id
    assert elite.spec.evaluation_split == "valid"
