"""Sprint-2 freeze: provenance, sealed test, and evidence integrity. Zero API spend."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from research_agent.experiments import ExperimentSpec
from research_agent.final_candidate import FINAL_EXPERIMENT_ID

SPEC = Path("configs/experiments/tiered_ensemble_valid.json")
SCORER = Path("src/research_agent/recommenders/tiered_ensemble_scorer.py")
EVIDENCE = Path("docs/evidence/sprint2_autonomous_sprint.json")
LIVE_ELITE = "rs-20260831T062638Z-939b7000-008"
FROZEN_SWA7_PRIMARY = 0.6023186326402106


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_freeze_is_validation_only_and_not_the_submission_candidate() -> None:
    spec = ExperimentSpec.from_path(SPEC)
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert spec.experiment_id != FINAL_EXPERIMENT_ID
    assert spec.implementation.entrypoint.endswith("tiered_ensemble_scorer.py")


def test_frozen_scorer_is_valid_python_with_the_candidate_cli() -> None:
    source = SCORER.read_text(encoding="utf-8")
    ast.parse(source)
    for flag in ("--data-dir", "--split", "--output-scores", "--seed", "--config"):
        assert flag in source


def test_frozen_scorer_records_its_autonomous_provenance() -> None:
    doc = ast.get_docstring(ast.parse(SCORER.read_text(encoding="utf-8"))) or ""
    assert LIVE_ELITE in doc
    assert "TEST IS SEALED" in doc


def test_frozen_scorer_never_requests_the_test_split() -> None:
    source = SCORER.read_text(encoding="utf-8")
    assert '"test"' not in source.replace('choices=["valid", "test"]', "")


def test_evidence_reports_the_live_elite_and_reproductions() -> None:
    data = _evidence()
    best = data["best_candidate"]
    assert best["experiment_id"] == LIVE_ELITE
    assert best["mechanism_executed"] is True
    assert best["repo_entrypoint"] == SCORER.as_posix()
    assert best["repo_spec"] == SPEC.as_posix()
    repro = data["reproduction"]
    assert repro["type_a_same_seed"]["bitwise_identical_to_live_elite"] is True
    assert repro["repo_copy"]["bitwise_identical_to_live_elite"] is True
    seeds = repro["type_b_different_seeds"]
    assert len({item["seed"] for item in seeds}) == 3
    # the mechanism must beat the incumbent at every seed tried, not just the best one
    assert all(item["primary"] > FROZEN_SWA7_PRIMARY for item in seeds)


def test_evidence_does_not_overclaim() -> None:
    data = _evidence()
    assert data["statistical_significance_claimed"] is False
    vs_incumbent = data["paired_user_bootstrap"]["vs_frozen_swa7"]
    # the interval against the incumbent genuinely includes zero; the doc must say so
    assert vs_incumbent["ci95_low"] < 0 < vs_incumbent["ci95_high"]
    assert vs_incumbent["delta"] > 0
    vs_root = data["paired_user_bootstrap"]["vs_fm_root"]
    assert vs_root["ci95_low"] > 0


def test_evidence_keeps_the_sealed_test_and_evaluator_hash() -> None:
    from conftest import EVALUATE_SHA256

    data = _evidence()
    assert data["evaluate_py_sha256_lf"] == EVALUATE_SHA256
    assert "SEALED" in data["test_policy"]
    assert data["manual_interventions"] == 0
    for item in data["experiments"]:
        assert item.get("evaluation_split", "valid") == "valid"


def test_evidence_separates_negative_results_from_failures() -> None:
    """A timeout or crash is not evidence against a hypothesis; a completed run is."""
    data = _evidence()
    by_id = {item["experiment_id"]: item for item in data["experiments"]}
    ranking = by_id["rs-20260831T062638Z-939b7000-001"]
    assert ranking["status"] == "success"
    assert ranking["used_lab_ranking_machinery"] is True
    assert ranking["delta_vs_frozen_swa7"] < 0  # genuine negative result, not a timeout
    failed = by_id["rs-20260831T062638Z-939b7000-005"]
    assert failed["status"] != "success"
    assert failed.get("primary") is None


@pytest.mark.parametrize("path", [SPEC, EVIDENCE])
def test_artifacts_are_wellformed_json(path: Path) -> None:
    json.loads(path.read_text(encoding="utf-8"))
