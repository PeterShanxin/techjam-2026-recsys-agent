"""Canonical evidence JSON is the number source. Judge-facing docs must not drift."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "docs" / "evidence" / "canonical_benchmark.json"
README = ROOT / "README.md"
EVALUATOR = "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"

JUDGE_FACING = (
    "README.md",
    "docs/BENCHMARK.md",
    "docs/DEVPOST.md",
    "docs/DEMO_SCRIPT.md",
    "docs/SUBMISSION_CHECKLIST.md",
)


def _canon() -> dict:
    return json.loads(CANON.read_text(encoding="utf-8"))


def test_canonical_benchmark_has_locked_values():
    data = _canon()
    assert data["evaluate_py_sha256"] == EVALUATOR
    assert data["manual_interventions"] == 0
    assert data["statistical_significance_claimed"] is False
    assert data["final_search_decision"] == "A"
    methods = {item["id"]: item for item in data["methods"]}
    assert methods["fm"]["best_primary"] == 0.601468756352959
    assert methods["phase3_sequential"]["best_primary"] == 0.6021109230359711
    assert methods["phase4_matched_sequential"]["best_primary"] == 0.6021109230359711
    assert methods["phase4_matched_sequential"]["delta_vs_starting_elite"] == 0.0
    assert methods["phase4_evolution"]["best_primary"] == 0.6023186326402106
    assert methods["phase4_evolution"]["new_evaluations"] == 6
    assert methods["phase4_matched_sequential"]["new_evaluations"] == 6
    assert methods["sprint2_evolution"]["best_primary"] == 0.6029037142533181
    assert methods["sprint2_evolution"]["new_evaluations"] == 7
    assert methods["sprint2_evolution"]["stop_reason"] == "converged"
    assert methods["sprint2_evolution"]["manual_interventions"] == 0


def test_final_candidate_is_the_sprint2_autonomous_elite():
    final = _canon()["final_candidate"]
    assert final["experiment_id"] == "final-tiered-ensemble"
    assert final["live_elite_id"] == "rs-20260831T062638Z-939b7000-008"
    assert final["primary"] == 0.6029037142533181
    assert final["seed"] == 42
    assert final["type_a_rerun"]["matches_live_elite"] is True
    assert final["type_a_rerun"]["primary"] == final["primary"]
    # the mechanism must beat the superseded candidate at every seed tried, not just the best
    seeds = final["type_b_different_seeds"]
    assert len({item["seed"] for item in seeds}) == 3
    assert all(item["primary"] > 0.6023186326402106 for item in seeds)


def test_superseded_candidate_is_kept_not_erased():
    data = _canon()
    old = data["superseded_phase4_candidate"]
    assert old["experiment_id"] == "final-swa7-ensemble"
    assert old["primary"] == 0.6023186326402106
    assert old["delta_vs_starting_elite_display"] == "+0.0002077"
    assert (ROOT / old["entrypoint"]).is_file()
    assert "--legacy-swa7" in old["still_runnable"]
    # the Phase 4 history rows are untouched
    methods = {item["id"]: item for item in data["methods"]}
    assert methods["phase4_evolution"]["best_experiment_id"] == "rs-20260830T133522Z-0e304128-004"


def test_deltas_are_arithmetically_consistent():
    data = _canon()
    fm = data["fm_root"]["primary"]
    old = data["superseded_phase4_candidate"]["primary"]
    new = data["final_candidate"]["primary"]
    assert data["final_candidate"]["delta_vs_fm"] == pytest.approx(new - fm, abs=1e-15)
    assert data["final_candidate"]["delta_vs_superseded_swa7"] == pytest.approx(new - old, abs=1e-12)
    assert data["superseded_phase4_candidate"]["delta_vs_fm"] == pytest.approx(old - fm, abs=1e-15)


@pytest.mark.parametrize(
    "baseline,candidate,ci_excludes_zero",
    [
        ("fm-root", "final-tiered-ensemble", True),
        ("final-swa7-ensemble", "final-tiered-ensemble", False),
        ("fm-root", "final-swa7-ensemble", False),
    ],
)
def test_paired_bootstrap_intervals_are_recorded_honestly(baseline, candidate, ci_excludes_zero):
    comparisons = _canon()["paired_user_bootstrap"]["comparisons"]
    match = [c for c in comparisons if c["baseline"] == baseline and c["candidate"] == candidate]
    assert len(match) == 1, (baseline, candidate)
    item = match[0]
    assert item["split"] == "valid"
    assert item["reps"] == 2000
    assert (item["ci95_low"] > 0) is ci_excludes_zero
    assert item["ci95_low"] < item["delta"] < item["ci95_high"]


def test_only_the_fm_root_comparison_is_claimed_significant():
    data = _canon()
    assert data["statistical_significance_claimed"] is False
    policy = data["significance_policy"]
    assert "fm-root" in policy["one_comparison_does_clear_95pct"]
    assert "0.990" in policy["one_comparison_does_clear_95pct"]
    assert "indistinguishable" in policy["test_split"]


def test_test_split_is_an_observation_that_did_not_pick_the_model():
    art = _canon()["submission_artifact"]
    assert art["split"] == "test"
    assert art["used_for_selection"] is False
    assert art["test_runs_executed"] == 1
    assert art["official_check"] == "pass"
    assert art["rows"] == 170588
    assert art["committed"] is False and art["gitignored"] is True
    obs = art["test_observation_not_for_selection"]
    # the honest part: the validation gain did NOT transfer, and the docs must say so
    assert obs["delta"] < 0
    assert "did NOT transfer" in obs["honest_reading"]
    assert obs["final-tiered-ensemble"]["primary"] == 0.5963753615661991
    assert obs["superseded_final-swa7-ensemble"]["primary"] == 0.596386214222103


def test_final_candidate_is_cheaper_on_the_same_harness():
    cost = _canon()["runtime_comparison"]["like_for_like"]
    for split in ("valid", "test"):
        row = cost[split]
        assert row["final-tiered-ensemble"] < row["final-swa7-ensemble"], split
        assert row["delta_seconds"] < 0, split
        assert row["relative"] < 0, split


def test_readme_uses_canonical_display_strings():
    data = _canon()
    readme = README.read_text(encoding="utf-8")
    for item in data["methods"]:
        assert item["primary_display"] in readme, item["id"]
    assert data["final_candidate"]["delta_vs_fm_display"] in readme
    assert data["final_candidate"]["delta_vs_superseded_swa7_display"] in readme
    assert data["superseded_phase4_candidate"]["primary_display"] in readme
    assert EVALUATOR in readme


def test_every_judge_facing_doc_agrees_on_the_headline_numbers():
    """One wrong decimal in one doc is the failure mode this guards."""
    data = _canon()
    obs = data["submission_artifact"]["test_observation_not_for_selection"]
    valid_display = data["final_candidate"]["primary_display"]
    test_display = obs["final-tiered-ensemble"]["primary_display"]
    for rel in JUDGE_FACING:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert valid_display in text, f"{rel} is missing the final validation primary"
        if rel != "README.md":
            assert "final-tiered-ensemble" in text or "tiered" in text, rel
    # the test observation must appear wherever the test result is discussed at all
    for rel in ("README.md", "docs/BENCHMARK.md", "docs/DEVPOST.md", "docs/DEMO_SCRIPT.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert test_display in text, f"{rel} is missing the test observation"


def test_no_judge_facing_doc_still_calls_the_ensemble_a_probability_average():
    """FM.predict returns logits. The wording was wrong and must stay corrected."""
    for rel in JUDGE_FACING + ("docs/SECOND_OPINION_SPRINT.md",):
        text = (ROOT / rel).read_text(encoding="utf-8").lower()
        assert "raw probability average" not in text, rel
        assert "raw probability mean" not in text, rel


def test_final_candidate_entrypoint_matches_json():
    data = _canon()
    entry = Path(data["final_candidate"]["entrypoint"])
    assert (ROOT / entry).is_file()
    assert not str(entry).startswith("runs/")
    for key in ("spec_valid", "spec_test"):
        assert (ROOT / data["final_candidate"][key]).is_file(), key
