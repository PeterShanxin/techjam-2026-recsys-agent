"""Canonical evidence JSON is the number source. README must not drift."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "docs" / "evidence" / "canonical_benchmark.json"
README = ROOT / "README.md"
EVALUATOR = "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"


def test_canonical_benchmark_has_locked_values():
    data = json.loads(CANON.read_text(encoding="utf-8"))
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
    assert data["final_candidate"]["type_a_rerun"]["matches_live_elite"] is True
    assert data["final_candidate"]["type_a_rerun"]["primary"] == data["final_candidate"]["primary"]


def test_readme_uses_canonical_display_strings():
    data = json.loads(CANON.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    for item in data["methods"]:
        assert item["primary_display"] in readme, item["id"]
    assert data["final_candidate"]["delta_vs_starting_elite_display"] in readme
    assert EVALUATOR in readme


def test_final_candidate_entrypoint_matches_json():
    data = json.loads(CANON.read_text(encoding="utf-8"))
    entry = Path(data["final_candidate"]["entrypoint"])
    assert (ROOT / entry).is_file()
    assert not str(entry).startswith("runs/")
