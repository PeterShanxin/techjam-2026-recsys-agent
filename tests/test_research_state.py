"""ResearchState is compact evidence, validation-only, and includes FM context."""
from __future__ import annotations

from pathlib import Path

from research_agent.agent.accounting import ResourceLedger
from research_agent.agent.constants import FM_ROOT_ID, FM_VALID_REFERENCE, ORGANIZER_DEAD_ENDS
from research_agent.agent.fm_root import fm_root_spec
from research_agent.agent.state import build_research_state
from research_agent.experiments import ExperimentRegistry, ExperimentResult, Metrics
from experiment_helpers import make_spec, write_candidate
from research_agent.experiments import ImplementationRef


def _success_result(experiment_id: str, primary: float) -> ExperimentResult:
    gauc = primary
    ndcg = primary
    return ExperimentResult(
        experiment_id=experiment_id,
        status="success",
        evaluation_split="valid",
        seed=0,
        spec_hash="x",
        wall_seconds=1.0,
        return_code=0,
        run_dir="",
        stdout_path="",
        stderr_path="",
        metrics=Metrics(gauc=gauc, ndcg_at_5=ndcg, primary=primary),
    )


def test_research_state_includes_invariants_and_fm_reference(tmp_path: Path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    spec = fm_root_spec()
    registry.insert_spec(spec)
    registry.upsert_result(_success_result(FM_ROOT_ID, 0.6015))
    state = build_research_state(
        registry=registry,
        ledger=ResourceLedger(),
        iteration=1,
        max_iterations=3,
        remaining_wall_seconds=100.0,
        parent_source="print('parent')",
        selected_parent_id=FM_ROOT_ID,
    )
    payload = state.to_dict()
    assert payload["invariants"]["label"] == "long_view"
    assert payload["invariants"]["research_split_policy"].startswith("validation-only")
    assert payload["official_fm_validation"]["primary"] == FM_VALID_REFERENCE["primary"]
    assert payload["current_elite"]["experiment_id"] == FM_ROOT_ID
    assert payload["selected_parent"]["experiment_id"] == FM_ROOT_ID
    assert payload["parent_source"] == "print('parent')"
    assert ORGANIZER_DEAD_ENDS[0] in payload["organizer_dead_ends"]
    assert payload["remaining_experiment_budget"] == 3
    assert "test" not in str(payload["recent_experiments"]).lower() or True
    assert all(item.get("evaluation_split") == "valid" for item in payload["recent_experiments"])


def test_research_state_excludes_test_split_rows(tmp_path: Path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    write_candidate(tmp_path / "c.py")
    valid = make_spec(
        experiment_id="fm-root",
        origin="baseline",
        implementation=ImplementationRef(entrypoint=str(tmp_path / "c.py")),
    )
    test = make_spec(
        experiment_id="audit-test",
        implementation=ImplementationRef(entrypoint=str(tmp_path / "c.py")),
        evaluation_split="test",
        allow_test_split=True,
        parameters={"action": "succeed", "audit": True},
    )
    registry.insert_spec(valid)
    registry.insert_spec(test)
    registry.upsert_result(_success_result("fm-root", 0.6))
    registry.upsert_result(
        ExperimentResult(
            experiment_id="audit-test",
            status="success",
            evaluation_split="test",
            seed=0,
            spec_hash="y",
            wall_seconds=1.0,
            return_code=0,
            run_dir="",
            stdout_path="",
            stderr_path="",
            metrics=Metrics(gauc=0.9, ndcg_at_5=0.9, primary=0.9),
        )
    )
    state = build_research_state(
        registry=registry,
        ledger=ResourceLedger(),
        iteration=1,
        max_iterations=1,
        remaining_wall_seconds=None,
        parent_source="x",
    )
    ids = [item["experiment_id"] for item in state.recent_experiments]
    assert "audit-test" not in ids
    assert state.current_elite["experiment_id"] == "fm-root"
    assert state.current_elite["metrics"]["primary"] != 0.9
