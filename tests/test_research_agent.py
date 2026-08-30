"""Sequential ResearchAgent with FakeProvider. Zero API spend."""
from __future__ import annotations

from pathlib import Path

from research_agent.agent import ResearchAgent
from research_agent.agent.constants import FM_ROOT_ID
from research_agent.agent.fm_root import fm_root_spec
from research_agent.llm import FakeProvider
from research_helpers import make_proposal_payload, make_runner, mini_root_spec


def _agent(tmp_path: Path, script: list, **kwargs) -> ResearchAgent:
    runner, _data = make_runner(tmp_path)
    provider = FakeProvider(script=script)
    return ResearchAgent(
        provider=provider,
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=kwargs.pop("max_iterations", 2),
        max_repairs=kwargs.pop("max_repairs", 2),
        manual_interventions=kwargs.pop("manual_interventions", 0),
        root_spec=kwargs.pop("root_spec", mini_root_spec(tmp_path)),
        experiment_timeout_seconds=30.0,
        **kwargs,
    )


def test_fm_root_spec_is_validation_baseline():
    spec = fm_root_spec()
    assert spec.experiment_id == FM_ROOT_ID
    assert spec.origin == "baseline"
    assert spec.parent_ids == ()
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert spec.implementation.entrypoint.endswith("fm_scorer.py")
    assert spec.parameters["k"] == 16


def test_fm_root_initialization_runs_once(tmp_path: Path):
    agent = _agent(tmp_path, script=[], max_iterations=0)
    first = agent.ensure_root()
    assert first.experiment_id == "fm-root"
    assert first.result_status == "success"
    assert first.result.metrics is not None
    again = agent.ensure_root()
    assert again.result.metrics.primary == first.result.metrics.primary
    assert agent.runner.registry.count() == 1


def test_successful_sequential_iterations(tmp_path: Path):
    script = [
        make_proposal_payload(hypothesis="first mutation"),
        make_proposal_payload(hypothesis="second mutation", mutation_summary="second change"),
    ]
    agent = _agent(tmp_path, script, max_iterations=2)
    run = agent.run()
    assert run.root.result_status == "success"
    assert [item.result_status for item in run.iterations] == ["success", "success"]
    assert agent.provider.calls[0].purpose == "research"
    assert agent.provider.calls[1].purpose == "research"
    second_prompt = agent.provider.calls[1].prompt
    assert "ra-001" in second_prompt
    assert "first mutation" in second_prompt
    elite = agent.runner.registry.elite()
    assert elite is not None
    assert elite.spec.evaluation_split == "valid"
    for spec_id in ("fm-root", "ra-001", "ra-002"):
        assert agent.runner.registry.get(spec_id).spec.evaluation_split == "valid"
        assert agent.runner.registry.get(spec_id).spec.allow_test_split is False


def test_syntax_failure_bounded_repair(tmp_path: Path):
    bad = make_proposal_payload(candidate_source="def broken(")
    good = make_proposal_payload(hypothesis="repaired candidate")
    agent = _agent(tmp_path, [bad, good], max_iterations=1, max_repairs=2)
    run = agent.run()
    assert run.iterations[0].result_status == "success"
    assert run.iterations[0].repair_calls == 1
    assert agent.provider.calls[0].purpose == "research"
    assert agent.provider.calls[1].purpose == "repair"
    assert agent.provider.calls[1].thinking_level == "high"
    assert run.ledger.repair_calls == 1
    assert run.ledger.research_calls == 1


def test_repair_bound_is_finite(tmp_path: Path):
    bad = make_proposal_payload(candidate_source="def broken(")
    agent = _agent(tmp_path, [bad, bad, bad], max_iterations=1, max_repairs=2)
    run = agent.run()
    assert run.iterations[0].result_status == "invalid"
    assert run.iterations[0].result is None
    assert run.iterations[0].repair_calls == 2
    assert run.ledger.llm_calls == 3
    assert agent.runner.registry.peek("ra-001") is None


def test_failed_experiment_continues_from_elite(tmp_path: Path):
    fail = make_proposal_payload(
        hypothesis="this will crash",
        experiment_parameters={"action": "fail"},
        abandon_or_continue_reasoning="abandon this crashy direction",
    )
    ok = make_proposal_payload(hypothesis="back to elite parent")
    agent = _agent(tmp_path, [fail, ok], max_iterations=2)
    run = agent.run()
    assert run.iterations[0].result_status == "failed"
    assert run.iterations[1].result_status == "success"
    assert run.iterations[1].parent_id == "fm-root"
    assert agent.runner.registry.elite().spec.experiment_id == "fm-root"
    assert run.ledger.failed_experiments >= 1
    assert run.ledger.completed_experiments >= 2  # root + second success


def test_validation_only_enforced(tmp_path: Path):
    agent = _agent(
        tmp_path,
        [make_proposal_payload()],
        max_iterations=1,
    )
    run = agent.run()
    spec = agent.runner.registry.get("ra-001").spec
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert run.iterations[0].result.evaluation_split == "valid"


def test_trace_persistence_and_cumulative_tokens(tmp_path: Path):
    agent = _agent(
        tmp_path,
        [make_proposal_payload(), make_proposal_payload(hypothesis="again")],
        max_iterations=2,
        manual_interventions=0,
    )
    run = agent.run()
    records = agent.trace.records()
    assert records[0]["experiment_id"] == "fm-root"
    assert records[1]["iteration"] == 1
    assert records[1]["hypothesis"]
    assert records[1]["token_counts"]["total_tokens"] > 0
    assert records[-1]["cumulative"]["total_tokens"] == run.ledger.total_tokens
    assert records[-1]["manual_interventions"] == 0
    assert agent.trace.report_path.is_file()
    assert agent.trace.summary_path.is_file()
    report = agent.trace.report_path.read_text(encoding="utf-8")
    assert "Phase 3 research trace" in report
    assert run.ledger.llm_calls == 2
    assert run.ledger.input_tokens > 0
    assert run.ledger.output_tokens > 0
    assert run.ledger.thinking_tokens > 0
    assert run.ledger.manual_interventions == 0
    dumped = str(run.summary) + agent.trace.path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" not in dumped


def test_manual_intervention_count_is_explicit(tmp_path: Path):
    agent = _agent(tmp_path, script=[], max_iterations=0, manual_interventions=2)
    run = agent.run()
    assert run.ledger.manual_interventions == 2
    assert run.summary["manual_interventions"] == 2
