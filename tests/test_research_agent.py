"""Sequential ResearchAgent with FakeProvider. Zero API spend."""
from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.agent import ResearchAgent, UnusableRootError
from research_agent.agent.constants import FM_ROOT_ID
from research_agent.agent.fm_root import fm_root_spec
from research_agent.llm import FakeProvider, LLMConfigError, LLMRateLimitError, LLMTransientError
from research_helpers import make_proposal_payload, make_runner, mini_root_spec


def _agent(tmp_path: Path, script: list, **kwargs) -> ResearchAgent:
    runner = kwargs.pop("runner", None)
    if runner is None:
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
        session_id=kwargs.pop("session_id", "rs-test"),
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
    assert "rs-test-001" in second_prompt
    assert "first mutation" in second_prompt
    elite = agent.runner.registry.elite()
    assert elite is not None
    assert elite.spec.evaluation_split == "valid"
    for spec_id in ("fm-root", "rs-test-001", "rs-test-002"):
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
    assert agent.runner.registry.peek("rs-test-001") is None


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
    spec = agent.runner.registry.get("rs-test-001").spec
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


def test_two_sessions_do_not_collide_or_overwrite(tmp_path: Path):
    runner, _data = make_runner(tmp_path)
    first = _agent(
        tmp_path,
        [make_proposal_payload(hypothesis="session one")],
        runner=runner,
        session_id="rs-aaa",
        max_iterations=1,
    )
    run1 = first.run()
    path1 = first.workspace.dest_for("rs-aaa-001")
    fingerprint1 = run1.iterations[0].materialized.fingerprint
    stored_fp = runner.registry.get("rs-aaa-001").result.source_fingerprint
    source1 = path1.read_text(encoding="utf-8")
    primary1 = runner.registry.get("rs-aaa-001").result.metrics.primary

    second = _agent(
        tmp_path,
        [make_proposal_payload(hypothesis="session two")],
        runner=runner,
        session_id="rs-bbb",
        max_iterations=1,
    )
    run2 = second.run()
    path2 = second.workspace.dest_for("rs-bbb-001")
    assert run1.iterations[0].experiment_id != run2.iterations[0].experiment_id
    assert path1.is_file() and path2.is_file()
    assert path1.read_text(encoding="utf-8") == source1
    assert runner.registry.get("rs-aaa-001").result.metrics.primary == primary1
    assert runner.registry.get("rs-aaa-001").result.source_fingerprint == stored_fp
    assert run1.iterations[0].materialized.fingerprint == fingerprint1
    assert runner.registry.peek("rs-bbb-001") is not None
    assert runner.registry.count() == 3  # fm-root + two mutations
    assert first.trace.path.is_file() and second.trace.path.is_file()
    assert first.trace.records()[1]["experiment_id"] == "rs-aaa-001"
    assert second.trace.records()[1]["experiment_id"] == "rs-bbb-001"


def test_failed_fm_root_is_not_reused(tmp_path: Path):
    runner, _data = make_runner(tmp_path)
    fail_spec = mini_root_spec(tmp_path)
    payload = fail_spec.to_dict()
    payload["parameters"] = {"action": "fail"}
    payload.pop("spec_hash", None)
    from research_agent.experiments import ExperimentSpec

    fail_spec = ExperimentSpec.from_dict(payload)
    failed_agent = _agent(
        tmp_path,
        script=[],
        runner=runner,
        root_spec=fail_spec,
        max_iterations=1,
        session_id="rs-fail",
    )
    with pytest.raises(UnusableRootError):
        failed_agent.run()
    stored = runner.registry.get("fm-root")
    assert stored.result.status == "failed"
    assert failed_agent.provider.calls == []

    ok_agent = _agent(
        tmp_path,
        [make_proposal_payload()],
        runner=runner,
        root_spec=mini_root_spec(tmp_path),
        max_iterations=1,
        session_id="rs-ok",
    )
    run = ok_agent.run()
    assert run.root.experiment_id == "fm-root-r001"
    assert run.root.result_status == "success"
    assert run.iterations[0].parent_id == "fm-root-r001"
    assert runner.registry.get("fm-root").result.status == "failed"


def test_non_fm_baseline_is_not_research_root(tmp_path: Path):
    runner, _data = make_runner(tmp_path)
    other = mini_root_spec(tmp_path, experiment_id="random-valid-seed0")
    other_result = runner.run(other)
    assert other_result.status == "success"
    agent = _agent(
        tmp_path,
        [make_proposal_payload()],
        runner=runner,
        max_iterations=1,
        session_id="rs-from-fm",
    )
    run = agent.run()
    assert run.root.experiment_id == "fm-root"
    assert run.iterations[0].parent_id == "fm-root"
    assert runner.registry.get("random-valid-seed0").result.status == "success"


def test_fatal_llm_error_persists_prior_usage(tmp_path: Path):
    class OnceThenRateLimit:
        name = "boom"

        def __init__(self):
            self.n = 0
            self._inner = FakeProvider(script=[make_proposal_payload()])

        def generate(self, request):
            self.n += 1
            if self.n == 1:
                return self._inner.generate(request)
            raise LLMRateLimitError("Gemini rate limited")

    runner, _data = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=OnceThenRateLimit(),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=2,
        max_repairs=0,
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id="rs-usage",
    )
    with pytest.raises(LLMRateLimitError):
        agent.run()
    assert agent.ledger.research_calls == 2
    assert agent.ledger.llm_calls == 2
    assert agent.trace.summary_path.is_file()
    summary = agent.trace.summary_path.read_text(encoding="utf-8")
    assert "rs-usage" in summary
    assert agent.ledger.input_tokens > 0


def test_env_file_is_gitignored():
    import subprocess

    from conftest import ROOT

    listed = subprocess.check_output(
        ["git", "-C", str(ROOT), "check-ignore", "-v", ".env"],
        text=True,
    )
    assert ".env" in listed
    example = subprocess.call(["git", "-C", str(ROOT), "check-ignore", "-q", ".env.example"])
    assert example == 1


def test_null_timeout_proposal_executes(tmp_path: Path):
    payload = make_proposal_payload()
    payload["timeout_seconds"] = None
    agent = _agent(tmp_path, [payload], max_iterations=1)
    run = agent.run()
    assert run.iterations[0].result_status == "success"
    assert run.iterations[0].proposal.timeout_seconds == 600.0


def test_config_error_is_not_repaired(tmp_path: Path):
    class BoomProvider:
        name = "boom"

        def generate(self, request):
            raise LLMConfigError("GEMINI_API_KEY is not set in the process environment or repo-root .env")

    runner, _data = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=BoomProvider(),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=1,
        max_repairs=2,
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id="rs-cfg",
    )
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        agent.run()
    assert agent.ledger.repair_calls == 0
    assert agent.ledger.research_calls == 1
    assert agent.ledger.llm_calls == 1


def test_research_trace_redacts_key_value(tmp_path: Path, monkeypatch):
    secret = "AIzaSyFakeSecretValueForTraceTest00000"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    agent = _agent(tmp_path, [make_proposal_payload()], max_iterations=1)
    run = agent.run()
    dumped = agent.trace.path.read_text(encoding="utf-8") + str(run.summary)
    assert secret not in dumped
    assert "GEMINI_API_KEY=" not in dumped
