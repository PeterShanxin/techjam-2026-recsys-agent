"""Sequential ResearchAgent. One Gemini research call per iteration. No populations."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from research_agent.experiments import ExperimentRunner, ExperimentSpec
from research_agent.experiments.splits import RESEARCH_SPLIT
from research_agent.llm.secrets import redact_text, sanitize
from research_agent.llm.types import (
    LLMAuthError,
    LLMConfigError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    UsageRecord,
)

from .accounting import ResourceLedger
from .constants import (
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_THINKING_LEVEL,
    FM_VALID_REFERENCE,
    MAX_REPAIRS_PER_ITERATION,
    REPAIR_THINKING_LEVEL,
)
from .environment import format_preflight_repair_message
from .fm_root import fm_root_spec
from .prompts import proposal_schema, repair_prompt, research_prompt
from .proposal import ProposalError, ResearchProposal
from .root import (
    UnusableRootError,
    find_usable_root,
    is_usable_root_result,
    next_root_experiment_id,
    spec_with_experiment_id,
)
from .safety import SafetyError, validate_candidate_source
from .session import experiment_id_for, new_research_session_id
from .state import ResearchState, build_research_state
from .trace import ResearchTrace
from .workspace import CandidateWorkspace, MaterializedCandidate

EmitFn = Callable[[dict[str, Any]], None]


@dataclass
class IterationOutcome:
    iteration: int
    experiment_id: str
    parent_id: str | None
    proposal: ResearchProposal | None
    result_status: str
    result: Any = None
    materialized: MaterializedCandidate | None = None
    usages: list[UsageRecord] = field(default_factory=list)
    error: str | None = None
    repair_calls: int = 0
    record: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchRun:
    root: IterationOutcome | None
    iterations: list[IterationOutcome]
    ledger: ResourceLedger
    trace_dir: Path
    summary: dict[str, Any]
    session_id: str = ""


class ResearchAgent:
    def __init__(
        self,
        *,
        provider: Any,
        runner: ExperimentRunner,
        workspace: CandidateWorkspace | None = None,
        trace: ResearchTrace | None = None,
        model: str = DEFAULT_RESEARCH_MODEL,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
        max_iterations: int = 3,
        max_repairs: int = MAX_REPAIRS_PER_ITERATION,
        escalate_repairs: bool = True,
        wall_clock_seconds: float | None = None,
        manual_interventions: int = 0,
        root_spec: ExperimentSpec | None = None,
        emit: EmitFn | None = None,
        experiment_timeout_seconds: float = 900.0,
        session_id: str | None = None,
    ) -> None:
        if max_iterations < 0:
            raise ValueError("max_iterations must be >= 0")
        if max_repairs < 0:
            raise ValueError("max_repairs must be >= 0")
        self.provider = provider
        self.runner = runner
        self.model = model
        self.thinking_level = thinking_level
        self.max_iterations = max_iterations
        self.max_repairs = max_repairs
        self.escalate_repairs = escalate_repairs
        self.wall_clock_seconds = wall_clock_seconds
        self.experiment_timeout_seconds = experiment_timeout_seconds
        self.root_spec = root_spec or fm_root_spec(timeout_seconds=experiment_timeout_seconds)
        self.session_id = session_id or new_research_session_id()
        self.ledger = ResourceLedger(manual_interventions=int(manual_interventions))
        self.rejected_directions: list[str] = []
        self.emit = emit or (lambda _event: None)
        runs_dir = Path(self.runner.runs_dir)
        self.workspace = workspace or CandidateWorkspace(runs_dir / "generated")
        research_dir = runs_dir / "research" / self.session_id
        self.trace = trace or ResearchTrace(
            path=research_dir / "trace.jsonl",
            report_path=research_dir / "report.md",
            summary_path=research_dir / "summary.json",
        )

    def run(self) -> ResearchRun:
        started = time.perf_counter()
        root: IterationOutcome | None = None
        outcomes: list[IterationOutcome] = []
        try:
            root = self.ensure_root()
            if not is_usable_root_result(root.result):
                self.ledger.research_wall_seconds = time.perf_counter() - started
                summary = self._summary(root, [])
                self.trace.write_exports(summary=summary)
                raise UnusableRootError(
                    f"research root {root.experiment_id} is not a successful validation result"
                )
            for iteration in range(1, self.max_iterations + 1):
                remaining_wall = self._remaining_wall(started)
                if remaining_wall is not None and remaining_wall <= 0:
                    self.emit({"type": "budget", "reason": "wall_clock", "iteration": iteration})
                    break
                outcome = self._run_iteration(iteration, started)
                outcomes.append(outcome)
            self.ledger.research_wall_seconds = time.perf_counter() - started
            summary = self._summary(root, outcomes)
            self.trace.write_exports(summary=summary)
            self.emit({"type": "done", "summary": summary})
            return ResearchRun(
                root=root,
                iterations=outcomes,
                ledger=self.ledger,
                trace_dir=self.trace.path.parent,
                summary=summary,
                session_id=self.session_id,
            )
        except (LLMConfigError, LLMAuthError, LLMRateLimitError, LLMTransientError, LLMProtocolError):
            self.ledger.research_wall_seconds = time.perf_counter() - started
            summary = self._summary(root, outcomes)
            self.trace.write_exports(summary=summary)
            raise

    def ensure_root(self) -> IterationOutcome:
        usable = find_usable_root(self.runner.registry, self.root_spec.experiment_id)
        if usable is not None and usable.result is not None:
            result = usable.result
            experiment_id = usable.spec.experiment_id
            self.emit(
                {
                    "type": "root",
                    "experiment_id": experiment_id,
                    "status": result.status,
                    "reused": True,
                }
            )
        else:
            experiment_id = next_root_experiment_id(
                self.runner.registry, self.root_spec.experiment_id
            )
            spec = spec_with_experiment_id(self.root_spec, experiment_id)
            self.emit({"type": "root", "experiment_id": experiment_id, "status": "running"})
            result = self.runner.run(spec)
            self.ledger.add_experiment(status=result.status, wall_seconds=result.wall_seconds)
        outcome = IterationOutcome(
            iteration=0,
            experiment_id=experiment_id,
            parent_id=None,
            proposal=None,
            result_status=result.status,
            result=result,
        )
        outcome.record = self._trace_record(outcome, usages=[], iteration=0)
        self.trace.append(outcome.record)
        self._emit_iteration(outcome)
        return outcome

    def _run_iteration(self, iteration: int, started: float) -> IterationOutcome:
        experiment_id = experiment_id_for(self.session_id, iteration)
        parent_id = self._default_parent_id()
        parent_source = self._parent_source(parent_id)
        state = build_research_state(
            registry=self.runner.registry,
            ledger=self.ledger,
            iteration=iteration,
            max_iterations=self.max_iterations,
            remaining_wall_seconds=self._remaining_wall(started),
            parent_source=parent_source,
            selected_parent_id=parent_id,
            rejected_directions=self.rejected_directions,
            repo_root=self.runner.repo_root,
        )
        self.emit(
            {
                "type": "iteration",
                "iteration": iteration,
                "max_iterations": self.max_iterations,
                "session_id": self.session_id,
                "parent": parent_id,
                "elite": None if state.current_elite is None else state.current_elite.get("experiment_id"),
            }
        )
        dest = self.workspace.dest_for(experiment_id)
        proposal, usages, error = self._propose(state, dest)
        repair_calls = sum(1 for item in usages if item.purpose == "repair")
        if proposal is None:
            outcome = IterationOutcome(
                iteration=iteration,
                experiment_id=experiment_id,
                parent_id=parent_id,
                proposal=None,
                result_status="invalid",
                usages=usages,
                error=error,
                repair_calls=repair_calls,
            )
            outcome.record = self._trace_record(outcome, usages, iteration)
            self.trace.append(outcome.record)
            self._emit_iteration(outcome)
            return outcome

        parent_id = self._resolve_parent_id(proposal.selected_parent_id)
        parent_source = self._parent_source(parent_id)
        try:
            materialized = self.workspace.materialize(
                experiment_id=experiment_id,
                source=proposal.candidate_source,
                parent_source=parent_source,
                repo_root=self.runner.repo_root,
            )
            (dest.parent / "proposal.json").write_text(proposal.to_json(), encoding="utf-8")
            (dest.parent / "diff.patch").write_text(materialized.diff_vs_parent, encoding="utf-8")
        except SafetyError as exc:
            outcome = IterationOutcome(
                iteration=iteration,
                experiment_id=experiment_id,
                parent_id=parent_id,
                proposal=proposal,
                result_status="invalid",
                usages=usages,
                error=redact_text(str(exc)),
                repair_calls=repair_calls,
            )
            outcome.record = self._trace_record(outcome, usages, iteration)
            self.trace.append(outcome.record)
            self._emit_iteration(outcome)
            return outcome

        spec = ExperimentSpec(
            experiment_id=experiment_id,
            implementation=materialized.implementation,
            hypothesis=proposal.hypothesis,
            rationale=proposal.rationale,
            origin="mutation",
            parent_ids=(parent_id,),
            parameters=dict(proposal.experiment_parameters),
            seed=proposal.seed,
            evaluation_split=RESEARCH_SPLIT,
            timeout_seconds=min(proposal.timeout_seconds, self.experiment_timeout_seconds),
            allow_test_split=False,
            tags=("phase3", "autonomous", self.session_id),
            notes=proposal.mutation_summary,
        )
        result = self.runner.run(spec)
        self.ledger.add_experiment(status=result.status, wall_seconds=result.wall_seconds)
        self._maybe_reject(proposal, result)
        outcome = IterationOutcome(
            iteration=iteration,
            experiment_id=experiment_id,
            parent_id=parent_id,
            proposal=proposal,
            result_status=result.status,
            result=result,
            materialized=materialized,
            usages=usages,
            error=_result_error(result),
            repair_calls=repair_calls,
        )
        outcome.record = self._trace_record(outcome, usages, iteration)
        self.trace.append(outcome.record)
        self._emit_iteration(outcome)
        return outcome

    def _propose(
        self,
        state: ResearchState,
        dest: Path,
    ) -> tuple[ResearchProposal | None, list[UsageRecord], str | None]:
        usages: list[UsageRecord] = []
        try:
            request = LLMRequest(
                prompt=research_prompt(state),
                response_schema=proposal_schema(),
                model=self.model,
                thinking_level=self.thinking_level,
                purpose="research",
            )
            response = self._generate(request, usages)
            last_error = None if response is not None else (usages[-1].error if usages else "provider error")
            previous_text = "" if response is None else response.text
            for repair in range(0, self.max_repairs + 1):
                if response is not None:
                    proposal, last_error = _try_parse_proposal(
                        response,
                        dest,
                        self.workspace.root,
                        environment=state.environment,
                    )
                    if proposal is not None:
                        return proposal, usages, None
                elif last_error is None:
                    last_error = usages[-1].error if usages else "provider error"
                if repair >= self.max_repairs:
                    break
                thinking = REPAIR_THINKING_LEVEL if self.escalate_repairs else self.thinking_level
                repair_request = LLMRequest(
                    prompt=repair_prompt(state, last_error or "invalid proposal", previous_text),
                    response_schema=proposal_schema(),
                    model=self.model,
                    thinking_level=thinking,
                    purpose="repair",
                )
                self.emit(
                    {
                        "type": "repair",
                        "attempt": repair + 1,
                        "thinking_level": thinking,
                        "error": last_error,
                    }
                )
                response = self._generate(repair_request, usages)
                previous_text = "" if response is None else response.text
            return None, usages, last_error
        finally:
            for usage in usages:
                self.ledger.add_usage(usage)

    def _generate(self, request: LLMRequest, usages: list[UsageRecord]) -> LLMResponse | None:
        try:
            response = self.provider.generate(request)
        except (LLMConfigError, LLMAuthError, LLMRateLimitError, LLMTransientError, LLMProtocolError) as exc:
            usage = UsageRecord(
                provider=getattr(self.provider, "name", "unknown"),
                model=request.model,
                thinking_level=request.thinking_level,
                purpose=request.purpose,
                status=_fatal_usage_status(exc),
                latency_seconds=0.0,
                error=redact_text(str(exc)),
            )
            usages.append(usage)
            raise
        except Exception as exc:
            usage = UsageRecord(
                provider=getattr(self.provider, "name", "unknown"),
                model=request.model,
                thinking_level=request.thinking_level,
                purpose=request.purpose,
                status="error",
                latency_seconds=0.0,
                error=redact_text(str(exc)),
            )
            usages.append(usage)
            return None
        retries = int(getattr(self.provider, "transport_retries", 0) or 0)
        if retries:
            self.ledger.transport_retries = retries
        usages.append(response.usage)
        return response

    def _parent_source(self, experiment_id: str) -> str:
        spec = self.runner.registry.get(experiment_id).spec
        return self.workspace.load_parent_source(spec, self.runner.repo_root)

    def _remaining_wall(self, started: float) -> float | None:
        if self.wall_clock_seconds is None:
            return None
        return max(0.0, float(self.wall_clock_seconds) - (time.perf_counter() - started))

    def _maybe_reject(self, proposal: ResearchProposal, result: Any) -> None:
        text = proposal.abandon_or_continue_reasoning.lower()
        if "abandon" in text and result.status != "success":
            note = f"{proposal.mutation_summary}: {proposal.hypothesis}"
            if note not in self.rejected_directions:
                self.rejected_directions.append(note)

    def _default_parent_id(self) -> str:
        elite = self.runner.registry.elite()
        if elite is not None:
            return elite.spec.experiment_id
        usable = find_usable_root(self.runner.registry, self.root_spec.experiment_id)
        if usable is not None:
            return usable.spec.experiment_id
        return self.root_spec.experiment_id

    def _resolve_parent_id(self, requested: str) -> str:
        entry = self.runner.registry.peek(requested)
        if entry is not None and is_usable_root_result(entry.result):
            return requested
        if entry is not None and entry.result is not None and entry.result.status == "success":
            return requested
        return self._default_parent_id()

    def _trace_record(
        self,
        outcome: IterationOutcome,
        usages: list[UsageRecord],
        iteration: int,
    ) -> dict[str, Any]:
        result = outcome.result
        metrics = None
        if result is not None and result.metrics is not None:
            metrics = {
                "GAUC": result.metrics.gauc,
                "nDCG@5": result.metrics.ndcg_at_5,
                "primary": result.metrics.primary,
            }
        parent_metrics = _parent_metrics(self.runner.registry, outcome.parent_id)
        fm_metrics = _usable_fm_metrics(self.runner.registry) or {
            "GAUC": FM_VALID_REFERENCE["GAUC"],
            "nDCG@5": FM_VALID_REFERENCE["nDCG@5"],
            "primary": FM_VALID_REFERENCE["primary"],
        }
        delta_parent = None
        delta_fm = None
        if metrics is not None and parent_metrics is not None:
            delta_parent = metrics["primary"] - parent_metrics["primary"]
        if metrics is not None:
            delta_fm = metrics["primary"] - float(fm_metrics["primary"])
        thinking = None
        model = self.model
        token_counts: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
        llm_latency = 0.0
        for usage in usages:
            thinking = usage.thinking_level
            model = usage.model
            llm_latency += usage.latency_seconds
            token_counts["input_tokens"] += usage.input_tokens or 0
            token_counts["output_tokens"] += usage.output_tokens or 0
            token_counts["thinking_tokens"] += usage.thinking_tokens or 0
            token_counts["cached_tokens"] += usage.cached_tokens or 0
            token_counts["total_tokens"] += usage.total_tokens or 0
        proposal = outcome.proposal
        return sanitize(
            {
                "session_id": self.session_id,
                "iteration": iteration,
                "experiment_id": outcome.experiment_id,
                "parent_id": outcome.parent_id,
                "reflection": None if proposal is None else proposal.reflection,
                "hypothesis": None if proposal is None else proposal.hypothesis,
                "rationale": None if proposal is None else proposal.rationale,
                "mutation_summary": None if proposal is None else proposal.mutation_summary,
                "source_diff": None if outcome.materialized is None else outcome.materialized.diff_vs_parent,
                "source_fingerprint": None
                if outcome.materialized is None
                else outcome.materialized.fingerprint,
                "metrics": metrics,
                "delta_vs_parent": delta_parent,
                "delta_vs_fm": delta_fm,
                "status": outcome.result_status,
                "error": outcome.error,
                "repair_calls": outcome.repair_calls,
                "model": model,
                "thinking_level": thinking,
                "token_counts": token_counts,
                "llm_calls": [item.to_dict() for item in usages],
                "llm_latency_seconds": llm_latency,
                "experiment_runtime_seconds": None if result is None else result.wall_seconds,
                "research_validity": _research_validity(outcome, iteration),
                "cumulative": self.ledger.to_dict(),
                "manual_interventions": self.ledger.manual_interventions,
            }
        )

    def _summary(self, root: IterationOutcome | None, outcomes: list[IterationOutcome]) -> dict[str, Any]:
        best = None
        elite = self.runner.registry.elite()
        if elite is not None and elite.result is not None and elite.result.metrics is not None:
            best = {
                "experiment_id": elite.spec.experiment_id,
                "primary": elite.result.metrics.primary,
                "GAUC": elite.result.metrics.gauc,
                "nDCG@5": elite.result.metrics.ndcg_at_5,
            }
        fm_metrics = _usable_fm_metrics(self.runner.registry)
        fm_primary = FM_VALID_REFERENCE["primary"] if fm_metrics is None else fm_metrics["primary"]
        improvement = None if best is None else best["primary"] - fm_primary
        return sanitize(
            {
                "session_id": self.session_id,
                "model": self.model,
                "thinking_level": self.thinking_level,
                "manual_interventions": self.ledger.manual_interventions,
                "research_wall_seconds": self.ledger.research_wall_seconds,
                "resources": self.ledger.to_dict(),
                "root_experiment_id": None if root is None else root.experiment_id,
                "best": best,
                "improvement_vs_fm": improvement,
                "iterations": [item.experiment_id for item in outcomes],
            }
        )

    def _emit_iteration(self, outcome: IterationOutcome) -> None:
        metrics = None
        if outcome.result is not None and outcome.result.metrics is not None:
            metrics = outcome.result.metrics
        last_usage = outcome.usages[-1] if outcome.usages else None
        self.emit(
            {
                "type": "result",
                "session_id": self.session_id,
                "iteration": outcome.iteration,
                "experiment_id": outcome.experiment_id,
                "parent_id": outcome.parent_id,
                "hypothesis": None if outcome.proposal is None else outcome.proposal.hypothesis,
                "status": outcome.result_status,
                "GAUC": None if metrics is None else metrics.gauc,
                "ndcg_at_5": None if metrics is None else metrics.ndcg_at_5,
                "primary": None if metrics is None else metrics.primary,
                "delta_vs_parent": outcome.record.get("delta_vs_parent"),
                "tokens": outcome.record.get("token_counts"),
                "thinking_level": None if last_usage is None else last_usage.thinking_level,
                "repair_calls": outcome.repair_calls,
                "error": outcome.error,
                "remaining_experiments": max(0, self.max_iterations - outcome.iteration),
                "cumulative": self.ledger.to_dict(),
            }
        )


def _try_parse_proposal(
    response: LLMResponse,
    dest: Path,
    workspace_root: Path,
    environment: Any | None = None,
) -> tuple[ResearchProposal | None, str | None]:
    if response.parsed is None:
        return None, response.usage.error or "structured output missing"
    try:
        proposal = ResearchProposal.from_dict(response.parsed)
    except (ProposalError, TypeError, ValueError) as exc:
        return None, redact_text(str(exc))
    try:
        validate_candidate_source(
            proposal.candidate_source,
            dest,
            workspace_root,
            environment=environment,
        )
    except SafetyError as exc:
        return None, format_preflight_repair_message(
            redact_text(str(exc)),
            hypothesis=proposal.hypothesis,
            environment=environment,
        )
    return proposal, None


def _research_validity(outcome: IterationOutcome, iteration: int) -> str:
    if iteration == 0:
        return "root"
    if outcome.result is not None:
        return "hypothesis_tested"
    return "not_executed"


def _usable_fm_metrics(registry: Any) -> dict[str, float] | None:
    usable = find_usable_root(registry)
    if usable is None or usable.result is None or usable.result.metrics is None:
        return None
    m = usable.result.metrics
    return {"GAUC": m.gauc, "nDCG@5": m.ndcg_at_5, "primary": m.primary}


def _parent_metrics(registry: Any, experiment_id: str | None) -> dict[str, float] | None:
    if not experiment_id:
        return None
    entry = registry.peek(experiment_id)
    if entry is None or entry.result is None or entry.result.metrics is None:
        return None
    m = entry.result.metrics
    return {"GAUC": m.gauc, "nDCG@5": m.ndcg_at_5, "primary": m.primary}


def _fatal_usage_status(exc: Exception) -> str:
    if isinstance(exc, LLMConfigError):
        return "config_error"
    if isinstance(exc, LLMAuthError):
        return "auth_error"
    if isinstance(exc, LLMRateLimitError):
        return "rate_limited"
    if isinstance(exc, LLMTransientError):
        return "transient"
    if isinstance(exc, LLMProtocolError):
        return "protocol_error"
    return "error"


def _result_error(result: Any) -> str | None:
    if result is None or result.failure is None:
        return None
    extra = ""
    stderr_path = result.stderr_path
    if stderr_path:
        path = Path(stderr_path)
        if path.is_file():
            extra = path.read_text(encoding="utf-8", errors="replace")[-2000:]
    message = result.failure.message
    if extra:
        message = f"{message}\n{extra}"
    return redact_text(message)
