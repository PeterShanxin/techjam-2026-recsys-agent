"""Compact ResearchState. Evidence for one Gemini research call. Not a chat dump."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_agent.experiments import ExperimentRegistry, ExperimentResult
from research_agent.experiments.splits import RESEARCH_SPLIT
from research_agent.llm.secrets import sanitize

from .accounting import ResourceLedger
from .constants import (
    BENCHMARK_INVARIANTS,
    FM_ROOT_ID,
    FM_VALID_REFERENCE,
    ORGANIZER_DEAD_ENDS,
    ORGANIZER_PROMISING_CATEGORIES,
)


@dataclass(frozen=True)
class ResearchState:
    iteration: int
    remaining_iterations: int
    remaining_wall_seconds: float | None
    current_elite: dict[str, Any] | None
    selected_parent: dict[str, Any] | None
    parent_source: str
    extra_source_snippets: list[dict[str, str]]
    top_successful: list[dict[str, Any]]
    recent_experiments: list[dict[str, Any]]
    recent_failures: list[dict[str, Any]]
    rejected_directions: list[str]
    lineage: list[dict[str, Any]]
    remaining_experiment_budget: int
    llm_usage: dict[str, Any]
    invariants: dict[str, Any] = field(default_factory=lambda: dict(BENCHMARK_INVARIANTS))
    official_fm_validation: dict[str, Any] = field(default_factory=lambda: dict(FM_VALID_REFERENCE))
    organizer_dead_ends: list[str] = field(default_factory=lambda: list(ORGANIZER_DEAD_ENDS))
    promising_categories: list[str] = field(
        default_factory=lambda: list(ORGANIZER_PROMISING_CATEGORIES)
    )

    def to_dict(self) -> dict[str, Any]:
        return sanitize(
            {
                "iteration": self.iteration,
                "remaining_iterations": self.remaining_iterations,
                "remaining_wall_seconds": self.remaining_wall_seconds,
                "remaining_experiment_budget": self.remaining_experiment_budget,
                "invariants": dict(self.invariants),
                "official_fm_validation": dict(self.official_fm_validation),
                "current_elite": self.current_elite,
                "selected_parent": self.selected_parent,
                "parent_source": self.parent_source,
                "extra_source_snippets": list(self.extra_source_snippets),
                "top_successful": list(self.top_successful),
                "recent_experiments": list(self.recent_experiments),
                "recent_failures": list(self.recent_failures),
                "rejected_directions": list(self.rejected_directions),
                "lineage": list(self.lineage),
                "llm_usage": dict(self.llm_usage),
                "organizer_dead_ends": list(self.organizer_dead_ends),
                "promising_categories": list(self.promising_categories),
                "guidance": (
                    "These organizer notes are research context, not a script. "
                    "Decide the next experiment from evidence. "
                    "Return one complete candidate Python file. Do not modify evaluate.py. "
                    "Write ordered scores for the requested split only."
                ),
            }
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True)


def build_research_state(
    *,
    registry: ExperimentRegistry,
    ledger: ResourceLedger,
    iteration: int,
    max_iterations: int,
    remaining_wall_seconds: float | None,
    parent_source: str,
    selected_parent_id: str | None = None,
    extra_source_snippets: list[dict[str, str]] | None = None,
    rejected_directions: list[str] | None = None,
    top_k: int = 5,
    recent_k: int = 5,
) -> ResearchState:
    elite_entry = registry.elite()
    elite = _summarize_entry(elite_entry) if elite_entry is not None else None
    parent_id = selected_parent_id
    if parent_id is None:
        parent_id = elite["experiment_id"] if elite else FM_ROOT_ID
    parent = None
    if registry.peek(parent_id) is not None:
        parent = _summarize_entry(registry.get(parent_id), vs_elite=elite, vs_fm=_fm_metrics(registry))
    ranked = [_summarize_entry(item, vs_fm=_fm_metrics(registry)) for item in registry.rank_validation()[:top_k]]
    recent = []
    failures = []
    lineage = []
    for experiment_id in list(registry.iter_ids()):
        entry = registry.get(experiment_id)
        if entry.spec.evaluation_split != RESEARCH_SPLIT:
            continue
        summary = _summarize_entry(entry, vs_fm=_fm_metrics(registry))
        recent.append(summary)
        if entry.result is not None and entry.result.status != "success":
            failures.append(summary)
        parents = registry.parents(experiment_id)
        lineage.append(
            {
                "experiment_id": experiment_id,
                "parent_ids": parents,
                "status": None if entry.result is None else entry.result.status,
                "primary": _primary(entry.result),
            }
        )
    return ResearchState(
        iteration=iteration,
        remaining_iterations=max(0, max_iterations - iteration + 1),
        remaining_wall_seconds=remaining_wall_seconds,
        remaining_experiment_budget=max(0, max_iterations - iteration + 1),
        current_elite=elite,
        selected_parent=parent,
        parent_source=parent_source,
        extra_source_snippets=list(extra_source_snippets or []),
        top_successful=ranked,
        recent_experiments=recent[-recent_k:],
        recent_failures=failures[-recent_k:],
        rejected_directions=list(rejected_directions or []),
        lineage=lineage[-20:],
        llm_usage=ledger.to_dict(),
    )


def _fm_metrics(registry: ExperimentRegistry) -> dict[str, float] | None:
    entry = registry.peek(FM_ROOT_ID)
    if entry is None or entry.result is None or entry.result.metrics is None:
        return dict(FM_VALID_REFERENCE)
    m = entry.result.metrics
    return {"GAUC": m.gauc, "nDCG@5": m.ndcg_at_5, "primary": m.primary}


def _primary(result: ExperimentResult | None) -> float | None:
    if result is None or result.metrics is None:
        return None
    return float(result.metrics.primary)


def _summarize_entry(entry: Any, *, vs_elite: dict[str, Any] | None = None, vs_fm: dict[str, Any] | None = None) -> dict[str, Any]:
    result = entry.result
    metrics = None if result is None or result.metrics is None else {
        "GAUC": result.metrics.gauc,
        "nDCG@5": result.metrics.ndcg_at_5,
        "primary": result.metrics.primary,
    }
    failure = None
    if result is not None and result.failure is not None:
        failure = {
            "kind": result.failure.kind,
            "message": _clip(result.failure.message, 800),
        }
    summary = {
        "experiment_id": entry.spec.experiment_id,
        "origin": entry.spec.origin,
        "parent_ids": list(entry.spec.parent_ids),
        "hypothesis": entry.spec.hypothesis,
        "rationale": entry.spec.rationale,
        "status": None if result is None else result.status,
        "metrics": metrics,
        "wall_seconds": None if result is None else result.wall_seconds,
        "failure": failure,
        "decision": entry.decision,
        "evaluation_split": entry.spec.evaluation_split,
    }
    if metrics and vs_fm and "primary" in vs_fm:
        summary["delta_vs_fm_primary"] = metrics["primary"] - float(vs_fm["primary"])
    if metrics and vs_elite and vs_elite.get("metrics"):
        summary["delta_vs_elite_primary"] = metrics["primary"] - float(vs_elite["metrics"]["primary"])
    return summary


def load_source_for_id(registry: ExperimentRegistry, experiment_id: str, repo_root: Path) -> str:
    entry = registry.get(experiment_id)
    from .workspace import CandidateWorkspace

    # Reuse resolver without requiring a workspace instance root.
    dummy = CandidateWorkspace(repo_root / "runs" / "generated")
    return dummy.load_parent_source(entry.spec, repo_root)


def _clip(text: str, n: int) -> str:
    text = text.replace("\x00", "")
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."
