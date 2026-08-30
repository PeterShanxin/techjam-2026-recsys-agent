"""Evolution population types. Registry remains lineage source of truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _tuple_str(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class PopulationMember:
    experiment_id: str
    parent_ids: tuple[str, ...]
    generation: int
    origin: str
    hypothesis: str
    rationale: str
    research_family: str
    mechanism_tags: tuple[str, ...]
    changed_axes: tuple[str, ...]
    source_fingerprint: str
    spec_hash: str
    metrics: dict[str, float] | None
    research_validity: str
    runtime_seconds: float | None
    resource_usage: dict[str, Any]
    status: str
    evaluation_split: str
    selection: str
    scientific_evidence: bool
    fitness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "parent_ids": list(self.parent_ids),
            "generation": self.generation,
            "origin": self.origin,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "research_family": self.research_family,
            "mechanism_tags": list(self.mechanism_tags),
            "changed_axes": list(self.changed_axes),
            "source_fingerprint": self.source_fingerprint,
            "spec_hash": self.spec_hash,
            "metrics": None if self.metrics is None else dict(self.metrics),
            "research_validity": self.research_validity,
            "runtime_seconds": self.runtime_seconds,
            "resource_usage": dict(self.resource_usage),
            "status": self.status,
            "evaluation_split": self.evaluation_split,
            "selection": self.selection,
            "scientific_evidence": self.scientific_evidence,
            "fitness": self.fitness,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PopulationMember":
        metrics = data.get("metrics")
        return cls(
            experiment_id=str(data["experiment_id"]),
            parent_ids=_tuple_str(data.get("parent_ids")),
            generation=int(data.get("generation", 0)),
            origin=str(data.get("origin", "mutation")),
            hypothesis=str(data.get("hypothesis", "")),
            rationale=str(data.get("rationale", "")),
            research_family=str(data.get("research_family", "")),
            mechanism_tags=_tuple_str(data.get("mechanism_tags")),
            changed_axes=_tuple_str(data.get("changed_axes")),
            source_fingerprint=str(data.get("source_fingerprint") or ""),
            spec_hash=str(data.get("spec_hash") or ""),
            metrics=None if metrics is None else dict(metrics),
            research_validity=str(data.get("research_validity", "not_executed")),
            runtime_seconds=data.get("runtime_seconds"),
            resource_usage=dict(data.get("resource_usage") or {}),
            status=str(data.get("status", "invalid")),
            evaluation_split=str(data.get("evaluation_split", "valid")),
            selection=str(data.get("selection", "pending")),
            scientific_evidence=bool(data.get("scientific_evidence", False)),
            fitness=data.get("fitness"),
        )

    def with_updates(self, **overrides: Any) -> "PopulationMember":
        payload = self.to_dict()
        payload.update(overrides)
        return PopulationMember.from_dict(payload)


@dataclass
class Population:
    members: list[PopulationMember] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"members": [item.to_dict() for item in self.members]}


@dataclass(frozen=True)
class SelectionDecision:
    generation: int
    operator: str
    parent_ids: tuple[str, ...]
    reason: str
    experiment_id: str | None = None
    fallback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "operator": self.operator,
            "parent_ids": list(self.parent_ids),
            "reason": self.reason,
            "experiment_id": self.experiment_id,
            "fallback": self.fallback,
        }


@dataclass
class GenerationRecord:
    generation: int
    member_ids: list[str]
    elite_ids: list[str]
    best_fitness: float | None
    improvement: float | None
    stagnation: int
    decisions: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "member_ids": list(self.member_ids),
            "elite_ids": list(self.elite_ids),
            "best_fitness": self.best_fitness,
            "improvement": self.improvement,
            "stagnation": self.stagnation,
            "decisions": list(self.decisions),
            "stop_reason": self.stop_reason,
        }


@dataclass
class EvolutionRun:
    population: Population
    all_members: list[PopulationMember]
    elites: list[PopulationMember]
    generations: list[GenerationRecord]
    diversity_events: list[dict[str, Any]]
    operator_decisions: list[dict[str, Any]]
    negative_scientific_hypotheses: list[str]
    stop_reason: str
    evaluated_offspring: int
    stagnation: int
    trace_dir: Path
    summary: dict[str, Any]
    session_id: str
