"""Validated structured research proposal. One Gemini call returns reflection + next experiment."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from research_agent.experiments.spec import EXPERIMENT_ID_RE
from research_agent.llm.secrets import sanitize

PROPOSAL_SCHEMA_VERSION = "1"

REQUIRED_STRING_FIELDS = (
    "reflection",
    "observation",
    "hypothesis",
    "rationale",
    "expected_mechanism",
    "selected_parent_id",
    "mutation_summary",
    "expected_effect",
    "candidate_source",
    "risk_notes",
    "abandon_or_continue_reasoning",
)

# JSON Schema for Gemini structured output. additionalProperties false keeps the trace compact.
PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reflection": {
            "type": "string",
            "description": "What the latest evidence says, including failures.",
        },
        "observation": {
            "type": "string",
            "description": "Concrete observation from elite/parent/recent runs.",
        },
        "hypothesis": {
            "type": "string",
            "description": "Testable ML hypothesis for the next experiment.",
        },
        "rationale": {
            "type": "string",
            "description": "Why this is the next experiment given remaining budget.",
        },
        "expected_mechanism": {
            "type": "string",
            "description": "How the change should move GAUC / nDCG@5.",
        },
        "selected_parent_id": {
            "type": "string",
            "description": "Existing experiment_id to mutate. Prefer elite over a failed child.",
        },
        "mutation_summary": {
            "type": "string",
            "description": "One meaningful research change versus the parent source.",
        },
        "expected_effect": {
            "type": "string",
            "description": "Expected metric movement, including risk of no gain.",
        },
        "candidate_source": {
            "type": "string",
            "description": "Complete replacement Python candidate file, including CLI.",
        },
        "experiment_parameters": {
            "type": "object",
            "description": "Free-form JSON written to the candidate --config file.",
        },
        "risk_notes": {"type": "string"},
        "abandon_or_continue_reasoning": {
            "type": "string",
            "description": "Whether to continue this direction or switch, and why.",
        },
        "seed": {"type": "integer"},
        "timeout_seconds": {"type": "number"},
        "operator": {
            "type": "string",
            "description": "mutation, crossover, or sequential. Controller still owns parent choice.",
        },
        "research_family": {
            "type": "string",
            "description": "Short structured family such as ensemble, ranking_loss, optimization.",
        },
        "mechanism_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Controlled tags for the claimed mechanism. Not free-text similarity.",
        },
        "changed_axes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Research axes changed versus the parent(s).",
        },
        "required_data_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Data fields the mechanism needs. Must exist on data.load tuples or be read from raw CSVs.",
        },
        "what_changed": {"type": "string"},
        "why": {"type": "string"},
        "evidence_motivated": {"type": "string"},
        "would_support": {"type": "string"},
        "would_refute": {"type": "string"},
        "selected_parent_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parent ids for crossover. Mutation uses selected_parent_id.",
        },
        "crossover_compatible": {"type": "boolean"},
        "parent_a_component": {"type": "string"},
        "parent_b_component": {"type": "string"},
        "crossover_conflicts": {"type": "string"},
        "crossover_inappropriate_reason": {"type": "string"},
    },
    "required": list(REQUIRED_STRING_FIELDS)
    + ["experiment_parameters", "research_family", "mechanism_tags", "changed_axes"],
}


class ProposalError(ValueError):
    """Structured proposal failed validation."""


@dataclass(frozen=True)
class ResearchProposal:
    reflection: str
    observation: str
    hypothesis: str
    rationale: str
    expected_mechanism: str
    selected_parent_id: str
    mutation_summary: str
    expected_effect: str
    candidate_source: str
    experiment_parameters: dict[str, Any] = field(default_factory=dict)
    risk_notes: str = ""
    abandon_or_continue_reasoning: str = ""
    seed: int = 0
    timeout_seconds: float = 600.0
    schema_version: str = PROPOSAL_SCHEMA_VERSION
    operator: str = "mutation"
    research_family: str = ""
    mechanism_tags: tuple[str, ...] = ()
    changed_axes: tuple[str, ...] = ()
    required_data_fields: tuple[str, ...] = ()
    what_changed: str = ""
    why: str = ""
    evidence_motivated: str = ""
    would_support: str = ""
    would_refute: str = ""
    selected_parent_ids: tuple[str, ...] = ()
    crossover_compatible: bool | None = None
    parent_a_component: str = ""
    parent_b_component: str = ""
    crossover_conflicts: str = ""
    crossover_inappropriate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return sanitize(
            {
                "schema_version": self.schema_version,
                "reflection": self.reflection,
                "observation": self.observation,
                "hypothesis": self.hypothesis,
                "rationale": self.rationale,
                "expected_mechanism": self.expected_mechanism,
                "selected_parent_id": self.selected_parent_id,
                "mutation_summary": self.mutation_summary,
                "expected_effect": self.expected_effect,
                "candidate_source": self.candidate_source,
                "experiment_parameters": dict(self.experiment_parameters),
                "risk_notes": self.risk_notes,
                "abandon_or_continue_reasoning": self.abandon_or_continue_reasoning,
                "seed": self.seed,
                "timeout_seconds": self.timeout_seconds,
                "operator": self.operator,
                "research_family": self.research_family,
                "mechanism_tags": list(self.mechanism_tags),
                "changed_axes": list(self.changed_axes),
                "required_data_fields": list(self.required_data_fields),
                "what_changed": self.what_changed,
                "why": self.why,
                "evidence_motivated": self.evidence_motivated,
                "would_support": self.would_support,
                "would_refute": self.would_refute,
                "selected_parent_ids": list(self.selected_parent_ids),
                "crossover_compatible": self.crossover_compatible,
                "parent_a_component": self.parent_a_component,
                "parent_b_component": self.parent_b_component,
                "crossover_conflicts": self.crossover_conflicts,
                "crossover_inappropriate_reason": self.crossover_inappropriate_reason,
            }
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True) + (
            "\n" if indent is not None else ""
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ResearchProposal":
        if not isinstance(data, Mapping):
            raise ProposalError("proposal must be a JSON object")
        missing = [name for name in REQUIRED_STRING_FIELDS if name not in data]
        if "experiment_parameters" not in data:
            missing.append("experiment_parameters")
        if missing:
            raise ProposalError(f"proposal missing fields: {missing}")
        strings = {}
        for name in REQUIRED_STRING_FIELDS:
            value = data[name]
            if not isinstance(value, str) or not value.strip():
                raise ProposalError(f"{name} must be a non-empty string")
            strings[name] = value.strip()
        params = data.get("experiment_parameters") or {}
        if not isinstance(params, dict):
            raise ProposalError("experiment_parameters must be a JSON object")
        if any(key in {"evaluation_split", "allow_test_split", "split"} for key in params):
            raise ProposalError("experiment_parameters must not set evaluation split")
        parent_id = strings["selected_parent_id"]
        if not EXPERIMENT_ID_RE.fullmatch(parent_id):
            raise ProposalError(f"invalid selected_parent_id {parent_id!r}")
        seed = _optional_int(data.get("seed", 0), "seed", default=0)
        timeout = _optional_float(data.get("timeout_seconds", 600.0), "timeout_seconds", default=600.0)
        if timeout <= 0:
            raise ProposalError("timeout_seconds must be positive")
        operator = str(data.get("operator") or "mutation").strip().lower()
        if operator not in {"mutation", "crossover", "sequential"}:
            raise ProposalError("operator must be mutation, crossover, or sequential")
        parent_ids = _optional_str_tuple(data.get("selected_parent_ids"), "selected_parent_ids")
        for extra_parent in parent_ids:
            if not EXPERIMENT_ID_RE.fullmatch(extra_parent):
                raise ProposalError(f"invalid selected_parent_ids entry {extra_parent!r}")
        # Diversity suppression and crossover parent choice both key off the semantic
        # signature. During the P0 sprint seven of eight proposals left these blank, the
        # signature collapsed to ("other", (), ()), and diversity.duplicate_reason
        # short-circuits on that. Requiring them re-arms both mechanisms.
        family = str(data.get("research_family") or "").strip()
        if not family or family.lower() == "other":
            raise ProposalError(
                "research_family must be a specific non-empty family, not 'other'"
            )
        mechanism_tags = _optional_str_tuple(data.get("mechanism_tags"), "mechanism_tags")
        if not mechanism_tags:
            raise ProposalError("mechanism_tags must list at least one mechanism tag")
        changed_axes = _optional_str_tuple(data.get("changed_axes"), "changed_axes")
        if not changed_axes:
            raise ProposalError("changed_axes must list at least one changed research axis")
        return cls(
            schema_version=str(data.get("schema_version", PROPOSAL_SCHEMA_VERSION)),
            experiment_parameters=dict(params),
            seed=seed,
            timeout_seconds=timeout,
            operator=operator,
            research_family=family,
            mechanism_tags=mechanism_tags,
            changed_axes=changed_axes,
            required_data_fields=_optional_str_tuple(
                data.get("required_data_fields"), "required_data_fields"
            ),
            what_changed=str(data.get("what_changed") or "").strip(),
            why=str(data.get("why") or "").strip(),
            evidence_motivated=str(data.get("evidence_motivated") or "").strip(),
            would_support=str(data.get("would_support") or "").strip(),
            would_refute=str(data.get("would_refute") or "").strip(),
            selected_parent_ids=parent_ids,
            crossover_compatible=_optional_bool(data.get("crossover_compatible")),
            parent_a_component=str(data.get("parent_a_component") or "").strip(),
            parent_b_component=str(data.get("parent_b_component") or "").strip(),
            crossover_conflicts=str(data.get("crossover_conflicts") or "").strip(),
            crossover_inappropriate_reason=str(
                data.get("crossover_inappropriate_reason") or ""
            ).strip(),
            **strings,
        )


def _optional_int(value: Any, name: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProposalError(f"{name} must be an int")
    return value


def _optional_float(value: Any, name: str, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProposalError(f"{name} must be a number") from exc


def _optional_str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        raise ProposalError(f"{name} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ProposalError(f"{name} must be a list of non-empty strings")
        out.append(item.strip())
    return tuple(out)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ProposalError("crossover_compatible must be a boolean")
