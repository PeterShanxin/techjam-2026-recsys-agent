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
    },
    "required": list(REQUIRED_STRING_FIELDS) + ["experiment_parameters"],
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
        return cls(
            schema_version=str(data.get("schema_version", PROPOSAL_SCHEMA_VERSION)),
            experiment_parameters=dict(params),
            seed=seed,
            timeout_seconds=timeout,
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
