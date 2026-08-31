"""Builders for Phase 4 evolution unit tests. Zero API spend."""
from __future__ import annotations

from research_agent.evolution.types import PopulationMember
from experiment_helpers import CANDIDATE_SOURCE
from research_helpers import make_proposal_payload


def unique_source(label: str) -> str:
    return CANDIDATE_SOURCE + f"\n# research-variant {label}\n"


def make_member(**overrides) -> PopulationMember:
    payload = {
        "experiment_id": "exp-a",
        "parent_ids": ("fm-root",),
        "generation": 1,
        "origin": "mutation",
        "hypothesis": "A ranking-loss mutation.",
        "rationale": "Evidence says pointwise logloss is misaligned.",
        "research_family": "ranking_loss",
        "mechanism_tags": ("bpr",),
        "changed_axes": ("objective",),
        "source_fingerprint": "fp-a",
        "spec_hash": "hash-a",
        "metrics": {"GAUC": 0.60, "nDCG@5": 0.50, "primary": 0.55},
        "research_validity": "hypothesis_tested",
        "runtime_seconds": 10.0,
        "resource_usage": {},
        "status": "success",
        "evaluation_split": "valid",
        "selection": "pending",
        "scientific_evidence": True,
    }
    payload.update(overrides)
    if "fitness" not in payload:
        payload["fitness"] = None
    return PopulationMember.from_dict(payload)


def evolution_proposal(
    *,
    label: str,
    hypothesis: str = "mutation",
    family: str = "ranking_loss",
    tags: tuple[str, ...] = ("bpr",),
    axes: tuple[str, ...] = ("objective",),
    action: str = "succeed",
    **overrides,
) -> dict:
    payload = make_proposal_payload(
        hypothesis=hypothesis,
        mutation_summary=f"semantic change {label}",
        candidate_source=unique_source(label),
        experiment_parameters={"action": action, "variant": label},
        research_family=family,
        mechanism_tags=list(tags),
        changed_axes=list(axes),
        operator="mutation",
        what_changed=f"Changed {label}",
        why="Evidence from the parent.",
        evidence_motivated="Parent metrics and failures.",
        would_support="Higher validation primary.",
        would_refute="Flat or worse primary with the mechanism firing.",
        required_data_fields=["long_view", "user_id"],
    )
    payload.update(overrides)
    return payload
