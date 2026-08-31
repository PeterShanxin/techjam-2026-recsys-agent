"""Audit-driven search-space changes: required diversity metadata, axis exposure, parent spread.

All zero-API. These lock in the fixes for the three blind spots the second-opinion audit
found in the P0 sprint: blank semantic signatures disabling diversity, no visibility into
which research axes were already exhausted, and elite-only parent selection.
"""
from __future__ import annotations

import json

import pytest

from research_agent.agent.constants import (
    AUDIT_FINDINGS,
    HEAVILY_SEARCHED_AXES,
    UNDEREXPLORED_AXES,
    VALIDATION_NOISE,
)
from research_agent.agent.proposal import ProposalError, ResearchProposal
from research_agent.evolution.controller import _mutation_parent
from research_agent.evolution.diversity import duplicate_reason
from research_agent.evolution.prompts import CROSSOVER_SYSTEM, MUTATION_SYSTEM
from tests.evolution_helpers import make_member
from tests.research_helpers import make_proposal_payload


# --- required diversity metadata -------------------------------------------------


def test_proposal_requires_specific_research_family() -> None:
    for bad in ("", "   ", "other", "OTHER"):
        payload = make_proposal_payload(research_family=bad)
        with pytest.raises(ProposalError, match="research_family"):
            ResearchProposal.from_dict(payload)


def test_proposal_requires_mechanism_tags_and_changed_axes() -> None:
    with pytest.raises(ProposalError, match="mechanism_tags"):
        ResearchProposal.from_dict(make_proposal_payload(mechanism_tags=[]))
    with pytest.raises(ProposalError, match="changed_axes"):
        ResearchProposal.from_dict(make_proposal_payload(changed_axes=[]))


def test_proposal_missing_metadata_keys_is_rejected() -> None:
    payload = make_proposal_payload()
    payload.pop("research_family")
    with pytest.raises(ProposalError):
        ResearchProposal.from_dict(payload)


def test_valid_metadata_still_parses() -> None:
    proposal = ResearchProposal.from_dict(
        make_proposal_payload(
            research_family="ranking_objective",
            mechanism_tags=["listwise_softmax", "within_user_groups"],
            changed_axes=["objective"],
        )
    )
    assert proposal.research_family == "ranking_objective"
    assert proposal.mechanism_tags == ("listwise_softmax", "within_user_groups")


def test_required_metadata_restores_duplicate_detection() -> None:
    """With blank signatures duplicate_reason short-circuits; with real ones it fires."""
    blank = make_member(
        experiment_id="a", research_family="other", mechanism_tags=(), changed_axes=()
    )
    blank_twin = make_member(
        experiment_id="b",
        research_family="other",
        mechanism_tags=(),
        changed_axes=(),
        source_fingerprint="fp-b",
        spec_hash="hash-b",
    )
    assert duplicate_reason(blank_twin, [blank]) is None

    real = make_member(
        experiment_id="c", research_family="ranking_objective", mechanism_tags=("listwise",)
    )
    real_twin = make_member(
        experiment_id="d",
        research_family="ranking_objective",
        mechanism_tags=("listwise",),
        source_fingerprint="fp-d",
        spec_hash="hash-d",
    )
    assert duplicate_reason(real_twin, [real]) == "semantic_signature"


# --- axis exposure ---------------------------------------------------------------


def test_axis_lists_are_disjoint_and_non_empty() -> None:
    assert HEAVILY_SEARCHED_AXES and UNDEREXPLORED_AXES
    assert not set(HEAVILY_SEARCHED_AXES) & set(UNDEREXPLORED_AXES)


def test_underexplored_axes_name_axes_not_settings() -> None:
    """The agent must reason out values itself. No answer may leak through this channel."""
    blob = " ".join(UNDEREXPLORED_AXES + AUDIT_FINDINGS + HEAVILY_SEARCHED_AXES).lower()
    for leaked in ("1e-5", "1e-05", "0.00001", "l2=", "alpha=0.3", "blend weight of"):
        assert leaked not in blob, f"scratch finding {leaked!r} leaked into agent state"


def test_validation_noise_sets_a_usable_floor() -> None:
    assert VALIDATION_NOISE["paired_delta_bootstrap_sd_typical"] == pytest.approx(0.0005)
    assert VALIDATION_NOISE["absolute_primary_bootstrap_sd"] > 0.002
    assert VALIDATION_NOISE["official_convergence_epsilon"] == 0.002


def test_research_state_surfaces_axes_and_noise(monkeypatch) -> None:
    from research_agent.agent import state as state_mod

    captured = {}

    class _Registry:
        def elite(self):
            return None

        def peek(self, _id):
            return None

        def rank_validation(self):
            return []

        def iter_ids(self):
            return []

    class _Ledger:
        def to_dict(self):
            return {}

    built = state_mod.build_research_state(
        registry=_Registry(),
        ledger=_Ledger(),
        iteration=1,
        max_iterations=8,
        remaining_wall_seconds=100.0,
        parent_source="# parent",
    )
    payload = json.loads(built.to_json())
    captured.update(payload)
    for key in (
        "heavily_searched_axes",
        "underexplored_axes",
        "validation_noise",
        "audit_findings",
    ):
        assert key in captured and captured[key]
    assert "near_identity_noop" in captured["guidance"]


def test_operator_prompts_forbid_the_alpha_zero_shape() -> None:
    for prompt in (MUTATION_SYSTEM, CROSSOVER_SYSTEM):
        assert "near_identity_noop" in prompt
    assert "alpha=0" in MUTATION_SYSTEM
    assert "GradientFM" in MUTATION_SYSTEM
    # no prescribed values may reach the operator
    assert "1e-5" not in MUTATION_SYSTEM and "1e-5" not in CROSSOVER_SYSTEM


# --- parent selection ------------------------------------------------------------


def _member(eid: str, family: str, primary: float):
    return make_member(
        experiment_id=eid,
        research_family=family,
        mechanism_tags=(family,),
        source_fingerprint=f"fp-{eid}",
        spec_hash=f"hash-{eid}",
        metrics={"GAUC": primary, "nDCG@5": primary, "primary": primary},
    )


def test_slot_zero_always_takes_the_current_elite() -> None:
    members = [
        _member("a", "fm_ensemble", 0.61),
        _member("b", "fm_ensemble", 0.60),
        _member("c", "ranking_objective", 0.55),
    ]
    assert _mutation_parent(members, 0, 2).experiment_id == "a"


def test_second_slot_leaves_the_elite_family_when_both_elites_match() -> None:
    """This is the FM-gravity fix: two same-family elites must not seed both offspring."""
    members = [
        _member("a", "fm_ensemble", 0.61),
        _member("b", "fm_ensemble", 0.60),
        _member("c", "ranking_objective", 0.55),
    ]
    assert _mutation_parent(members, 1, 2).experiment_id == "c"


def test_second_slot_keeps_the_runner_up_when_families_already_differ() -> None:
    members = [
        _member("a", "fm_ensemble", 0.61),
        _member("b", "ranking_objective", 0.60),
    ]
    assert _mutation_parent(members, 1, 2).experiment_id == "b"


def test_parent_selection_falls_back_when_every_member_shares_a_family() -> None:
    members = [_member("a", "fm_ensemble", 0.61), _member("b", "fm_ensemble", 0.60)]
    assert _mutation_parent(members, 1, 2).experiment_id == "b"


def test_parent_selection_is_deterministic() -> None:
    members = [
        _member("a", "fm_ensemble", 0.61),
        _member("b", "fm_ensemble", 0.60),
        _member("c", "ranking_objective", 0.55),
    ]
    picks = [_mutation_parent(members, slot, 2).experiment_id for slot in range(6)]
    assert picks == ["a", "c", "a", "c", "a", "c"]
