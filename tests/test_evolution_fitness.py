"""Deterministic fitness and elite rules. FakeProvider only. Zero API spend."""
from __future__ import annotations

import pytest

from evolution_helpers import make_member
from research_agent.evolution.fitness import (
    compute_fitness,
    is_elite_eligible,
    rank_members,
    select_elites,
)


def test_default_fitness_is_validation_primary():
    member = make_member(metrics={"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015})
    assert compute_fitness(member) == pytest.approx(0.6015)
    assert compute_fitness(member, efficiency_penalty=0.0) == pytest.approx(0.6015)


def test_failed_timeout_invalid_not_executed_have_no_fitness():
    for status, validity in (
        ("failed", "implementation_failure"),
        ("timeout", "implementation_failure"),
        ("invalid", "implementation_failure"),
        ("success", "not_executed"),
    ):
        member = make_member(
            status=status,
            research_validity=validity,
            scientific_evidence=False,
            metrics={"GAUC": 0.9, "nDCG@5": 0.9, "primary": 0.9} if status == "success" else None,
        )
        assert compute_fitness(member) is None
        assert is_elite_eligible(member) is False


def test_research_invalid_and_semantic_noop_cannot_be_elite():
    noop = make_member(
        research_validity="semantic_noop",
        metrics={"GAUC": 0.9, "nDCG@5": 0.9, "primary": 0.9},
        scientific_evidence=False,
    )
    invalid = make_member(
        research_validity="research_invalid",
        metrics={"GAUC": 0.9, "nDCG@5": 0.9, "primary": 0.9},
        scientific_evidence=False,
    )
    assert is_elite_eligible(noop) is False
    assert is_elite_eligible(invalid) is False
    assert compute_fitness(noop) is None
    assert compute_fitness(invalid) is None


def test_test_split_cannot_enter_fitness():
    member = make_member(
        evaluation_split="test",
        metrics={"GAUC": 0.99, "nDCG@5": 0.99, "primary": 0.99},
    )
    assert compute_fitness(member) is None
    assert is_elite_eligible(member) is False


def test_elite_selection_is_deterministic_and_preserves_top_k():
    members = [
        make_member(experiment_id="c", metrics={"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.50}, runtime_seconds=5),
        make_member(experiment_id="a", metrics={"GAUC": 0.6, "nDCG@5": 0.6, "primary": 0.60}, runtime_seconds=9),
        make_member(experiment_id="b", metrics={"GAUC": 0.6, "nDCG@5": 0.6, "primary": 0.60}, runtime_seconds=3),
        make_member(
            experiment_id="fail",
            status="failed",
            research_validity="implementation_failure",
            scientific_evidence=False,
            metrics=None,
            runtime_seconds=1,
        ),
    ]
    ranked = rank_members(members)
    assert [item.experiment_id for item in ranked] == ["b", "a", "c"]
    elites = select_elites(members, elite_count=2)
    assert [item.experiment_id for item in elites] == ["b", "a"]


def test_implementation_failure_is_not_scientific_evidence():
    member = make_member(
        status="failed",
        research_validity="implementation_failure",
        scientific_evidence=False,
        hypothesis="BPR pairwise loss",
        metrics=None,
    )
    assert member.scientific_evidence is False
    assert is_elite_eligible(member) is False
