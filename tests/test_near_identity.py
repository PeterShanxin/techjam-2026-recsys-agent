"""Near-identity rejection: a re-weighted parent is not a scientific experiment."""
from __future__ import annotations

import numpy as np
import pytest

from research_agent.evolution.identity import (
    NEAR_IDENTITY_VALIDITY,
    assess_identity,
    first_near_identity,
    load_scores,
    rank_change_fraction,
)

USERS = ["u1", "u1", "u1", "u2", "u2", "u3"]
PARENT = np.array([0.9, 0.5, 0.1, 0.7, 0.2, 0.4])


def test_identical_scores_change_no_ordering() -> None:
    assert rank_change_fraction(PARENT, PARENT, USERS) == 0.0


def test_monotone_per_user_transform_changes_no_ordering() -> None:
    """The official metric is invariant to these, so the controller must be too."""
    for transformed in (
        PARENT * 1000.0 + 7.0,
        1.0 / (1.0 + np.exp(-PARENT)),
        np.array([9.0, 5.0, 1.0, 7.0, 2.0, 4.0]),
    ):
        assert rank_change_fraction(transformed, PARENT, USERS) == 0.0


def test_alpha_near_zero_residual_is_near_identity() -> None:
    """A residual too small to reorder anything is a reparameterised parent."""
    child = PARENT + 1e-9 * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    report = assess_identity(child, PARENT, USERS, parent_id="p", primary_delta=8.8e-6)
    assert report.near_identity is True
    assert report.rank_change_fraction == 0.0


def test_real_reordering_is_not_near_identity() -> None:
    child = PARENT.copy()
    child[0], child[2] = child[2], child[0]  # flips u1's top and bottom
    report = assess_identity(child, PARENT, USERS, parent_id="p", primary_delta=0.0001)
    assert report.near_identity is False
    assert report.rank_change_fraction == pytest.approx(2 / 6)


def test_material_metric_move_survives_even_with_tiny_reordering() -> None:
    """A small but genuinely large-effect change must not be discarded."""
    child = PARENT.copy()
    report = assess_identity(child, PARENT, USERS, parent_id="p", primary_delta=0.004)
    assert report.near_identity is False


def test_cross_user_reordering_alone_is_still_identity() -> None:
    """Shifting one user's whole list cannot change within-user order, so it is a no-op."""
    child = PARENT.copy()
    child[3:5] += 100.0
    assert rank_change_fraction(child, PARENT, USERS) == 0.0


def test_ties_resolve_by_row_order_like_the_official_evaluator() -> None:
    parent = np.array([0.5, 0.5, 0.5])
    child = np.array([0.5, 0.5, 0.5])
    assert rank_change_fraction(child, parent, ["u", "u", "u"]) == 0.0


def test_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError):
        rank_change_fraction(np.zeros(3), np.zeros(4), ["a", "b", "c"])
    with pytest.raises(ValueError):
        rank_change_fraction(np.zeros(3), np.zeros(3), ["a", "b"])


def test_first_near_identity_scans_all_parents() -> None:
    other = np.array([0.1, 0.9, 0.5, 0.2, 0.7, 0.4])
    child = PARENT + 1e-9
    hit = first_near_identity(
        child, [("other", other), ("elite", PARENT)], USERS, min_rows=1
    )
    assert hit is not None and hit.parent_id == "elite"
    miss = first_near_identity(child, [("other", other)], USERS, min_rows=1)
    assert miss is None


def test_first_near_identity_skips_unusable_parents() -> None:
    assert (
        first_near_identity(PARENT, [("a", None), ("b", np.zeros(2))], USERS, min_rows=1)
        is None
    )


def test_gate_is_disabled_on_splits_too_small_to_be_informative() -> None:
    """On a handful of impressions two unrelated vectors agree by chance. Do not judge."""
    assert first_near_identity(PARENT, [("elite", PARENT)], USERS) is None
    assert (
        first_near_identity(PARENT, [("elite", PARENT)], USERS, min_rows=len(USERS))
        is not None
    )


def test_default_min_rows_is_far_below_the_real_split() -> None:
    from research_agent.evolution.identity import DEFAULT_MIN_ROWS

    assert 100 <= DEFAULT_MIN_ROWS < 124_909


def test_load_scores_is_tolerant(tmp_path) -> None:
    assert load_scores(None) is None
    assert load_scores(str(tmp_path / "missing.npy")) is None
    path = tmp_path / "s.npy"
    np.save(path, np.array([1.0, 2.0]))
    np.testing.assert_allclose(load_scores(str(path)), [1.0, 2.0])


def test_near_identity_validity_is_not_elite_eligible() -> None:
    from research_agent.evolution.fitness import compute_fitness, is_elite_eligible
    from tests.evolution_helpers import make_member

    member = make_member(
        experiment_id="child-1",
        metrics={"GAUC": 0.9, "nDCG@5": 0.9, "primary": 0.9},
        research_validity=NEAR_IDENTITY_VALIDITY,
    )
    assert is_elite_eligible(member) is False
    assert compute_fitness(member) is None
