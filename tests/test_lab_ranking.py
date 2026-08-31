"""Within-user grouping and gradient-driven FM. Machinery only, no objective baked in."""
from __future__ import annotations

import numpy as np
import pytest

from research_agent.lab import GradientFM, user_groups
from research_agent.lab.capabilities import LAB_EXAMPLE, lab_contract_dict


def test_user_groups_segments_by_user_and_preserves_row_order() -> None:
    users = ["b", "a", "b", "c", "a", "b"]
    groups = user_groups(users)
    assert groups.n_groups == 3
    assert groups.n_rows == 6
    assert list(groups.group_sizes) == [2, 3, 1]
    assert list(groups.group_ids) == ["a", "b", "c"]
    # within a user the original relative row order survives the stable sort
    assert list(groups.order) == [1, 4, 0, 2, 5, 3]


def test_user_groups_empty_split() -> None:
    groups = user_groups([])
    assert groups.n_groups == 0
    assert groups.n_rows == 0
    assert groups.group_sum(np.zeros(0)).size == 0


def test_group_reductions_match_manual_segments() -> None:
    users = ["u1", "u1", "u2", "u2", "u2"]
    groups = user_groups(users)
    values = np.array([1.0, 3.0, 5.0, 2.0, 9.0])
    ordered = values[groups.order]
    assert list(groups.group_sum(ordered)) == [4.0, 16.0]
    assert list(groups.group_max(ordered)) == [3.0, 9.0]
    assert list(groups.broadcast(np.array([10.0, 20.0]))) == [10.0, 10.0, 20.0, 20.0, 20.0]


def test_group_reductions_reject_wrong_length() -> None:
    groups = user_groups(["a", "a", "b"])
    with pytest.raises(ValueError):
        groups.group_sum(np.zeros(2))
    with pytest.raises(ValueError):
        groups.broadcast(np.zeros(5))


def test_select_returns_whole_groups_never_splitting_a_user() -> None:
    users = ["a", "a", "b", "c", "c", "c"]
    groups = user_groups(users)
    rows = groups.select([0, 2])
    # group 0 is 2 rows starting at 0, group 2 is 3 rows starting at 3
    assert list(rows) == [0, 1, 3, 4, 5]
    assert list(groups.select([])) == []


def test_gradient_fm_logits_match_official_fm_algebra() -> None:
    baseline = pytest.importorskip("baseline")
    X = np.array([[0, 3, 5], [1, 2, 4]], dtype=np.int32)
    ours = GradientFM(8, k=4, seed=11)
    theirs = baseline.FM(8, k=4, seed=11)
    theirs.V, theirs.W, theirs.b = ours.V.copy(), ours.W.copy(), np.float32(ours.b)
    np.testing.assert_allclose(ours.logits(X)[0], theirs.logits(X)[0], rtol=1e-6)


def test_gradient_fm_with_bce_gradient_reproduces_official_step() -> None:
    """Feeding the pointwise BCE gradient must match baseline.FM.step exactly.

    This is the contract that lets a candidate swap in a different objective and trust
    the parameter update. It is not evidence that BCE is the right objective.
    """
    baseline = pytest.importorskip("baseline")
    rng = np.random.default_rng(3)
    X = rng.integers(0, 12, size=(32, 3)).astype(np.int32)
    y = (rng.random(32) < 0.4).astype(np.float32)
    ours = GradientFM(12, k=4, lr=0.01, l2=1e-5, seed=5)
    theirs = baseline.FM(12, k=4, lr=0.01, l2=1e-5, seed=5)
    theirs.V, theirs.W, theirs.b = ours.V.copy(), ours.W.copy(), np.float32(ours.b)
    z, _, _ = ours.logits(X)
    grad = ((1.0 / (1.0 + np.exp(-z)) - y) / len(y)).astype(np.float32)
    ours.apply(X, grad)
    theirs.step(X, y)
    np.testing.assert_allclose(ours.V, theirs.V, atol=1e-6)
    np.testing.assert_allclose(ours.W, theirs.W, atol=1e-6)
    np.testing.assert_allclose(float(ours.b), float(theirs.b), atol=1e-6)


def test_gradient_fm_rejects_bad_gradient() -> None:
    model = GradientFM(6, k=2, seed=0)
    X = np.array([[0, 2], [1, 3]], dtype=np.int32)
    with pytest.raises(ValueError):
        model.apply(X, np.zeros(3))
    with pytest.raises(ValueError):
        model.apply(X, np.array([np.nan, 0.0]))


def test_gradient_fm_state_roundtrip() -> None:
    model = GradientFM(6, k=2, seed=1)
    X = np.array([[0, 2]], dtype=np.int32)
    before = model.predict(X).copy()
    state = model.state()
    model.apply(X, np.array([0.5]))
    assert not np.allclose(before, model.predict(X))
    model.load_state(state)
    np.testing.assert_allclose(before, model.predict(X))


def test_gradient_fm_carries_no_objective() -> None:
    """The class must not smuggle in a loss. Only the caller decides the objective."""
    names = {name for name in dir(GradientFM) if not name.startswith("_")}
    assert names == {"logits", "predict", "apply", "state", "load_state"}


def test_lab_contract_advertises_ranking_machinery_and_example() -> None:
    contract = lab_contract_dict()
    names = {item["name"] for item in contract["capabilities"]}
    assert {"user_groups", "GradientFM"} <= names
    assert contract["example"] == LAB_EXAMPLE
    assert "GradientFM" in contract["example"]
    # the worked example must stay plumbing-only: no ranking objective handed over
    assert "softmax" not in LAB_EXAMPLE.lower()
    assert contract["not_a_ranker"] is True
