"""Synthetic checks that official evaluate.py wiring and metric semantics hold."""
from __future__ import annotations

import hashlib
import math

import pytest

from conftest import EVALUATE_SHA256, evaluate_py_canonical_bytes
from evaluate import evaluate


def test_evaluate_py_bytes_unchanged():
    digest = hashlib.sha256(evaluate_py_canonical_bytes()).hexdigest()
    assert digest == EVALUATE_SHA256


def test_primary_is_mean_of_gauc_and_ndcg():
    out = evaluate(["u", "u"], [1, 0], [0.9, 0.1])
    assert out["primary"] == (out["GAUC"] + out["nDCG@5"]) / 2.0


def test_perfect_and_all_negative_users():
    # Discriminative user, perfect order; all-negative user is nDCG=0 and skipped by GAUC.
    users = ["a", "a", "b", "b"]
    labels = [1, 0, 0, 0]
    scores = [0.9, 0.1, 0.5, 0.2]
    out = evaluate(users, labels, scores)
    assert out["GAUC"] == 1.0
    assert out["nDCG@5"] == 0.5
    assert out["primary"] == 0.75
    assert out["users"] == 2
    assert out["rows"] == 4


def test_all_positive_user_excluded_from_gauc():
    users = ["p", "p", "d", "d"]
    labels = [1, 1, 1, 0]
    scores = [0.2, 0.1, 0.9, 0.1]
    out = evaluate(users, labels, scores)
    assert out["GAUC"] == 1.0
    assert out["nDCG@5"] == 1.0


def test_inverted_pair_ndcg_and_auc():
    out = evaluate(["u", "u"], [1, 0], [0.1, 0.9])
    assert out["GAUC"] == 0.0
    assert out["nDCG@5"] == pytest.approx(1.0 / math.log2(3))


def test_gauc_is_positive_weighted():
    # User A: 2 pos, perfect, weight 2. User B: 1 pos, inverted, weight 1.
    users = ["a", "a", "a", "b", "b"]
    labels = [1, 1, 0, 1, 0]
    scores = [0.9, 0.8, 0.1, 0.1, 0.9]
    out = evaluate(users, labels, scores)
    assert out["GAUC"] == pytest.approx(2.0 / 3.0)
