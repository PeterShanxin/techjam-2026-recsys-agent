"""Data-present smoke checks for official split sizes and random harness range."""
from __future__ import annotations

from baseline import run_random

EXPECTED_SPLIT_SIZES = {"train": 1141112, "valid": 124909, "test": 170588}
RANDOM_PRIMARY_REF = 0.4753
RANDOM_PRIMARY_TOL = 0.001


def test_official_split_sizes_and_date_windows(kuairand_splits):
    assert {k: len(v) for k, v in kuairand_splits.items()} == EXPECTED_SPLIT_SIZES
    windows = {
        "train": (20220408, 20220421),
        "valid": (20220422, 20220428),
        "test": (20220429, 20220508),
    }
    for name, (lo, hi) in windows.items():
        dates = [row[0] for row in kuairand_splits[name]]
        assert lo <= min(dates) <= max(dates) <= hi


def test_test_user_video_pairs_are_not_unique(kuairand_splits):
    pairs = [(row[1], row[2]) for row in kuairand_splits["test"]]
    assert len(pairs) != len(set(pairs))


def test_random_test_primary_in_official_range(kuairand_splits):
    out = run_random(kuairand_splits, seed=0)["test"]
    assert abs(out["primary"] - RANDOM_PRIMARY_REF) <= RANDOM_PRIMARY_TOL
