"""Validation/test split policy.

Autonomous research uses validation only. Test is reserved for explicit
final, audit, or submission workflows and must never enter elite ranking.
"""
from __future__ import annotations

from .errors import ForbiddenTestSplit, SpecError

DEFAULT_EVALUATION_SPLIT = "valid"
EVALUATION_SPLITS = frozenset({"valid", "test"})
RESEARCH_SPLIT = "valid"
AUDIT_SPLIT = "test"


def normalize_evaluation_split(split: str) -> str:
    if split not in EVALUATION_SPLITS:
        raise SpecError(
            f"evaluation_split must be one of {sorted(EVALUATION_SPLITS)}, got {split!r}"
        )
    return split


def is_research_split(split: str) -> bool:
    return split == RESEARCH_SPLIT


def assert_split_allowed(split: str, allow_test: bool) -> None:
    split = normalize_evaluation_split(split)
    if split == RESEARCH_SPLIT:
        return
    if split == AUDIT_SPLIT and allow_test:
        return
    raise ForbiddenTestSplit(
        "evaluation_split='test' requires explicit opt-in "
        "(ExperimentSpec.allow_test_split=True or ExperimentRunner(allow_test=True))"
    )
