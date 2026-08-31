"""Frozen validation-selected final candidate. Test is opt-in submission only.

Phase 5 candidate is the sprint-2 autonomous elite `rs-20260831T062638Z-939b7000-008`
(a catalog-balanced tiered FM ensemble). It replaced the Phase 4 SWA+7-seed winner on
validation primary, the project's standing selection rule.

The superseded Phase 4 candidate stays runnable through `swa7_candidate_spec()` so its
historical result remains reproducible. Nothing about that result is rewritten.
"""
from __future__ import annotations

from research_agent.agent.constants import FM_ROOT_PARAMETERS
from research_agent.experiments import ExperimentSpec, ImplementationRef

FINAL_EXPERIMENT_ID = "final-tiered-ensemble"
FINAL_ENTRYPOINT = "src/research_agent/recommenders/tiered_ensemble_scorer.py"
FINAL_SEED = 42
LIVE_ELITE_ID = "rs-20260831T062638Z-939b7000-008"
LIVE_PARENT_IDS = (
    "rs-20260831T062638Z-939b7000-007",
    "rs-20260831T062638Z-939b7000-006",
)

# Superseded Phase 4 candidate. Historical evidence, not the submission.
LEGACY_SWA7_EXPERIMENT_ID = "final-swa7-ensemble"
LEGACY_SWA7_ENTRYPOINT = "src/research_agent/recommenders/fm_swa7_ensemble_scorer.py"
LEGACY_SWA7_ELITE_ID = "rs-20260830T133522Z-0e304128-004"
LEGACY_SWA7_PARENT_ID = "rs-20260830T133522Z-0e304128-003"


def _guard_test_split(evaluation_split: str, allow_test_split: bool) -> None:
    if evaluation_split == "test" and not allow_test_split:
        raise ValueError("test split requires allow_test_split=True after the candidate is frozen")


def final_candidate_spec(
    *,
    experiment_id: str = FINAL_EXPERIMENT_ID,
    evaluation_split: str = "valid",
    allow_test_split: bool = False,
    timeout_seconds: float = 1800.0,
    seed: int = FINAL_SEED,
) -> ExperimentSpec:
    _guard_test_split(evaluation_split, allow_test_split)
    return ExperimentSpec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=FINAL_ENTRYPOINT),
        hypothesis=(
            "Split an 8-member SWA FM ensemble across three tiers that differ in both "
            "train-row selection and L2 strength, keeping half the members on the full "
            "catalog, and average the members in probability space."
        ),
        rationale=(
            "Sprint-2 autonomous evolutionary search selected this mechanism on validation "
            f"as elite {LIVE_ELITE_ID}. Frozen for submission; test is not used for selection."
        ),
        origin="crossover",
        parent_ids=LIVE_PARENT_IDS,
        # Empty on purpose: every tier setting is baked into the frozen entrypoint, so an
        # empty config reproduces the live elite bitwise (config fingerprint = sha256 of {}).
        parameters={},
        seed=seed,
        evaluation_split=evaluation_split,
        timeout_seconds=timeout_seconds,
        allow_test_split=allow_test_split,
        tags=(
            "phase5",
            "final",
            "family:ensemble",
            "mech:swa",
            "mech:bagging",
            "mech:tiered_variance_filtering",
            "mech:tier_adaptive_l2",
        ),
        notes="Canonical repo copy of the sprint-2 autonomous elite. Not an ephemeral generated path.",
    )


def swa7_candidate_spec(
    *,
    experiment_id: str = LEGACY_SWA7_EXPERIMENT_ID,
    evaluation_split: str = "valid",
    allow_test_split: bool = False,
    timeout_seconds: float = 1800.0,
    seed: int = 0,
) -> ExperimentSpec:
    """Superseded Phase 4 winner. Kept reproducible as historical evidence."""
    _guard_test_split(evaluation_split, allow_test_split)
    params = dict(FM_ROOT_PARAMETERS)
    params["num_models"] = 7
    params["top_k_checkpoints"] = 2
    return ExperimentSpec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=LEGACY_SWA7_ENTRYPOINT),
        hypothesis=(
            "Average raw FM probabilities from seven seeds after intra-seed averaging "
            "of the top-2 validation-primary checkpoints."
        ),
        rationale=(
            "Phase 4 matched evolutionary search selected this mechanism on validation "
            f"as elite {LEGACY_SWA7_ELITE_ID}. Superseded by {FINAL_EXPERIMENT_ID} in sprint 2; "
            "retained so the historical number stays reproducible."
        ),
        origin="mutation",
        parent_ids=(LEGACY_SWA7_PARENT_ID,),
        parameters=params,
        seed=seed,
        evaluation_split=evaluation_split,
        timeout_seconds=timeout_seconds,
        allow_test_split=allow_test_split,
        tags=("phase4", "superseded", "family:ensemble", "mech:swa", "mech:bagging"),
        notes="Historical Phase 4 SWA + 7-seed winner. Not the Phase 5 submission candidate.",
    )
