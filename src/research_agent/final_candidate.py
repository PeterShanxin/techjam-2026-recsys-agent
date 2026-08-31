"""Frozen validation-selected final candidate. Test is opt-in submission only."""
from __future__ import annotations

from research_agent.agent.constants import FM_ROOT_PARAMETERS
from research_agent.experiments import ExperimentSpec, ImplementationRef

FINAL_EXPERIMENT_ID = "final-swa7-ensemble"
FINAL_ENTRYPOINT = "src/research_agent/recommenders/fm_swa7_ensemble_scorer.py"
LIVE_PARENT_ID = "rs-20260830T133522Z-0e304128-003"
LIVE_ELITE_ID = "rs-20260830T133522Z-0e304128-004"


def final_candidate_spec(
    *,
    experiment_id: str = FINAL_EXPERIMENT_ID,
    evaluation_split: str = "valid",
    allow_test_split: bool = False,
    timeout_seconds: float = 1800.0,
    seed: int = 0,
) -> ExperimentSpec:
    if evaluation_split == "test" and not allow_test_split:
        raise ValueError("test split requires allow_test_split=True after the candidate is frozen")
    params = dict(FM_ROOT_PARAMETERS)
    params["num_models"] = 7
    params["top_k_checkpoints"] = 2
    return ExperimentSpec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=FINAL_ENTRYPOINT),
        hypothesis=(
            "Average raw FM probabilities from seven seeds after intra-seed averaging "
            "of the top-2 validation-primary checkpoints."
        ),
        rationale=(
            "Phase 4 matched evolutionary search selected this mechanism on validation "
            f"as elite {LIVE_ELITE_ID}. Frozen for submission; test is not used for selection."
        ),
        origin="mutation",
        parent_ids=(LIVE_PARENT_ID,),
        parameters=params,
        seed=seed,
        evaluation_split=evaluation_split,
        timeout_seconds=timeout_seconds,
        allow_test_split=allow_test_split,
        tags=("phase5", "final", "family:ensemble", "mech:swa", "mech:bagging"),
        notes="Canonical repo copy of the Phase 4 SWA + 7-seed winner. Not an ephemeral generated path.",
    )
