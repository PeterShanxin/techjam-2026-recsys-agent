"""Reproducible Phase 4 seed: verified 3-seed FM bagging winner."""
from __future__ import annotations

from research_agent.experiments import ExperimentSpec, ImplementationRef

from research_agent.agent.constants import FM_ROOT_ID, FM_ROOT_PARAMETERS

ENSEMBLE_SEED_ID = "fm-ensemble-3seed"
ENSEMBLE_ENTRYPOINT = "src/research_agent/recommenders/fm_ensemble_scorer.py"
ENSEMBLE_REFERENCE_PRIMARY = 0.6021


def ensemble_seed_spec(
    *,
    experiment_id: str = ENSEMBLE_SEED_ID,
    parent_id: str = FM_ROOT_ID,
    timeout_seconds: float = 1800.0,
    seed: int = 0,
) -> ExperimentSpec:
    params = dict(FM_ROOT_PARAMETERS)
    params["num_models"] = 3
    return ExperimentSpec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=ENSEMBLE_ENTRYPOINT),
        hypothesis=(
            "Average scores from three official FM models trained with distinct seeds "
            "to reduce seed variance on within-user ranking."
        ),
        rationale=(
            "Phase 3 sequential search found 3-seed FM bagging at validation primary "
            f"{ENSEMBLE_REFERENCE_PRIMARY} (delta vs FM +0.0006). Phase 4 reseeds that "
            "mechanism as a reproducible mutation of the FM root."
        ),
        origin="mutation",
        parent_ids=(parent_id,),
        parameters=params,
        seed=seed,
        evaluation_split="valid",
        timeout_seconds=timeout_seconds,
        allow_test_split=False,
        tags=("phase4", "seed", "family:ensemble", "axis:ensembling", "mech:bagging"),
        notes="Reconstructed Phase 3 bagging winner. Not a manufactured extra seed.",
    )
