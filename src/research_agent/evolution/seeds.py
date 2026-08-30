"""Reproducible Phase 4 seed: verified 3-seed FM bagging winner."""
from __future__ import annotations

from typing import Any

from research_agent.agent.constants import FM_ROOT_ID, FM_ROOT_PARAMETERS
from research_agent.agent.root import UnusableRootError, is_usable_root_result
from research_agent.experiments import ExperimentSpec, ImplementationRef

ENSEMBLE_SEED_ID = "fm-ensemble-3seed"
ENSEMBLE_ENTRYPOINT = "src/research_agent/recommenders/fm_ensemble_scorer.py"
ENSEMBLE_REFERENCE_PRIMARY = 0.6021
MATCHED_STARTING_SEED_IDS = (FM_ROOT_ID, ENSEMBLE_SEED_ID)


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


def ensure_prior_spec(runner: Any, spec: ExperimentSpec) -> Any:
    """Run a prior seed if missing. Does not consume sequential iteration budget."""
    existing = runner.registry.peek(spec.experiment_id)
    if existing is not None and existing.result is not None:
        return existing
    runner.run(spec)
    return runner.registry.get(spec.experiment_id)


def ensure_matched_starting_seeds(agent: Any, *, ensemble_spec: ExperimentSpec | None = None) -> tuple[Any, Any]:
    """Give sequential search the same Generation-0 priors as evolution: FM root + ensemble."""
    root = agent.ensure_root()
    if not is_usable_root_result(root.result):
        raise UnusableRootError(
            f"research root {root.experiment_id} is not a successful validation result"
        )
    spec = ensemble_spec or ensemble_seed_spec(parent_id=root.experiment_id)
    seed = ensure_prior_spec(agent.runner, spec)
    return root, seed
