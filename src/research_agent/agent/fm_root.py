"""FM validation root spec. Sequential research starts here, not from random."""
from __future__ import annotations

from research_agent.experiments import ExperimentSpec, ImplementationRef

from .constants import (
    FM_ROOT_ENTRYPOINT,
    FM_ROOT_ID,
    FM_ROOT_PARAMETERS,
    FM_VALID_REFERENCE,
)


def fm_root_spec(
    *,
    experiment_id: str = FM_ROOT_ID,
    timeout_seconds: float = 900.0,
    seed: int = 0,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=FM_ROOT_ENTRYPOINT),
        hypothesis=(
            "Official KuaiRand Factorization Machine is the research root. "
            "Within-user ranking of long_view using 5 fields and pointwise logloss."
        ),
        rationale=(
            f"Phase 1 reproduced validation GAUC {FM_VALID_REFERENCE['GAUC']}, "
            f"nDCG@5 {FM_VALID_REFERENCE['nDCG@5']}, primary {FM_VALID_REFERENCE['primary']}."
        ),
        origin="baseline",
        parent_ids=(),
        parameters=dict(FM_ROOT_PARAMETERS),
        seed=seed,
        evaluation_split="valid",
        timeout_seconds=timeout_seconds,
        allow_test_split=False,
        tags=("phase3", "fm", "root"),
        notes="Organizer-compatible FM candidate. Runner owns official metrics.",
    )
