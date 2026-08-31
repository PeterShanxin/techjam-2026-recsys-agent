"""Validated starting priors. Configurable ExperimentSpecs, not one-off winners."""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Iterable

from research_agent.agent.constants import FM_ROOT_ID, FM_ROOT_PARAMETERS
from research_agent.agent.root import UnusableRootError, is_usable_root_result
from research_agent.experiments import ExperimentSpec, ImplementationRef
from research_agent.final_candidate import FINAL_EXPERIMENT_ID, final_candidate_spec

from .config import DEFAULT_STARTING_PRIOR_IDS

ENSEMBLE_SEED_ID = "fm-ensemble-3seed"
ENSEMBLE_ENTRYPOINT = "src/research_agent/recommenders/fm_ensemble_scorer.py"
ENSEMBLE_REFERENCE_PRIMARY = 0.6021
FINAL_PRIOR_ID = FINAL_EXPERIMENT_ID
MATCHED_STARTING_SEED_IDS = (FM_ROOT_ID, *DEFAULT_STARTING_PRIOR_IDS)


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


def final_swa7_prior_spec(
    *,
    experiment_id: str = FINAL_PRIOR_ID,
    timeout_seconds: float = 1800.0,
    seed: int = 0,
) -> ExperimentSpec:
    spec = final_candidate_spec(
        experiment_id=experiment_id,
        evaluation_split="valid",
        allow_test_split=False,
        timeout_seconds=timeout_seconds,
        seed=seed,
    )
    return spec


def prior_spec_for(prior_id: str, **kwargs: Any) -> ExperimentSpec:
    if prior_id == ENSEMBLE_SEED_ID:
        return ensemble_seed_spec(**kwargs)
    if prior_id == FINAL_PRIOR_ID:
        allowed = {key: kwargs[key] for key in ("experiment_id", "timeout_seconds", "seed") if key in kwargs}
        return final_swa7_prior_spec(**allowed)
    raise KeyError(f"unknown starting prior id: {prior_id}")


def resolve_prior_specs(prior_ids: Iterable[str] | None = None) -> list[ExperimentSpec]:
    ids = tuple(DEFAULT_STARTING_PRIOR_IDS if prior_ids is None else prior_ids)
    return [prior_spec_for(item) for item in ids]


def ensure_prior_spec(runner: Any, spec: ExperimentSpec) -> Any:
    """Run a prior seed if missing. Does not consume sequential iteration budget.

    Always return an object with ``spec`` and ``result``. If the id already exists
    without a stored result and the incoming spec hash collides, keep the
    ``runner.run()`` collision result instead of assuming the registry row is complete.
    """
    existing = runner.registry.peek(spec.experiment_id)
    if existing is not None and existing.result is not None:
        return existing
    result = runner.run(spec)
    if result is None:
        raise RuntimeError(f"prior {spec.experiment_id} produced no result")
    entry = runner.registry.peek(spec.experiment_id)
    if entry is not None and entry.result is not None:
        return entry
    spec_used = existing.spec if existing is not None else spec
    return SimpleNamespace(spec=spec_used, result=result)


def ensure_matched_starting_seeds(
    agent: Any,
    *,
    ensemble_spec: ExperimentSpec | None = None,
    prior_specs: Iterable[ExperimentSpec] | None = None,
) -> tuple[Any, ...]:
    """Give sequential search the same Generation-0 priors as evolution.

    Priors are not new evaluations. Their wall time is folded into ``agent.run()``
    via ``_prior_wall_seconds`` so matched ``--wall-clock`` budgets stay fair.
    Passing ``ensemble_spec`` keeps the old one-extra-prior test hook.
    """
    started = time.perf_counter()
    root = agent.ensure_root()
    if not is_usable_root_result(root.result):
        raise UnusableRootError(
            f"research root {root.experiment_id} is not a successful validation result"
        )
    if prior_specs is not None:
        specs = list(prior_specs)
    elif ensemble_spec is not None:
        specs = [ensemble_spec]
    else:
        specs = resolve_prior_specs()
    seeds = []
    for spec in specs:
        existing = agent.runner.registry.peek(spec.experiment_id)
        reused = existing is not None and existing.result is not None
        seed = ensure_prior_spec(agent.runner, spec)
        if seed.result is None:
            raise RuntimeError(f"prior {spec.experiment_id} has no result")
        if not reused:
            agent.ledger.add_experiment(status=seed.result.status, wall_seconds=seed.result.wall_seconds)
        seeds.append(seed)
    prior = float(getattr(agent, "_prior_wall_seconds", 0.0) or 0.0)
    agent._prior_wall_seconds = prior + (time.perf_counter() - started)
    return (root, *seeds)
