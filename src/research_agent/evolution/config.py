"""Pilot and competition-representable evolution budgets."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

COMPETITION_MAX_EVALUATIONS = 50
COMPETITION_WALL_SECONDS = 6 * 3600.0
COMPETITION_EPSILON = 0.002
COMPETITION_PATIENCE = 3
DEFAULT_STARTING_PRIOR_IDS = ("fm-ensemble-3seed", "final-swa7-ensemble")


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 4
    elite_count: int = 2
    generations: int = 2
    max_new_evaluations: int = 6
    include_ensemble_seed: bool = True
    starting_prior_ids: tuple[str, ...] | None = None
    fill_to_size_on_init: bool = True
    token_budget: int | None = None
    wall_clock_seconds: float | None = None
    convergence_epsilon: float = 0.002
    convergence_patience: int = 3
    efficiency_penalty: float = 0.0
    prefer_crossover_from_generation: int = 2
    max_repairs: int = 2
    experiment_timeout_seconds: float = 900.0
    # Behavioural no-op gate. A child whose within-user ordering matches a parent's on
    # all but this share of rows is a reparameterisation, not a hypothesis. Disabled on
    # splits below near_identity_min_rows, where ordering agreement is coincidence.
    near_identity_min_rank_change: float = 0.001
    near_identity_min_rows: int = 1000

    def __post_init__(self) -> None:
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.elite_count < 1:
            raise ValueError("elite_count must be >= 1")
        if self.elite_count > self.population_size:
            raise ValueError("elite_count cannot exceed population_size")
        if self.generations < 0:
            raise ValueError("generations must be >= 0")
        if self.max_new_evaluations < 0:
            raise ValueError("max_new_evaluations must be >= 0")
        if self.starting_prior_ids is not None:
            object.__setattr__(self, "starting_prior_ids", tuple(self.starting_prior_ids))

    def resolved_starting_prior_ids(self) -> tuple[str, ...]:
        if self.starting_prior_ids is not None:
            return tuple(self.starting_prior_ids)
        if self.include_ensemble_seed:
            return DEFAULT_STARTING_PRIOR_IDS
        return ()

    @property
    def offspring_per_generation(self) -> int:
        return max(1, self.population_size - self.elite_count)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvolutionConfig":
        allowed = {item.name for item in fields(cls)}
        payload = {key: value for key, value in dict(data).items() if key in allowed}
        return cls(**payload)

    @classmethod
    def competition(cls, **overrides: Any) -> "EvolutionConfig":
        payload = {
            "population_size": 4,
            "elite_count": 2,
            "generations": COMPETITION_MAX_EVALUATIONS,
            "max_new_evaluations": COMPETITION_MAX_EVALUATIONS,
            "include_ensemble_seed": True,
            "fill_to_size_on_init": True,
            "wall_clock_seconds": COMPETITION_WALL_SECONDS,
            "convergence_epsilon": COMPETITION_EPSILON,
            "convergence_patience": COMPETITION_PATIENCE,
            "efficiency_penalty": 0.0,
            "prefer_crossover_from_generation": 2,
            "max_repairs": 2,
            "experiment_timeout_seconds": 1800.0,
        }
        payload.update(overrides)
        return cls(**payload)
