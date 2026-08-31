"""Pilot and competition-representable evolution budgets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 4
    elite_count: int = 2
    generations: int = 2
    max_new_evaluations: int = 6
    include_ensemble_seed: bool = True
    fill_to_size_on_init: bool = True
    token_budget: int | None = None
    wall_clock_seconds: float | None = None
    convergence_epsilon: float = 0.002
    convergence_patience: int = 3
    efficiency_penalty: float = 0.0
    prefer_crossover_from_generation: int = 2
    max_repairs: int = 2
    experiment_timeout_seconds: float = 900.0

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

    @property
    def offspring_per_generation(self) -> int:
        return max(1, self.population_size - self.elite_count)
