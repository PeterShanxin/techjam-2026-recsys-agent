"""Deterministic fitness. Official validation primary dominates. Test never enters."""
from __future__ import annotations

import math
from typing import Iterable

from research_agent.experiments.splits import RESEARCH_SPLIT

from .types import PopulationMember

ELITE_VALIDITY = {"root", "hypothesis_tested"}
BLOCKED_STATUS = {"failed", "timeout", "invalid", "not_executed"}


def is_elite_eligible(member: PopulationMember) -> bool:
    if member.status != "success":
        return False
    if member.evaluation_split != RESEARCH_SPLIT:
        return False
    if member.research_validity not in ELITE_VALIDITY:
        return False
    if member.metrics is None or "primary" not in member.metrics:
        return False
    try:
        primary = float(member.metrics["primary"])
    except (TypeError, ValueError):
        return False
    return math.isfinite(primary)


def compute_fitness(
    member: PopulationMember,
    *,
    efficiency_penalty: float = 0.0,
) -> float | None:
    if not is_elite_eligible(member):
        return None
    primary = float(member.metrics["primary"])  # type: ignore[index]
    runtime = float(member.runtime_seconds or 0.0)
    return primary - float(efficiency_penalty) * runtime


def rank_members(
    members: Iterable[PopulationMember],
    *,
    efficiency_penalty: float = 0.0,
) -> list[PopulationMember]:
    scored: list[tuple[float, float, str, PopulationMember]] = []
    for member in members:
        fitness = member.fitness if member.fitness is not None else compute_fitness(
            member, efficiency_penalty=efficiency_penalty
        )
        if fitness is None:
            continue
        runtime = float(member.runtime_seconds) if member.runtime_seconds is not None else float("inf")
        scored.append((fitness, runtime, member.experiment_id, member.with_updates(fitness=fitness)))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored]


def select_elites(
    members: Iterable[PopulationMember],
    elite_count: int,
    *,
    efficiency_penalty: float = 0.0,
) -> list[PopulationMember]:
    if elite_count <= 0:
        return []
    ranked = rank_members(members, efficiency_penalty=efficiency_penalty)
    return ranked[:elite_count]
