"""FM/research root reuse rules. Failed roots are not research parents."""
from __future__ import annotations

import math
from typing import Any

from research_agent.experiments import ExperimentSpec
from research_agent.experiments.splits import RESEARCH_SPLIT

from .constants import FM_ROOT_ID


class UnusableRootError(RuntimeError):
    """No successful validation root exists; research must not call the LLM."""


def is_usable_root_result(result: Any) -> bool:
    if result is None:
        return False
    if getattr(result, "status", None) != "success":
        return False
    if getattr(result, "evaluation_split", None) != RESEARCH_SPLIT:
        return False
    metrics = getattr(result, "metrics", None)
    if metrics is None:
        return False
    for value in (metrics.gauc, metrics.ndcg_at_5, metrics.primary):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number):
            return False
    return True


def is_root_identity(experiment_id: str, base_id: str = FM_ROOT_ID) -> bool:
    if experiment_id == base_id:
        return True
    prefix = f"{base_id}-r"
    if not experiment_id.startswith(prefix):
        return False
    suffix = experiment_id[len(prefix) :]
    return suffix.isdigit() and len(suffix) >= 1


def find_usable_root(registry: Any, preferred_id: str = FM_ROOT_ID) -> Any | None:
    preferred = registry.peek(preferred_id)
    if preferred is not None and is_usable_root_result(preferred.result):
        return preferred
    for experiment_id in registry.iter_ids():
        if not is_root_identity(experiment_id, preferred_id):
            continue
        entry = registry.get(experiment_id)
        if is_usable_root_result(entry.result):
            return entry
    return None


def next_root_experiment_id(registry: Any, base_id: str = FM_ROOT_ID) -> str:
    if registry.peek(base_id) is None:
        return base_id
    n = 1
    while True:
        candidate = f"{base_id}-r{n:03d}"
        if registry.peek(candidate) is None:
            return candidate
        n += 1


def spec_with_experiment_id(spec: ExperimentSpec, experiment_id: str) -> ExperimentSpec:
    payload = spec.to_dict()
    payload["experiment_id"] = experiment_id
    payload.pop("spec_hash", None)
    return ExperimentSpec.from_dict(payload)
