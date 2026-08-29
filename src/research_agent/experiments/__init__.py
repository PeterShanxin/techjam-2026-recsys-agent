"""Phase 2 experiment harness: spec, runner, result, registry."""

from .errors import ForbiddenTestSplit, RegistryError, SpecError
from .registry import ExperimentRegistry, RegistryEntry
from .result import ExperimentResult, FailureInfo, Metrics
from .runner import ExperimentRunner
from .spec import ExperimentSpec, ImplementationRef
from .splits import DEFAULT_EVALUATION_SPLIT, RESEARCH_SPLIT

__all__ = [
    "DEFAULT_EVALUATION_SPLIT",
    "RESEARCH_SPLIT",
    "ExperimentRegistry",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentSpec",
    "FailureInfo",
    "ImplementationRef",
    "Metrics",
    "RegistryEntry",
    "RegistryError",
    "ForbiddenTestSplit",
    "SpecError",
]
