"""Evolution Controller: deterministic survival, diversity, budgets."""

from .config import EvolutionConfig
from .controller import EvolutionController
from .diversity import duplicate_reason, semantic_signature
from .fitness import compute_fitness, is_elite_eligible, rank_members, select_elites
from .lineage import format_lineage, lineage_forest
from .seeds import ENSEMBLE_SEED_ID, ensemble_seed_spec
from .types import EvolutionRun, GenerationRecord, Population, PopulationMember, SelectionDecision

__all__ = [
    "ENSEMBLE_SEED_ID",
    "EvolutionConfig",
    "EvolutionController",
    "EvolutionRun",
    "GenerationRecord",
    "Population",
    "PopulationMember",
    "SelectionDecision",
    "compute_fitness",
    "duplicate_reason",
    "ensemble_seed_spec",
    "format_lineage",
    "is_elite_eligible",
    "lineage_forest",
    "rank_members",
    "select_elites",
    "semantic_signature",
]
