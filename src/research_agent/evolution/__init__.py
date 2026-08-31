"""Evolution Controller: deterministic survival, diversity, budgets."""

from .config import EvolutionConfig
from .controller import EvolutionController
from .diversity import duplicate_reason, semantic_signature
from .fitness import compute_fitness, is_elite_eligible, rank_members, select_elites
from .lineage import format_lineage, lineage_forest, scoped_lineage_ids, session_lineage_forest
from .seeds import ENSEMBLE_SEED_ID, MATCHED_STARTING_SEED_IDS, ensemble_seed_spec, ensure_matched_starting_seeds
from .types import EvolutionRun, GenerationRecord, Population, PopulationMember, SelectionDecision

__all__ = [
    "MATCHED_STARTING_SEED_IDS",
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
    "ensure_matched_starting_seeds",
    "ensemble_seed_spec",
    "format_lineage",
    "is_elite_eligible",
    "lineage_forest",
    "rank_members",
    "scoped_lineage_ids",
    "select_elites",
    "semantic_signature",
    "session_lineage_forest",
]
