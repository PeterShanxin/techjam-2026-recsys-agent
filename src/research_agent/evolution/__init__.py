"""Evolution Controller: deterministic survival, diversity, budgets."""

from .config import (
    COMPETITION_EPSILON,
    COMPETITION_MAX_EVALUATIONS,
    COMPETITION_PATIENCE,
    COMPETITION_WALL_SECONDS,
    DEFAULT_STARTING_PRIOR_IDS,
    EvolutionConfig,
)
from .controller import EvolutionController
from .diversity import duplicate_reason, semantic_signature
from .fitness import compute_fitness, is_elite_eligible, rank_members, select_elites
from .lineage import format_lineage, lineage_forest, scoped_lineage_ids, session_lineage_forest
from .seeds import (
    ENSEMBLE_SEED_ID,
    SWA7_PRIOR_ID,
    TIERED_PRIOR_ID,
    MATCHED_STARTING_SEED_IDS,
    ensemble_seed_spec,
    ensure_matched_starting_seeds,
    final_swa7_prior_spec,
    prior_spec_for,
    resolve_prior_specs,
)
from .types import EvolutionRun, GenerationRecord, Population, PopulationMember, SelectionDecision

__all__ = [
    "COMPETITION_EPSILON",
    "COMPETITION_MAX_EVALUATIONS",
    "COMPETITION_PATIENCE",
    "COMPETITION_WALL_SECONDS",
    "DEFAULT_STARTING_PRIOR_IDS",
    "MATCHED_STARTING_SEED_IDS",
    "ENSEMBLE_SEED_ID",
    "SWA7_PRIOR_ID",
    "TIERED_PRIOR_ID",
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
    "final_swa7_prior_spec",
    "prior_spec_for",
    "resolve_prior_specs",
    "format_lineage",
    "is_elite_eligible",
    "lineage_forest",
    "rank_members",
    "scoped_lineage_ids",
    "select_elites",
    "semantic_signature",
    "session_lineage_forest",
]
