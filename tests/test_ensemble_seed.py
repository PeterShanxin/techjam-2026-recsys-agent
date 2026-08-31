"""Phase 4 ensemble seed is a mutation of FM, not a manufactured extra winner."""
from __future__ import annotations

from research_agent.agent.constants import FM_ROOT_ID
from research_agent.evolution.seeds import ENSEMBLE_SEED_ID, ensemble_seed_spec


def test_ensemble_seed_spec_is_verified_bagging_mutation():
    spec = ensemble_seed_spec()
    assert spec.experiment_id == ENSEMBLE_SEED_ID
    assert spec.origin == "mutation"
    assert spec.parent_ids == (FM_ROOT_ID,)
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert spec.parameters["num_models"] == 3
    assert spec.implementation.entrypoint.endswith("fm_ensemble_scorer.py")
    assert "family:ensemble" in spec.tags
    assert "axis:ensembling" in spec.tags
