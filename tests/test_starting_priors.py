"""Starting priors are configurable and do not consume the new-eval budget."""
from __future__ import annotations

from pathlib import Path

from evolution_helpers import make_member
from experiment_helpers import make_spec
from research_agent.agent import ResearchAgent
from research_agent.agent.constants import FM_ROOT_ID
from research_agent.evolution import EvolutionConfig, EvolutionController
from research_agent.evolution.diversity import duplicate_reason
from research_agent.evolution.seeds import (
    ENSEMBLE_SEED_ID,
    FINAL_PRIOR_ID,
    final_swa7_prior_spec,
    prior_spec_for,
    resolve_prior_specs,
)
from research_agent.experiments import ExperimentSpec, ImplementationRef
from research_agent.final_candidate import FINAL_EXPERIMENT_ID
from research_agent.llm import FakeProvider
from research_helpers import make_runner, mini_root_spec
from research_agent.evolution.seeds import ensure_matched_starting_seeds


def _mini_prior(tmp_path: Path, experiment_id: str, *, family: str, extra_params: dict | None = None):
    params = {"action": "succeed", "family": family}
    if extra_params:
        params.update(extra_params)
    return make_spec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=str(tmp_path / "root_candidate.py")),
        origin="mutation",
        parent_ids=(FM_ROOT_ID,),
        hypothesis=f"mini prior {experiment_id}",
        rationale="tests",
        parameters=params,
        timeout_seconds=30.0,
        tags=("test", "seed", f"family:{family}", "axis:ensembling", "mech:bagging"),
    )


def test_final_candidate_is_a_starting_prior_spec():
    spec = final_swa7_prior_spec()
    assert spec.experiment_id == FINAL_EXPERIMENT_ID == FINAL_PRIOR_ID
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert spec.parameters["num_models"] == 7
    assert spec.parameters["top_k_checkpoints"] == 2
    ids = [item.experiment_id for item in resolve_prior_specs()]
    assert ids == [ENSEMBLE_SEED_ID, FINAL_PRIOR_ID]
    assert prior_spec_for(FINAL_PRIOR_ID).experiment_id == FINAL_PRIOR_ID


def test_starting_priors_do_not_consume_new_evaluation_budget(tmp_path: Path):
    runner, _ = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=FakeProvider(script=[]),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=2,
        max_repairs=0,
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id="ev-priors",
    )
    priors = [
        _mini_prior(tmp_path, ENSEMBLE_SEED_ID, family="ensemble"),
        _mini_prior(tmp_path, FINAL_PRIOR_ID, family="ensemble", extra_params={"swa": True}),
    ]
    root, *seeds = ensure_matched_starting_seeds(agent, prior_specs=priors)
    assert root.experiment_id == FM_ROOT_ID
    assert [item.spec.experiment_id for item in seeds] == [ENSEMBLE_SEED_ID, FINAL_PRIOR_ID]
    assert agent.ledger.research_calls == 0
    ctl = EvolutionController(
        agent=agent,
        config=EvolutionConfig(
            population_size=4,
            elite_count=2,
            generations=0,
            max_new_evaluations=8,
            include_ensemble_seed=False,
            fill_to_size_on_init=False,
        ),
        prior_specs=priors,
    )
    run = ctl.run()
    assert run.evaluated_offspring == 0
    ids = {item.experiment_id for item in run.population.members}
    assert {FM_ROOT_ID, ENSEMBLE_SEED_ID, FINAL_PRIOR_ID} <= ids
    assert FINAL_PRIOR_ID in {item.experiment_id for item in run.all_members}


def test_frozen_best_can_be_generation_zero_elite():
    members = [
        make_member(
            experiment_id=FM_ROOT_ID,
            metrics={"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6014688},
            research_validity="root",
            origin="baseline",
        ),
        make_member(
            experiment_id=ENSEMBLE_SEED_ID,
            metrics={"GAUC": 0.6680, "nDCG@5": 0.5361, "primary": 0.6021109},
            research_validity="hypothesis_tested",
            research_family="ensemble",
            mechanism_tags=("bagging",),
            changed_axes=("ensembling",),
        ),
        make_member(
            experiment_id=FINAL_PRIOR_ID,
            metrics={"GAUC": 0.668366, "nDCG@5": 0.536271, "primary": 0.6023186},
            research_validity="hypothesis_tested",
            research_family="ensemble",
            mechanism_tags=("swa", "bagging"),
            changed_axes=("ensembling", "checkpoint_average"),
        ),
    ]
    ctl = object.__new__(EvolutionController)
    ctl.config = EvolutionConfig(elite_count=2)
    marked = EvolutionController._mark_elites(ctl, members)
    elites = [item.experiment_id for item in marked if item.selection == "elite"]
    assert elites[0] == FINAL_PRIOR_ID


def test_multiple_model_family_signatures_can_coexist():
    existing = [
        make_member(
            experiment_id="fm-bag",
            spec_hash="h1",
            source_fingerprint="fp1",
            research_family="ensemble",
            mechanism_tags=("bagging",),
            changed_axes=("ensembling",),
        ),
        make_member(
            experiment_id="hist",
            spec_hash="h2",
            source_fingerprint="fp2",
            research_family="history_recency",
            mechanism_tags=("recency",),
            changed_axes=("history",),
        ),
    ]
    pairwise = make_member(
        experiment_id="bpr",
        spec_hash="h3",
        source_fingerprint="fp3",
        research_family="pairwise_ranking",
        mechanism_tags=("bpr",),
        changed_axes=("objective",),
    )
    twin = make_member(
        experiment_id="hist2",
        spec_hash="h4",
        source_fingerprint="fp4",
        research_family="history_recency",
        mechanism_tags=("recency",),
        changed_axes=("history",),
    )
    assert duplicate_reason(pairwise, existing) is None
    assert duplicate_reason(twin, existing) == "semantic_signature"


def test_p0_affinity_freeze_is_valid_only_not_submission():
    spec = ExperimentSpec.from_path(Path("configs/experiments/p0_affinity_residual_valid.json"))
    assert spec.experiment_id == "p0-affinity-residual"
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False
    assert spec.implementation.entrypoint.endswith("fm_affinity_residual_scorer.py")
    assert spec.experiment_id != FINAL_EXPERIMENT_ID
