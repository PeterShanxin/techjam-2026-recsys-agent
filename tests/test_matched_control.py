"""Matched sequential control shares evolution Generation-0 priors. Zero API spend."""
from __future__ import annotations

from pathlib import Path

from experiment_helpers import make_spec, write_candidate
from research_agent.agent import ResearchAgent
from research_agent.agent.constants import FM_ROOT_ID
from research_agent.evolution.config import DEFAULT_STARTING_PRIOR_IDS, EvolutionConfig
from research_agent.evolution.seeds import (
    ENSEMBLE_SEED_ID,
    FINAL_PRIOR_ID,
    MATCHED_STARTING_SEED_IDS,
    ensure_matched_starting_seeds,
    ensure_prior_spec,
)
from research_agent.experiments import ExperimentResult, ImplementationRef, Metrics
from research_agent.llm import FakeProvider
from research_helpers import make_proposal_payload, make_runner, mini_root_spec


def _mini_ensemble_spec(tmp_path: Path, parent_id: str = FM_ROOT_ID):
    return make_spec(
        experiment_id=ENSEMBLE_SEED_ID,
        implementation=ImplementationRef(entrypoint=str(tmp_path / "root_candidate.py")),
        origin="mutation",
        parent_ids=(parent_id,),
        hypothesis="mini ensemble prior",
        rationale="tests",
        parameters={"action": "succeed", "num_models": 3},
        timeout_seconds=30.0,
        tags=("test", "seed"),
    )


def _agent(tmp_path: Path, script: list, **kwargs) -> ResearchAgent:
    runner = kwargs.pop("runner", None)
    if runner is None:
        runner, _data = make_runner(tmp_path)
    return ResearchAgent(
        provider=FakeProvider(script=script),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=kwargs.pop("max_iterations", 6),
        max_repairs=kwargs.pop("max_repairs", 0),
        root_spec=kwargs.pop("root_spec", mini_root_spec(tmp_path)),
        experiment_timeout_seconds=30.0,
        session_id=kwargs.pop("session_id", "rs-seq"),
        **kwargs,
    )


def test_matched_starting_seed_ids_match_evolution_defaults():
    assert MATCHED_STARTING_SEED_IDS == (FM_ROOT_ID, ENSEMBLE_SEED_ID, FINAL_PRIOR_ID)
    assert DEFAULT_STARTING_PRIOR_IDS == (ENSEMBLE_SEED_ID, FINAL_PRIOR_ID)
    assert EvolutionConfig().include_ensemble_seed is True
    assert EvolutionConfig().resolved_starting_prior_ids() == DEFAULT_STARTING_PRIOR_IDS
    assert EvolutionConfig(include_ensemble_seed=False).resolved_starting_prior_ids() == ()


def test_ensure_matched_starting_seeds_inserts_fm_and_ensemble(tmp_path: Path):
    agent = _agent(tmp_path, script=[], max_iterations=6)
    root, seed = ensure_matched_starting_seeds(agent, ensemble_spec=_mini_ensemble_spec(tmp_path))
    assert root.experiment_id == FM_ROOT_ID
    assert seed.spec.experiment_id == ENSEMBLE_SEED_ID
    assert agent.max_iterations == 6
    assert agent.ledger.research_calls == 0
    assert agent.ledger.completed_experiments == 2
    assert agent._prior_wall_seconds > 0
    ids = set(agent.runner.registry.iter_ids())
    assert {FM_ROOT_ID, ENSEMBLE_SEED_ID} <= ids
    assert seed.spec.parent_ids == (FM_ROOT_ID,)


def test_ensemble_prior_does_not_consume_new_evaluation_budget(tmp_path: Path):
    script = [make_proposal_payload(hypothesis=f"seq {i}") for i in range(6)]
    agent = _agent(tmp_path, script, max_iterations=6)
    ensure_matched_starting_seeds(agent, ensemble_spec=_mini_ensemble_spec(tmp_path))
    run = agent.run()
    assert len(run.iterations) == 6
    assert agent.ledger.research_calls == 6
    assert agent.ledger.completed_experiments == 8
    assert agent.ledger.mutation_calls == 0
    assert agent.ledger.crossover_calls == 0
    ids = set(agent.runner.registry.iter_ids())
    assert FM_ROOT_ID in ids
    assert ENSEMBLE_SEED_ID in ids
    for i in range(1, 7):
        assert f"rs-seq-{i:03d}" in ids


def test_sequential_and_evolution_registries_stay_isolated(tmp_path: Path):
    evo_root = tmp_path / "evo"
    seq_root = tmp_path / "seq"
    evo_root.mkdir()
    seq_root.mkdir()
    write_candidate(evo_root / "root_candidate.py")
    write_candidate(seq_root / "root_candidate.py")
    evo_runner, _ = make_runner(evo_root)
    seq_runner, _ = make_runner(seq_root)
    evo_agent = _agent(evo_root, script=[], runner=evo_runner, session_id="rs-evo", max_iterations=0)
    seq_agent = _agent(seq_root, script=[], runner=seq_runner, session_id="rs-seq", max_iterations=6)
    evo_agent.ensure_root()
    evo_runner.registry.insert_spec(
        make_spec(
            experiment_id="rs-evo-004",
            origin="mutation",
            parent_ids=(FM_ROOT_ID,),
            implementation=ImplementationRef(entrypoint=str(evo_root / "root_candidate.py")),
            parameters={"secret": "evolution-only"},
        )
    )
    ensure_matched_starting_seeds(seq_agent, ensemble_spec=_mini_ensemble_spec(seq_root))
    assert evo_runner.registry.peek("rs-evo-004") is not None
    assert seq_runner.registry.peek("rs-evo-004") is None
    assert seq_runner.registry.peek(ENSEMBLE_SEED_ID) is not None
    assert evo_runner.registry.peek(ENSEMBLE_SEED_ID) is None


def test_sequential_default_parent_prefers_stronger_ensemble_prior(tmp_path: Path):
    agent = _agent(tmp_path, script=[], max_iterations=1)
    ensure_matched_starting_seeds(agent, ensemble_spec=_mini_ensemble_spec(tmp_path))
    agent.runner.registry.upsert_result(
        ExperimentResult(
            experiment_id=FM_ROOT_ID,
            status="success",
            evaluation_split="valid",
            seed=0,
            spec_hash="h",
            wall_seconds=1.0,
            return_code=0,
            run_dir="",
            stdout_path="",
            stderr_path="",
            metrics=Metrics(gauc=0.60, ndcg_at_5=0.60, primary=0.6015),
        )
    )
    agent.runner.registry.upsert_result(
        ExperimentResult(
            experiment_id=ENSEMBLE_SEED_ID,
            status="success",
            evaluation_split="valid",
            seed=0,
            spec_hash="h",
            wall_seconds=1.0,
            return_code=0,
            run_dir="",
            stdout_path="",
            stderr_path="",
            metrics=Metrics(gauc=0.67, ndcg_at_5=0.54, primary=0.6021),
        )
    )
    assert agent._default_parent_id() == ENSEMBLE_SEED_ID
    elite = agent.runner.registry.elite()
    assert elite is not None
    assert elite.spec.experiment_id == ENSEMBLE_SEED_ID


def test_ensure_root_does_not_retrace_after_matched_priors(tmp_path: Path):
    agent = _agent(tmp_path, script=[], max_iterations=0)
    ensure_matched_starting_seeds(agent, ensemble_spec=_mini_ensemble_spec(tmp_path))
    first = agent.ensure_root()
    again = agent.ensure_root()
    assert again is first
    run = agent.run()
    roots = [row for row in agent.trace.records() if row.get("iteration") == 0]
    assert len(roots) == 1
    assert run.root is first


def test_prior_wall_counts_toward_wall_clock_not_iteration_budget(tmp_path: Path):
    script = [make_proposal_payload(hypothesis=f"seq {i}") for i in range(6)]
    agent = _agent(tmp_path, script, max_iterations=6, wall_clock_seconds=10.0)
    ensure_matched_starting_seeds(agent, ensemble_spec=_mini_ensemble_spec(tmp_path))
    assert agent.ledger.research_calls == 0
    assert agent.ledger.completed_experiments == 2
    agent._prior_wall_seconds = 10.0
    run = agent.run()
    assert len(run.iterations) == 0
    assert agent.ledger.research_calls == 0
    assert run.summary["resources"]["research_wall_seconds"] >= 10.0


def test_ensure_prior_spec_keeps_run_result_when_registry_row_has_no_result(tmp_path: Path):
    runner, _ = make_runner(tmp_path)
    parked = make_spec(
        experiment_id=ENSEMBLE_SEED_ID,
        implementation=ImplementationRef(entrypoint=str(tmp_path / "root_candidate.py")),
        parameters={"action": "succeed", "parked": True},
    )
    runner.registry.insert_spec(parked)
    colliding = make_spec(
        experiment_id=ENSEMBLE_SEED_ID,
        implementation=ImplementationRef(entrypoint=str(tmp_path / "root_candidate.py")),
        parameters={"action": "succeed", "parked": False},
    )
    entry = ensure_prior_spec(runner, colliding)
    assert entry.result is not None
    assert entry.result.status == "invalid"
    assert entry.result.failure is not None
    assert entry.result.failure.kind == "id_collision"
    assert runner.registry.peek(ENSEMBLE_SEED_ID).result is None
