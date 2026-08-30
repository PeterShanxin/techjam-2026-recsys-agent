"""EvolutionController: population, lineage, budgets, operators. FakeProvider only."""
from __future__ import annotations

from pathlib import Path

import pytest
from evolution_helpers import evolution_proposal, make_member
from research_agent.agent import ResearchAgent, UnusableRootError
from research_agent.evolution import EvolutionConfig, EvolutionController, Population
from research_agent.evolution.lineage import format_lineage, lineage_forest
from research_agent.llm import FakeProvider
from research_helpers import make_runner, mini_root_spec
from conftest import EVALUATE_SHA256


def _controller(tmp_path: Path, script: list, **config_kw) -> EvolutionController:
    runner, _data = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=FakeProvider(script=script),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=config_kw.get("max_new_evaluations", 6),
        max_repairs=config_kw.pop("max_repairs", 2),
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id=config_kw.pop("session_id", "ev-test"),
    )
    config = EvolutionConfig(
        population_size=config_kw.get("population_size", 4),
        elite_count=config_kw.get("elite_count", 2),
        generations=config_kw.get("generations", 1),
        max_new_evaluations=config_kw.get("max_new_evaluations", 6),
        include_ensemble_seed=False,
        fill_to_size_on_init=config_kw.get("fill_to_size_on_init", False),
        token_budget=config_kw.get("token_budget"),
        wall_clock_seconds=config_kw.get("wall_clock_seconds"),
        convergence_epsilon=config_kw.get("convergence_epsilon", 0.002),
        convergence_patience=config_kw.get("convergence_patience", 3),
        prefer_crossover_from_generation=config_kw.get("prefer_crossover_from_generation", 2),
    )
    return EvolutionController(agent=agent, config=config)


def test_population_initialization_starts_from_fm_root(tmp_path: Path):
    ctl = _controller(tmp_path, script=[], generations=0, max_new_evaluations=0)
    run = ctl.run()
    assert [m.experiment_id for m in run.population.members] == ["fm-root"]
    assert run.population.members[0].origin == "baseline"
    assert run.population.members[0].parent_ids == ()
    assert run.stop_reason in {"generation_limit", "evaluation_budget"}


def test_mutation_has_one_parent(tmp_path: Path):
    ctl = _controller(
        tmp_path,
        script=[evolution_proposal(label="m1", family="ranking_loss")],
        population_size=2,
        elite_count=1,
        generations=1,
        max_new_evaluations=1,
    )
    run = ctl.run()
    child = next(m for m in run.population.members if m.experiment_id != "fm-root")
    assert child.origin == "mutation"
    assert len(child.parent_ids) == 1
    assert child.parent_ids[0] == "fm-root"
    spec = ctl.agent.runner.registry.get(child.experiment_id).spec
    assert spec.origin == "mutation"
    assert spec.parent_ids == ("fm-root",)
    assert spec.evaluation_split == "valid"
    assert spec.allow_test_split is False


def test_crossover_has_two_parents_when_compatible(tmp_path: Path):
    script = [
        evolution_proposal(label="loss", family="ranking_loss", tags=("bpr",), axes=("objective",)),
        evolution_proposal(
            label="cross",
            family="hybrid",
            tags=("bpr", "bagging"),
            axes=("objective", "ensembling"),
            operator="crossover",
            crossover_compatible=True,
            parent_a_component="ranking loss",
            parent_b_component="ensemble robustness",
            selected_parent_id="fm-root",
        ),
    ]
    ctl = _controller(
        tmp_path,
        script=script,
        population_size=3,
        elite_count=2,
        generations=2,
        max_new_evaluations=2,
        prefer_crossover_from_generation=2,
    )
    run = ctl.run()
    crossed = [m for m in run.all_members if m.origin == "crossover"]
    assert len(crossed) == 1
    assert len(crossed[0].parent_ids) == 2
    spec = ctl.agent.runner.registry.get(crossed[0].experiment_id).spec
    assert spec.origin == "crossover"
    assert len(spec.parent_ids) == 2


def test_incompatible_crossover_falls_back_to_mutation(tmp_path: Path):
    script = [
        evolution_proposal(label="loss", family="ranking_loss"),
        evolution_proposal(
            label="nope",
            family="hybrid",
            operator="crossover",
            crossover_compatible=False,
            crossover_inappropriate_reason="Loss change and ensemble change conflict on the training loop.",
            parent_a_component="loss",
            parent_b_component="ensemble",
        ),
        evolution_proposal(label="fallback", family="optimization", tags=("lr",), axes=("optimization",)),
    ]
    ctl = _controller(
        tmp_path,
        script=script,
        population_size=3,
        elite_count=2,
        generations=2,
        max_new_evaluations=2,
        prefer_crossover_from_generation=2,
    )
    run = ctl.run()
    assert [m.origin for m in run.all_members if m.generation == 2] == ["mutation"]
    assert any("incompatible_crossover" in (item.get("reason") or "") for item in run.diversity_events) or any(
        rec.get("fallback") == "mutation" for rec in run.operator_decisions
    )


def test_failed_crossover_proposal_falls_back_to_mutation(tmp_path: Path):
    script = [
        evolution_proposal(label="loss", family="ranking_loss"),
        {"reflection": "not a full proposal"},
        evolution_proposal(label="fallback", family="optimization", tags=("lr",), axes=("optimization",)),
    ]
    ctl = _controller(
        tmp_path,
        script=script,
        population_size=3,
        elite_count=2,
        generations=2,
        max_new_evaluations=2,
        prefer_crossover_from_generation=2,
        max_repairs=0,
    )
    run = ctl.run()
    assert [m.origin for m in run.all_members if m.generation == 2] == ["mutation"]
    assert any(rec.get("reason") == "crossover_proposal_failed" for rec in run.operator_decisions)
    assert any(rec.get("fallback") == "mutation" for rec in run.operator_decisions)


def test_exact_duplicate_is_suppressed(tmp_path: Path):
    first = evolution_proposal(
        label="same",
        family="optimization",
        tags=("lr",),
        axes=("optimization",),
        hypothesis="exact-source duplicate A",
    )
    duplicate = dict(first)
    third = evolution_proposal(
        label="other",
        family="ranking_loss",
        hypothesis="distinct ranking-loss branch",
    )
    ctl = _controller(
        tmp_path,
        script=[first, duplicate, third],
        population_size=3,
        elite_count=1,
        generations=0,
        max_new_evaluations=3,
        fill_to_size_on_init=True,
    )
    run = ctl.run()
    reasons = [item.get("reason") for item in run.diversity_events]
    assert "spec_hash" in reasons or "source_fingerprint" in reasons
    executed = [m for m in run.all_members if m.generation >= 0 and m.origin == "mutation"]
    ids = [m.hypothesis for m in executed]
    assert ids.count(first["hypothesis"]) <= 1


def test_semantic_duplicate_is_suppressed(tmp_path: Path):
    first = evolution_proposal(label="ens-a", family="ensemble", tags=("bagging",), axes=("ensembling",))
    twin = evolution_proposal(label="ens-b", family="ensemble", tags=("bagging",), axes=("ensembling",))
    other = evolution_proposal(label="loss", family="ranking_loss", tags=("bpr",), axes=("objective",))
    ctl = _controller(
        tmp_path,
        script=[first, twin, other],
        population_size=3,
        elite_count=1,
        generations=0,
        max_new_evaluations=3,
        fill_to_size_on_init=True,
    )
    run = ctl.run()
    assert any(item.get("reason") == "semantic_signature" for item in run.diversity_events)


def test_invalid_candidates_cannot_become_elite(tmp_path: Path):
    ctl = _controller(
        tmp_path,
        script=[evolution_proposal(label="crash", action="fail", family="ranking_loss")],
        population_size=2,
        elite_count=1,
        generations=1,
        max_new_evaluations=1,
    )
    run = ctl.run()
    elite_ids = [m.experiment_id for m in run.elites]
    assert "fm-root" in elite_ids
    failed = [m for m in run.all_members if m.status == "failed"]
    assert failed
    assert failed[0].experiment_id not in elite_ids
    assert failed[0].scientific_evidence is False


def test_failed_implementation_is_not_negative_science(tmp_path: Path):
    ctl = _controller(
        tmp_path,
        script=[evolution_proposal(label="crash", action="fail", hypothesis="BPR pairwise ranking")],
        population_size=2,
        elite_count=1,
        generations=1,
        max_new_evaluations=1,
    )
    run = ctl.run()
    failed = next(m for m in run.all_members if m.status == "failed")
    assert failed.scientific_evidence is False
    assert failed.research_validity == "implementation_failure"
    assert "BPR pairwise ranking" not in " ".join(run.negative_scientific_hypotheses)


def test_failed_fm_root_stops_evolution(tmp_path: Path):
    from research_agent.experiments import ExperimentSpec

    fail_spec = mini_root_spec(tmp_path)
    payload = fail_spec.to_dict()
    payload["parameters"] = {"action": "fail"}
    payload.pop("spec_hash", None)
    ctl = _controller(tmp_path, script=[], generations=0, max_new_evaluations=0)
    ctl.agent.root_spec = ExperimentSpec.from_dict(payload)
    with pytest.raises(UnusableRootError):
        ctl.run()
    assert ctl.agent.provider.calls == []


def test_parameter_only_mutation_is_not_semantic_noop(tmp_path: Path):
    from experiment_helpers import CANDIDATE_SOURCE
    from research_helpers import make_proposal_payload

    param_only = make_proposal_payload(
        hypothesis="Same candidate source, different embedding size.",
        candidate_source=CANDIDATE_SOURCE,
        experiment_parameters={"action": "succeed", "k": 32},
        research_family="factorization_machine",
        mechanism_tags=["fm"],
        changed_axes=["capacity"],
        what_changed="k=32 instead of the parent default.",
    )
    ctl = _controller(
        tmp_path,
        script=[param_only],
        population_size=2,
        elite_count=1,
        generations=1,
        max_new_evaluations=1,
    )
    ctl.agent.workspace.load_parent_source = lambda spec, repo: CANDIDATE_SOURCE  # type: ignore[method-assign]
    run = ctl.run()
    child = next(m for m in run.all_members if m.origin == "mutation")
    assert child.research_validity == "hypothesis_tested"
    assert child.scientific_evidence is True


def test_parameter_only_mutation_of_non_root_is_not_suppressed(tmp_path: Path):
    first = evolution_proposal(label="loss", family="ranking_loss")
    param_only = dict(first)
    param_only["hypothesis"] = "Same ranking-loss source, different learning rate."
    param_only["experiment_parameters"] = {**dict(first["experiment_parameters"]), "lr": 0.01}
    param_only["what_changed"] = "lr=0.01 on the same candidate source."
    param_only["changed_axes"] = ["optimization"]
    ctl = _controller(
        tmp_path,
        script=[first, param_only],
        population_size=3,
        elite_count=1,
        generations=0,
        max_new_evaluations=3,
        fill_to_size_on_init=True,
    )
    run = ctl.run()
    children = [m for m in run.all_members if m.origin == "mutation"]
    assert len(children) == 2
    assert all(m.research_validity == "hypothesis_tested" for m in children)
    fps = {m.source_fingerprint for m in children}
    assert len(fps) == 1


def test_marked_elites_are_fitness_ranked(tmp_path: Path):
    ctl = _controller(tmp_path, script=[], generations=0, max_new_evaluations=0)
    members = [
        make_member(
            experiment_id="weak",
            metrics={"GAUC": 0.50, "nDCG@5": 0.50, "primary": 0.50},
            runtime_seconds=1.0,
            research_validity="hypothesis_tested",
        ),
        make_member(
            experiment_id="strong",
            metrics={"GAUC": 0.70, "nDCG@5": 0.70, "primary": 0.70},
            runtime_seconds=8.0,
            research_validity="hypothesis_tested",
        ),
        make_member(
            experiment_id="mid",
            metrics={"GAUC": 0.60, "nDCG@5": 0.60, "primary": 0.60},
            runtime_seconds=2.0,
            research_validity="hypothesis_tested",
        ),
    ]
    marked = ctl._mark_elites(members)
    elites = [m for m in marked if m.selection == "elite"]
    assert [m.experiment_id for m in elites] == ["strong", "mid"]
    finished = ctl._finish(Population(members=marked))
    assert [m.experiment_id for m in finished.elites] == ["strong", "mid"]


def test_semantic_noop_does_not_become_elite(tmp_path: Path):
    from experiment_helpers import CANDIDATE_SOURCE
    from research_helpers import make_proposal_payload

    noop = make_proposal_payload(
        hypothesis="Claimed change that copies the parent.",
        candidate_source=CANDIDATE_SOURCE,
        experiment_parameters={"action": "succeed"},
        research_family="factorization_machine",
        mechanism_tags=["fm"],
        changed_axes=["optimization"],
        what_changed="Nothing material.",
    )
    ctl = _controller(
        tmp_path,
        script=[noop],
        population_size=2,
        elite_count=1,
        generations=1,
        max_new_evaluations=1,
    )
    # Force parent source to match candidate so the controller marks a no-op.
    ctl.agent.workspace.load_parent_source = lambda spec, repo: CANDIDATE_SOURCE  # type: ignore[method-assign]
    run = ctl.run()
    noops = [m for m in run.all_members if m.research_validity == "semantic_noop"]
    assert noops
    assert all(m.experiment_id not in {e.experiment_id for e in run.elites} for m in noops)


def test_lineage_reconstruction_from_registry(tmp_path: Path):
    script = [
        evolution_proposal(label="a", family="ranking_loss"),
        evolution_proposal(
            label="cross",
            family="hybrid",
            operator="crossover",
            crossover_compatible=True,
            parent_a_component="A",
            parent_b_component="B",
        ),
    ]
    ctl = _controller(
        tmp_path,
        script=script,
        population_size=3,
        elite_count=2,
        generations=2,
        max_new_evaluations=2,
        prefer_crossover_from_generation=2,
    )
    run = ctl.run()
    forest = lineage_forest(ctl.agent.runner.registry)
    text = format_lineage(forest)
    assert "fm-root" in text
    child_ids = [m.experiment_id for m in run.all_members if m.origin != "baseline"]
    for cid in child_ids:
        assert cid in text
    reconstructed = EvolutionController.reconstruct(run.trace_dir, ctl.agent.runner.registry)
    assert [m.experiment_id for m in reconstructed.population.members] == [
        m.experiment_id for m in run.population.members
    ]


def test_evaluation_budget_stop_reason(tmp_path: Path):
    ctl = _controller(
        tmp_path,
        script=[evolution_proposal(label="only")],
        population_size=4,
        elite_count=2,
        generations=2,
        max_new_evaluations=1,
    )
    run = ctl.run()
    assert run.stop_reason == "evaluation_budget"
    assert run.evaluated_offspring <= 1


def test_token_budget_stop_reason(tmp_path: Path):
    ctl = _controller(
        tmp_path,
        script=[evolution_proposal(label="t1"), evolution_proposal(label="t2")],
        population_size=4,
        elite_count=2,
        generations=2,
        max_new_evaluations=6,
        token_budget=100,
    )
    run = ctl.run()
    assert run.stop_reason == "token_budget"


def test_convergence_counter_and_stop(tmp_path: Path):
    script = [
        evolution_proposal(label=f"g{i}", family="optimization", tags=("lr",), axes=("optimization",))
        for i in range(6)
    ]
    ctl = _controller(
        tmp_path,
        script=script,
        population_size=2,
        elite_count=1,
        generations=5,
        max_new_evaluations=6,
        convergence_epsilon=1.0,
        convergence_patience=2,
        prefer_crossover_from_generation=99,
    )
    run = ctl.run()
    assert run.stop_reason == "converged"
    assert run.stagnation >= 2


def test_generation_limit_stop_reason(tmp_path: Path):
    ctl = _controller(
        tmp_path,
        script=[evolution_proposal(label="a"), evolution_proposal(label="b")],
        population_size=3,
        elite_count=1,
        generations=1,
        max_new_evaluations=6,
        prefer_crossover_from_generation=99,
        fill_to_size_on_init=False,
    )
    run = ctl.run()
    assert run.stop_reason == "generation_limit"


def test_phase3_research_agent_still_works_alongside_evolution(tmp_path: Path):
    from research_helpers import make_proposal_payload

    runner, _data = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=FakeProvider(script=[make_proposal_payload(hypothesis="phase3 still")]),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=1,
        max_repairs=0,
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id="rs-still",
    )
    run = agent.run()
    assert run.iterations[0].result_status == "success"
    assert run.iterations[0].record["research_validity"] == "hypothesis_tested"


def test_fitness_never_reads_test_split_rows(tmp_path: Path):
    ctl = _controller(tmp_path, script=[], generations=0, max_new_evaluations=0)
    run = ctl.run()
    for member in run.all_members:
        assert member.evaluation_split != "test"
        if member.fitness is not None:
            assert member.evaluation_split == "valid"


def test_evaluator_hash_unchanged():
    from conftest import evaluate_py_canonical_bytes
    import hashlib

    digest = hashlib.sha256(evaluate_py_canonical_bytes()).hexdigest()
    assert digest == EVALUATE_SHA256
