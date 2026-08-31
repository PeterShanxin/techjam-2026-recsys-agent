"""End-to-end: the controller rejects a child that only re-labels its parent's ranking.

FakeProvider only, zero API spend. The mini fixtures elsewhere are a handful of rows, so
the gate is configured off there; here it is switched on explicitly to prove the wiring.
"""
from __future__ import annotations

from pathlib import Path

from evolution_helpers import evolution_proposal
from research_agent.agent import ResearchAgent
from research_agent.evolution import EvolutionConfig, EvolutionController
from research_agent.evolution.identity import NEAR_IDENTITY_VALIDITY
from research_agent.llm import FakeProvider
from research_helpers import make_runner, mini_root_spec

# A candidate that ignores its own config and re-emits a fixed ranking. This is exactly
# the shape of "parent + alpha * residual" once alpha is tuned to zero on validation.
IDENTITY_SOURCE = '''\
import argparse
import json
from pathlib import Path

import numpy as np
from data import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output-scores", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = load(args.data_dir)[args.split]
    np.save(args.output_scores, np.random.default_rng(0).random(len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _controller(tmp_path: Path, script: list, *, min_rows: int) -> EvolutionController:
    runner, _data = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=FakeProvider(script=script),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=2,
        max_repairs=0,
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id="ni-test",
    )
    config = EvolutionConfig(
        population_size=2,
        elite_count=1,
        generations=1,
        max_new_evaluations=2,
        include_ensemble_seed=False,
        near_identity_min_rows=min_rows,
    )
    return EvolutionController(agent=agent, config=config)


def _identity_pair() -> list:
    """Two proposals with different text and params but the same emitted ranking."""
    first = evolution_proposal(
        label="ranker", family="ranking_objective", tags=("listwise",), axes=("objective",)
    )
    first["candidate_source"] = IDENTITY_SOURCE
    second = evolution_proposal(
        label="residual", family="train_affinity", tags=("residual",), axes=("features",)
    )
    second["candidate_source"] = IDENTITY_SOURCE + "\n# alpha tuned to zero\n"
    second["experiment_parameters"] = {"action": "succeed", "alpha": 0.0}
    return [first, second]


def test_identical_ranking_child_is_rejected_as_near_identity(tmp_path: Path):
    ctl = _controller(tmp_path, script=_identity_pair(), min_rows=1)
    run = ctl.run()
    children = [m for m in run.all_members if m.origin == "mutation"]
    assert len(children) == 2
    later = children[-1]
    assert later.status == "success"
    assert later.research_validity == NEAR_IDENTITY_VALIDITY
    assert later.scientific_evidence is False
    assert later.fitness is None  # cannot become an elite
    events = [e for e in ctl.diversity_events if e.get("reason") == NEAR_IDENTITY_VALIDITY]
    assert events and events[0]["rank_change_fraction"] == 0.0


def test_rejected_child_is_marked_in_the_registry(tmp_path: Path):
    ctl = _controller(tmp_path, script=_identity_pair(), min_rows=1)
    run = ctl.run()
    later = [m for m in run.all_members if m.origin == "mutation"][-1]
    entry = ctl.agent.runner.registry.get(later.experiment_id)
    assert entry.decision == "rejected"


def test_gate_off_on_small_splits_keeps_the_old_classification(tmp_path: Path):
    ctl = _controller(tmp_path, script=_identity_pair(), min_rows=10_000)
    run = ctl.run()
    children = [m for m in run.all_members if m.origin == "mutation"]
    assert all(m.research_validity == "hypothesis_tested" for m in children)


def test_distinct_ranking_child_survives(tmp_path: Path):
    """Control: a child that genuinely reorders rows must not be swept up by the gate."""
    script = _identity_pair()
    script[1]["candidate_source"] = IDENTITY_SOURCE.replace(
        "default_rng(0)", "default_rng(12345)"
    )
    ctl = _controller(tmp_path, script=script, min_rows=1)
    run = ctl.run()
    children = [m for m in run.all_members if m.origin == "mutation"]
    assert children[-1].research_validity == "hypothesis_tested"
    assert children[-1].scientific_evidence is True
