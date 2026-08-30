"""Official Track 2 budget envelope. FakeProvider / zero wall-clock only."""
from __future__ import annotations

import json
from pathlib import Path

from research_agent.evolution import (
    COMPETITION_EPSILON,
    COMPETITION_MAX_EVALUATIONS,
    COMPETITION_PATIENCE,
    COMPETITION_WALL_SECONDS,
    EvolutionConfig,
    EvolutionController,
)
from research_agent.evolution.controller import STOP_WALL
from research_agent.agent import ResearchAgent
from research_agent.llm import FakeProvider
from research_helpers import make_runner, mini_root_spec

ROOT = Path(__file__).resolve().parents[1]


def test_competition_config_matches_official_envelope():
    cfg = EvolutionConfig.competition()
    assert cfg.max_new_evaluations == COMPETITION_MAX_EVALUATIONS == 50
    assert cfg.wall_clock_seconds == COMPETITION_WALL_SECONDS == 6 * 3600
    assert cfg.convergence_epsilon == COMPETITION_EPSILON == 0.002
    assert cfg.convergence_patience == COMPETITION_PATIENCE == 3
    assert cfg.generations == 50


def test_competition_json_loads_to_same_envelope():
    payload = json.loads(
        (ROOT / "configs" / "research" / "competition.json").read_text(encoding="utf-8")
    )
    cfg = EvolutionConfig.from_mapping(payload)
    assert cfg.max_new_evaluations == 50
    assert cfg.wall_clock_seconds == 21600
    assert cfg.convergence_epsilon == 0.002
    assert cfg.convergence_patience == 3


def test_zero_wall_clock_stops_before_new_evaluations(tmp_path: Path):
    runner, _data = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=FakeProvider(script=[]),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=50,
        max_repairs=0,
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id="ev-wall",
    )
    ctl = EvolutionController(
        agent=agent,
        config=EvolutionConfig.competition(
            wall_clock_seconds=0.0,
            include_ensemble_seed=False,
            fill_to_size_on_init=False,
            max_repairs=0,
            experiment_timeout_seconds=30.0,
        ),
    )
    run = ctl.run()
    assert run.stop_reason == STOP_WALL
    assert run.evaluated_offspring == 0
