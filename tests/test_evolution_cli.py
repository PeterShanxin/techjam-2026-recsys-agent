"""Evolution CLI fail-fast. Zero API spend."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    path = ROOT / "scripts" / "run_evolution.py"
    spec = importlib.util.spec_from_file_location("run_evolution_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evolution_cli_missing_key_fails_before_runner(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("ExperimentRunner must not be constructed without credentials")

    monkeypatch.setattr(cli, "ExperimentRunner", boom)
    rc = cli.main(["--generations", "0"])
    assert rc == 2


def test_evolution_cli_fake_provider_skips_credentials(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class DummyRun:
        stop_reason = "generation_limit"
        evaluated_offspring = 0
        elites = []
        population = type("P", (), {"members": []})()
        trace_dir = tmp_path
        summary = {"lineage": ""}

    class DummyController:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return DummyRun()

    class DummyAgent:
        session_id = "ev-fake"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(cli, "ResearchAgent", DummyAgent)
    monkeypatch.setattr(cli, "EvolutionController", DummyController)
    monkeypatch.setattr(cli, "ExperimentRunner", lambda **_kwargs: object())
    rc = cli.main(["--provider", "fake", "--generations", "0"])
    assert rc == 0


def test_sequential_control_uses_independent_runs_dir(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    seen_dirs = []

    class DummyElite:
        experiment_id = "child-1"
        fitness = 0.61
        metrics = {"primary": 0.61}

        def to_dict(self):
            return {"experiment_id": self.experiment_id, "fitness": self.fitness}

    class DummyRun:
        stop_reason = "generation_limit"
        evaluated_offspring = 3
        elites = [DummyElite()]
        population = type("P", (), {"members": [type("M", (), {"status": "success"})()]})()
        trace_dir = tmp_path
        summary = {"lineage": "", "resources": {}}
        all_members = [type("M", (), {"research_family": "ranking_loss"})()]

    class DummyController:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return DummyRun()

    class DummyAgent:
        session_id = "ev-fake"
        max_iterations = None

        def __init__(self, **kwargs):
            DummyAgent.max_iterations = kwargs.get("max_iterations")

        def run(self):
            return type("SR", (), {"summary": {"session_id": "seq-control"}})()

    def capture_runner(**kwargs):
        seen_dirs.append(kwargs.get("runs_dir"))
        return object()

    monkeypatch.setattr(cli, "ResearchAgent", DummyAgent)
    monkeypatch.setattr(cli, "EvolutionController", DummyController)
    monkeypatch.setattr(cli, "ExperimentRunner", capture_runner)
    priors = []

    def fake_priors(agent, **_kwargs):
        priors.append(agent.max_iterations if hasattr(agent, "max_iterations") else 3)
        return object(), object()

    monkeypatch.setattr(cli, "ensure_matched_starting_seeds", fake_priors)
    rc = cli.main(["--provider", "fake", "--generations", "0", "--sequential-control"])
    assert rc == 0
    assert priors == [3]
    assert len(seen_dirs) == 2
    assert seen_dirs[0] == tmp_path / "runs"
    assert seen_dirs[1] == tmp_path / "runs" / "sequential-control"
    assert seen_dirs[1] != seen_dirs[0]
    assert (tmp_path / "sequential_control.json").is_file()
    payload = json.loads((tmp_path / "sequential_control.json").read_text(encoding="utf-8"))
    assert payload["starting_seeds"] == ["fm-root", "fm-ensemble-3seed"]
    assert payload["new_evaluations"] == 3


def test_competition_flag_uses_official_envelope(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    seen = {}

    class DummyRun:
        stop_reason = "wall_clock_budget"
        evaluated_offspring = 0
        elites = []
        population = type("P", (), {"members": []})()
        trace_dir = tmp_path
        summary = {"lineage": ""}

    class DummyController:
        def __init__(self, **kwargs):
            seen["config"] = kwargs.get("config")

        def run(self):
            return DummyRun()

    class DummyAgent:
        session_id = "ev-comp"

        def __init__(self, **kwargs):
            DummyAgent.max_iterations = kwargs.get("max_iterations")
            DummyAgent.wall_clock_seconds = kwargs.get("wall_clock_seconds")

    monkeypatch.setattr(cli, "ResearchAgent", DummyAgent)
    monkeypatch.setattr(cli, "EvolutionController", DummyController)
    monkeypatch.setattr(cli, "ExperimentRunner", lambda **_kwargs: object())
    rc = cli.main(["--provider", "fake", "--competition"])
    assert rc == 0
    config = seen["config"]
    assert config.max_new_evaluations == 50
    assert config.generations == 50
    assert config.wall_clock_seconds == 21600.0
    assert config.convergence_epsilon == 0.002
    assert config.convergence_patience == 3
    assert DummyAgent.max_iterations == 50
    assert DummyAgent.wall_clock_seconds == 21600.0


def test_competition_sequential_control_inherits_wall(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    walls = []

    class DummyElite:
        experiment_id = "child-1"
        fitness = 0.61
        metrics = {"primary": 0.61}

        def to_dict(self):
            return {"experiment_id": self.experiment_id, "fitness": self.fitness}

    class DummyRun:
        stop_reason = "wall_clock_budget"
        evaluated_offspring = 2
        elites = [DummyElite()]
        population = type("P", (), {"members": [type("M", (), {"status": "success"})()]})()
        trace_dir = tmp_path
        summary = {"lineage": "", "resources": {}}
        all_members = [type("M", (), {"research_family": "ensemble"})()]

    class DummyController:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return DummyRun()

    class DummyAgent:
        session_id = "ev-comp-seq"

        def __init__(self, **kwargs):
            walls.append(kwargs.get("wall_clock_seconds"))

        def run(self):
            return type("SR", (), {"summary": {"session_id": "seq-control"}})()

    monkeypatch.setattr(cli, "ResearchAgent", DummyAgent)
    monkeypatch.setattr(cli, "EvolutionController", DummyController)
    monkeypatch.setattr(cli, "ExperimentRunner", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "ensure_matched_starting_seeds", lambda *_a, **_k: (object(), object()))
    rc = cli.main(["--provider", "fake", "--competition", "--sequential-control"])
    assert rc == 0
    assert walls == [21600.0, 21600.0]
