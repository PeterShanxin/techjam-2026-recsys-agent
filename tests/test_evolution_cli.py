"""Evolution CLI fail-fast. Zero API spend."""
from __future__ import annotations

import importlib.util
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
