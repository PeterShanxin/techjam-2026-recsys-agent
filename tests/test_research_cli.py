"""Research CLI fail-fast. Must not train FM or call Gemini without credentials."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    path = ROOT / "scripts" / "run_research_agent.py"
    spec = importlib.util.spec_from_file_location("run_research_agent_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_missing_key_fails_before_runner(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("ExperimentRunner must not be constructed without credentials")

    monkeypatch.setattr(cli, "ExperimentRunner", boom)
    rc = cli.main(["--iterations", "1"])
    assert rc == 2


def test_cli_fake_provider_skips_credentials(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class DummyRoot:
        result_status = "success"

    class DummyRun:
        root = DummyRoot()

    class DummyAgent:
        session_id = "rs-fake"

        def __init__(self, **_kwargs):
            pass

        def run(self):
            return DummyRun()

    monkeypatch.setattr(cli, "ResearchAgent", DummyAgent)
    monkeypatch.setattr(cli, "ExperimentRunner", lambda **_kwargs: object())
    rc = cli.main(["--provider", "fake", "--iterations", "0"])
    assert rc == 0


def test_cli_dotenv_is_enough(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file-cli\n", encoding="utf-8")

    class DummyRoot:
        result_status = "success"

    class DummyRun:
        root = DummyRoot()

    class DummyAgent:
        session_id = "rs-env"

        def __init__(self, **_kwargs):
            pass

        def run(self):
            return DummyRun()

    captured = {}

    def fake_provider(**kwargs):
        captured["repo_root"] = kwargs.get("repo_root")
        return object()

    monkeypatch.setattr(cli, "GeminiProvider", fake_provider)
    monkeypatch.setattr(cli, "ResearchAgent", DummyAgent)
    monkeypatch.setattr(cli, "ExperimentRunner", lambda **_kwargs: object())
    rc = cli.main(["--iterations", "1"])
    assert rc == 0
    assert captured["repo_root"] == tmp_path
