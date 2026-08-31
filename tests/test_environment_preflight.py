"""Environment capabilities, import preflight, and silent-fallback rejection. No API spend."""
from __future__ import annotations

from pathlib import Path

import pytest

from experiment_helpers import CANDIDATE_SOURCE
from research_agent.agent.accounting import ResourceLedger
from research_agent.agent.constants import FM_ROOT_ID
from research_agent.agent.environment import discover_environment
from research_agent.agent.fm_root import fm_root_spec
from research_agent.agent.safety import SafetyError, validate_candidate_source
from research_agent.agent.state import build_research_state
from research_agent.experiments import ExperimentRegistry, ExperimentResult, Metrics


def _dest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "generated"
    dest = root / "rs-test-001" / "candidate.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest, root


def _success_result(experiment_id: str, primary: float) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        status="success",
        evaluation_split="valid",
        seed=0,
        spec_hash="x",
        wall_seconds=1.0,
        return_code=0,
        run_dir="",
        stdout_path="",
        stderr_path="",
        metrics=Metrics(gauc=primary, ndcg_at_5=primary, primary=primary),
    )


def test_discover_environment_lists_numpy_not_torch():
    env = discover_environment()
    payload = env.to_dict()
    assert payload["python_version"]
    assert payload["platform"]
    assert payload["architecture"]
    assert "numpy" in payload["allowed_third_party"]
    assert "research_agent" in payload["project_modules"]
    assert "torch" in payload["unsupported_or_unavailable"]
    assert "numpy" not in payload["unsupported_or_unavailable"]
    rule = payload["rule"].lower()
    assert "must stay within" in rule or "must use only" in rule
    assert "fail explicitly" in rule
    assert "silently" in rule


def test_research_state_includes_environment_capabilities(tmp_path: Path):
    registry = ExperimentRegistry(tmp_path / "registry.sqlite")
    spec = fm_root_spec()
    registry.insert_spec(spec)
    registry.upsert_result(_success_result(FM_ROOT_ID, 0.6015))
    state = build_research_state(
        registry=registry,
        ledger=ResourceLedger(),
        iteration=1,
        max_iterations=3,
        remaining_wall_seconds=100.0,
        parent_source="print('parent')",
        selected_parent_id=FM_ROOT_ID,
        repo_root=tmp_path,
    )
    payload = state.to_dict()
    env = payload["environment"]
    assert env["python_version"]
    assert "numpy" in env["allowed_third_party"]
    assert "torch" in env["unsupported_or_unavailable"]
    assert "fail explicitly" in env["rule"].lower()


def test_stdlib_and_numpy_imports_allowed(tmp_path: Path):
    dest, root = _dest(tmp_path)
    validate_candidate_source(CANDIDATE_SOURCE, dest, root)


def test_research_agent_lab_import_allowed(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = "from research_agent.lab import SplitSafeStore, recency_weight\n" + CANDIDATE_SOURCE
    validate_candidate_source(src, dest, root)


def test_unsupported_torch_import_rejected(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = "import torch\n" + CANDIDATE_SOURCE
    with pytest.raises(SafetyError, match=r"unsupported_dependency:\s*torch"):
        validate_candidate_source(src, dest, root)


def test_from_torch_import_rejected(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = "from torch import nn\n" + CANDIDATE_SOURCE
    with pytest.raises(SafetyError, match=r"unsupported_dependency:\s*torch"):
        validate_candidate_source(src, dest, root)


def test_silent_torch_fallback_to_fm_rejected(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = (
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    from baseline import FM\n"
        + CANDIDATE_SOURCE
    )
    with pytest.raises(SafetyError, match="silent_dependency_fallback"):
        validate_candidate_source(src, dest, root)
