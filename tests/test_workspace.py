"""Generated workspace, syntax checks, and evaluator-tamper rejection."""
from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.agent.safety import SafetyError, validate_candidate_source
from research_agent.agent.workspace import CandidateWorkspace, unified_diff
from experiment_helpers import CANDIDATE_SOURCE


def test_materialize_fingerprints_and_diffs(tmp_path: Path):
    workspace = CandidateWorkspace(tmp_path / "generated")
    parent = CANDIDATE_SOURCE
    child = CANDIDATE_SOURCE + "\n# mutation\n"
    result = workspace.materialize(
        experiment_id="ra-001",
        source=child,
        parent_source=parent,
        repo_root=tmp_path,
    )
    assert result.dest.is_file()
    assert result.dest.read_text(encoding="utf-8") == child
    assert result.fingerprint
    assert "# mutation" in result.diff_vs_parent
    assert result.implementation.entrypoint.endswith("candidate.py")
    assert "evaluate.py" not in result.dest.as_posix()


def test_syntax_error_rejected(tmp_path: Path):
    dest = tmp_path / "generated" / "ra-001" / "candidate.py"
    with pytest.raises(SafetyError, match="syntax"):
        validate_candidate_source("def (", dest, tmp_path / "generated")


def test_missing_cli_rejected(tmp_path: Path):
    dest = tmp_path / "generated" / "ra-001" / "candidate.py"
    src = "print('no argparse flags here')\n"
    with pytest.raises(SafetyError, match="CLI"):
        validate_candidate_source(src, dest, tmp_path / "generated")


def test_evaluator_write_rejected(tmp_path: Path):
    dest = tmp_path / "generated" / "ra-001" / "candidate.py"
    src = '''
import argparse
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir")
ap.add_argument("--split")
ap.add_argument("--output-scores")
ap.add_argument("--seed")
ap.add_argument("--config")
Path("starter/kuairand/evaluate.py").write_text("hacked")
'''
    with pytest.raises(SafetyError, match="evaluate.py"):
        validate_candidate_source(src, dest, tmp_path / "generated")


def test_path_escape_rejected(tmp_path: Path):
    dest = tmp_path / "outside" / "candidate.py"
    with pytest.raises(SafetyError, match="escapes workspace"):
        validate_candidate_source(CANDIDATE_SOURCE, dest, tmp_path / "generated")


def test_unified_diff_readable():
    diff = unified_diff("a\n", "b\n", from_name="parent.py", to_name="candidate.py")
    assert "-a" in diff
    assert "+b" in diff
