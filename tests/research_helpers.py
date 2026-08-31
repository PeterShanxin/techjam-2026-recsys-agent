"""Shared builders for Phase 3 research-agent tests."""
from __future__ import annotations

from pathlib import Path

from experiment_helpers import CANDIDATE_SOURCE, make_spec, write_candidate, write_mini_dataset
from research_agent.agent.proposal import ResearchProposal
from research_agent.experiments import ExperimentRunner, ImplementationRef


def make_proposal_payload(**overrides) -> dict:
    payload = {
        "reflection": "Elite is the FM root. No later evidence yet.",
        "observation": "Validation primary is the official FM number.",
        "hypothesis": "A tiny candidate still has to honor the score CLI contract.",
        "rationale": "Harness wiring must stay valid while we mutate later.",
        "expected_mechanism": "Scores remain finite and ordered by split row.",
        "selected_parent_id": "fm-root",
        "mutation_summary": "Keep the CLI and write deterministic-length scores.",
        "expected_effect": "Valid experiment; not expected to beat FM.",
        "candidate_source": CANDIDATE_SOURCE,
        "experiment_parameters": {"action": "succeed"},
        "risk_notes": "Plumbing mutation only.",
        "abandon_or_continue_reasoning": "continue with sequential search",
        "seed": 0,
        "timeout_seconds": 30.0,
        # Required since the diversity signature became load-bearing: an empty family
        # collapses duplicate detection and crossover parent choice to a no-op.
        "research_family": "harness_plumbing",
        "mechanism_tags": ["cli_contract"],
        "changed_axes": ["plumbing"],
    }
    payload.update(overrides)
    return payload


def make_proposal(**overrides) -> ResearchProposal:
    return ResearchProposal.from_dict(make_proposal_payload(**overrides))


def make_runner(tmp_path: Path) -> tuple[ExperimentRunner, Path]:
    data_dir = write_mini_dataset(tmp_path)
    write_candidate(tmp_path / "root_candidate.py")
    runner = ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
        allow_test=False,
    )
    return runner, data_dir


def mini_root_spec(tmp_path: Path, experiment_id: str = "fm-root"):
    return make_spec(
        experiment_id=experiment_id,
        implementation=ImplementationRef(entrypoint=str(tmp_path / "root_candidate.py")),
        origin="baseline",
        parent_ids=(),
        hypothesis="mini root",
        rationale="tests",
        parameters={"action": "succeed"},
        timeout_seconds=30.0,
        tags=("test", "root"),
    )
