"""Research Agent sees the real KuaiRand loader contract. Zero API spend."""
from __future__ import annotations

from pathlib import Path

import pytest

from experiment_helpers import CANDIDATE_SOURCE
from research_agent.agent.accounting import ResourceLedger
from research_agent.agent.constants import FM_ROOT_ID
from research_agent.agent.data_contract import (
    DataContractError,
    discover_data_contract,
    format_data_contract_repair_message,
    validate_proposal_data_claims,
)
from research_agent.agent.fm_root import fm_root_spec
from research_agent.agent.proposal import ResearchProposal
from research_agent.agent.safety import SafetyError, validate_candidate_source
from research_agent.agent.state import build_research_state
from research_agent.experiments import ExperimentRegistry, ExperimentResult, Metrics
from research_helpers import make_proposal, make_proposal_payload, make_runner
from research_agent.agent import ResearchAgent
from research_agent.llm import FakeProvider
from research_helpers import mini_root_spec


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


def _dest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "generated"
    dest = root / "rs-test-001" / "candidate.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest, root


def test_data_contract_matches_starter_load_tuple():
    contract = discover_data_contract()
    payload = contract.to_dict()
    fields = payload["load"]["tuple_fields"]
    by_index = {item["index"]: item["name"] for item in fields}
    assert by_index == {
        0: "date",
        1: "user_id",
        2: "video_id",
        3: "author_id",
        4: "tab",
        5: "duration_ms",
        6: "long_view",
    }
    assert payload["load"]["tuple_length"] == 7
    assert payload["encode"]["fields"] == [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "dur_bucket",
    ]
    assert payload["official_target"] == "long_view"
    assert payload["evaluation"]["task"] == "within-user ranking"
    assert payload["evaluation"]["metrics"] == ["GAUC", "nDCG@5"]
    assert payload["evaluation"]["primary"] == "mean(GAUC, nDCG@5)"
    assert "is_like" in payload["not_available_via_load"]
    assert "play_time_ms" in payload["not_available_via_load"]
    assert "is_click" in payload["not_available_via_load"]
    assert "long_view" in payload["available_via_load"]
    assert "is_like" not in payload["available_via_load"]
    assert "raw data" not in str(payload).lower() or "do not dump" in payload["rule"].lower()
    assert payload["load"]["returns"].startswith("dict")


def test_research_state_includes_data_contract(tmp_path: Path):
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
    contract = payload["data_contract"]
    assert contract["official_target"] == "long_view"
    assert contract["load"]["tuple_fields"][6]["name"] == "long_view"
    assert "is_like" in contract["not_available_via_load"]
    assert "play_time_ms" in contract["not_available_via_load"]
    assert "mechanism" in contract["rule"].lower()


def test_sequential_prompt_includes_data_capabilities(tmp_path: Path):
    runner, _data = make_runner(tmp_path)
    agent = ResearchAgent(
        provider=FakeProvider(script=[make_proposal_payload()]),
        runner=runner,
        model="fake-model",
        thinking_level="medium",
        max_iterations=1,
        max_repairs=0,
        root_spec=mini_root_spec(tmp_path),
        experiment_timeout_seconds=30.0,
        session_id="rs-dc",
    )
    agent.run()
    prompt = agent.provider.calls[0].prompt
    assert "data_contract" in prompt
    assert "is_like" in prompt
    assert "play_time_ms" in prompt
    assert "not_available_via_load" in prompt
    assert "long_view" in prompt


def test_soft_label_claim_without_loader_fields_is_rejected(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = (
        "def extract_column(rows, name):\n"
        "    return None\n"
        "y_soft = extract_column(rows, 'is_like')\n"
        + CANDIDATE_SOURCE
    )
    proposal = make_proposal(
        hypothesis="Soft labels from is_like and play_time_ms watch ratio.",
        expected_mechanism="Blend long_view with is_like when play_time_ms is high.",
        candidate_source=src,
        required_data_fields=["is_like", "play_time_ms"],
    )
    with pytest.raises((SafetyError, DataContractError), match="unavailable_data_field"):
        validate_proposal_data_claims(proposal, discover_data_contract())
    with pytest.raises(SafetyError, match="unavailable_data_field"):
        validate_candidate_source(
            src,
            dest,
            root,
            proposal=proposal,
            data_contract=discover_data_contract(),
        )


def test_loader_only_candidate_is_allowed(tmp_path: Path):
    dest, root = _dest(tmp_path)
    proposal = make_proposal(
        hypothesis="Use official long_view from data.load tuples.",
        expected_mechanism="Train on tuple field 6 (long_view).",
        required_data_fields=["long_view", "user_id"],
    )
    validate_proposal_data_claims(proposal, discover_data_contract())
    validate_candidate_source(
        proposal.candidate_source,
        dest,
        root,
        proposal=proposal,
        data_contract=discover_data_contract(),
    )


def test_raw_csv_reader_may_claim_like_column(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = (
        "import csv\n"
        "path = 'log_standard_4_08_to_4_21_pure.csv'\n"
        "with open(path) as fh:\n"
        "    for row in csv.DictReader(fh):\n"
        "        like = row['is_like']\n"
        + CANDIDATE_SOURCE
    )
    proposal = make_proposal(
        hypothesis="Read is_like from the raw log CSV, not from data.load tuples.",
        expected_mechanism="Auxiliary like labels from log_standard files.",
        candidate_source=src,
        required_data_fields=["is_like"],
    )
    validate_proposal_data_claims(proposal, discover_data_contract())
    validate_candidate_source(
        src,
        dest,
        root,
        proposal=proposal,
        data_contract=discover_data_contract(),
    )


def test_repair_message_names_missing_fields():
    msg = format_data_contract_repair_message(
        fields=("is_like", "play_time_ms"),
        hypothesis="Soft labels from auxiliary behaviors.",
        contract=discover_data_contract(),
    )
    assert "is_like" in msg
    assert "play_time_ms" in msg
    assert "data.load" in msg
    assert "long_view" in msg
    assert "Preserve the original hypothesis" in msg or "reimplement" in msg.lower()


def test_kuairand_splits_match_contract_tuple(kuairand_splits):
    row = kuairand_splits["train"][0]
    assert len(row) == 7
    assert isinstance(row[0], int)
    assert row[6] in (0, 1)
