"""Research-space inventory, split-safe lab, and leakage walls. Zero API spend."""
from __future__ import annotations

from pathlib import Path

import pytest

from experiment_helpers import CANDIDATE_SOURCE, write_lab_dataset
from research_agent.agent.data_contract import (
    DataContractError,
    discover_data_contract,
    validate_proposal_data_claims,
)
from research_agent.agent.environment import discover_environment
from research_agent.agent.safety import SafetyError, validate_candidate_source
from research_agent.lab import (
    LeakageError,
    SealedSplitError,
    SplitSafeStore,
    field_inventory,
    recency_weight,
)
from research_helpers import make_proposal


def _dest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "generated"
    dest = root / "rs-lab-001" / "candidate.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest, root


def test_field_inventory_names_history_and_unsafe_stats(tmp_path: Path):
    data_dir = write_lab_dataset(tmp_path)
    records = field_inventory(data_dir)
    names = {item["field"] for item in records}
    assert "user history" in names
    assert "long_view" in names
    assert "unscoped engagement counters" in names
    leaky = next(item for item in records if item["field"] == "unscoped engagement counters")
    assert leaky["leakage_risk"] == "high"
    assert leaky["safe_for_valid_research"] is False
    meta = next(item for item in records if item["field"] == "_inventory_meta")
    assert "log_standard_4_08_to_4_21_pure.csv" in meta["headers"]["files"]
    assert "raw data" not in str(records).lower() or "not a row dump" in str(records).lower()


def test_data_contract_lists_lab_and_sealed_test():
    payload = discover_data_contract().to_dict()
    assert payload["test_sealed"] is True
    lab = payload["lab"]
    assert lab["import"].startswith("from research_agent.lab")
    assert lab["test_sealed"] is True
    assert lab["not_a_ranker"] is True
    names = {item["name"] for item in lab["capabilities"]}
    assert "get_user_history" in names
    assert "build_pairwise_samples" in names
    assert "train_popularity" in names
    assert any("statistic" in item.lower() for item in lab["unavailable_or_unsafe"])


def test_history_and_popularity_use_train_only(tmp_path: Path):
    store = SplitSafeStore(write_lab_dataset(tmp_path))
    history = store.get_user_history("u1")
    videos = {item.video_id for item in history}
    assert videos == {"v1", "v2", "v3", "v4"}
    assert "v9" not in videos
    assert "v_test" not in videos
    assert store.train_popularity("v9") == 0.0
    assert store.train_popularity("v_test") == 0.0
    assert store.train_popularity("v1") == 2.0
    assert store.train_popularity("v1", kind="long_view") == 1.0
    assert store.train_author_affinity("u1", "a1") == 1.0
    stats = store.video_statistics_unscoped("v9")
    assert stats.get("show_cnt") == "999999"
    assert stats.get("leakage_risk") == "high"


def test_validation_and_test_labels_are_blocked(tmp_path: Path):
    store = SplitSafeStore(write_lab_dataset(tmp_path))
    with pytest.raises(LeakageError, match="validation labels"):
        store.labels("valid")
    with pytest.raises(SealedSplitError, match="sealed"):
        store.labels("test")
    with pytest.raises(LeakageError):
        SplitSafeStore(write_lab_dataset(tmp_path / "other"), feature_source="valid")
    rows = store.inference_rows("valid")
    assert len(rows) == 5
    leaked = [row for row in rows if row.video_id == "v9"]
    assert leaked
    payload = leaked[0].to_dict()
    assert "long_view" not in payload
    assert "is_like" not in payload
    assert payload["hourmin"] == "1600"
    assert all(key not in payload for key in ("is_like", "play_time_ms", "long_view"))
    train_y = store.labels("train")
    assert train_y == [1, 0, 1, 0, 0, 1]


def test_pairwise_samples_are_train_only(tmp_path: Path):
    store = SplitSafeStore(write_lab_dataset(tmp_path))
    pairs = store.build_pairwise_samples(max_pairs=20, negatives_per_positive=1, seed=0)
    assert pairs
    videos = {item.pos_video_id for item in pairs} | {item.neg_video_id for item in pairs}
    assert "v9" not in videos
    assert "v_test" not in videos
    assert all(item.provenance == "train" for item in pairs)
    u1 = [item for item in pairs if item.user_id == "u1"]
    assert u1
    assert {item.pos_video_id for item in u1} <= {"v1", "v3"}


def test_recency_future_event_is_zero():
    assert recency_weight(20220420, 20220410) == 0.0
    assert recency_weight(20220410, 20220410) == 1.0
    assert 0.0 < recency_weight(20220408, 20220414, half_life_days=3.0) < 1.0


def test_lab_import_passes_preflight(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = "from research_agent.lab import SplitSafeStore\n" + CANDIDATE_SOURCE
    validate_candidate_source(src, dest, root)
    env = discover_environment()
    assert "research_agent.lab" in env.project_modules
    assert "research_agent.lab" in env.to_dict()["lab_import"]


def test_lab_history_api_may_claim_like_column(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = (
        "from research_agent.lab import SplitSafeStore\n"
        "store = SplitSafeStore('data')\n"
        "for event in store.get_user_history('u1'):\n"
        "    like = (event.aux or {}).get('is_like')\n"
        + CANDIDATE_SOURCE
    )
    proposal = make_proposal(
        hypothesis="Use train is_like from lab history, not valid labels.",
        expected_mechanism="Train-only is_like aux from SplitSafeStore.get_user_history.",
        candidate_source=src,
        required_data_fields=["is_like"],
    )
    validate_proposal_data_claims(proposal, discover_data_contract())
    validate_candidate_source(src, dest, root, proposal=proposal, data_contract=discover_data_contract())


def test_lab_comment_does_not_unlock_like_column(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = (
        "from research_agent.lab import SplitSafeStore\n"
        "# train_aux is_like play_time_ms\n"
        + CANDIDATE_SOURCE
    )
    proposal = make_proposal(
        hypothesis="Soft labels from is_like via a comment, not a lab call.",
        expected_mechanism="Mention train_aux and is_like only.",
        candidate_source=src,
        required_data_fields=["is_like"],
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


def test_fake_lab_method_does_not_unlock_like_column(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = (
        "from research_agent.lab import SplitSafeStore\n"
        "store = SplitSafeStore('data')\n"
        "store.train_behavior('is_like')\n"
        + CANDIDATE_SOURCE
    )
    proposal = make_proposal(
        hypothesis="Soft labels from is_like via a fake train_behavior helper.",
        expected_mechanism="Local train_behavior call plus lab import.",
        candidate_source=src,
        required_data_fields=["is_like"],
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


def test_inference_rows_may_claim_hourmin_not_like(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = (
        "from research_agent.lab import SplitSafeStore\n"
        "store = SplitSafeStore('data')\n"
        "rows = store.inference_rows('valid')\n"
        "hour = rows[0].hourmin\n"
        "stamp = rows[0].time_ms\n"
        + CANDIDATE_SOURCE
    )
    ok = make_proposal(
        hypothesis="Use serving-time hourmin and time_ms from inference_rows.",
        expected_mechanism="Impression context clock, not labels.",
        candidate_source=src,
        required_data_fields=["hourmin", "time_ms"],
    )
    validate_proposal_data_claims(ok, discover_data_contract())
    validate_candidate_source(src, dest, root, proposal=ok, data_contract=discover_data_contract())
    bad = make_proposal(
        hypothesis="Soft labels from is_like via inference_rows.",
        expected_mechanism="inference_rows has no like label.",
        candidate_source=src,
        required_data_fields=["is_like"],
    )
    with pytest.raises((SafetyError, DataContractError), match="unavailable_data_field"):
        validate_proposal_data_claims(bad, discover_data_contract())


def test_lab_import_alone_does_not_unlock_like_column(tmp_path: Path):
    dest, root = _dest(tmp_path)
    src = "from research_agent.lab import SplitSafeStore\n" + CANDIDATE_SOURCE
    proposal = make_proposal(
        hypothesis="Soft labels from is_like without calling a lab field API.",
        expected_mechanism="Mention is_like only.",
        candidate_source=src,
        required_data_fields=["is_like"],
    )
    with pytest.raises((SafetyError, DataContractError), match="unavailable_data_field"):
        validate_proposal_data_claims(proposal, discover_data_contract())


def test_unavailable_fields_still_rejected_without_raw_or_lab(tmp_path: Path):
    dest, root = _dest(tmp_path)
    proposal = make_proposal(
        hypothesis="Soft labels from is_like and play_time_ms.",
        expected_mechanism="Blend long_view with is_like.",
        required_data_fields=["is_like", "play_time_ms"],
    )
    with pytest.raises((SafetyError, Exception), match="unavailable_data_field"):
        validate_candidate_source(
            proposal.candidate_source,
            dest,
            root,
            proposal=proposal,
            data_contract=discover_data_contract(),
        )
