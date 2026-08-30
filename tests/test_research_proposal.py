"""ResearchProposal schema validation."""
from __future__ import annotations

import pytest

from research_agent.agent.proposal import ProposalError, ResearchProposal
from research_helpers import make_proposal, make_proposal_payload


def test_valid_proposal_round_trip():
    proposal = make_proposal()
    again = ResearchProposal.from_dict(proposal.to_dict())
    assert again.hypothesis == proposal.hypothesis
    assert again.selected_parent_id == "fm-root"
    assert "argparse" in again.candidate_source


def test_missing_field_rejected():
    payload = make_proposal_payload()
    del payload["hypothesis"]
    with pytest.raises(ProposalError, match="missing fields"):
        ResearchProposal.from_dict(payload)


def test_empty_candidate_source_rejected():
    with pytest.raises(ProposalError, match="candidate_source"):
        ResearchProposal.from_dict(make_proposal_payload(candidate_source="   "))


def test_split_in_parameters_rejected():
    with pytest.raises(ProposalError, match="evaluation split"):
        ResearchProposal.from_dict(
            make_proposal_payload(experiment_parameters={"split": "test"})
        )


def test_invalid_parent_id_rejected():
    with pytest.raises(ProposalError, match="selected_parent_id"):
        ResearchProposal.from_dict(make_proposal_payload(selected_parent_id="bad id"))


def test_null_timeout_uses_default():
    payload = make_proposal_payload()
    payload["timeout_seconds"] = None
    proposal = ResearchProposal.from_dict(payload)
    assert proposal.timeout_seconds == 600.0


def test_null_seed_uses_default():
    payload = make_proposal_payload()
    payload["seed"] = None
    proposal = ResearchProposal.from_dict(payload)
    assert proposal.seed == 0


def test_bad_timeout_type_is_proposal_error():
    with pytest.raises(ProposalError, match="timeout_seconds"):
        ResearchProposal.from_dict(make_proposal_payload(timeout_seconds="nope"))


def test_null_experiment_parameters_defaults_to_empty_object():
    payload = make_proposal_payload()
    payload["experiment_parameters"] = None
    proposal = ResearchProposal.from_dict(payload)
    assert proposal.experiment_parameters == {}
