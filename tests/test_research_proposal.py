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
