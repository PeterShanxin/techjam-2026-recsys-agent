"""Research session identity must be unique across persistent sessions."""
from __future__ import annotations

from datetime import datetime, timezone

from research_agent.agent.session import experiment_id_for, new_research_session_id


def test_session_ids_are_unique():
    ids = {new_research_session_id() for _ in range(20)}
    assert len(ids) == 20
    assert all(item.startswith("rs-") for item in ids)


def test_experiment_ids_are_session_scoped_and_ordered():
    now = datetime(2026, 8, 30, 8, 12, 12, tzinfo=timezone.utc)
    session = new_research_session_id(now=now)
    assert session.startswith("rs-20260830T081212Z-")
    assert experiment_id_for(session, 1) == f"{session}-001"
    assert experiment_id_for(session, 2) == f"{session}-002"
    assert experiment_id_for("rs-aaa", 1) != experiment_id_for("rs-bbb", 1)
