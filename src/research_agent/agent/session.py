"""Research session identity. Experiment IDs must be unique across sessions."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def new_research_session_id(*, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"rs-{stamp}-{uuid.uuid4().hex[:8]}"


def experiment_id_for(session_id: str, iteration: int) -> str:
    if iteration < 1:
        raise ValueError("research iteration must be >= 1")
    return f"{session_id}-{iteration:03d}"
