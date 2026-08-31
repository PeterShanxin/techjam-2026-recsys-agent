"""Date / recency primitives. Not a temporal model."""
from __future__ import annotations

import datetime


def date_to_ordinal(yyyymmdd: int) -> int:
    value = int(yyyymmdd)
    year, month, day = value // 10000, (value // 100) % 100, value % 100
    return datetime.date(year, month, day).toordinal()


def days_between(later: int, earlier: int) -> int:
    return date_to_ordinal(int(later)) - date_to_ordinal(int(earlier))


def recency_weight(event_date: int, reference_date: int, half_life_days: float = 3.0) -> float:
    """Exponential decay from event_date toward reference_date. Future events get 0."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    delta = days_between(reference_date, event_date)
    if delta < 0:
        return 0.0
    return float(0.5 ** (delta / float(half_life_days)))
