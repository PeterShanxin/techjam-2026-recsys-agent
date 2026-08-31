"""Split-safe research lab. Instruments, not a hidden winning ranker."""

from .audit import field_inventory
from .capabilities import (
    LAB_CAPABILITIES,
    LAB_CONTRACT,
    LAB_IMPORT,
    LAB_MODULE,
    lab_contract_dict,
)
from .errors import LeakageError, SealedSplitError
from .ranking import GradientFM, UserGroups, user_groups
from .store import ImpressionContext, PairwiseSample, SplitSafeStore, TrainEvent
from .timeutil import date_to_ordinal, days_between, recency_weight

__all__ = [
    "GradientFM",
    "ImpressionContext",
    "LAB_CAPABILITIES",
    "LAB_CONTRACT",
    "LAB_IMPORT",
    "LAB_MODULE",
    "LeakageError",
    "PairwiseSample",
    "SealedSplitError",
    "SplitSafeStore",
    "TrainEvent",
    "UserGroups",
    "date_to_ordinal",
    "days_between",
    "field_inventory",
    "lab_contract_dict",
    "recency_weight",
    "user_groups",
]
