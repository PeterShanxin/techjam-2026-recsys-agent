"""Compact research-space inventory. Headers and meanings, not row dumps."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from research_agent.evaluation.official import STARTER

from .store import AUX_FIELDS, CONTEXT_RAW_FIELDS, SPLIT_DATES, VIDEO_STATS

_KNOWN: tuple[dict[str, Any], ...] = (
    {
        "source": "log_standard_4_08_to_4_21_pure.csv",
        "field": "user_id",
        "meaning": "user identifier",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "data.load + lab",
    },
    {
        "source": "log_standard_*",
        "field": "video_id",
        "meaning": "video identifier",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "data.load + lab",
    },
    {
        "source": "log_standard_*",
        "field": "date",
        "meaning": "YYYYMMDD impression date",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "data.load + lab",
    },
    {
        "source": "log_standard_*",
        "field": "tab",
        "meaning": "recommendation tab",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "data.load + lab",
    },
    {
        "source": "log_standard_*",
        "field": "duration_ms",
        "meaning": "video duration milliseconds",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "data.load + lab",
    },
    {
        "source": "log_standard_*",
        "field": "long_view",
        "meaning": "official binary target",
        "train_time": True,
        "inference_time": False,
        "safe_for_valid_research": False,
        "leakage_risk": "high",
        "access": "train labels via lab.labels('train'); valid/test sealed from features",
    },
    {
        "source": "video_features_basic_pure.csv",
        "field": "author_id",
        "meaning": "uploader id joined by data.load",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "data.load join + lab video catalog",
    },
    {
        "source": "log_standard_*",
        "field": "hourmin",
        "meaning": "impression clock",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "medium",
        "access": "raw CSV / lab impression context",
    },
    {
        "source": "log_standard_*",
        "field": "play_time_ms",
        "meaning": "watch time of this impression",
        "train_time": True,
        "inference_time": False,
        "safe_for_valid_research": False,
        "leakage_risk": "high",
        "access": "raw train CSV / lab.train_aux",
    },
    {
        "source": "log_standard_*",
        "field": "is_like",
        "meaning": "like on this impression",
        "train_time": True,
        "inference_time": False,
        "safe_for_valid_research": False,
        "leakage_risk": "high",
        "access": "raw train CSV / lab.train_aux",
    },
    {
        "source": "user_features_pure.csv",
        "field": "user catalog",
        "meaning": "static user metadata",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "lab.get_user_features",
    },
    {
        "source": "video_features_basic_pure.csv",
        "field": "video catalog",
        "meaning": "static video metadata besides author_id",
        "train_time": True,
        "inference_time": True,
        "safe_for_valid_research": True,
        "leakage_risk": "low",
        "access": "lab.get_video_features",
    },
    {
        "source": VIDEO_STATS,
        "field": "unscoped engagement counters",
        "meaning": "undated show/play/like/share counts",
        "train_time": "unknown",
        "inference_time": "unknown",
        "safe_for_valid_research": False,
        "leakage_risk": "high",
        "access": "lab.video_statistics_unscoped (explicit leaky API)",
    },
    {
        "source": "train interactions",
        "field": "user history",
        "meaning": "prior train events for a user",
        "train_time": True,
        "inference_time": "train history only",
        "safe_for_valid_research": True,
        "leakage_risk": "high if valid/test mixed in",
        "access": "lab.get_user_history",
    },
)


def field_inventory(data_dir: str | Path | None = None) -> list[dict[str, Any]]:
    records = [dict(item) for item in _KNOWN]
    if data_dir is None:
        return records
    root = Path(data_dir)
    if not root.is_dir():
        return records
    observed: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.csv")):
        observed[path.name] = _header(path)
    for item in records:
        source = str(item.get("source") or "")
        if source.endswith(".csv") and source in observed:
            item["observed_in_header"] = item.get("field") in observed[source] or item["field"] in {
                "user catalog",
                "video catalog",
                "unscoped engagement counters",
            }
    item_headers = {
        "files": {name: cols[:24] for name, cols in observed.items()},
        "official_split_dates": {k: list(v) for k, v in SPLIT_DATES.items()},
        "aux_fields": list(AUX_FIELDS),
        "context_raw_fields": list(CONTEXT_RAW_FIELDS),
        "starter_data_py": str((STARTER / "data.py").as_posix()),
    }
    records.append(
        {
            "source": "local_headers",
            "field": "_inventory_meta",
            "meaning": "Observed CSV headers only. Not a row dump.",
            "headers": item_headers,
            "safe_for_valid_research": True,
            "leakage_risk": "none",
        }
    )
    return records


def _header(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.reader(handle), [])
        return [item.strip() for item in row if item.strip()]
    except OSError:
        return []
