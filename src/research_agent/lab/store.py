"""Split-safe KuaiRand facts. Train-derived indexes, not a ranker."""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from research_agent.evaluation.official import official_load

from .errors import LeakageError, SealedSplitError

FEATURE_SOURCE_TRAIN = "train"
OFFICIAL_SPLITS = ("train", "valid", "test")
INFERENCE_FIELDS = ("date", "user_id", "video_id", "author_id", "tab", "duration_ms")
CONTEXT_RAW_FIELDS = ("hourmin", "time_ms")
AUX_FIELDS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
)
LABEL_FIELDS = frozenset(("long_view", *AUX_FIELDS))
TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"
VALID_TEST_LOG = "log_standard_4_22_to_5_08_pure.csv"
USER_CATALOG = "user_features_pure.csv"
VIDEO_CATALOG = "video_features_basic_pure.csv"
VIDEO_STATS = "video_features_statistic_pure.csv"
SPLIT_DATES = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}


@dataclass(frozen=True)
class TrainEvent:
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    long_view: int
    hourmin: str | None = None
    time_ms: str | None = None
    aux: dict[str, str] | None = None
    provenance: str = FEATURE_SOURCE_TRAIN

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "date": self.date,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "author_id": self.author_id,
            "tab": self.tab,
            "duration_ms": self.duration_ms,
            "long_view": self.long_view,
            "hourmin": self.hourmin,
            "time_ms": self.time_ms,
            "provenance": self.provenance,
        }
        if self.aux:
            payload["aux"] = dict(self.aux)
        return payload


@dataclass(frozen=True)
class ImpressionContext:
    index: int
    split: str
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    hourmin: str | None = None
    time_ms: str | None = None
    provenance: str = "inference_visible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "split": self.split,
            "date": self.date,
            "user_id": self.user_id,
            "video_id": self.video_id,
            "author_id": self.author_id,
            "tab": self.tab,
            "duration_ms": self.duration_ms,
            "hourmin": self.hourmin,
            "time_ms": self.time_ms,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class PairwiseSample:
    user_id: str
    pos_video_id: str
    neg_video_id: str
    provenance: str = FEATURE_SOURCE_TRAIN


class SplitSafeStore:
    """Train-only aggregates + serving-visible lookups.

    Does not score rows. Does not expose validation or test labels.
    """

    def __init__(self, data_dir: str | Path, *, feature_source: str = FEATURE_SOURCE_TRAIN) -> None:
        if feature_source != FEATURE_SOURCE_TRAIN:
            raise LeakageError("train-derived features must use feature_source='train'")
        self.data_dir = Path(data_dir)
        self.feature_source = feature_source
        self.provenance = {
            "feature_source": FEATURE_SOURCE_TRAIN,
            "test_sealed": True,
            "validation_labels": "evaluator_only",
        }
        self._splits = official_load(self.data_dir)
        self._raw_by_split = _align_raw_logs(self.data_dir, self._splits)
        self._events = self._build_train_events()
        self._history: dict[str, tuple[TrainEvent, ...]] = {}
        self._item_impr: dict[str, int] = defaultdict(int)
        self._item_pos: dict[str, int] = defaultdict(int)
        self._author_impr: dict[str, int] = defaultdict(int)
        self._author_pos: dict[str, int] = defaultdict(int)
        self._user_author_impr: dict[tuple[str, str], int] = defaultdict(int)
        self._user_author_pos: dict[tuple[str, str], int] = defaultdict(int)
        self._field_pos: dict[tuple[str, str], int] = defaultdict(int)
        self._field_impr: dict[tuple[str, str], int] = defaultdict(int)
        self._index_train()
        self._users = _read_catalog(self.data_dir / USER_CATALOG, "user_id")
        self._videos = _read_catalog(self.data_dir / VIDEO_CATALOG, "video_id")
        self._video_stats: dict[str, dict[str, str]] | None = None

    def official_tuple_count(self, split: str) -> int:
        return len(self._require_split(split))

    def inference_rows(self, split: str) -> list[ImpressionContext]:
        rows = self._require_split(split)
        raw = self._raw_by_split.get(split) or []
        out: list[ImpressionContext] = []
        for i, row in enumerate(rows):
            extra = raw[i] if i < len(raw) else {}
            out.append(
                ImpressionContext(
                    index=i,
                    split=split,
                    date=int(row[0]),
                    user_id=str(row[1]),
                    video_id=str(row[2]),
                    author_id=str(row[3]),
                    tab=str(row[4]),
                    duration_ms=float(row[5]),
                    hourmin=_optional(extra.get("hourmin")),
                    time_ms=_optional(extra.get("time_ms")),
                )
            )
        return out

    def train_events(self) -> tuple[TrainEvent, ...]:
        return self._events

    def train_aux(self, index: int) -> dict[str, str]:
        event = self._events[index]
        return dict(event.aux or {})

    def get_user_history(self, user_id: str) -> tuple[TrainEvent, ...]:
        return self._history.get(str(user_id), ())

    def train_popularity(self, video_id: str, *, kind: str = "impressions") -> float:
        key = str(video_id)
        impressions = float(self._item_impr.get(key, 0))
        positives = float(self._item_pos.get(key, 0))
        if kind == "impressions":
            return impressions
        if kind == "long_view":
            return positives
        if kind in {"rate", "long_view_rate"}:
            return positives / impressions if impressions else 0.0
        raise ValueError(f"unknown popularity kind: {kind}")

    def train_author_popularity(self, author_id: str, *, kind: str = "impressions") -> float:
        key = str(author_id)
        impressions = float(self._author_impr.get(key, 0))
        positives = float(self._author_pos.get(key, 0))
        if kind == "impressions":
            return impressions
        if kind == "long_view":
            return positives
        if kind in {"rate", "long_view_rate"}:
            return positives / impressions if impressions else 0.0
        raise ValueError(f"unknown popularity kind: {kind}")

    def train_author_affinity(self, user_id: str, author_id: str) -> float:
        key = (str(user_id), str(author_id))
        impressions = self._user_author_impr.get(key, 0)
        if not impressions:
            return 0.0
        return self._user_author_pos.get(key, 0) / impressions

    def train_target_rate(self, field: str, value: str) -> float:
        if field in LABEL_FIELDS:
            raise LeakageError(f"{field} is a label/aux field, not a grouping key for target rate")
        key = (field, str(value))
        impressions = self._field_impr.get(key, 0)
        if not impressions:
            return 0.0
        return self._field_pos.get(key, 0) / impressions

    def get_user_features(self, user_id: str) -> dict[str, str]:
        return dict(self._users.get(str(user_id), {}))

    def get_video_features(self, video_id: str) -> dict[str, str]:
        return dict(self._videos.get(str(video_id), {}))

    def get_duration_ms(self, video_id: str) -> float | None:
        feats = self.get_video_features(video_id)
        raw = feats.get("video_duration") or feats.get("duration_ms")
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def video_statistics_unscoped(self, video_id: str) -> dict[str, str]:
        """Undated catalog counters. Prefer train_popularity for research features."""
        if self._video_stats is None:
            self._video_stats = _read_catalog(self.data_dir / VIDEO_STATS, "video_id")
        row = dict(self._video_stats.get(str(video_id), {}))
        if row:
            row["leakage_risk"] = "high"
            row["provenance"] = "catalog_unscoped"
        return row

    def labels(self, split: str) -> list[int]:
        if split == "test":
            raise SealedSplitError("test labels are sealed")
        if split == "valid":
            raise LeakageError("validation labels are not feature inputs; the official evaluator owns them")
        if split != "train":
            raise LeakageError(f"labels are only available for train, not {split}")
        return [int(row[6]) for row in self._splits["train"]]

    def build_pairwise_samples(
        self,
        *,
        max_pairs: int = 100_000,
        negatives_per_positive: int = 1,
        seed: int = 0,
    ) -> list[PairwiseSample]:
        import numpy as np

        if negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be >= 1")
        by_user: dict[str, list[TrainEvent]] = defaultdict(list)
        catalog: set[str] = set()
        for event in self._events:
            by_user[event.user_id].append(event)
            catalog.add(event.video_id)
        catalog_list = list(catalog)
        rng = np.random.default_rng(seed)
        out: list[PairwiseSample] = []
        users = list(by_user)
        rng.shuffle(users)
        for user_id in users:
            events = by_user[user_id]
            pos = [item.video_id for item in events if item.long_view == 1]
            neg = [item.video_id for item in events if item.long_view == 0]
            seen = {item.video_id for item in events}
            if not pos:
                continue
            for pos_id in pos:
                pool = list(neg) if neg else [vid for vid in catalog_list if vid not in seen]
                if not pool:
                    continue
                take = min(negatives_per_positive, len(pool))
                chosen = rng.choice(pool, size=take, replace=False)
                for neg_id in chosen:
                    out.append(PairwiseSample(user_id=user_id, pos_video_id=pos_id, neg_video_id=str(neg_id)))
                    if len(out) >= max_pairs:
                        return out
        return out

    def _require_split(self, split: str) -> list:
        if split not in self._splits:
            raise ValueError(f"unknown split {split}")
        return self._splits[split]

    def _build_train_events(self) -> tuple[TrainEvent, ...]:
        rows = self._splits["train"]
        raw = self._raw_by_split.get("train") or []
        events: list[TrainEvent] = []
        for i, row in enumerate(rows):
            extra = raw[i] if i < len(raw) else {}
            aux = {name: extra[name] for name in AUX_FIELDS if name in extra}
            events.append(
                TrainEvent(
                    date=int(row[0]),
                    user_id=str(row[1]),
                    video_id=str(row[2]),
                    author_id=str(row[3]),
                    tab=str(row[4]),
                    duration_ms=float(row[5]),
                    long_view=int(row[6]),
                    hourmin=_optional(extra.get("hourmin")),
                    time_ms=_optional(extra.get("time_ms")),
                    aux=aux or None,
                )
            )
        return tuple(events)

    def _index_train(self) -> None:
        history: dict[str, list[TrainEvent]] = defaultdict(list)
        for event in self._events:
            history[event.user_id].append(event)
            self._item_impr[event.video_id] += 1
            self._author_impr[event.author_id] += 1
            self._user_author_impr[(event.user_id, event.author_id)] += 1
            if event.long_view:
                self._item_pos[event.video_id] += 1
                self._author_pos[event.author_id] += 1
                self._user_author_pos[(event.user_id, event.author_id)] += 1
            for field, value in (
                ("video_id", event.video_id),
                ("author_id", event.author_id),
                ("tab", event.tab),
                ("user_id", event.user_id),
            ):
                key = (field, value)
                self._field_impr[key] += 1
                if event.long_view:
                    self._field_pos[key] += 1
        self._history = {user: tuple(items) for user, items in history.items()}


def _align_raw_logs(data_dir: Path, splits: Mapping[str, list]) -> dict[str, list[dict[str, str]]]:
    files = {
        "train": data_dir / TRAIN_LOG,
        "valid": data_dir / VALID_TEST_LOG,
        "test": data_dir / VALID_TEST_LOG,
    }
    out: dict[str, list[dict[str, str]]] = {name: [] for name in OFFICIAL_SPLITS}
    seen_valid_test = False
    for split in OFFICIAL_SPLITS:
        path = files[split]
        if not path.is_file():
            continue
        if split in {"valid", "test"}:
            if seen_valid_test:
                continue
            seen_valid_test = True
            lo_v, hi_v = SPLIT_DATES["valid"]
            lo_t, hi_t = SPLIT_DATES["test"]
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    date = _row_date(row)
                    if date is None:
                        continue
                    if lo_v <= date <= hi_v:
                        out["valid"].append(row)
                    elif lo_t <= date <= hi_t:
                        out["test"].append(row)
        else:
            lo, hi = SPLIT_DATES["train"]
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    date = _row_date(row)
                    if date is None:
                        continue
                    if lo <= date <= hi:
                        out["train"].append(row)
    for split in OFFICIAL_SPLITS:
        official_n = len(splits.get(split) or [])
        if official_n and len(out[split]) != official_n:
            # Keep alignment only when row counts match official load order.
            out[split] = []
    return out


def _row_date(row: Mapping[str, str]) -> int | None:
    raw = row.get("date")
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_catalog(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ident = row.get(key)
            if ident:
                out[str(ident)] = dict(row)
    return out


def _optional(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def iter_inference_dicts(rows: Iterable[ImpressionContext]) -> Iterator[dict[str, Any]]:
    for row in rows:
        yield row.to_dict()
