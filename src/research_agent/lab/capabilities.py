"""Compact lab instrument list for ResearchState. Facts, not a winning ranker."""
from __future__ import annotations

from typing import Any

LAB_MODULE = "research_agent.lab"
LAB_IMPORT = "from research_agent.lab import SplitSafeStore, GradientFM, user_groups"

LAB_CONTRACT = (
    "SplitSafeStore is a lab instrument, not a recommender. "
    "Train-derived facts use the train split only. "
    "Inference-visible fields are the current row plus catalogs and train history. "
    "Validation labels must not enter features. TEST IS SEALED. "
    "Do not implement a hidden 'best score' helper. Decide the model yourself."
)

LAB_CAPABILITIES: tuple[dict[str, str], ...] = (
    {
        "name": "SplitSafeStore",
        "call": "SplitSafeStore(data_dir)",
        "provenance": "feature_source=train",
        "role": "Cached train indexes + inference-visible lookups",
    },
    {
        "name": "inference_rows",
        "call": "store.inference_rows(split)",
        "provenance": "inference_visible",
        "role": "Current-row context without labels",
    },
    {
        "name": "get_user_history",
        "call": "store.get_user_history(user_id)",
        "provenance": "train",
        "role": "Prior train interactions for one user",
    },
    {
        "name": "train_popularity",
        "call": "store.train_popularity(video_id, kind=impressions|long_view|rate)",
        "provenance": "train",
        "role": "Train item exposure / target counts",
    },
    {
        "name": "train_author_affinity",
        "call": "store.train_author_affinity(user_id, author_id)",
        "provenance": "train",
        "role": "User-author long_view rate on train",
    },
    {
        "name": "train_target_rate",
        "call": "store.train_target_rate(field, value)",
        "provenance": "train",
        "role": "P(long_view|field=value) on train",
    },
    {
        "name": "get_user_features",
        "call": "store.get_user_features(user_id)",
        "provenance": "catalog",
        "role": "Raw user catalog lookup",
    },
    {
        "name": "get_video_features",
        "call": "store.get_video_features(video_id)",
        "provenance": "catalog",
        "role": "Raw video catalog lookup (basic, not statistic file)",
    },
    {
        "name": "train_aux",
        "call": "store.train_aux(index) or event.aux",
        "provenance": "train",
        "role": "Train-log aux actions / play_time_ms",
    },
    {
        "name": "build_pairwise_samples",
        "call": "store.build_pairwise_samples(...)",
        "provenance": "train",
        "role": "User-wise pos/neg pairs from train labels only",
    },
    {
        "name": "recency_weight",
        "call": "recency_weight(event_date, reference_date, half_life_days)",
        "provenance": "time_util",
        "role": "Date-difference decay. Not a score.",
    },
    {
        "name": "video_statistics_unscoped",
        "call": "store.video_statistics_unscoped(video_id)",
        "provenance": "catalog_unscoped",
        "role": "Optional leaky global counters. Prefer train_popularity.",
    },
    {
        "name": "user_groups",
        "call": "user_groups(users_from_encode) -> UserGroups",
        "provenance": "structural",
        "role": (
            "Vectorized within-user segmentation of an encoded split. Gives .order, "
            ".group_start, .group_sizes plus .group_sum/.group_max/.broadcast/.select. "
            "Makes a group-structured objective cost O(rows) instead of O(pairs)."
        ),
    },
    {
        "name": "GradientFM",
        "call": "GradientFM(dim, k=..., lr=..., l2=..., seed=...); m.logits(X); m.apply(X, grad_logits)",
        "provenance": "structural",
        "role": (
            "Official FM forward pass plus Adam, driven by a caller-supplied per-row "
            "dL/dlogit. Contains NO loss function and cannot train on its own. You choose "
            "the objective, the targets and the regularization. Also has predict/state/load_state."
        ),
    },
)

# One worked example. Shows the plumbing only: the gradient shown is plain pointwise BCE,
# which just reproduces the existing FM. Whether some other objective is worth training,
# and what it should be, is the research question this example does not answer.
LAB_EXAMPLE = """\
from data import encode, load
from research_agent.lab import GradientFM, user_groups
import numpy as np

splits = load(data_dir)
enc, dim = encode(splits)
Xtr, ytr, utr = enc["train"]
groups = user_groups(utr)                # within-user segmentation of the train split
Xg = Xtr[groups.order]                   # rows reordered so each user is contiguous
yg = ytr[groups.order]

model = GradientFM(dim, k=16, lr=0.001, l2=1e-6, seed=0)
rows = groups.select(np.arange(64))      # a whole-user minibatch: 64 users, no user split
z, _, _ = model.logits(Xg[rows])
g = (1.0 / (1.0 + np.exp(-z)) - yg[rows]) / len(rows)   # <- pointwise BCE; replace this
model.apply(Xg[rows], g)
"""

TRAIN_DERIVED = (
    "user history",
    "item/author popularity",
    "target rates",
    "user-author affinity",
    "pairwise/listwise train samples",
    "train aux behaviors / play_time_ms",
)

INFERENCE_VISIBLE = (
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
    "date (current row)",
    "hourmin / time_ms (current row, if raw log aligned)",
    "user catalog",
    "video basic catalog",
    "train history of that user",
)

UNAVAILABLE_OR_UNSAFE = (
    "valid/test long_view as a feature",
    "valid/test is_like/is_click/play_time_ms of the current impression as a feature",
    "test labels (sealed)",
    "video_features_statistic_pure.csv as default popularity (unscoped, undated)",
)

PRACTICAL_FAMILIES = (
    "history_recency",
    "pairwise_ranking",
    "listwise_ranking",
    "context_residual",
    "train_affinity",
    "duration_watch_bias",
    "interaction_beyond_fm",
)


def lab_contract_dict() -> dict[str, Any]:
    return {
        "module": LAB_MODULE,
        "import": LAB_IMPORT,
        "rule": LAB_CONTRACT,
        "example": LAB_EXAMPLE,
        "capabilities": [dict(item) for item in LAB_CAPABILITIES],
        "train_derived": list(TRAIN_DERIVED),
        "inference_visible": list(INFERENCE_VISIBLE),
        "unavailable_or_unsafe": list(UNAVAILABLE_OR_UNSAFE),
        "practical_families": list(PRACTICAL_FAMILIES),
        "test_sealed": True,
        "not_a_ranker": True,
    }
