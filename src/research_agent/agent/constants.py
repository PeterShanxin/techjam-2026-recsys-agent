"""Benchmark invariants and organizer research context. Not a hard-coded experiment script."""
from __future__ import annotations

from typing import Any

FM_ROOT_ID = "fm-root"
DEFAULT_RESEARCH_MODEL = "gemini-3.7-flash"
DEFAULT_THINKING_LEVEL = "medium"
REPAIR_THINKING_LEVEL = "high"
MAX_REPAIRS_PER_ITERATION = 2
CANDIDATE_FILENAME = "candidate.py"

# Phase 1 reproduced official FM validation (seed 0). Research starts here, not random.
FM_VALID_REFERENCE: dict[str, Any] = {
    "experiment_id": FM_ROOT_ID,
    "split": "valid",
    "seed": 0,
    "GAUC": 0.6671,
    "nDCG@5": 0.5358,
    "primary": 0.6015,
    "k": 16,
    "lr": 0.001,
    "batch": 8192,
    "max_epochs": 40,
    "patience": 4,
    "fields": ["user_id", "video_id", "author_id", "tab", "dur_bucket"],
}

BENCHMARK_INVARIANTS: dict[str, Any] = {
    "task": "within-user ranking",
    "label": "long_view",
    "metrics": ["GAUC", "nDCG@5", "primary=mean(GAUC, nDCG@5)"],
    "research_split_policy": "validation-only; never rank or select on test",
    "evaluator": "starter/kuairand/evaluate.py is immutable; candidates write scores only",
    "external_data": "forbidden",
}

ORGANIZER_DEAD_ENDS: list[str] = [
    "static feature expansion (CWM-style extra fields) did not materially help",
    "embedding dimension k=8/16/32 did not materially help",
    "pure user-side first-order terms do not change within-user order",
]

ORGANIZER_PROMISING_CATEGORIES: list[str] = [
    "ranking-aligned objectives such as BPR or listwise losses",
    "user history / sequence modelling",
    "multi-task behavioral signals (click/like/follow/comment/forward)",
    "watch-time modelling",
    "temporal / distribution-shift ideas",
]

FM_ROOT_PARAMETERS: dict[str, Any] = {
    "k": 16,
    "lr": 0.001,
    "epochs": 40,
    "batch": 8192,
    "patience": 4,
    "l2": 1e-6,
}

FM_ROOT_ENTRYPOINT = "src/research_agent/recommenders/fm_scorer.py"

FROZEN_VALID_BEST: dict[str, Any] = {
    "experiment_id": "final-swa7-ensemble",
    "split": "valid",
    "GAUC": 0.6683660080655603,
    "nDCG@5": 0.5362712572148608,
    "primary": 0.6023186326402106,
    "mechanism": "7-seed official FM + top-2 checkpoint SWA + raw probability mean",
    "note": "Beat this on validation. Test is sealed.",
}

KNOWN_NEGATIVE_EVIDENCE: list[str] = [
    "soft-label claims without reading raw CSVs or lab train-aux APIs are invalid, not negative science",
    "silent torch/ImportError fallback to FM is a no-op, not a method result",
    "homogeneous FM bagging/SWA refinements produced only +0.00085 vs FM root",
    "P0: BPR-FM 7-seed and 3-seed timed out at 600s — implementation_failure, not negative science",
    "P0: dual user-author + user-tab residual (0.60213) did not beat user-author residual (0.60233)",
    "P0: recency-decayed user-author affinity (0.60197) lost to static train affinity",
    "P0: treating FM outputs as probabilities then taking logit/sigmoid destroyed ranking (impl bug)",
]

TEST_SEALED_POLICY = (
    "TEST IS SEALED. Do not evaluate, inspect, or select on test. "
    "Use train + valid only. Official fitness is validation primary."
)
