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

# Axes this project has already spent most of its evaluations on. Repeating them is
# cheap to propose and has repeatedly returned deltas inside the noise floor.
HEAVILY_SEARCHED_AXES: list[str] = [
    "number of bagged FM seeds (3-seed and 7-seed both measured)",
    "checkpoint averaging / SWA inside a seed",
    "score-space combination (raw logit mean vs probability mean vs percentile rank)",
    "additive train-count residuals on top of the elite (user-author affinity, "
    "user-tab, author quality prior, item popularity), including recency-decayed variants",
    "embedding dimension k (organizer measured 8/16/32)",
    "static catalog feature expansion (organizer measured, no material help)",
]

# Axes the search has NOT meaningfully entered. Listing them is not an instruction to
# use any particular one, and no value or setting is suggested here.
UNDEREXPLORED_AXES: list[str] = [
    "capacity control / regularization strength of the FM fit "
    "(the current fit's validation primary peaks several epochs before early stop "
    "while training loss is still falling, which is an overfitting signature)",
    "the training objective itself: the loss is pointwise BCE on long_view while the "
    "official metric is within-user ranking; ranking-aligned objectives have never "
    "completed a run here (every attempt died on runtime, not on evidence)",
    "which train rows are used and how they are weighted "
    "(for example rows that carry no within-user ranking information)",
    "feature encoding and bucketing choices inside the encoder itself",
    "multi-task or auxiliary supervision from train-only aux columns",
    "heterogeneous ensembles whose members differ by objective rather than by seed",
]

# Measured on this validation split. Sets the bar for what counts as a result.
VALIDATION_NOISE: dict[str, Any] = {
    "users": 22377,
    "rows": 124909,
    "absolute_primary_bootstrap_sd": 0.00216,
    "paired_delta_bootstrap_sd_typical": 0.0005,
    "interpretation": (
        "Resampling validation users gives an absolute primary standard deviation of "
        "about 0.0022. A paired comparison against a shared baseline is tighter, about "
        "0.0005. A candidate whose best realistic case is a delta below ~0.0005 is not "
        "measurable here and is not worth an evaluation slot. Deltas near 1e-5 are noise."
    ),
    "official_convergence_epsilon": 0.002,
}

# Independent second-opinion audit of the P0 sprint. Evidence about what has been ruled
# out, so evaluation slots are not spent re-deriving it. Not a list of answers.
AUDIT_FINDINGS: list[str] = [
    "GAUC and nDCG@5 are both computed strictly inside one user's impression list. Any "
    "strictly monotone per-user transform of the score vector (global sigmoid, affine "
    "rescale, per-user z-score, per-user rank) leaves both metrics bitwise unchanged. "
    "Verified. Do not propose calibration, normalization or rank transforms as mechanisms.",
    "A per-user constant cannot change within-user order. The FM's first-order user "
    "weight and global bias contribute nothing to the metric, and user-catalog columns "
    "are inert unless crossed with something that varies inside the user.",
    "Additive blends of train-derived count features on top of the frozen elite were "
    "swept with the blend weight chosen on validation (an optimistic upper bound). Item "
    "rate, author rate, video-by-tab, author-by-tab, user-author affinity, user-tab, "
    "duration-bucket rate, item support, user duration preference and video age all "
    "peaked at weight zero. Tab, tag and hour-of-day peaked below +1.2e-4. This family "
    "is informationally redundant with the FM and is closed.",
    "long_view is about 98% a deterministic function of play_time_ms and duration_ms "
    "with a threshold near 18 seconds. Four duration encodings, including one with an "
    "explicit 18000 ms knot and one with a short-video indicator, all landed inside "
    "noise. The duration-bucketing axis is closed.",
    "Validation impressions per user are sparse: mean 5.58, median 4. 30.3% of users are "
    "all-negative and 11.9% all-positive, so 42% of nDCG@5 weight cannot be moved by any "
    "model. Users with 6-20 impressions hold 63.5% of GAUC weight and 65% of the "
    "realizable nDCG@5 gain. GAUC and nDCG@5 favour the same users; there is no tradeoff.",
    "Every prior ranking-objective attempt failed on runtime. The pairwise candidates "
    "built O(pairs) loops over roughly 1.9M sampled pairs per epoch with plain SGD and "
    "hit the timeout. research_agent.lab.ranking now exposes vectorized within-user "
    "grouping and a gradient-driven FM so an O(rows) objective is practical. It supplies "
    "no loss function; choosing the objective is a research decision.",
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
    "experiment_id": "final-tiered-ensemble",
    "split": "valid",
    "GAUC": 0.6690881485589812,
    "nDCG@5": 0.536719279947655,
    "primary": 0.6029037142533181,
    "mechanism": (
        "8 official FM members, top-2 checkpoint SWA each, tiered by train-row "
        "selection and L2 strength, averaged as raw FM scores"
    ),
    "superseded": {
        "experiment_id": "final-swa7-ensemble",
        "primary": 0.6023186326402106,
        "note": "Phase 4 winner. Beaten on validation by +0.0005851 (paired P=0.888).",
    },
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
    "P0 pattern: 'parent + alpha * residual' with alpha grid-searched on validation and "
    "alpha=0 inside the grid cannot score below its parent. Five P0 offspring used it. "
    "The controller now rejects children whose within-user ordering barely differs from a "
    "parent's, so this shape no longer earns fitness. Propose a mechanism, not a bounded "
    "reparameterisation of the elite.",
    "Audit: the whole pipeline's gain over the FM root (+0.00085) does not clear a 95% "
    "paired user bootstrap (CI [-0.00025, +0.00184]). Treat sub-0.0005 deltas as unproven.",
]

TEST_SEALED_POLICY = (
    "TEST IS SEALED. Do not evaluate, inspect, or select on test. "
    "Use train + valid only. Official fitness is validation primary."
)
