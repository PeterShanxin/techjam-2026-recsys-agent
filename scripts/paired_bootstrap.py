"""Paired user bootstrap between two validation score vectors. Test is never touched.

Why paired: resampling validation users gives an absolute primary standard deviation of
about 0.0022, which swamps every delta this project has produced. Two models scored on
the *same* resampled users share that variance, so the paired difference is far tighter
and is the only honest way to ask whether one candidate beats another here.

Usage:
    python scripts/paired_bootstrap.py --baseline runs/final-swa7-ensemble/scores.npy \\
        --candidate runs/<id>/scores.npy [--reps 2000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_agent.evaluation.official import ensure_starter_on_path, official_load

RESEARCH_SPLIT = "valid"


def _per_user(scores: np.ndarray, users: list, labels: list) -> tuple[np.ndarray, ...]:
    """Per-user nDCG@5, per-user AUC (nan when undefined) and its GAUC weight."""
    ensure_starter_on_path()
    from evaluate import auc, ndcg_at_k

    buckets: dict = {}
    for index, user in enumerate(users):
        buckets.setdefault(user, []).append(index)
    ndcg = np.empty(len(buckets))
    gauc = np.full(len(buckets), np.nan)
    weight = np.zeros(len(buckets))
    for slot, rows in enumerate(buckets.values()):
        ordered = sorted(((scores[i], labels[i]) for i in rows), key=lambda item: -item[0])
        labs = [int(label) for _, label in ordered]
        positives = sum(labs)
        ndcg[slot] = ndcg_at_k(labs, 5)
        if 0 < positives < len(labs):
            gauc[slot] = auc(labs, [score for score, _ in ordered])
            weight[slot] = positives
    return ndcg, gauc, weight


def _primary(ndcg: np.ndarray, gauc: np.ndarray, weight: np.ndarray, pick: np.ndarray) -> float:
    defined = ~np.isnan(gauc[pick])
    weights = weight[pick][defined]
    if not weights.sum():
        return float(ndcg[pick].mean())
    weighted = float((gauc[pick][defined] * weights).sum() / weights.sum())
    return (float(ndcg[pick].mean()) + weighted) / 2.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Paired user bootstrap. Validation by default; test needs --allow-test."
    )
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--split", default=RESEARCH_SPLIT, choices=("valid", "test"))
    ap.add_argument(
        "--allow-test",
        action="store_true",
        help="Required with --split test. Observation only; never for selection.",
    )
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "starter" / "kuairand" / "KuaiRand-Pure" / "data"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label-baseline", default="baseline")
    ap.add_argument("--label-candidate", default="candidate")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        raise SystemExit("--split test requires --allow-test; test never informs selection")

    rows = official_load(args.data_dir)[args.split]
    users = [row[1] for row in rows]
    labels = [row[6] for row in rows]
    base = np.asarray(np.load(args.baseline), dtype=np.float64).reshape(-1)
    cand = np.asarray(np.load(args.candidate), dtype=np.float64).reshape(-1)
    if len(base) != len(rows) or len(cand) != len(rows):
        raise SystemExit(
            f"score length mismatch: baseline {len(base)}, candidate {len(cand)}, split {len(rows)}"
        )

    base_stats = _per_user(base, users, labels)
    cand_stats = _per_user(cand, users, labels)
    n_users = len(base_stats[0])
    everyone = np.arange(n_users)
    base_primary = _primary(*base_stats, everyone)
    cand_primary = _primary(*cand_stats, everyone)
    delta = cand_primary - base_primary

    rng = np.random.default_rng(args.seed)
    boots = np.empty(args.reps)
    for rep in range(args.reps):
        pick = rng.integers(0, n_users, n_users)
        boots[rep] = _primary(*cand_stats, pick) - _primary(*base_stats, pick)

    report = {
        "split": args.split,
        "users": int(n_users),
        "rows": int(len(rows)),
        "reps": int(args.reps),
        "seed": int(args.seed),
        args.label_baseline: base_primary,
        args.label_candidate: cand_primary,
        "delta": float(delta),
        "paired_bootstrap_sd": float(boots.std(ddof=1)),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "p_delta_gt_0": float(np.mean(boots > 0)),
    }
    print(f"{args.label_baseline:24s} primary = {base_primary:.7f}")
    print(f"{args.label_candidate:24s} primary = {cand_primary:.7f}")
    print(f"delta                    = {delta:+.7f}")
    print(f"paired bootstrap sd      = {report['paired_bootstrap_sd']:.7f}")
    print(f"95% CI                   = [{report['ci95_low']:+.7f}, {report['ci95_high']:+.7f}]")
    print(f"P(delta > 0)             = {report['p_delta_gt_0']:.4f}")
    if args.split == "test":
        report["selection_use"] = "forbidden: post-freeze observation only"
        print("split                    = test (observation only, not selection)")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
