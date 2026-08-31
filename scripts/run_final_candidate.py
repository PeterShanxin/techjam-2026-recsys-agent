"""Run the frozen final candidate through the Phase 2 harness.

Validation is the default. Test requires --allow-test and is for the official
CSV only — never for selecting experiments.

--legacy-swa7 runs the superseded Phase 4 winner instead, so its historical
number stays reproducible. It is not the submission candidate.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_agent.experiments import ExperimentRunner
from research_agent.final_candidate import (
    FINAL_EXPERIMENT_ID,
    LEGACY_SWA7_EXPERIMENT_ID,
    final_candidate_spec,
    swa7_candidate_spec,
)


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("KUAI_RAND_DATA_DIR")
    if env:
        return Path(env)
    return ROOT / "starter" / "kuairand" / "KuaiRand-Pure" / "data"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Execute the frozen SWA+7-seed candidate.")
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument(
        "--allow-test",
        action="store_true",
        help="Required when --split test. Does not feed elite ranking.",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument(
        "--legacy-swa7",
        action="store_true",
        help="Run the superseded Phase 4 SWA+7-seed winner instead of the Phase 5 candidate.",
    )
    args = ap.parse_args(argv)

    if args.split == "test" and not args.allow_test:
        print("test split requires --allow-test after the candidate is frozen", file=sys.stderr)
        return 2

    build = swa7_candidate_spec if args.legacy_swa7 else final_candidate_spec
    base_id = LEGACY_SWA7_EXPERIMENT_ID if args.legacy_swa7 else FINAL_EXPERIMENT_ID
    spec = build(
        experiment_id=args.experiment_id
        or (base_id if args.split == "valid" else f"{base_id}-test"),
        evaluation_split=args.split,
        allow_test_split=bool(args.allow_test),
    )
    runner = ExperimentRunner(
        repo_root=ROOT,
        runs_dir=Path(args.runs_dir) if args.runs_dir else ROOT / "runs",
        data_dir=_resolve_data_dir(args.data_dir),
        allow_test=args.allow_test,
    )
    result = runner.run(spec, allow_test=args.allow_test)
    metrics = result.metrics
    print(f"experiment_id  {result.experiment_id}")
    print(f"status         {result.status}")
    print(f"split          {result.evaluation_split}")
    if metrics is None:
        print("GAUC           -")
        print("nDCG@5         -")
        print("primary        -")
    else:
        print(f"GAUC           {metrics.gauc:.4f}")
        print(f"nDCG@5         {metrics.ndcg_at_5:.4f}")
        print(f"primary        {metrics.primary:.4f}")
    print(f"runtime        {result.wall_seconds:.3f}s")
    print(f"run_dir        {result.run_dir}")
    if result.failure:
        print(f"failure        {result.failure.kind}: {result.failure.message}")
    if args.split == "test" and result.status == "success" and result.scores_path:
        print(
            "pack CSV with: .\\.venv\\Scripts\\python.exe scripts\\make_submission.py "
            f"--scores {result.scores_path} --split test --output submission.csv"
        )
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
