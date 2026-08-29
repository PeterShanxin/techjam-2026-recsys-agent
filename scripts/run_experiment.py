"""Run one declarative experiment through the Phase 2 harness."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_agent.experiments import ExperimentRunner, ExperimentSpec


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("KUAI_RAND_DATA_DIR")
    if env:
        return Path(env)
    return ROOT / "starter" / "kuairand" / "KuaiRand-Pure" / "data"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Execute one ExperimentSpec and persist the result.")
    ap.add_argument("--spec", required=True, help="Path to ExperimentSpec JSON")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument(
        "--allow-test",
        action="store_true",
        help="Opt in to evaluation_split=test (final/audit only)",
    )
    ap.add_argument("--experiment-id", default=None, help="Override spec experiment_id")
    args = ap.parse_args(argv)

    spec = ExperimentSpec.from_path(Path(args.spec))
    if args.experiment_id:
        payload = spec.to_dict()
        payload["experiment_id"] = args.experiment_id
        payload.pop("spec_hash", None)
        spec = ExperimentSpec.from_dict(payload)

    data_dir = _resolve_data_dir(args.data_dir)
    runs_dir = Path(args.runs_dir) if args.runs_dir else ROOT / "runs"
    if args.registry:
        from research_agent.experiments import ExperimentRegistry

        registry = ExperimentRegistry(Path(args.registry))
    else:
        registry = None

    runner = ExperimentRunner(
        repo_root=ROOT,
        runs_dir=runs_dir,
        registry=registry,
        data_dir=data_dir,
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
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
