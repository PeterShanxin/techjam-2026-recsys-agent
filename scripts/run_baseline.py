"""Run official KuaiRand baselines from the repo root.

Does not copy or modify starter/kuairand/evaluate.py. It only adds the
organizer starter directory to sys.path and calls the official functions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter" / "kuairand"
DEFAULT_DATA_DIR = STARTER / "KuaiRand-Pure" / "data"


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("KUAI_RAND_DATA_DIR")
    if env:
        return Path(env)
    return DEFAULT_DATA_DIR


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Official KuaiRand-Pure baseline runner (random / pop / fm)."
    )
    ap.add_argument("--model", default="random", choices=["random", "pop", "fm"])
    ap.add_argument("--data_dir", default=None, help="Path to KuaiRand-Pure/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--json-out", default=None, help="Optional path for machine-readable metrics")
    args = ap.parse_args(argv)

    data_dir = _resolve_data_dir(args.data_dir)
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        print("Download KuaiRand-Pure into starter/kuairand/ or pass --data_dir.", file=sys.stderr)
        return 2

    sys.path.insert(0, str(STARTER))
    import numpy as np
    from data import FIELDS, load
    from baseline import run_fm, run_pop, run_random

    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"data_dir {data_dir}")
    print(f"loading {data_dir} ...")
    t0 = time.perf_counter()
    splits = load(str(data_dir))
    print({k: len(v) for k, v in splits.items()}, f"fields={FIELDS}")

    if args.model == "random":
        res = run_random(splits, seed=args.seed)
    elif args.model == "pop":
        res = run_pop(splits)
    else:
        res = run_fm(splits, k=args.k, lr=args.lr, epochs=args.epochs, seed=args.seed)

    elapsed = time.perf_counter() - t0
    print(f"\n=== {args.model} (seed={args.seed}) ===")
    for sp in ("valid", "test"):
        r = res[sp]
        print(
            f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | "
            f"primary {r['primary']:.4f}"
        )
    print(f"wall_seconds {elapsed:.3f}")

    if args.json_out:
        payload = {
            "model": args.model,
            "seed": args.seed,
            "data_dir": str(data_dir),
            "wall_seconds": elapsed,
            "metrics": _jsonable(res),
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
