"""Official-style random candidate.

Writes an ordered score vector for the requested split. Does not call evaluate().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_starter() -> None:
    repo = Path(__file__).resolve().parents[3]
    starter = repo / "starter" / "kuairand"
    if str(starter) not in sys.path:
        sys.path.insert(0, str(starter))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write random scores for one KuaiRand split.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", required=True, choices=["valid", "test"])
    ap.add_argument("--output-scores", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=None, help="JSON file of free-form parameters")
    return ap.parse_args(argv)


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_starter()
    import numpy as np
    from data import load

    config = load_config(args.config)
    splits = load(args.data_dir)
    rows = splits[args.split]
    rng = np.random.default_rng(args.seed)
    scores = rng.random(len(rows)).astype(np.float64)
    out = Path(args.output_scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    print(
        f"random_scorer wrote {len(scores)} scores split={args.split} "
        f"seed={args.seed} scorer={config.get('scorer', 'random')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
