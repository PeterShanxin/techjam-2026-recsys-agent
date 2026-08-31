"""Capability smoke: train history + recency. Not a submission winner.

Scores a row from train-only user history and train item rate.
No FM. No validation labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write history/recency scores for one KuaiRand split.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", required=True, choices=["valid", "test"])
    ap.add_argument("--output-scores", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=None)
    return ap.parse_args(argv)


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[3]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import numpy as np
    from research_agent.lab import SplitSafeStore, recency_weight

    config = load_config(args.config)
    half_life = float(config.get("half_life_days", 3.0))
    author_w = float(config.get("author_weight", 0.35))
    pop_w = float(config.get("popularity_weight", 0.15))
    store = SplitSafeStore(args.data_dir)
    rows = store.inference_rows(args.split)
    scores = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        score = 0.0
        for event in store.get_user_history(row.user_id):
            decay = recency_weight(event.date, row.date, half_life_days=half_life)
            if event.video_id == row.video_id:
                score += decay * (1.0 + event.long_view)
            if event.author_id == row.author_id:
                score += author_w * decay * (0.5 + event.long_view)
        score += pop_w * store.train_popularity(row.video_id, kind="long_view_rate")
        score += 0.05 * store.train_author_affinity(row.user_id, row.author_id)
        scores[i] = score
    out = Path(args.output_scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    print(f"history_recency_scorer wrote {len(scores)} scores split={args.split} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
