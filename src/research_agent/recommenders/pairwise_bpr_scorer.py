"""Capability smoke: train pairwise BPR on ids. Not a submission winner.

Uses SplitSafeStore.build_pairwise_samples (train labels only).
No FM. No validation labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write pairwise BPR scores for one KuaiRand split.")
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
    from research_agent.lab import SplitSafeStore

    config = load_config(args.config)
    dim = int(config.get("k", 8))
    lr = float(config.get("lr", 0.05))
    epochs = int(config.get("epochs", 2))
    max_pairs = int(config.get("max_pairs", 30000))
    store = SplitSafeStore(args.data_dir)
    samples = store.build_pairwise_samples(
        max_pairs=max_pairs,
        negatives_per_positive=int(config.get("negatives_per_positive", 1)),
        seed=args.seed,
    )
    users = sorted({item.user_id for item in samples} | {ev.user_id for ev in store.train_events()})
    items = sorted({item.pos_video_id for item in samples} | {item.neg_video_id for item in samples})
    items = sorted(set(items) | {ev.video_id for ev in store.train_events()})
    u_index = {name: i for i, name in enumerate(users)}
    i_index = {name: i for i, name in enumerate(items)}
    rng = np.random.default_rng(args.seed)
    U = rng.normal(0.0, 0.01, size=(max(len(users), 1), dim)).astype(np.float64)
    V = rng.normal(0.0, 0.01, size=(max(len(items), 1), dim)).astype(np.float64)
    if samples:
        order = rng.permutation(len(samples))
        for _ in range(max(1, epochs)):
            for idx in order:
                sample = samples[int(idx)]
                u = u_index[sample.user_id]
                i = i_index[sample.pos_video_id]
                j = i_index[sample.neg_video_id]
                x = float(U[u] @ V[i] - U[u] @ V[j])
                sig = 1.0 / (1.0 + np.exp(min(30.0, max(-30.0, x))))
                grad = lr * sig
                u_vec = U[u].copy()
                U[u] += grad * (V[i] - V[j])
                V[i] += grad * u_vec
                V[j] -= grad * u_vec
    rows = store.inference_rows(args.split)
    scores = np.zeros(len(rows), dtype=np.float64)
    mean_item = V.mean(axis=0) if len(items) else np.zeros(dim)
    mean_user = U.mean(axis=0) if len(users) else np.zeros(dim)
    for n, row in enumerate(rows):
        u_vec = U[u_index[row.user_id]] if row.user_id in u_index else mean_user
        v_vec = V[i_index[row.video_id]] if row.video_id in i_index else mean_item
        scores[n] = float(u_vec @ v_vec)
    out = Path(args.output_scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    print(
        f"pairwise_bpr_scorer wrote {len(scores)} scores split={args.split} "
        f"pairs={len(samples)} seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
