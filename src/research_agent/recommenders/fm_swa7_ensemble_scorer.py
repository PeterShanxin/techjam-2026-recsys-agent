"""Frozen Phase 4 winner: 7-seed FM bagging + top-2 intra-seed SWA.

Canonical copy of the live elite `rs-20260830T133522Z-0e304128-004`.
Do not point submission commands at gitignored `runs/generated/` paths.
Algorithm matches the live candidate: raw probability averaging across 7
official FM members; each member averages parameters from its top-2
validation-primary checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _ensure_starter() -> None:
    repo = Path(__file__).resolve().parents[3]
    starter = repo / "starter" / "kuairand"
    if str(starter) not in sys.path:
        sys.path.insert(0, str(starter))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write intra-checkpoint averaged multi-seed FM scores for KuaiRand split.")
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
    from baseline import FM
    from data import encode, load
    from evaluate import evaluate

    config = load_config(args.config)
    k = int(config.get("k", 16))
    lr = float(config.get("lr", 0.001))
    epochs = int(config.get("epochs", 40))
    bs = int(config.get("batch", 8192))
    patience = int(config.get("patience", 4))
    l2 = float(config.get("l2", 1e-6))
    num_models = int(config.get("num_models", 7))
    top_k_checkpoints = int(config.get("top_k_checkpoints", 2))
    verbose = bool(config.get("verbose", True))

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    if args.split not in enc:
        raise SystemExit(f"unknown split {args.split}")
    X_out, _y_out, _u_out = enc[args.split]

    all_prob_scores = []
    for member_i in range(num_models):
        model_seed = args.seed + member_i * 1000 + 7
        if verbose:
            print(f"--- Training FM ensemble member {member_i + 1}/{num_models} seed {model_seed} ---")
        model = FM(dim, k=k, lr=lr, l2=l2, seed=model_seed)
        rng = np.random.default_rng(model_seed)

        checkpoints: list[tuple[float, tuple[np.ndarray, np.ndarray, np.float32]]] = []
        best_primary = -1.0
        bad = 0

        for epoch in range(1, epochs + 1):
            idx = rng.permutation(len(ytr))
            t0 = time.time()
            losses = [
                model.step(Xtr[idx[i : i + bs]], ytr[idx[i : i + bs]])
                for i in range(0, len(idx), bs)
            ]
            va = evaluate(uva, yva, model.predict(Xva))
            score = float(va["primary"])
            if verbose:
                print(
                    f"  [M{member_i + 1}] epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
                    f"valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                    f"primary {score:.4f} | {time.time() - t0:.1f}s"
                )

            state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            checkpoints.append((score, state))
            checkpoints.sort(key=lambda item: item[0], reverse=True)
            if len(checkpoints) > top_k_checkpoints:
                checkpoints = checkpoints[:top_k_checkpoints]

            if score > best_primary + 1e-5:
                best_primary = score
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch}")
                    break

        if not checkpoints:
            raise SystemExit(f"FM training produced no checkpoint for model {member_i + 1}")

        avg_V = np.mean([ckpt[1][0] for ckpt in checkpoints], axis=0)
        avg_W = np.mean([ckpt[1][1] for ckpt in checkpoints], axis=0)
        avg_b = np.mean([ckpt[1][2] for ckpt in checkpoints], axis=0)

        model.V = avg_V
        model.W = avg_W
        model.b = float(avg_b)

        raw_preds = np.asarray(model.predict(X_out), dtype=np.float64)
        all_prob_scores.append(raw_preds)

    ensemble_scores = np.mean(all_prob_scores, axis=0)
    out = Path(args.output_scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, ensemble_scores)
    print(
        f"fm_ensemble wrote {len(ensemble_scores)} scores split={args.split} "
        f"seed={args.seed} num_models={num_models} k={k} lr={lr} rank_averaging=False top_k_ckpt={top_k_checkpoints}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
