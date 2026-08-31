"""P0 validation candidate: 7-seed FM SWA plus train-only user-author affinity.

Canonical copy of live elite `rs-20260831T031604Z-3c1f1f0f-003`.
Not a submission replacement. Test stays sealed.

On valid, the live run grid-searched alpha and selected 0.1. The log line
prints the config default alpha (0.05); written scores use the selected
alpha. Train affinity uses `splits['train']` only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def _ensure_starter() -> None:
    repo = Path(__file__).resolve().parents[3]
    starter = repo / "starter" / "kuairand"
    if str(starter) not in sys.path:
        sys.path.insert(0, str(starter))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Write intra-checkpoint averaged multi-seed FM scores with train-derived user-author affinity residual for KuaiRand split."
    )
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


def compute_train_affinity(
    train_tuples: list, smoothing: float = 5.0
) -> tuple[dict[tuple[str, str], float], float]:
    """Compute Laplace-smoothed centered user-author affinity strictly from train split."""
    total_pos = sum(row[6] for row in train_tuples)
    total_count = len(train_tuples)
    global_prior = float(total_pos) / max(total_count, 1)

    user_author_pos = defaultdict(int)
    user_author_tot = defaultdict(int)

    for row in train_tuples:
        u = row[1]
        a = row[3]
        y = row[6]
        user_author_tot[(u, a)] += 1
        if y == 1:
            user_author_pos[(u, a)] += 1

    affinity_map = {}
    for key, tot in user_author_tot.items():
        pos = user_author_pos[key]
        smoothed_rate = (pos + smoothing * global_prior) / (tot + smoothing)
        affinity_map[key] = float(smoothed_rate - global_prior)

    return affinity_map, global_prior


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
    alpha = float(config.get("alpha", 0.05))
    verbose = bool(config.get("verbose", True))

    splits = load(args.data_dir)

    # 1. Compute train-derived User-Author Affinity lookup map (TRAIN ONLY)
    if verbose:
        print("--- Computing train-derived User-Author affinity residual map ---")
    train_affinity_map, global_prior = compute_train_affinity(
        splits["train"], smoothing=5.0
    )

    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    if args.split not in enc:
        raise SystemExit(f"unknown split {args.split}")
    X_out, _y_out, _u_out = enc[args.split]

    # Pre-extract target split (user_id, author_id) for fast affinity vector construction
    target_split_rows = splits[args.split]
    aff_vector = np.array(
        [train_affinity_map.get((r[1], r[3]), 0.0) for r in target_split_rows],
        dtype=np.float64,
    )

    # 2. Train 7-seed FM Ensemble with intra-seed top-k SWA
    all_prob_scores = []
    for member_i in range(num_models):
        model_seed = args.seed + member_i * 1000 + 7
        if verbose:
            print(
                f"--- Training FM ensemble member {member_i + 1}/{num_models} seed {model_seed} ---"
            )
        model = FM(dim, k=k, lr=lr, l2=l2, seed=model_seed)
        rng = np.random.default_rng(model_seed)

        checkpoints: list[
            tuple[float, tuple[np.ndarray, np.ndarray, np.float32]]
        ] = []
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
            raise SystemExit(
                f"FM training produced no checkpoint for model {member_i + 1}"
            )

        avg_V = np.mean([ckpt[1][0] for ckpt in checkpoints], axis=0)
        avg_W = np.mean([ckpt[1][1] for ckpt in checkpoints], axis=0)
        avg_b = np.mean([ckpt[1][2] for ckpt in checkpoints], axis=0)

        model.V = avg_V
        model.W = avg_W
        model.b = float(avg_b)

        raw_preds = np.asarray(model.predict(X_out), dtype=np.float64)
        all_prob_scores.append(raw_preds)

    base_ensemble_scores = np.mean(all_prob_scores, axis=0)

    # 3. Apply residual affinity boosting with validation hyperparameter selection
    if args.split == "valid":
        valid_aff_vector = np.array(
            [train_affinity_map.get((r[1], r[3]), 0.0) for r in splits["valid"]],
            dtype=np.float64,
        )
        best_alpha = alpha
        best_val_primary = -1.0
        for candidate_alpha in [0.0, 0.01, 0.02, 0.05, 0.08, 0.1]:
            cand_scores = base_ensemble_scores + candidate_alpha * valid_aff_vector
            va = evaluate(uva, yva, cand_scores)
            if verbose:
                print(
                    f"Validation alpha search: alpha={candidate_alpha:.2f} -> "
                    f"GAUC={va['GAUC']:.4f}, nDCG@5={va['nDCG@5']:.4f}, Primary={va['primary']:.4f}"
                )
            if va["primary"] > best_val_primary:
                best_val_primary = va["primary"]
                best_alpha = candidate_alpha
        if verbose:
            print(f"Selected best alpha on validation: {best_alpha}")
        final_ensemble_scores = base_ensemble_scores + best_alpha * aff_vector
    else:
        final_ensemble_scores = base_ensemble_scores + alpha * aff_vector

    out = Path(args.output_scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, final_ensemble_scores)
    print(
        f"fm_ensemble_affinity wrote {len(final_ensemble_scores)} scores split={args.split} "
        f"seed={args.seed} num_models={num_models} k={k} alpha={alpha} top_k_ckpt={top_k_checkpoints}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
