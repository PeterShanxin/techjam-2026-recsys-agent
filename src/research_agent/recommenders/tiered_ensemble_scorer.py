"""Sprint-2 live elite: catalog-balanced tiered FM ensemble.

Canonical copy of live elite `rs-20260831T062638Z-939b7000-008`, discovered
autonomously by the Gemini research agent on 2026-08-31 (crossover of 007 x 006).

Mechanism: 8 official FM members, each with top-2 checkpoint SWA, averaged in
probability space. Members are split across three tiers that differ in BOTH the
train rows they see and their L2 strength:
  2x Strict   - users with >= 3 impressions and mixed labels, l2 = 1e-4
  2x Moderate - users with >= 2 impressions and mixed labels, l2 = 1e-5
  4x Full     - unfiltered train split,                       l2 = 1e-6

The agent reached the regularization and train-row-selection axes on its own;
no value in this file was supplied by a human. Validation only. TEST IS SEALED.
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
        description="Catalog-Balanced Tiered Heterogeneous Ensemble: 2 Strict + 2 Moderate + 4 Full FM with Tier-Adaptive L2 and SWA."
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


def filter_tiered_train_rows(Xtr, ytr, utr, min_impressions: int):
    import numpy as np
    user_pos = defaultdict(int)
    user_tot = defaultdict(int)
    for u, y in zip(utr, ytr):
        user_tot[u] += 1
        if y > 0.5:
            user_pos[u] += 1

    informative_users = {
        u for u, tot in user_tot.items()
        if tot >= min_impressions and 0 < user_pos[u] < tot
    }
    
    mask_arr = np.array([u in informative_users for u in utr], dtype=bool)
    
    if mask_arr.sum() == 0:
        print(f"Warning: Filtering min_imp={min_impressions} resulted in 0 rows. Falling back to full training set.")
        return Xtr, ytr
        
    print(f"Train row filtering (min_imp={min_impressions}): kept {mask_arr.sum()}/{len(ytr)} rows ({mask_arr.mean():.2%}) "
          f"from {len(informative_users)} informative users.")
    return Xtr[mask_arr], ytr[mask_arr]


def train_swa_fm(
    dim: int,
    Xtr,
    ytr,
    Xva,
    yva,
    uva,
    X_out,
    model_seed: int,
    k: int,
    lr: float,
    l2: float,
    epochs: int,
    bs: int,
    patience: int,
    top_k_checkpoints: int,
    verbose: bool,
    member_name: str,
):
    import numpy as np
    from baseline import FM
    from evaluate import evaluate

    if verbose:
        print(f"--- Training FM member [{member_name}] seed {model_seed} (l2={l2}) ---")
    model = FM(dim, k=k, lr=lr, l2=l2, seed=model_seed)
    rng = np.random.default_rng(model_seed)

    checkpoints = []
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
                f"  [{member_name}] epoch {epoch:2d} | loss {np.mean(losses):.4f} | "
                f"valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                f"primary {score:.4f} | {time.time() - t0:.1f}s"
            )

        state = (model.V.copy(), model.W.copy(), float(model.b))
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
        raise SystemExit(f"FM training produced no checkpoint for [{member_name}]")

    avg_V = np.mean([ckpt[1][0] for ckpt in checkpoints], axis=0)
    avg_W = np.mean([ckpt[1][1] for ckpt in checkpoints], axis=0)
    avg_b = np.mean([ckpt[1][2] for ckpt in checkpoints], axis=0)

    model.V = avg_V
    model.W = avg_W
    model.b = float(avg_b)

    return np.asarray(model.predict(X_out), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _ensure_starter()
    import numpy as np
    from data import encode, load

    config = load_config(args.config)
    k = int(config.get("k", 16))
    lr = float(config.get("lr", 0.001))
    epochs = int(config.get("epochs", 40))
    bs = int(config.get("batch", 8192))
    patience = int(config.get("patience", 4))
    top_k_checkpoints = int(config.get("top_k_checkpoints", 2))
    verbose = bool(config.get("verbose", True))

    # Tier setup: Crossover between Parent A (3 tiers + adaptive L2) and Parent B (50% full catalog weighting)
    num_strict = int(config.get("num_strict", 2))
    num_moderate = int(config.get("num_moderate", 2))
    num_full = int(config.get("num_full", 4))

    l2_strict = float(config.get("l2_strict", 1e-4))
    l2_moderate = float(config.get("l2_moderate", 1e-5))
    l2_full = float(config.get("l2_full", 1e-6))

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr_raw, ytr_raw, utr_raw = enc["train"]
    Xva, yva, uva = enc["valid"]
    if args.split not in enc:
        raise SystemExit(f"unknown split {args.split}")
    X_out, _y_out, _u_out = enc[args.split]

    # Tier 1: Strict variance filtering (tot >= 3)
    Xtr_strict, ytr_strict = filter_tiered_train_rows(
        Xtr_raw, ytr_raw, utr_raw, min_impressions=3
    )

    # Tier 2: Moderate variance filtering (tot >= 2)
    Xtr_mod, ytr_mod = filter_tiered_train_rows(
        Xtr_raw, ytr_raw, utr_raw, min_impressions=2
    )

    all_prob_scores = []

    # 1. Train Strict-filtered models (Parent A mechanism)
    for member_i in range(num_strict):
        model_seed = args.seed + member_i * 1000 + 11
        preds = train_swa_fm(
            dim=dim,
            Xtr=Xtr_strict,
            ytr=ytr_strict,
            Xva=Xva,
            yva=yva,
            uva=uva,
            X_out=X_out,
            model_seed=model_seed,
            k=k,
            lr=lr,
            l2=l2_strict,
            epochs=epochs,
            bs=bs,
            patience=patience,
            top_k_checkpoints=top_k_checkpoints,
            verbose=verbose,
            member_name=f"Strict-{member_i+1}/{num_strict}",
        )
        all_prob_scores.append(preds)

    # 2. Train Moderate-filtered models (Parent A mechanism)
    for member_i in range(num_moderate):
        model_seed = args.seed + (member_i + 50) * 1000 + 22
        preds = train_swa_fm(
            dim=dim,
            Xtr=Xtr_mod,
            ytr=ytr_mod,
            Xva=Xva,
            yva=yva,
            uva=uva,
            X_out=X_out,
            model_seed=model_seed,
            k=k,
            lr=lr,
            l2=l2_moderate,
            epochs=epochs,
            bs=bs,
            patience=patience,
            top_k_checkpoints=top_k_checkpoints,
            verbose=verbose,
            member_name=f"Mod-{member_i+1}/{num_moderate}",
        )
        all_prob_scores.append(preds)

    # 3. Train Full-data models (Parent B mechanism: 50% ensemble capacity allocated to full catalog)
    for member_i in range(num_full):
        model_seed = args.seed + (member_i + 100) * 1000 + 33
        preds = train_swa_fm(
            dim=dim,
            Xtr=Xtr_raw,
            ytr=ytr_raw,
            Xva=Xva,
            yva=yva,
            uva=uva,
            X_out=X_out,
            model_seed=model_seed,
            k=k,
            lr=lr,
            l2=l2_full,
            epochs=epochs,
            bs=bs,
            patience=patience,
            top_k_checkpoints=top_k_checkpoints,
            verbose=verbose,
            member_name=f"Full-{member_i+1}/{num_full}",
        )
        all_prob_scores.append(preds)

    ensemble_scores = np.mean(all_prob_scores, axis=0)
    out = Path(args.output_scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, ensemble_scores)
    print(
        f"catalog_balanced_tiered_ensemble wrote {len(ensemble_scores)} scores "
        f"split={args.split} seed={args.seed} "
        f"num_strict={num_strict} num_mod={num_moderate} num_full={num_full}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())