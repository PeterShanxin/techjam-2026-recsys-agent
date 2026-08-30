"""Official-style FM candidate.

Trains the organizer FM on official train, writes ordered scores for the
requested split. Authoritative metrics stay with ExperimentRunner.

Early-stop uses organizer evaluate() as part of the official training loop.
That is not the ExperimentResult metric path.
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
    ap = argparse.ArgumentParser(description="Write official FM scores for one KuaiRand split.")
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
    verbose = bool(config.get("verbose", True))

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    if args.split not in enc:
        raise SystemExit(f"unknown split {args.split}")
    X_out, _y_out, _u_out = enc[args.split]

    model = FM(dim, k=k, lr=lr, l2=l2, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        losses = [
            model.step(Xtr[idx[i : i + bs]], ytr[idx[i : i + bs]])
            for i in range(0, len(idx), bs)
        ]
        va = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(
                f"  epoch {epoch:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s"
            )
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break
    if best_state is None:
        raise SystemExit("FM training produced no checkpoint")
    model.V, model.W, model.b = best_state
    scores = np.asarray(model.predict(X_out), dtype=np.float64)
    out = Path(args.output_scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    print(
        f"fm_scorer wrote {len(scores)} scores split={args.split} "
        f"seed={args.seed} k={k} lr={lr}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
