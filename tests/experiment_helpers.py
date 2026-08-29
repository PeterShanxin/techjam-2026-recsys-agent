"""Shared builders for Phase 2 harness tests."""
from __future__ import annotations

from pathlib import Path

from research_agent.experiments import ExperimentSpec, ImplementationRef

CANDIDATE_SOURCE = '''\
import argparse
import json
import time
from pathlib import Path

import numpy as np
from data import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output-scores", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    action = cfg.get("action", "succeed")
    if action == "fail":
        raise SystemExit(2)
    if action == "sleep":
        time.sleep(float(cfg.get("sleep", 30)))
    rows = load(args.data_dir)[args.split]
    n = len(rows)
    if action == "missing":
        return 0
    if action == "wrong_length":
        np.save(args.output_scores, np.zeros(n + 1))
        return 0
    if action == "wrong_shape":
        np.save(args.output_scores, np.zeros((n, 2)))
        return 0
    if action == "nan":
        scores = np.zeros(n)
        scores[0] = np.nan
        np.save(args.output_scores, scores)
        return 0
    if action == "inf":
        scores = np.zeros(n)
        scores[0] = np.inf
        np.save(args.output_scores, scores)
        return 0
    rng = np.random.default_rng(args.seed)
    np.save(args.output_scores, rng.random(n))
    print("ok", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_mini_dataset(root: Path) -> Path:
    data = root / "data"
    data.mkdir(parents=True)
    (data / "video_features_basic_pure.csv").write_text(
        "video_id,author_id\nv1,a1\nv2,a2\n",
        encoding="utf-8",
    )
    (data / "log_standard_4_08_to_4_21_pure.csv").write_text(
        "date,user_id,video_id,tab,duration_ms,long_view\n"
        "20220410,u1,v1,0,1000,1\n"
        "20220410,u1,v2,0,2000,0\n",
        encoding="utf-8",
    )
    (data / "log_standard_4_22_to_5_08_pure.csv").write_text(
        "date,user_id,video_id,tab,duration_ms,long_view\n"
        "20220423,u1,v1,0,1000,1\n"
        "20220423,u1,v2,0,2000,0\n"
        "20220423,u2,v1,0,1000,0\n"
        "20220423,u2,v2,0,2000,0\n"
        "20220430,u1,v1,0,1000,1\n"
        "20220430,u1,v2,0,2000,0\n",
        encoding="utf-8",
    )
    return data


def write_candidate(path: Path) -> Path:
    path.write_text(CANDIDATE_SOURCE, encoding="utf-8")
    return path


def make_spec(**overrides) -> ExperimentSpec:
    impl = overrides.pop("implementation", None)
    if impl is None:
        impl = ImplementationRef(entrypoint="src/research_agent/recommenders/random_scorer.py")
    payload = {
        "experiment_id": overrides.pop("experiment_id", "exp-a"),
        "implementation": impl,
        "hypothesis": "test",
        "rationale": "test",
        "origin": "manual",
        "parameters": {"action": "succeed"},
        "seed": 0,
        "evaluation_split": "valid",
        "timeout_seconds": 30.0,
    }
    payload.update(overrides)
    return ExperimentSpec(**payload)
