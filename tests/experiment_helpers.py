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


def write_lab_dataset(root: Path) -> Path:
    """Mini dump with history, aux, catalogs, and valid/test-only items."""
    data = root / "data"
    data.mkdir(parents=True)
    (data / "video_features_basic_pure.csv").write_text(
        "video_id,author_id,video_type,music_id,video_duration,tag\n"
        "v1,a1,normal,m1,1000,t1\n"
        "v2,a2,normal,m2,2000,t2\n"
        "v3,a1,normal,m1,1500,t1\n"
        "v4,a2,ad,m3,1800,t3\n"
        "v9,a9,normal,m9,3000,t9\n"
        "v_test,a9,normal,m9,4000,t9\n",
        encoding="utf-8",
    )
    (data / "user_features_pure.csv").write_text(
        "user_id,user_active_degree,follow_user_num_range,onehot_feat0\n"
        "u1,high,0-10,1\n"
        "u2,low,11-20,0\n",
        encoding="utf-8",
    )
    (data / "video_features_statistic_pure.csv").write_text(
        "video_id,show_cnt,like_cnt\n"
        "v1,10,1\n"
        "v9,999999,999\n"
        "v_test,888888,888\n",
        encoding="utf-8",
    )
    (data / "log_standard_4_08_to_4_21_pure.csv").write_text(
        "date,user_id,video_id,tab,duration_ms,long_view,hourmin,time_ms,is_like,is_click,play_time_ms\n"
        "20220410,u1,v1,0,1000,1,1000,1,1,1,800\n"
        "20220411,u1,v2,0,2000,0,1100,2,0,1,100\n"
        "20220412,u1,v3,0,1500,1,1200,3,1,1,900\n"
        "20220413,u1,v4,0,1800,0,1300,4,0,0,50\n"
        "20220410,u2,v1,0,1000,0,1000,5,0,1,200\n"
        "20220411,u2,v2,0,2000,1,1100,6,1,1,1500\n",
        encoding="utf-8",
    )
    (data / "log_standard_4_22_to_5_08_pure.csv").write_text(
        "date,user_id,video_id,tab,duration_ms,long_view,hourmin,time_ms,is_like,is_click,play_time_ms\n"
        "20220423,u1,v1,0,1000,1,1400,7,1,1,800\n"
        "20220423,u1,v2,0,2000,0,1500,8,0,1,100\n"
        "20220423,u1,v9,0,3000,1,1600,9,1,1,2000\n"
        "20220423,u2,v1,0,1000,0,1400,10,0,1,200\n"
        "20220423,u2,v2,0,2000,0,1500,11,0,1,100\n"
        "20220430,u1,v1,0,1000,1,1700,12,1,1,800\n"
        "20220430,u1,v_test,0,4000,1,1800,13,1,1,3000\n",
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
