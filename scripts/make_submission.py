"""Pack official Track 2 CSV from a 1-D scores.npy. Does not train or select."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
STARTER = ROOT / "starter" / "kuairand"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))

from research_agent.submission import load_score_vector, write_official_csv


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("KUAI_RAND_DATA_DIR")
    if env:
        return Path(env)
    return ROOT / "starter" / "kuairand" / "KuaiRand-Pure" / "data"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write row_id,user_id,video_id,score CSV from scores.npy.")
    ap.add_argument("--scores", required=True, help="Path to 1-D scores.npy")
    ap.add_argument("--split", required=True, choices=("valid", "test"))
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)

    from data import load

    data_dir = _resolve_data_dir(args.data_dir)
    rows = load(str(data_dir))[args.split]
    scores = load_score_vector(Path(args.scores))
    dest = write_official_csv(Path(args.output), rows, scores)
    print(f"wrote {dest} rows={len(rows)} split={args.split}")
    print("next: starter/kuairand/submit.py --check --split", args.split, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
