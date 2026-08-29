# Phase 1 — Official KuaiRand baseline reproduction

Trusted local benchmark harness for TikTok TechJam 2026 Track 2. Official evaluator semantics are unchanged. Modeling improvements are out of scope for this phase.

Machine-readable copy: [`baseline_reproduction.json`](baseline_reproduction.json).

## Environment

| | |
|---|---|
| OS | Windows 11 (build 26200) |
| Architecture | ARM64 |
| Python | 3.14.2 (`CPython`, 64-bit ARM64) |
| Virtualenv | `.venv/` at repo root (gitignored) |
| pip | 26.2.1 |
| numpy | 2.5.2 |
| pytest | 9.1.1 |

Create the same environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install "numpy>=1.26,<3" pytest
```

The official starter needs only NumPy. pytest is a test extra. No PyTorch / pandas / sklearn.

No organizer-code patch was required on this machine.

## Dataset

Follow `starter/kuairand/README.md`.

```powershell
curl.exe -L --fail -o starter\kuairand\KuaiRand-Pure.tar.gz "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1"
tar -xzf starter\kuairand\KuaiRand-Pure.tar.gz -C starter\kuairand
```

| | |
|---|---|
| Source | https://zenodo.org/records/10439422 |
| Archive MD5 | `0820331067a3784d9691136f772b35a7` (matches Zenodo) |
| Default path | `starter/kuairand/KuaiRand-Pure/data` |
| Override | `--data_dir` or env `KUAI_RAND_DATA_DIR` |

Do not commit the archive, extract, checkpoints, or submission CSVs. `.gitignore` already excludes `KuaiRand-Pure/`, `*.tar.gz`, `submission.csv`, and `.venv/`.

Observed official `data.load()` split sizes:

| split | rows | date min | date max |
|---|---|---|---|
| train | 1,141,112 | 20220409 | 20220421 |
| valid | 124,909 | 20220422 | 20220428 |
| test | 170,588 | 20220429 | 20220508 |

The advertised train window starts on 20220408. The Pure standard log has no 20220408 rows; the first train date is 20220409. Filter semantics in `data.py` were not changed.

test `(user_id, video_id)` is not unique: 5,133 duplicate pairs (3.10%), max repeat 12. That is why submissions key on `row_id`.

## Commands and seeds

Run from `starter/kuairand` so official imports resolve, or use the repo-root wrapper.

### Random sanity check (seed 0)

```powershell
# official entry, cwd = starter/kuairand
C:\FormerD\Repos\techjam-2026-recsys-agent\.venv\Scripts\python.exe baseline.py --model random --seed 0 --data_dir ./KuaiRand-Pure/data
```

Equivalent wrapper from repo root:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --model random --seed 0
```

### Official FM (seed 0)

```powershell
C:\FormerD\Repos\techjam-2026-recsys-agent\.venv\Scripts\python.exe baseline.py --model fm --seed 0 --data_dir ./KuaiRand-Pure/data
```

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --model fm --seed 0
```

FM config is the official default: `k=16`, `lr=0.001`, `batch=8192`, `max_epochs=40`, `patience=4`, fields `user_id, video_id, author_id, tab, dur_bucket`.

## Measured results

### Random (seed 0)

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.4990 | 0.4663 | 0.4827 |
| test | 0.4999 | 0.4514 | **0.4757** |

Wall time: **8.346 s** (includes CSV load).

### FM (seed 0)

Early stop at epoch 11.

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.6671 | 0.5358 | 0.6015 |
| test | 0.6621 | 0.5286 | **0.5953** |

Wall time: **79.220 s** (includes load and training). README estimate was ~40 s on a single CPU core; this ARM64 Windows box is slower, not a correctness issue.

## Reference comparison

Published test references (`starter/kuairand/baseline_scores.json`):

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (mean seeds 0–4) | 0.4996 | 0.4511 | 0.4753 |
| FM (official) | 0.6610 | 0.5282 | 0.5946 |
| FM std over 5 seeds | 0.0008 | 0.0008 | 0.0008 |

Random test primary 0.4757 is **+0.0004** from 0.4753, inside the required **±0.001** sanity window.

FM seed 0 vs published mean:

| metric | measured | reference | delta | vs 0.0008 std |
|---|---|---|---|---|
| GAUC | 0.6621 | 0.6610 | +0.0011 | ~1.4σ |
| nDCG@5 | 0.5286 | 0.5282 | +0.0004 | ~0.5σ |
| primary | 0.5953 | 0.5946 | +0.0007 | ~0.9σ |

Not bit-for-bit equal (published numbers are 5-seed means). Deltas are inside known seed noise. No harness investigation was required.

## Submission / evaluator wiring

Official `submit.py` was exercised without `--make` (that retrains FM). A random-score CSV was written with `write_submission` and checked with the official CLI.

Verified:

- `row_id` is 0-based and contiguous in `data.load()[split]` order
- `user_id` / `video_id` match the evaluation rows
- finite scores round-trip (format `%.6g`)
- `python submit.py --check --split test` accepts a correct file (170,588 rows)
- `python submit.py --score --split valid` accepts a correct file and matches in-memory `evaluate()`
- misaligned `video_id` is rejected
- `NaN` / `Inf` scores are rejected

`starter/kuairand/evaluate.py` SHA-256: `735b429e6223572ecb48c0f44953b0fbc39b9f6e4e7a1e9b06326d7fe0d0f58c` (2593 bytes). Unchanged.

## How to re-check

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\run_baseline.py --model random --seed 0
```

The smoke suite covers evaluator hash/semantics, submission alignment, official split sizes, and the random primary window. It does not retrain FM.

## Organizer code

No semantic edits to `starter/kuairand/evaluate.py`, `data.py`, `baseline.py`, or `submit.py`. The wrapper only puts `starter/kuairand` on `sys.path` so those modules can be imported from the repo root.
