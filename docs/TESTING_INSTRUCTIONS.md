# Testing and reproducibility

Python 3.9+. Official starter needs NumPy only.

Never commit `.env`. Never paste `GEMINI_API_KEY` into chat, logs, or git.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install "numpy>=1.26,<3" pytest
```

Dataset (not in git):

```powershell
curl.exe -L --fail -o starter\kuairand\KuaiRand-Pure.tar.gz "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1"
tar -xzf starter\kuairand\KuaiRand-Pure.tar.gz -C starter\kuairand
```

Optional live Gemini: copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Process env wins over the file.

## Zero-cost tests (no API money)

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Uses `FakeProvider` and a mini dataset. Does **not** call Gemini. Does **not** train full KuaiRand.

Evaluator lock:

`starter/kuairand/evaluate.py` SHA-256 (LF) = `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de`

## Baseline (CPU, no Gemini)

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --model random --seed 0
.\.venv\Scripts\python.exe scripts\run_baseline.py --model fm --seed 0
```

Needs KuaiRand-Pure. FM is ~1 minute on this machine.

## One experiment (CPU, no Gemini)

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --spec configs\experiments\fm_valid.json
.\.venv\Scripts\python.exe scripts\run_experiment.py --spec configs\experiments\final_tiered_valid.json
```

The frozen candidate trains 8 tiered FM members with SWA. Expect ~3 minutes (173.9s measured).
The superseded Phase 4 candidate is `configs\experiments\final_swa7_valid.json` (7 members, 225.0s).

`--allow-test` is required for any test split. Do not use test to pick a model.

## Sequential agent (**spends Gemini money**)

```powershell
.\.venv\Scripts\python.exe scripts\run_research_agent.py --iterations 3 --model gemini-3.6-flash --thinking medium
```

Intended model remains `gemini-3.7-flash`. Pass it explicitly only if a **one-shot** smoke succeeds. Do not probe 3.7 in a loop.

## Evolutionary search (**spends Gemini money**)

Pilot (what we actually ran):

```powershell
.\.venv\Scripts\python.exe scripts\run_evolution.py --model gemini-3.6-flash --thinking medium --generations 2 --max-new-evaluations 6
```

Competition envelope (software; do not fire a 50-eval live run unless a human decides it is worth it):

```powershell
.\.venv\Scripts\python.exe scripts\run_evolution.py --provider fake --competition
.\.venv\Scripts\python.exe scripts\run_evolution.py --model gemini-3.6-flash --thinking medium --competition
```

The FakeProvider line spends $0 and will stop on empty script / wall. The Gemini line can run up to 50 evals / 6h.

Matched sequential control:

```powershell
.\.venv\Scripts\python.exe scripts\run_research_agent.py --model gemini-3.6-flash --thinking medium --iterations 6 --timeout 1800 --runs-dir runs\sequential-matched --with-ensemble-prior
```

## Final artifact (CPU; test split is opt-in)

Validation freeze (no Gemini):

```powershell
.\.venv\Scripts\python.exe scripts\run_final_candidate.py --split valid
```

Superseded Phase 4 candidate, kept reproducible as historical evidence:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_candidate.py --split valid --legacy-swa7
```

Official test CSV **after** the candidate is frozen. This does **not** feed elite ranking:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_candidate.py --split test --allow-test
.\.venv\Scripts\python.exe scripts\make_submission.py --scores runs\final-tiered-ensemble-test\scores.npy --split test --output submission.csv
.\.venv\Scripts\python.exe starter\kuairand\submit.py --check --split test --data_dir starter\kuairand\KuaiRand-Pure\data submission.csv
```

`submission.csv` is gitignored. Do not commit it. Do not `--score` the test file to choose a model. `--score` is only meaningful on local **valid**.

## What spends money

| Command | API | Heavy CPU |
| --- | --- | --- |
| `pytest tests` | no | no |
| `run_baseline.py` / `run_experiment.py` / `run_final_candidate.py` | no | yes if KuaiRand |
| `--provider fake` | no | only if it actually trains |
| `run_research_agent.py` / `run_evolution.py` with Gemini | **yes** | yes |

## Numbers

Do not copy scores from memory. Use [`evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json).
