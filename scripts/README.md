# Scripts

## Official baseline

`run_baseline.py` is a repo-root wrapper around the organizer starter. It does not change `evaluate.py`.

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline.py --model random --seed 0
.\.venv\Scripts\python.exe scripts\run_baseline.py --model fm --seed 0
```

Default data path is `starter/kuairand/KuaiRand-Pure/data`. Override with `--data_dir` or `KUAI_RAND_DATA_DIR`.

Optional `--json-out path.json` writes metrics and wall time.

See [`docs/BASELINE_REPRODUCTION.md`](../docs/BASELINE_REPRODUCTION.md).

## Experiment harness

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --spec configs\experiments\random_valid.json
```

`--allow-test` is required for `evaluation_split=test`. Details: [`docs/EXPERIMENT_HARNESS.md`](../docs/EXPERIMENT_HARNESS.md).

## Research agent

```powershell
.\.venv\Scripts\python.exe scripts\run_research_agent.py --iterations 3 --model gemini-3.7-flash --thinking medium
.\.venv\Scripts\python.exe scripts\run_gemini_smoke.py
```

Requires `GEMINI_API_KEY` in the process environment or repo-root `.env`. Unit tests use `FakeProvider` and spend no API money. Intended model is `gemini-3.7-flash`. First live validation used `--model gemini-3.6-flash` after Developer API 3.7 high-demand errors; see [`docs/RESEARCH_AGENT.md`](../docs/RESEARCH_AGENT.md).

## Evolution controller

```powershell
.\.venv\Scripts\python.exe scripts\run_evolution.py --model gemini-3.6-flash --thinking medium --generations 2 --max-new-evaluations 6
```

Deterministic population/fitness/elitism. Gemini only proposes semantic mutation/crossover. Details: [`docs/EVOLUTION.md`](../docs/EVOLUTION.md).

Competition envelope (50 evals, 6h, ε=0.002, patience=3). FakeProvider spends no API money:

```powershell
.\.venv\Scripts\python.exe scripts\run_evolution.py --provider fake --competition
```

## Frozen final candidate

In-repo copy of the sprint-2 autonomous elite `final-tiered-ensemble`. Not a gitignored
`runs/generated/` path. `--legacy-swa7` runs the superseded Phase 4 winner instead.

```powershell
.\.venv\Scripts\python.exe scripts\run_final_candidate.py --split valid
.\.venv\Scripts\python.exe scripts\run_final_candidate.py --split valid --legacy-swa7
.\.venv\Scripts\python.exe scripts\run_final_candidate.py --split test --allow-test
.\.venv\Scripts\python.exe scripts\make_submission.py --scores runs\final-tiered-ensemble-test\scores.npy --split test --output submission.csv
```

## Paired user bootstrap

Validation deltas in this project are smaller than the absolute bootstrap SD (~0.0022),
so comparisons must be paired on the same resampled users.

```powershell
.\.venv\Scripts\python.exe scripts\paired_bootstrap.py --baseline runs\fm-root\scores.npy --candidate runs\sprint2-tiered-ensemble\scores.npy
```

`--split test` additionally requires `--allow-test` and is post-freeze observation only.

Test split is submission-only. `--allow-test` does not feed elite ranking. `submission.csv` is gitignored.

