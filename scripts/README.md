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
