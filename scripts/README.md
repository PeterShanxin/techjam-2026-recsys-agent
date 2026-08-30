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
