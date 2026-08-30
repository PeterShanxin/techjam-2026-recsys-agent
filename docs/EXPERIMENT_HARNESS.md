# Phase 2 — Experiment harness

Stable substrate for later ResearchAgent / EvolutionController code. No LLM, no evolution policy, no model search.

Machine-readable validation record: [`phase2_validation.json`](phase2_validation.json) (written after the real KuaiRand run).

## Data flow

```text
ExperimentSpec
    → ExperimentRunner
    → candidate subprocess
    → scores.npy
    → official evaluate(user_ids, labels, scores)
    → ExperimentResult
    → ExperimentRegistry (SQLite) + run artifacts
```

Public import:

```python
from research_agent.experiments import (
    ExperimentSpec,
    ExperimentRunner,
    ExperimentResult,
    ExperimentRegistry,
)
```

## ExperimentSpec

Model-agnostic request. Do not put `model=FM` / `loss=BPR` on the schema. Those belong in `parameters` or the candidate file.

| field | role |
| --- | --- |
| `experiment_id` | unique run identity |
| `spec_hash` | execution fingerprint (computed) |
| `parent_ids` | 0 / 1 / many |
| `origin` | `baseline` \| `manual` \| `mutation` \| `crossover` |
| `implementation` | `entrypoint`, optional `source_root`, `extra_paths` |
| `parameters` | free-form JSON object |
| `seed` | candidate seed |
| `evaluation_split` | `valid` (default) or `test` |
| `allow_test_split` | explicit test opt-in on the spec |
| `timeout_seconds` | subprocess budget |
| `hypothesis`, `rationale`, `tags`, `notes` | prose only |

`spec_hash` hashes only:

- `schema_version`
- `implementation`
- `parameters`
- `seed`
- `evaluation_split`

It does **not** include `experiment_id`, hypothesis, origin, parents, tags, notes, or timeout. Two reruns may share a hash and differ in id.

Parent rules:

- `baseline`: zero parents
- `mutation`: exactly one
- `crossover`: two or more
- `manual`: any

## Candidate CLI

```text
python <entrypoint> \
  --data-dir <path> \
  --split valid \
  --output-scores <attempt_dir>/scores.npy \
  --seed <seed> \
  --config <run_dir>/config.json
```

`--config` is a JSON file (Windows-safe). The candidate writes a 1-d finite score vector in requested-split row order. It must not call `evaluate()`.

Example candidate: `src/research_agent/recommenders/random_scorer.py`.

## ExperimentRunner

1. Reject `experiment_id` collision **before** any disk/registry mutation. Identity is immutable once registered.
2. Insert spec (same `spec_hash` rerun is allowed; different hash is not)
3. Create `runs/<experiment_id>/` and write identity files from the **registered** spec
4. Start a fresh `attempts/<attempt_id>/` and clear published execution artifacts (`scores.npy`, logs, metadata)
5. Fingerprint source + config
6. Run candidate as a subprocess with `--output-scores` pointing at the **attempt** file
7. Capture stdout / stderr / return code
8. Load only that attempt's scores; reject missing, wrong length/shape, NaN/Inf, or leftover published files
9. Load the official split via `data.load()`
10. Call only `evaluate(user_ids, labels, scores)`
11. Write `ExperimentResult` and persist to SQLite

A later attempt that exits 0 without writing scores is `invalid`. It never evaluates a previous `scores.npy`.

Isolation is filesystem + subprocess + timeout. Not a security sandbox.

## ExperimentResult

Raw outcome. No `accepted` / `rejected` here.

Statuses: `success`, `failed`, `timeout`, `invalid`.

Success stores official `GAUC`, `nDCG@5`, `primary`. Other statuses store structured `failure`.

## ExperimentRegistry

SQLite file, default `runs/registry.sqlite`.

Primitives:

- `insert_spec` / `upsert_result` / `get`
- `query_by_status` / `successful`
- `find_by_spec_hash`
- `parents` / `children` / `ancestry`
- `mark_decision` (`pending` \| `accepted` \| `rejected`)
- `elite` / `rank_validation` — validation only
- `rollback_target` — first parent

Elite = highest successful **validation** primary among non-rejected rows. Ties: earlier `created_at`, then `experiment_id`. Test rows never enter elite or validation rank.

## Validation / test policy

Autonomous research uses `evaluation_split="valid"`.

Test requires explicit opt-in:

- `ExperimentSpec.allow_test_split=True`, or
- `ExperimentRunner(allow_test=True)` / CLI `--allow-test`

Without that, the runner records `invalid` and does not score.

## Example command

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --spec configs\experiments\random_valid.json
```

Prints `experiment_id`, `status`, `split`, `GAUC`, `nDCG@5`, `primary`, `runtime`, `run_dir`.

Observed random / valid / seed 0: GAUC 0.4990, nDCG@5 0.4663, primary 0.4827 (9.727s). Matches Phase 1.

## Run artifact layout

```text
runs/<experiment_id>/
  spec.json
  config.json
  metadata.json
  stdout.log
  stderr.log
  scores.npy          # published only if THIS attempt wrote scores
  result.json
  attempts/<attempt_id>/
    scores.npy
    stdout.log
    stderr.log
    metadata.json
    result.json
runs/registry.sqlite
```

All of the above are gitignored.

## Phase 3 notes

Implemented in [`RESEARCH_AGENT.md`](RESEARCH_AGENT.md). The agent emits `ExperimentSpec` JSON and calls `ExperimentRunner.run`. Evidence comes from `ExperimentResult` plus `ExperimentRegistry.rank_validation()` / `elite()`. Generated code does not own official metrics. Search stays on `evaluation_split="valid"`.
