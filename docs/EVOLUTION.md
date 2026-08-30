# Phase 4 — Evolutionary experiment search

LLM-guided evolutionary research. **Not** a raw-code genetic algorithm.

One sentence: the Research Agent decides what experiments mean. The Evolution Controller decides what survives.

## Responsibility split

### Research Agent

- Observes evidence (including the compact data contract)
- Proposes a hypothesis and why
- Semantic mutation of one parent
- Semantic crossover of two parents when the controller asks
- Writes candidate code
- Reflects on scientific results
- Bounded repair of implementation failures

The agent does **not** pick elites, fitness, or budgets.

### Evolution Controller

- Owns the population
- Fitness = validation primary (test never enters)
- Elite preservation
- Parent choice
- Duplicate / diversity suppression
- Compute and token budgets
- Stop / convergence
- Lineage bookkeeping via the Phase 2 registry `parent_ids`

The controller does **not** invent ML hypotheses.

## Data contract

`ResearchState.data_contract` is derived from `starter/kuairand/data.py`.

`data.load()` returns split → list of 7-tuples:

| index | field |
| --- | --- |
| 0 | date |
| 1 | user_id |
| 2 | video_id |
| 3 | author_id (joined) |
| 4 | tab |
| 5 | duration_ms |
| 6 | long_view (official target) |

`is_like`, `play_time_ms`, and other aux log columns are **not** on those tuples. A proposal that claims them without reading the raw CSVs is rejected before execution. That blocks the Phase 3 soft-label no-op from counting as evidence.

## Pilot defaults

```text
population_size = 4
elite_count = 2
generations = 2
max_new_evaluations = 6
```

Seeds: official FM root + verified 3-seed FM ensemble (`fm-ensemble-3seed`).

Fitness: validation primary. Runtime/tokens are tie-breakers and reporting, not the objective.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_evolution.py --model gemini-3.6-flash --thinking medium --generations 2 --max-new-evaluations 6
```

Intended production model remains `gemini-3.7-flash`. Pass `--model` explicitly when Developer API serving of 3.7 is constrained.

Traces: `runs/evolution/<session>/summary.json`, `population.json`, `generations.jsonl`, `tree.txt`.

Matched sequential control (same new-evaluation count, no population):

```powershell
.\.venv\Scripts\python.exe scripts\run_evolution.py --model gemini-3.6-flash --thinking medium --sequential-control
```
