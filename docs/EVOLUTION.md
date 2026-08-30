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

`is_like`, `play_time_ms`, and other aux log columns are **not** on those tuples. A proposal that claims them is allowed only if the candidate actually reads raw CSVs (`csv.DictReader` or `csv.reader`, a raw filename, and the claimed field in source). Filename comments or `data.load()` snippets are not enough. That blocks the Phase 3 soft-label no-op from counting as evidence.

Same source as a parent is a semantic no-op only when parameters and seed also match. A parameter-only mutation still counts as `hypothesis_tested`.

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

Traces: `runs/evolution/<session>/summary.json`, `population.json`, `generations.jsonl`, `tree.txt`. Lineage export is **session-scoped**: the registry stays the global source of truth; `tree.txt` includes the current session plus required seed ancestors only.

Matched sequential control is a Phase-3-style sequential search with the **same Generation-0 priors** as evolution (`fm-root` + verified `fm-ensemble-3seed`) and the same number of **new** evaluations. The ensemble seed is prior knowledge; it does not consume the new-evaluation budget. Control uses an independent registry so it cannot see evolution-only children. Old unmatched FM-only sequential runs are not the benchmark.

```powershell
.\.venv\Scripts\python.exe scripts\run_research_agent.py --model gemini-3.6-flash --thinking medium --iterations 6 --timeout 1800 --runs-dir runs\sequential-matched --with-ensemble-prior
```

Or via the evolution CLI (runs evolution first, then a fresh sequential registry):

```powershell
.\.venv\Scripts\python.exe scripts\run_evolution.py --model gemini-3.6-flash --thinking medium --sequential-control
```
