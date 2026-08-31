# 3-minute demo script

Target: ~3:00. One story. Do not overstuff.

Numbers from [`evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json).

## Narrative

### 0:00–0:25 — Problem

ML research loops are slow. A human repeatedly hypothesizes, writes code, runs KuaiRand, reads GAUC / nDCG@5, and decides the next try.

Track 2 asks for an **autonomous research agent**, not a one-shot model.

### 0:25–0:55 — Architecture

We split the job:

- **Gemini** is the semantic researcher. It observes evidence, forms a hypothesis, writes candidate code, mutates or crosses parents, repairs broken implementations, and reflects.
- A **deterministic Evolution Controller** owns survival: fitness, population, elites, parent selection, diversity, lineage, and budgets.

This is not generic AutoML. The LLM never picks the elite. The controller never invents the ML idea.

Show [`diagrams/architecture.mmd`](diagrams/architecture.mmd).

### 0:55–1:35 — One real autonomous loop

Show one Phase 4 turn from the live trace (session `rs-20260830T133522Z-0e304128`):

1. Hypothesis (percentile-rank parent → SWA + 7-seed raw average).
2. Generated candidate (short diff).
3. Official evaluator on **validation**.
4. Result: primary **0.6023186**.

Say out loud: test split is locked. Research never sees it.

### 1:35–2:10 — Evolutionary tree

Show [`diagrams/lineage.mmd`](diagrams/lineage.mmd) / `evidence/lineage_tree.txt`.

Point at:

- `fm-root` baseline
- `fm-ensemble-3seed` starting elite
- negative branches (logit average, soft labels)
- mutation winner `…-004`
- crossover `…-005` implementation failure
- crossover `…-006` ran, scored worse

Elites survived. Failures stayed in the log.

### 2:10–2:35 — Matched comparison

Same priors: `fm-root` + 3-seed ensemble. Six new evaluations each.

- Sequential best stayed **0.6021109**.
- Evolution reached **0.6023186** (+0.0002077 vs the starting elite).

Line: *Under the same prior knowledge and six new experiment evaluations, evolutionary search found an additional validation improvement while the matched sequential search did not surpass the starting elite.*

Do **not** say statistically significant.

### 2:35–3:00 — Feasibility

- Live model: Gemini 3.6 Flash after 3.7 Developer API high-demand. Intended model remains 3.7. No hidden steering.
- Evolution: 6 LLM calls, 139830 tokens, ~33 min, 0 GPU-hours, **0 manual interventions**.
- Software enforces 50 evals / 6h / ε=0.002 / patience=3. We did not burn the full budget for a sub-epsilon hunt.
- Repo is reproducible: FakeProvider tests are free; live Gemini needs `GEMINI_API_KEY`.

End on the architecture, not the 4th decimal.

## Shot list (screen recording)

| Time | Shot | File / command | Voice |
| --- | --- | --- | --- |
| 0:00 | README top (problem + one-liner) | `README.md` | Humans are the bottleneck |
| 0:25 | Architecture mermaid | `docs/diagrams/architecture.mmd` | LLM researches; controller governs |
| 0:55 | Trace snippet + `result.json` metrics | `runs/rs-20260830T133522Z-0e304128-004/result.json` if local; else `docs/evidence/phase4_evolution.json` | Hypothesis → code → official metric |
| 1:35 | Lineage tree | `docs/evidence/lineage_tree.txt` | Branches, elite, crossover, negatives |
| 2:10 | Benchmark table | `docs/BENCHMARK.md` | Matched 6 vs 6 |
| 2:35 | Resource row + testing commands | `docs/TESTING_INSTRUCTIONS.md` | Flash, tokens, zero interventions, FakeProvider |

Keep each shot 10–20 seconds. No live 40-epoch FM training in the video unless already cached on screen.

## Out of scope for the take

- 50-iteration live run
- Claiming significance
- Showing `.env` or any key
- Scoring the test split as if it chose the model
