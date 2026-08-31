# 3-minute demo script

Target: ~3:00. One story. Do not overstuff.

Numbers from [`evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json).

## Narrative

### 0:00–0:20 — Problem

ML research loops are slow. A human repeatedly hypothesizes, writes code, runs KuaiRand, reads GAUC / nDCG@5, and decides the next try.

Track 2 asks for an **autonomous research agent**, not a one-shot model.

### 0:20–0:45 — Architecture

We split the job:

- **Gemini** is the semantic researcher. It observes evidence, forms a hypothesis, writes candidate code, mutates or crosses parents, repairs broken implementations, and reflects.
- A **deterministic Evolution Controller** owns survival: fitness, population, elites, parent selection, diversity, lineage, and budgets.

This is not generic AutoML. The LLM never picks the elite. The controller never invents the ML idea.

Show [`diagrams/architecture.mmd`](diagrams/architecture.mmd).

### 0:45–1:15 — One real autonomous loop

Show one live turn from session `rs-20260831T062638Z-939b7000`:

1. Hypothesis (crossover of two surviving ensemble variants).
2. Generated candidate (short diff).
3. Official evaluator on **validation**.
4. Result: primary **0.6029037** — frozen as `final-tiered-ensemble`, an 8-member tiered FM
   ensemble the agent designed itself.

Say out loud: test split is locked. Research never sees it.

### 1:15–2:00 — The turn of the story: we audited our own search

This is the centrepiece. Do not rush it.

We had concluded that our architecture was strong and further optimization was pointless.
We ran an **independent second-opinion review against our own conclusion**. It found three
reachability defects in our search space:

1. Regularization and the training objective were **unreachable** — no proposal could touch them.
2. `parent + alpha * residual` with `alpha` tuned on validation **cannot score below its parent**,
   so fitness rewarded a degenerate shape unconditionally.
3. Blank diversity metadata had **silently disabled** duplicate suppression and crossover.

Plus a runtime bug that made every ranking-objective attempt time out instead of producing evidence.

We fixed the **affordances, not the architecture** — and a test asserts that no measured value
from the audit leaks into anything the agent reads.

Then the agent found the axes by itself: a within-user listwise objective (a genuine negative
result at last, not another timeout), tier-adaptive L2 unprompted, and an independent
rediscovery that varying embedding dimension does not help. It stopped on **convergence** with
budget still on the table.

### 2:00–2:30 — Evidence, including the part that did not work

Show [`BENCHMARK.md`](BENCHMARK.md).

- Matched 6-vs-6: sequential stayed **0.6021109**; evolution reached **0.6023186**.
- Post-audit sprint 2 reached **0.6029037**.
- Paired user bootstrap: vs the FM root, **P(Δ>0) = 0.990** — the first result here that clears
  95%. Vs the previous candidate, P = 0.888, interval includes zero. **We claim no significance
  there.**

Then say the uncomfortable part out loud:

> We ran the frozen candidate on test once. The validation gain **did not transfer** —
> 0.5963754 versus 0.5963862, P(Δ>0) = 0.515. On test the two candidates are indistinguishable.
> We kept the validation-selected one anyway, because changing it on the strength of a test
> number is the exact failure mode this system exists to prevent.

Do **not** say statistically significant about the candidate-vs-candidate comparison.

### 2:30–3:00 — Feasibility

- Live model: Gemini 3.6 Flash after 3.7 Developer API high-demand. Intended model remains 3.7. No hidden steering.
- Sprint 2: 12 LLM calls, 415105 tokens, ~46 min, 0 GPU-hours, **0 manual interventions**, stop reason **converged**.
- The final candidate is also **cheaper** than the one it replaced: 173.9s vs 225.0s on validation, same harness, despite one more member.
- Software enforces 50 evals / 6h / ε=0.002 / patience=3. We did not burn the full budget for a sub-epsilon hunt.
- Repo is reproducible: FakeProvider tests are free; live Gemini needs `GEMINI_API_KEY`.

End on the architecture and the audit, not the 4th decimal.

## Shot list (screen recording)

| Time | Shot | File / command | Voice |
| --- | --- | --- | --- |
| 0:00 | README top (problem + one-liner) | `README.md` | Humans are the bottleneck |
| 0:20 | Architecture mermaid | `docs/diagrams/architecture.mmd` | LLM researches; controller governs |
| 0:45 | Trace snippet + `result.json` metrics | `runs/rs-20260831T062638Z-939b7000-008/result.json` if local; else `docs/evidence/sprint2_autonomous_sprint.json` | Hypothesis → code → official metric |
| 1:15 | Audit findings table | `docs/SECOND_OPINION_SPRINT.md` §1 | Three blind spots, all ours |
| 1:35 | The fix diff + leak test | `src/research_agent/evolution/identity.py`, `tests/test_search_space_exposure.py` | Affordances, not architecture. No value leaked |
| 2:00 | Benchmark + bootstrap tables | `docs/BENCHMARK.md` | Matched 6 vs 6; P=0.990 vs root, P=0.888 vs previous |
| 2:20 | Test observation table | `docs/BENCHMARK.md` §Test observation | It did not transfer, and we said so |
| 2:30 | Resource row + testing commands | `docs/TESTING_INSTRUCTIONS.md` | Flash, tokens, zero interventions, converged, cheaper |

Keep each shot 10–20 seconds. No live 40-epoch FM training in the video unless already cached on screen.

## Out of scope for the take

- 50-iteration live run
- Claiming significance for the candidate-vs-candidate delta
- Showing `.env` or any key
- Scoring the test split as if it chose the model
- Hiding the non-transferring test result
