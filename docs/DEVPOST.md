# Devpost draft — TikTok TechJam 2026 Track 2

Do not submit this from the agent unless a human authorizes it.

Numbers: [`evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json).

## Project name

LLM-guided evolutionary research for KuaiRand (Track 2)

## Tagline

Gemini proposes semantic experiments. A deterministic controller decides what survives.

## Inspiration / problem

Recommender research is a loop: hypothesize, implement, train, read GAUC and nDCG@5, try again. Humans are slow and inconsistent at that loop. Track 2 asks for an autonomous research agent on KuaiRand-Pure with a frozen official evaluator.

We did not want a giant AutoML search over random hyperparameters. We wanted a system that **does research** under budgets.

## What it does

The agent inspects validation evidence, writes a hypothesis, emits candidate Python, runs it through one experiment harness, and records metrics, diffs, tokens, and lineage.

A second component, the Evolution Controller, keeps a small population, preserves elites, suppresses duplicates, requests mutation or crossover, and stops on evaluation count, wall-clock, tokens, or convergence (ε=0.002, patience=3).

## Architecture

```
Research state
  → Gemini Research Agent (semantic mutation / crossover / repair / reflection)
  → generated candidate
  → deterministic ExperimentRunner
  → official evaluate.py
  → registry
  → Evolution Controller (population, elite, diversity, budgets)
  → next research state
```

Gemini never ranks elites. The controller never invents the loss or the model.

## How autonomy works

Zero manual candidate edits in the live pilots. `manual_interventions = 0`.

Intended production model: **Gemini 3.7 Flash**. Developer API capacity returned server-side high-demand (HTTP 500/503) while model listing still showed 3.7. Live evidence therefore used **Gemini 3.6 Flash**, medium thinking, bounded high-thinking repair. That is provider resilience, not a hidden human researcher.

## Evolutionary novelty

Sequential search keeps one parent chain. Evolution keeps a population, so a later crossover can combine two surviving ideas, and a failed child does not erase the elite.

In the matched pilot, crossover did **not** beat the best mutation. The architectural result still stands: the controller can request crossover, log incompatibility, and keep negatives as evidence.

In sprint 2 it did: the final candidate is a crossover of two surviving ensemble variants,
combining one parent's tier-adaptive regularization with the other's full-catalog member
allocation. Neither parent alone scored as well.

## The part we are proudest of: auditing our own search

Late in the project we ran an independent second-opinion review against our own conclusion
that the architecture was strong and further optimization was pointless. It found that the
search space had three **reachability** defects, all of them ours:

1. **Regularization and the training objective were unreachable.** `l2` sat pinned in the root
   parameters, appeared in no guidance list, and the operator prompt discouraged touching
   hyperparameters without evidence. The model was visibly overfitting and no proposal could
   act on it.
2. **`parent + alpha * residual` was a dominant degenerate strategy.** Five of seven offspring
   in one sprint tuned `alpha` on validation with `alpha = 0` inside the grid. That shape
   cannot score below its parent, so fitness rewarded it unconditionally — and the validity
   check only compared source text, so all of them counted as real hypotheses.
3. **Diversity was inert.** Seven of eight proposals left their family and mechanism metadata
   blank. The semantic signature collapsed to a single degenerate value, which short-circuits
   duplicate detection and starves crossover of two distinct parents. Both features had
   silently stopped working.

Every previous ranking-objective attempt had also failed on **runtime**, not evidence: the
pairwise candidates built per-sample Python loops over ~1.9M sampled pairs and hit the timeout.
That is an implementation failure and was never counted as science.

We fixed the **affordances**, not the architecture: named the unexplored axes without naming
any value, added a rejection rule for children whose within-user ordering barely differs from a
parent's, made family/mechanism/axis metadata required, stopped drawing every offspring from one
family, and added vectorized within-user grouping so a full 1.14M-row epoch costs 1.7s. A test
asserts that no measured value from the audit leaks into anything the agent reads, and that the
gradient FM we exposed contains **no loss function** — the objective stays the agent's choice.

Then we ran one more autonomous sprint. The agent reached both blind spots on its own: it tested
a within-user listwise softmax objective (a real negative result at last, not another timeout),
introduced tier-adaptive L2 unprompted, and independently rediscovered the organizer's finding
that varying embedding dimension does not help. It stopped on **convergence** with budget
remaining.

## Scientific-integrity safeguards

- Official `evaluate.py` fingerprint `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de`
- Research split is **valid** only
- Test requires an explicit opt-in and is for the final CSV only
- Unsupported imports (e.g. torch) are rejected before subprocess
- Claimed extra log fields must actually be read from raw CSVs
- Duplicate / no-op copies are not scientific evidence and cannot become elite
- Invalid / NaN scores never enter fitness

## Quantitative results (validation)

| Method | New evals | Best primary | Δ vs FM | Δ vs starting elite |
| --- | --- | --- | --- | --- |
| Reproduced FM | 0 | 0.6014688 | 0 | — |
| Phase 3 sequential (3-seed bagging) | 3 | 0.6021109 | +0.0006422 | +0.0006422 |
| Phase 4 matched sequential | 6 | 0.6021109 | +0.0006422 | 0 |
| Phase 4 evolution | 6 | 0.6023186 | +0.0008499 | +0.0002077 |
| Sprint 2 evolution (post-audit) | 7 | **0.6029037** | +0.0014350 | +0.0005851 |

Under the same prior knowledge and six new experiment evaluations, evolutionary search found an additional validation improvement while the matched sequential search did not surpass the starting elite. After the audit fixes, one further sprint improved it again and converged on its own.

Final candidate `final-tiered-ensemble`: 8 official FM members, top-2 checkpoint SWA each,
averaged as raw FM scores, split across three tiers differing in **both** train-row selection
and L2 strength (2× strict / 2× moderate / 4× full catalog). Discovered autonomously.

### How much of that is real?

We measured the noise floor before believing anything. Bootstrapping validation users gives an
**absolute** primary standard deviation of ~0.0022 — larger than every delta this project has
produced. Paired against a shared baseline it drops to ~0.0005. So we report paired user
bootstraps, 2000 reps:

| Comparison | Δ | 95% CI | P(Δ>0) |
| --- | --- | --- | --- |
| Final vs FM root | +0.0014350 | [+0.00026, +0.00262] | **0.990** |
| Final vs Phase 4 candidate | +0.0005851 | [−0.00034, +0.00150] | 0.888 |
| Phase 4 candidate vs FM root | +0.0008499 | [−0.00021, +0.00187] | 0.934 |

The final candidate is the first here whose margin over the FM root clears a 95% paired
interval. Its margin over the previous candidate does not, so **we claim no significance
there**, and none against the organizer ε=0.002.

### The test result, reported honestly

Selection was closed on validation. The candidate then ran on test **once**, for the official
CSV. Test primary **0.5963754**, versus the superseded candidate's 0.5963862.

**The validation gain did not transfer.** The test difference is −0.0000109 — 35× smaller than
the paired test bootstrap SD, P(Δ>0) = 0.515. On test the two candidates are statistically
indistinguishable.

We kept the validation-selected candidate anyway. Swapping it now because a test number came
back marginally lower would be exactly the test-driven selection the whole system is built to
prevent. Reporting this is the point: a research agent whose evidence standard only holds when
the answer is flattering is not an evidence standard.

## Resource efficiency

Phase 4 evolution: 6 LLM calls, 139830 tokens, ~33 minutes wall-clock, 0 GPU-hours, 0 manual interventions.

Sprint 2: 12 LLM calls (9 research + 3 repair), 415105 tokens, ~46 minutes, 0 GPU-hours,
**0 manual interventions**, 1 transport retry. It stopped on **convergence** at 7 of 8
evaluations — it declined to spend the rest of its own budget.

The final candidate is also cheaper to train than the one it replaced, despite having one more
member: **173.9s vs 225.0s** on validation (−22.7%) and 208.6s vs 216.7s on test (−3.7%), on the
same harness. Four of its eight members train on filtered rows and all of them early-stop
sooner. Better on validation and cheaper is a real Pareto improvement even where the metric
gain is not significant.

We did not spend a 50-iteration / 6-hour live budget. The software can enforce that envelope; FakeProvider tests prove the stop reasons.

## Reproducibility

Clone, install NumPy + pytest, download KuaiRand-Pure, run FakeProvider tests for free.

Live Gemini needs `GEMINI_API_KEY` in `.env` (never committed).

Final candidate: `scripts/run_final_candidate.py` then `scripts/make_submission.py`. See [`TESTING_INSTRUCTIONS.md`](TESTING_INSTRUCTIONS.md).

## Challenges

- NumPy-only starter blocked deep models without violating integrity.
- First live sequential session had a silent torch fallback; we treated it as invalid science and added import guards.
- 3.7 Flash was capacity-blocked on the Developer API; 3.6 Flash completed the pilots.
- Score deltas are small relative to organizer ε=0.002, so we had to build a paired bootstrap
  before we could tell a result from noise.
- Our own search space had blind spots we could not see from inside it. Finding them took an
  independent adversarial review of our own conclusion.

## Limitations

- Matched benchmark is six new evaluations, not 50.
- The final candidate's delta over the previous one is +0.0005851 on validation. **Not
  significant** (P=0.888), and it did not transfer to test (−0.0000109, P=0.515).
- Crossover did not outperform the best mutation in the first live pilot. It did in sprint 2.
- The winner's two ingredients — tiered row filtering and tier-adaptive L2 — are confounded in
  one candidate. Nobody ablated them.
- Three of nine sprint-2 research calls produced unusable proposals that claimed unavailable
  data fields. The data contract caught all three, but that is a third of the LLM budget.
- Sub-0.0005 differences are unmeasurable on this split. That is a property of the benchmark,
  not a defect we can fix.
- Longer runs may explore more; we chose not to burn compute for a sub-epsilon chase.

The architectural contribution still matters: a judge can see a real autonomous loop with lineage, budgets, and a matched sequential control.

## Future work

- Ablate the winner: tiered row filtering and tier-adaptive L2 separately, to learn which one
  actually carries the gain.
- Rewrite the lab store. Its 13s construction and per-call scalar API deterred every proposal
  from touching raw log fields; that is an affordance defect we found but chose not to fix.
- If 3.7 serving recovers, a single smoke then a bounded search — not repeated probing.
- A benchmark with enough users to resolve 0.0005 would make all of this measurable.

## Built with

Python, NumPy, official KuaiRand starter, Gemini (3.6 Flash live / 3.7 Flash intended), pytest.
