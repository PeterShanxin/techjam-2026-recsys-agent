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

Under the same prior knowledge and six new experiment evaluations, evolutionary search found an additional validation improvement while the matched sequential search did not surpass the starting elite.

We do **not** claim statistical significance.

## Resource efficiency

Phase 4 evolution: 6 LLM calls, 139830 tokens, ~33 minutes wall-clock, 0 GPU-hours, 0 manual interventions.

We did not spend a 50-iteration / 6-hour live budget. The software can enforce that envelope; FakeProvider tests prove the stop reasons.

## Reproducibility

Clone, install NumPy + pytest, download KuaiRand-Pure, run FakeProvider tests for free.

Live Gemini needs `GEMINI_API_KEY` in `.env` (never committed).

Final candidate: `scripts/run_final_candidate.py` then `scripts/make_submission.py`. See [`TESTING_INSTRUCTIONS.md`](TESTING_INSTRUCTIONS.md).

## Challenges

- NumPy-only starter blocked deep models without violating integrity.
- First live sequential session had a silent torch fallback; we treated it as invalid science and added import guards.
- 3.7 Flash was capacity-blocked on the Developer API; 3.6 Flash completed the pilots.
- Score deltas are small relative to organizer ε=0.002.

## Limitations

- Matched benchmark is six new evaluations, not 50.
- Evolution delta vs starting elite is +0.0002077. Not significant.
- Crossover did not outperform the best mutation in the first live pilot.
- Longer runs may explore more; we chose not to burn compute for a sub-epsilon chase.

The architectural contribution still matters: a judge can see a real autonomous loop with lineage, budgets, and a matched sequential control.

## Future work

- One optional same-seed re-run of the frozen candidate for Type A reproducibility.
- If 3.7 serving recovers, a single smoke then a bounded search — not repeated probing.
- Pairwise / listwise losses if they stay inside NumPy or a declared extra dependency.

## Built with

Python, NumPy, official KuaiRand starter, Gemini (3.6 Flash live / 3.7 Flash intended), pytest.
