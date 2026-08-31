# P0 performance sprint

Epic #13 items P0-1 and P0-2. Validation only. Test sealed. Phase 5 `canonical_benchmark.json` is unchanged. Submission candidate stays `final-swa7-ensemble`.

Machine-readable copy: [`evidence/p0_performance_sprint.json`](evidence/p0_performance_sprint.json).

## Starting best

`final-swa7-ensemble` validation primary **0.6023186326402106**.

## Expanded lab (P0-1)

Helpers in `src/research_agent/lab/`: split-safe history, popularity, affinity, catalogs, pairwise samples, recency utilities. Facts only. No hidden ranker.

Smoke (not winners):

| Mechanism | ID | GAUC | nDCG@5 | Primary |
| --- | --- | --- | --- | --- |
| History / recency | smoke-history-recency | 0.6373390087984021 | 0.5220386099468598 | 0.5796888093726309 |
| Pairwise BPR | smoke-pairwise-bpr | 0.498921630506908 | 0.46735264286811085 | 0.48313713668750946 |

## Live sprint (P0-2)

Session `rs-20260831T031604Z-3c1f1f0f`. Model `gemini-3.6-flash`, thinking medium. Population 4, elites 2, 3 generations, max 8 new evals. Stop: `generation_limit`. New evals used: 7. Manual interventions: 0.

Priors (no new-eval spend): `fm-root`, `fm-ensemble-3seed`, `final-swa7-ensemble`.

| ID | Class | Family | Primary | Delta vs 0.6023186 |
| --- | --- | --- | --- | --- |
| 001 | B timeout | pairwise BPR-FM 7-seed | — | — |
| 002 | B timeout | pairwise BPR-FM 3-seed | — | — |
| 003 | A tiny positive | train user-author residual + SWA7 | 0.6023274346172034 | +8.80e-6 |
| 004 | A negative | user-author + user-tab residual | 0.6021332955877018 | -1.85e-4 |
| 005 | D / proposal fail | crossover proposal failed | — | — |
| 006 | B write bug | logit residual; FM logits treated as probs | 0.5465376095606709 | not science |
| 007 | A negative | recency-decayed user-author residual | 0.6019741685838249 | -3.44e-4 |
| 008 | C extra prior unused | gamma=0; scores bitwise = 003 | 0.6023274346172034 | +8.80e-6 |

003 mechanism did run: 4223 / 124909 valid rows differ from frozen SWA (user-author overlap). Score corr vs frozen ≈ 0.9999995. Lift is far below FM seed noise (~0.0008).

008 independently retrained seed=0 and wrote **identical** scores to 003. That is a same-seed reproduce of 003, not a new author-quality prior.

Frozen repo copy: `src/research_agent/recommenders/fm_affinity_residual_scorer.py`. Valid spec: `configs/experiments/p0_affinity_residual_valid.json`.

Repo-copy valid rerun `p0-affinity-residual-valid-rerun`: primary **0.6023274346172034**, scores bitwise identical to 003. Do not run test. Do not replace `submission.csv`.

## P0-3

Not justified. No second competitive family. Natural crossovers already happened and stayed inside FM + tiny residual.

## Integrity

Test not touched. Evaluator SHA-256 `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de`. Timeouts and the 006 write bug are not negative science.
