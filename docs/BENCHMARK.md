# Canonical benchmark

Source: [`evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json). If another doc disagrees, trust that file.

All **selection** scores are validation. Test was never used to pick experiments. The one
test run is reported in [Test observation](#test-observation-not-selection) below.

## Primary table (validation)

| Method | Starting priors | New evals | Best GAUC | Best nDCG@5 | Primary | Δ vs FM | Δ vs starting elite | LLM calls | Tokens | Wall-clock | Manual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Official / reproduced FM | none (root) | 0 | 0.6671326 | 0.5358049 | **0.6014688** | 0 | — | 0 | 0 | ~84s | 0 |
| Phase 3 sequential autonomous | fm-root | 3 | 0.6680555 | 0.5361663 | **0.6021109** | +0.0006422 | +0.0006422 | 3 | 47295 | ~413s | 0 |
| Phase 4 matched sequential | fm-root + 3-seed ensemble | 6 | 0.6680555 | 0.5361663 | **0.6021109** | +0.0006422 | 0 | 6 | 124036 | ~2492s | 0 |
| Phase 4 evolutionary search | fm-root + 3-seed ensemble | 6 | 0.6683660 | 0.5362713 | **0.6023186** | +0.0008499 | **+0.0002077** | 6 | 139830 | ~1985s | 0 |
| Sprint 2 evolutionary search | + frozen SWA7 elite | 7 | 0.6690881 | 0.5367193 | **0.6029037** | +0.0014350 | **+0.0005851** | 12 | 415105 | ~2761s | 0 |

Phase 3 discovered 3-seed bagging. The table uses the later **verified** `fm-ensemble-3seed` metrics, not the rounded 0.6021 in `phase3_acceptance.json`.

Sprint 2 ran only after an independent second-opinion audit found and fixed three
reachability defects in the search space. It stopped on **convergence**, not budget
(7 of 8 evaluations, stagnation 3). Its 12 LLM calls are 9 research + 3 repair.
See [`SECOND_OPINION_SPRINT.md`](SECOND_OPINION_SPRINT.md).

## Final candidate

`final-tiered-ensemble`, the sprint-2 autonomous elite `rs-20260831T062638Z-939b7000-008`
(crossover of 007 × 006). Selected on validation primary, the project's standing rule.

Eight official FM members, each averaging its top-2 validation-primary checkpoints (SWA),
split across three tiers that differ in **both** train-row selection and L2 strength:

| Members | Train rows | L2 |
| --- | --- | --- |
| 2 | users with ≥3 impressions and mixed labels | 1e-4 |
| 2 | users with ≥2 impressions and mixed labels | 1e-5 |
| 4 | full train split | 1e-6 |

Member scores are averaged as **raw FM scores (logits)**. `starter/kuairand/baseline.py`
`FM.predict` returns logits, so every ensemble in this project averages logits; earlier
docs said "probability mean", which was wrong. Nothing about any score or model changed —
only the description. See `aggregation_space_correction` in the canonical JSON.

Frozen at [`tiered_ensemble_scorer.py`](../src/research_agent/recommenders/tiered_ensemble_scorer.py),
specs [`final_tiered_valid.json`](../configs/experiments/final_tiered_valid.json) and
[`final_tiered_test.json`](../configs/experiments/final_tiered_test.json). Both specs pass an
**empty** config, because every tier setting is baked into the entrypoint — that is what makes
the live elite reproduce bitwise.

The superseded Phase 4 winner `final-swa7-ensemble` (0.6023186) is retained as historical
evidence and stays runnable:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_candidate.py --split valid --legacy-swa7
```

## Significance

Paired user bootstrap over validation users, 2000 reps, seed 7 (`scripts/paired_bootstrap.py`).
Paired because resampling users gives an **absolute** primary SD of about 0.0022 — larger than
every delta in this project. Two models scored on the same resampled users share that variance.

| Comparison | Δ | 95% CI | P(Δ>0) |
| --- | --- | --- | --- |
| `final-tiered-ensemble` vs `fm-root` | +0.0014350 | [+0.0002577, +0.0026169] | **0.990** |
| `final-tiered-ensemble` vs `final-swa7-ensemble` | +0.0005851 | [−0.0003427, +0.0015007] | 0.888 |
| `final-swa7-ensemble` vs `fm-root` (reference) | +0.0008499 | [−0.0002086, +0.0018664] | 0.934 |

Read this carefully:

- The final candidate is the **first** in this project whose margin over the FM root clears a
  95% paired interval. The Phase 4 candidate's did not.
- Its margin over the Phase 4 candidate does **not** clear 95%. **No significance is claimed
  there**, and none against the organizer convergence ε=0.002.
- Official FM 5-seed std on test is 0.0008, the same order as these deltas.

## Test observation (not selection)

Model selection was frozen on validation first. The candidate was then run on test **once**,
for the official CSV. These numbers did not change the model and no candidate was ever
compared on test before the freeze.

| Candidate | GAUC | nDCG@5 | Primary |
| --- | --- | --- | --- |
| `final-tiered-ensemble` (submitted) | 0.6631900 | 0.5295608 | **0.5963754** |
| `final-swa7-ensemble` (superseded) | 0.6631817 | 0.5295907 | 0.5963862 |

**The +0.00059 validation gain did not transfer.** On test the difference is −0.0000109 —
35× smaller than the paired test bootstrap SD (0.00038), with 95% CI [−0.00075, +0.00075] and
P(Δ>0) = 0.515. On test the two candidates are **statistically indistinguishable**.

The validation-selected candidate was kept anyway. Switching to the other one now, on the
strength of a test number, would be exactly the test-driven selection this project forbids.

Both candidates lose about 0.006 from validation to test (−0.00653 and −0.00593). That shift
is a property of the split, not of either model.

## Cost

Like-for-like: both candidates run as committed repo files through the same harness in the
foreground.

| Split | `final-tiered-ensemble` | `final-swa7-ensemble` | Δ |
| --- | --- | --- | --- |
| valid | **173.9s** | 225.0s | −51.1s (−22.7%) |
| test | **208.6s** | 216.7s | −8.1s (−3.7%) |

The final candidate is cheaper despite having one more member (8 vs 7), because four of its
members train on filtered rows and its members early-stop sooner. The test gap is smaller
because scoring 170,588 rows with 8 members costs more than with 7, offsetting part of the
training saving. 0 GPU-hours either way.

The live-elite figures (183.7s vs 383.4s) ran inside evolution sprints under different load
and are **not** a fair head-to-head. Use the table above.

## Robustness (separate from the primary table)

| Check | What it is | What it is not |
| --- | --- | --- |
| Mini-dataset scorer tests | CLI contract, finite scores, split length | Not KuaiRand metrics |
| Same-seed re-execution of `final-tiered-ensemble` | Type A: deterministic reproduction. Primary matched the live elite **exactly**: 0.6029037142533181 | Not Type B: different-seed variance. 1 replicate |
| Different-seed replicates of the final candidate | Type B: seeds 42 / 7 / 2024 → 0.6029037 / 0.6029838 / 0.6031161. **All three beat the superseded candidate** | Not a significance test. 3 replicates |
| Same-seed re-execution of `final-swa7-ensemble` | Type A. Primary matched its live elite exactly: 0.6023186326402106 | Same caveat |
| Organizer FM 5-seed std 0.0008 | Official starter evidence | Not our ensemble's std |

Seed 42 is kept because it is the agent's own choice. Seeds 7 and 2024 scored higher;
picking one after seeing the sweep would be selection across replicates.

## Resource roll-up

| | Phase 4 evolution | Phase 4 matched sequential | Sprint 2 evolution |
| --- | --- | --- | --- |
| LLM calls | 6 | 6 | 12 (9 research + 3 repair) |
| Input tokens | 91544 | 59612 | 298971 |
| Output tokens | 15745 | 17165 | 41662 |
| Thinking tokens | 32541 | 47259 | 74472 |
| Total tokens | 139830 | 124036 | 415105 |
| Wall-clock | 1984.5s | 2491.5s | 2761.4s |
| New evaluations | 6 | 6 | 7 |
| Stop reason | generation limit | generation limit | **converged** |
| GPU-hours | 0 | 0 | 0 |
| Manual interventions | 0 | 0 | 0 |
