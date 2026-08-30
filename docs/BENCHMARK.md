# Canonical benchmark

Source: [`evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json). If another doc disagrees, trust that file.

All scores are **validation**. Test was not used to pick experiments.

## Primary table

| Method | Starting priors | New evals | Best GAUC | Best nDCG@5 | Primary | Δ vs FM | Δ vs starting elite | LLM calls | Tokens | Wall-clock | Manual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Official / reproduced FM | none (root) | 0 | 0.6671326 | 0.5358049 | **0.6014688** | 0 | — | 0 | 0 | ~84s | 0 |
| Phase 3 sequential autonomous | fm-root | 3 | 0.6680555 | 0.5361663 | **0.6021109** | +0.0006422 | +0.0006422 | 3 | 47295 | ~413s | 0 |
| Phase 4 matched sequential | fm-root + 3-seed ensemble | 6 | 0.6680555 | 0.5361663 | **0.6021109** | +0.0006422 | 0 | 6 | 124036 | ~2492s | 0 |
| Phase 4 evolutionary search | fm-root + 3-seed ensemble | 6 | 0.6683660 | 0.5362713 | **0.6023186** | +0.0008499 | **+0.0002077** | 6 | 139830 | ~1985s | 0 |

Phase 3 discovered 3-seed bagging. The table uses the later **verified** `fm-ensemble-3seed` metrics, not the rounded 0.6021 in `phase3_acceptance.json`.

## Cautious conclusion

Under the same prior knowledge and six new experiment evaluations, evolutionary search found an additional validation improvement while the matched sequential search did not surpass the starting elite.

Do **not** call the score difference statistically significant. Official FM 5-seed std on **test** is 0.0008; our evolution delta vs the starting elite is +0.0002077, below the organizer ε=0.002.

## Robustness (separate from the primary table)

| Check | What it is | What it is not |
| --- | --- | --- |
| Mini-dataset scorer tests | CLI contract, finite scores, split length | Not KuaiRand metrics |
| Same-seed re-execution of frozen SWA+7-seed (`final-swa7-ensemble`) | Type A: deterministic reproduction | Not Type B: different-seed variance. 1 replicate. Primary matched live elite **exactly**: 0.6023186326402106 |
| Organizer FM 5-seed std 0.0008 | Official starter evidence | Not our SWA+7-seed std |

Do not claim repeated-seed robustness from rerunning the same seed.

## Resource roll-up (Phase 4 matched pair)

| | Evolution | Matched sequential |
| --- | --- | --- |
| LLM calls | 6 | 6 |
| Input tokens | 91544 | 59612 |
| Output tokens | 15745 | 17165 |
| Thinking tokens | 32541 | 47259 |
| Total tokens | 139830 | 124036 |
| Wall-clock | 1984.5s | 2491.5s |
| New experiments | 6 | 6 |
| GPU-hours | 0 | 0 |
| Manual interventions | 0 | 0 |
