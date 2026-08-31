# Canonical submission evidence

This folder is the **committed, sanitized** number source for README, demo, and Devpost.

Not included: API keys, datasets, `scores.npy`, huge generated trees, gitignored `runs/`.

| File | What |
| --- | --- |
| `canonical_benchmark.json` | Exact table used everywhere else, including the final candidate, the paired bootstraps, the runtime comparison and the single post-freeze test observation |
| `phase4_evolution.json` | Compact Phase 4 evolution summary |
| `phase4_matched_sequential.json` | Compact matched sequential control |
| `p0_performance_sprint.json` | P0 optimization sprint |
| `sprint2_autonomous_sprint.json` | Post-audit sprint 2: every experiment, reproduction, resources |
| `sprint2_008_vs_fmroot.json` | Paired bootstrap, final candidate vs FM root (valid) |
| `sprint2_008_vs_swa7.json` | Paired bootstrap, final candidate vs Phase 4 candidate (valid) |
| `swa7_vs_fmroot_valid.json` | Paired bootstrap, Phase 4 candidate vs FM root (valid, reference) |
| `final_tiered_vs_swa7_test.json` | Paired bootstrap on **test**, post-freeze observation only |
| `lineage_tree.txt` | Session-scoped Phase 4 lineage |
| `environment.json` | Machine / NumPy / evaluator |
| `evaluator.json` | `evaluate.py` fingerprint |

Older Phase 1–3 records stay in `docs/` (`baseline_reproduction.json`, `phase3_acceptance.json`). If a rounded figure disagrees with `canonical_benchmark.json`, **trust this folder**.
