# Canonical submission evidence

This folder is the **committed, sanitized** number source for README, demo, and Devpost.

Not included: API keys, datasets, `scores.npy`, huge generated trees, gitignored `runs/`.

| File | What |
| --- | --- |
| `canonical_benchmark.json` | Exact table used everywhere else |
| `phase4_evolution.json` | Compact Phase 4 evolution summary |
| `phase4_matched_sequential.json` | Compact matched sequential control |
| `lineage_tree.txt` | Session-scoped Phase 4 lineage |
| `environment.json` | Machine / NumPy / evaluator |
| `evaluator.json` | `evaluate.py` fingerprint |

Older Phase 1–3 records stay in `docs/` (`baseline_reproduction.json`, `phase3_acceptance.json`). If a rounded figure disagrees with `canonical_benchmark.json`, **trust this folder**.
