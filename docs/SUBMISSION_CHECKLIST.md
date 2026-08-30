# Track 2 submission checklist

Classified against official KuaiRand starter instructions (`starter/kuairand/README.md`, `submit.py`) and GitHub Issue #6.

Status key: **READY** / **NEEDS WORK** / **OPTIONAL** / **BLOCKED**.

Canonical numbers: [`docs/evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json).

## Required source / code

| Item | Status | Notes |
| --- | --- | --- |
| Research Agent + Evolution Controller | READY | Frozen. Phase 5 does not redesign them. |
| Official starter under `starter/kuairand/` | READY | Unmodified `evaluate.py`. |
| KuaiRand-Pure usage | READY | Local data dir; dataset **not** committed. |
| Reproducible final candidate in repo | READY | `src/research_agent/recommenders/fm_swa7_ensemble_scorer.py` + `configs/experiments/final_swa7_valid.json`. Not a gitignored `runs/generated/` path. |

## Evaluation contract

| Item | Status | Notes |
| --- | --- | --- |
| Official metrics GAUC / nDCG@5 / primary | READY | Runner calls organizer `evaluate()`. |
| Validation-only research | READY | Test requires `--allow-test` / `allow_test_split`. Elite ranking never uses test. |
| Hidden-test / test-split rules | READY | Test used only after freeze, official CSV role. |
| Official CSV schema `row_id,user_id,video_id,score` | READY | `scripts/make_submission.py` + `submit.py --check`. |
| Row count / order / finite scores | READY | Packer + tests. Test split has 170588 rows. |
| Final prediction file generated | READY | Local `submission.csv` (gitignored). Official `submit.py --check` passed: 170588 test rows. Generate with `scripts/run_final_candidate.py --split test --allow-test` then `scripts/make_submission.py`. Do not commit the CSV. |
| Checkpoint bundle | OPTIONAL | Track 2 asks for the CSV of scores, not a torch checkpoint. Code is the repo. |

## Competition search envelope

| Item | Status | Notes |
| --- | --- | --- |
| Max 50 iterations | READY | `EvolutionConfig.competition()`, `configs/research/competition.json`, `--competition`. |
| Max 6h wall-clock | READY | 21600s. Proven with FakeProvider `wall_clock_seconds=0`. |
| Convergence ε=0.002 | READY | Default and competition config. |
| Patience N=3 | READY | Same. |
| 50-iteration **live** Gemini run | OPTIONAL | Decision **A**: do not burn hours. Delta vs elite is +0.0002077 ≪ 0.002. |

## Iteration-level logs

| Item | Status | Notes |
| --- | --- | --- |
| Hypothesis / rationale | READY | Spec + traces. |
| Code diff | READY | Workspace diffs vs parent. |
| Metrics | READY | `result.json`. |
| Errors / recovery | READY | Failure kinds + repair traces. |
| Manual intervention count | READY | Ledger field; live runs = 0. |
| LLM input/output/thinking tokens | READY | Resource ledger. |
| Wall-clock | READY | Session summaries. |
| Iteration / evaluation count | READY | Sequential iterations; evolution `evaluated_offspring`. |
| GPU-hours | READY | 0. CPU NumPy only. |

## Docs / demo / Devpost

| Item | Status | Notes |
| --- | --- | --- |
| Judge-first README | READY | This phase. |
| Reproducibility instructions | READY | `docs/TESTING_INSTRUCTIONS.md`. |
| Architecture + lineage diagrams | READY | `docs/diagrams/`. |
| 3-minute demo **script** + shot list | READY | `docs/DEMO_SCRIPT.md`. |
| Demo **video recorded** | NEEDS WORK | Human records from the script. Software cannot upload the video. |
| Devpost draft | READY | `docs/DEVPOST.md`. |
| Devpost **submit** | BLOCKED | Human only. Do not submit unless authorized. |
| Testing instructions | READY | Includes which commands spend API money. |

## Integrity

| Item | Status | Notes |
| --- | --- | --- |
| Evaluator fingerprint | READY | `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de` |
| No test leakage into selection | READY | Guard + tests. |
| Secrets not committed | READY | `.env` gitignored. Never print `GEMINI_API_KEY`. |
| Dataset not committed | READY | `.gitignore`. |

## Human / process

| Item | Status | Notes |
| --- | --- | --- |
| Phase 4 merged to `main` | BLOCKED | GitHub `main` is still Phase 3 (`68357aa`). PR #11 is open. Phase 5 branched from Phase 4 HEAD, not stale main. |
| Phase 5 PR merged | BLOCKED | Open the PR. Do not self-merge. |
| Local valid re-run of frozen candidate | READY | Type A: primary matched live elite exactly (`0.6023186326402106`). |

## Longer search decision (locked)

**A — existing 6-evaluation matched pilot is enough.**

Phase 4 evolution wall ~33 min, 139830 tokens. Matched sequential ~42 min, 124036 tokens. A near-full 50-eval run would be many hours and still likely below ε=0.002. Software can enforce the official envelope without spending it.
