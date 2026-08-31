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
| Reproducible final candidate in repo | READY | `src/research_agent/recommenders/tiered_ensemble_scorer.py` + `configs/experiments/final_tiered_valid.json` / `final_tiered_test.json`. Not a gitignored `runs/generated/` path. |
| Superseded candidate retained | READY | `fm_swa7_ensemble_scorer.py` + `final_swa7_*.json` kept as historical evidence; still runnable via `--legacy-swa7`. Provenance not rewritten. |

## Evaluation contract

| Item | Status | Notes |
| --- | --- | --- |
| Official metrics GAUC / nDCG@5 / primary | READY | Runner calls organizer `evaluate()`. |
| Validation-only research | READY | Test requires `--allow-test` / `allow_test_split`. Elite ranking never uses test. |
| Hidden-test / test-split rules | READY | Test used only after freeze, official CSV role. |
| Official CSV schema `row_id,user_id,video_id,score` | READY | `scripts/make_submission.py` + `submit.py --check`. |
| Row count / order / finite scores | READY | Packer + tests. Test split has 170588 rows. |
| Final prediction file generated | READY | Local `submission.csv` (gitignored), scored by `final-tiered-ensemble-test`. Official `submit.py --check` passed: 170588 test rows. Generate with `scripts/run_final_candidate.py --split test --allow-test` then `scripts/make_submission.py`. Do not commit the CSV. |
| Test split run exactly once | READY | One post-freeze run, after model selection closed on validation. Recorded as an observation in `canonical_benchmark.json`; it did not change the candidate. |
| Checkpoint bundle | OPTIONAL | Track 2 asks for the CSV of scores, not a torch checkpoint. Code is the repo. |

## Competition search envelope

| Item | Status | Notes |
| --- | --- | --- |
| Max 50 iterations | READY | `EvolutionConfig.competition()`, `configs/research/competition.json`, `--competition`. |
| Max 6h wall-clock | READY | 21600s. Proven with FakeProvider `wall_clock_seconds=0`. |
| Convergence ε=0.002 | READY | Default and competition config. |
| Patience N=3 | READY | Same. |
| 50-iteration **live** Gemini run | OPTIONAL | Decision **A**: do not burn hours. Sprint 2 stopped itself on **convergence** at 7 of 8 evaluations; delta vs the previous elite is +0.0005851 ≪ 0.002. |

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
| Devpost **submit** | BLOCKED | Human only. Not submitted. |
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
| Phase 4 + Phase 5 merged to `main` | READY | PR #11 and PR #12 both merged 2026-08-31. `main` is at Phase 5 (`ace0ca9`). |
| Post-Phase-5 optimization PR merged | BLOCKED | PR #14 open and up to date with `main`. Cursor Security + Approval pass, approved. Human merges; do not self-merge. |
| Local valid re-run of frozen candidate | READY | Type A: primary matched the live elite exactly (`0.6029037142533181`). Superseded candidate also still matches exactly (`0.6023186326402106`). |
| Paired-bootstrap evidence recorded | READY | `scripts/paired_bootstrap.py`; four comparisons in `canonical_benchmark.json`. Significance claimed only vs the FM root (P=0.990). |

## Longer search decision (locked)

**A — the matched pilot plus one post-audit sprint is enough. Technical search is closed.**

Phase 4 evolution wall ~33 min, 139830 tokens. Matched sequential ~42 min, 124036 tokens.
Sprint 2 ~46 min, 415105 tokens, and it stopped on **convergence** with a spare evaluation
in hand. A near-full 50-eval run would be many hours and still likely below ε=0.002. Software
can enforce the official envelope without spending it.

## Honest caveats a judge will ask about

| Question | Answer |
| --- | --- |
| Is the final candidate significantly better than the one it replaced? | **No.** +0.0005851 on validation, P(Δ>0)=0.888, CI includes zero. |
| Did the validation gain transfer to test? | **No.** −0.0000109, P(Δ>0)=0.515 — indistinguishable. Reported in full in `BENCHMARK.md`. |
| Then why keep it? | The selection rule (validation primary) was fixed before the test run. Switching after seeing a test number is test-driven selection. |
| Is anything significant? | Yes: the final candidate vs the FM root, P(Δ>0)=0.990, CI excludes zero. It is the first result here that clears 95%. |
| Any other advantage? | It is cheaper: 173.9s vs 225.0s on validation on the same harness, despite one more member. |
