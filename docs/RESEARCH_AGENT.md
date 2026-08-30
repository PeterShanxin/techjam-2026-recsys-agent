# Phase 3 — Sequential Research Agent

Closed-loop ML researcher. Gemini reasons. Phase 2 harness still owns execution, official metrics, registry, elite, and split policy.

## Loop

```text
ResearchState
    → one Gemini 3.7 Flash call (thinking=medium)
    → structured ResearchProposal (reflection + next experiment + full candidate source)
    → isolated generated workspace
    → ExperimentSpec
    → ExperimentRunner
    → official validation evaluate()
    → ExperimentRegistry
    → next ResearchState
```

No population. No crossover. No evolutionary selection pressure. That is Phase 4.

## LLM

Production: `GeminiProvider` talks to the official Interactions REST API with `urllib` (`POST /v1beta/interactions`). No `google-genai` SDK.

- intended model: `gemini-3.7-flash`
- thinking: `medium` on research calls (`generation_config.thinking_level`)
- `high` only on bounded repair
- credential: `GEMINI_API_KEY` from process env, else repo-root `.env` (`override=False`)
- missing key fails fast before FM training
- structured JSON via `response_format` + JSON schema
- responses parsed from completed `steps` / `model_output` (not legacy `outputs`)
- usage metadata is first-party; missing fields stay `None` (no token estimates)

There is no automatic model fallback. If `gemini-3.7-flash` is unavailable on the Developer API, pass `--model` explicitly. First live validation used `gemini-3.6-flash` for that reason; see [Live validation](#live-validation).

Tests use `FakeProvider` or a scripted HTTP transport. They must not call Gemini.

Experiment IDs are session-scoped: `rs-<timestamp>-<rand>-001`. Generated candidate files are never overwritten.

## Code mutation

Each iteration writes a full replacement `candidate.py` under `runs/generated/<id>/`.

The file is syntax-checked, fingerprinted, and diffed against the selected parent. It never writes into `starter/` or `src/` during a research run. This is a controlled workspace, not a security sandbox.

The official evaluator stays immutable. Candidates write `scores.npy` only.

Failed, timeout, invalid, or corrupt stored roots are not reused. A later session allocates `fm-root-r001`, `fm-root-r002`, … and must succeed on validation before any paid research call.

## FM root

Research starts from a usable `fm-root` (success + valid split + finite metrics):

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --spec configs\experiments\fm_valid.json
```

Expected validation (seed 0, Phase 1): GAUC ≈ 0.6671, nDCG@5 ≈ 0.5358, primary ≈ 0.6015.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_research_agent.py --iterations 3 --model gemini-3.7-flash --thinking medium
```

Smoke (requires `GEMINI_API_KEY`):

```powershell
.\.venv\Scripts\python.exe scripts\run_gemini_smoke.py
```

Traces: `runs/research/<session-id>/trace.jsonl`, `report.md`, `summary.json`.

## Live validation

Intended production model stays `gemini-3.7-flash`. The first closed-loop KuaiRand validation did **not** use it.

On 30 Aug 2026 the same `GEMINI_API_KEY` could list `gemini-3.7-flash` and complete AI Studio chat, but Developer API serving failed:

- Interactions `POST /v1beta/interactions` → HTTP 500 high-demand
- `generateContent` `POST /v1beta/models/gemini-3.7-flash:generateContent` → HTTP 503 high-demand
- Control: `generateContent` `gemini-3.6-flash` → HTTP 200

Phase 3 was time-boxed onto `gemini-3.6-flash` (`--model gemini-3.6-flash --thinking medium`). No Vertex path. No automatic fallback code.

Machine-readable record: [`phase3_live_validation.json`](phase3_live_validation.json). Full traces stay under gitignored `runs/research/rs-20260830T095957Z-c4eebeaf/`.

Session `rs-20260830T095957Z-c4eebeaf`: reused usable `fm-root`, 3 research calls, 0 repairs, 0 manual edits. Best remained `fm-root` (primary 0.6015). Iteration 2 failed (`import torch` missing). Iterations 2 and 3 did change direction from prior evidence, imperfectly: iteration 1 silently fell back to official FM when torch was absent, so its “no gain” was not a real BPR test.
