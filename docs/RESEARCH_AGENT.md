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

## Scientific integrity

ResearchState includes a compact `environment` snapshot: Python version, platform, architecture, allowed third-party packages (from project deps that actually import, currently `numpy`), and known unsupported packages such as `torch`.

Before `ExperimentRunner` launches, generated code is AST-checked:

- stdlib, `numpy`, and starter modules `data` / `baseline` / `evaluate` are allowed
- other third-party imports (for example `import torch`) become `unsupported_dependency` and trigger bounded repair, not a subprocess `ImportError`
- `try: import torch except ImportError: ...` style silent FM/parent fallback is `silent_dependency_fallback` and is also repaired, never stored as a successful hypothesis test

Traces record `research_validity`: `root` for the FM baseline, `hypothesis_tested` only when the runner actually executed the candidate, `not_executed` when preflight/parse rejected it. Execution `success` on the official evaluator is unchanged. A silent FM substitute must not count as evidence for the claimed method.

Repair prompts tell the model to keep the original hypothesis and reimplement with NumPy when the blocked package was only an implementation choice.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_research_agent.py --iterations 3 --model gemini-3.7-flash --thinking medium
```

Smoke (requires `GEMINI_API_KEY`):

```powershell
.\.venv\Scripts\python.exe scripts\run_gemini_smoke.py
```

Traces: `runs/research/<session-id>/trace.jsonl`, `report.md`, `summary.json`.

ResearchState includes a compact **data contract** derived from `starter/kuairand/data.py`: `data.load()` 7-tuples, encode fields, official `long_view` target, and columns that are **not** on the loader (including `is_like` and `play_time_ms`). Candidates may import `research_agent.lab.SplitSafeStore` for train-only history, popularity, catalogs, and pairwise samples. A proposal that claims aux fields is rejected unless the candidate actually reads the raw CSVs **or** calls those lab train-aux/history APIs. Filename comments are not enough. Test is sealed. See [`EVOLUTION.md`](EVOLUTION.md) for the Phase 4 controller and [`RESEARCH_SPACE_AUDIT.md`](RESEARCH_SPACE_AUDIT.md) for the field inventory.

## Live validation

Intended production model stays `gemini-3.7-flash`. The first closed-loop KuaiRand validation did **not** use it.

On 30 Aug 2026 the same `GEMINI_API_KEY` could list `gemini-3.7-flash` and complete AI Studio chat, but Developer API serving failed:

- Interactions `POST /v1beta/interactions` → HTTP 500 high-demand
- `generateContent` `POST /v1beta/models/gemini-3.7-flash:generateContent` → HTTP 503 high-demand
- Control: `generateContent` `gemini-3.6-flash` → HTTP 200

Phase 3 was time-boxed onto `gemini-3.6-flash` (`--model gemini-3.6-flash --thinking medium`). No Vertex path. No automatic fallback code.

Machine-readable record: [`phase3_live_validation.json`](phase3_live_validation.json). Full traces stay under gitignored `runs/research/rs-20260830T095957Z-c4eebeaf/`.

Session `rs-20260830T095957Z-c4eebeaf` is the **first pilot**, not a Phase 3 PASS. Iteration 1 reported harness success at primary 0.6015 identical to FM because the candidate caught missing `torch` and trained official FM. That is a semantic no-op, not a BPR result. Iteration 2 crashed on `import torch`. Iteration 3 ran a real BPR+BCE candidate (primary 0.5982). Environment-aware preflight was added after that pilot.

Integrity acceptance session `rs-20260830T105206Z-2fb7f092` (`gemini-3.6-flash`, thinking medium): 3 executed research iterations, 0 repairs, 0 torch imports, 0 silent ImportError fallbacks. Iteration 2 (batch/lr schedule) and iteration 3 (3-seed FM bagging) actually changed the method; bagging is the new elite at primary 0.6021. Iteration 1 claimed soft labels but `data.load` rows have no `is_like` / `play_time_ms`, so labels stayed binary FM targets. Machine-readable copy: [`phase3_acceptance.json`](phase3_acceptance.json).
