# TikTok TechJam 2026 - Autonomous ML Research Agent for Recommender Systems

Track 2 submission workspace for TikTok TechJam 2026.

## Goal

Build an autonomous research agent that can inspect a recommender-system task, propose hypotheses, modify experiment code, run and evaluate experiments, reflect on results, and iteratively improve the model with minimal human intervention.

The architecture is an **LLM-guided evolutionary research system**:

- The **Research Agent** owns semantic ML reasoning, hypothesis generation, meaningful mutation and conditional crossover proposals.
- The **Evolution Controller** owns deterministic fitness, selection, elitism, diversity, lineage and resource-budget enforcement.
- The evolutionary layer is intentionally lightweight and hypothesis-driven, not brute-force AutoML.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the frozen responsibility split and staged V1-V3 design. Phase 2 harness contract: [`docs/EXPERIMENT_HARNESS.md`](docs/EXPERIMENT_HARNESS.md).

## Benchmark

The official starter benchmark uses **KuaiRand-Pure** and evaluates within-user ranking with:

- GAUC
- nDCG@5
- Primary score = mean(GAUC, nDCG@5)

Official FM test baseline: **0.5946 primary**.

The unmodified organizer starter kit is kept under `starter/kuairand/` for reproducibility and attribution.

## Repository layout

```text
configs/                    Experiment and agent configs
docs/                       Architecture and design notes
notebooks/                  Exploratory notebooks only
scripts/                    Entry points and utility scripts
src/research_agent/
  agent/                     Research planning, reflection, orchestration
  evolution/                 Selection, mutation, crossover, diversity
  experiments/               Experiment specs, runner, registry
  recommenders/              Candidate recommender implementations
  evaluation/                Metric adapters and result analysis
starter/kuairand/            Official KuaiRand-Pure starter kit
tests/                       Automated tests
```

## Development principles

1. Preserve the official evaluator and task definition.
2. Make every experiment reproducible and machine-readable.
3. Track hypothesis, code/config diff, metrics, runtime, token usage, lineage and outcome.
4. Prefer autonomous research decisions over manual intervention.
5. Let the LLM propose meaningful semantic changes; keep selection and budgets deterministic.
6. Use evolutionary search only where it improves search quality - not as brute-force hyperparameter tuning.
7. Optimize for the official ranking metrics and compute efficiency.

## Quick start

Python 3.9+ is required by the official starter kit.

```bash
python scripts/run_baseline.py --model random --seed 0
python scripts/run_baseline.py --model fm --seed 0
python scripts/run_experiment.py --spec configs/experiments/random_valid.json
```

The dataset is intentionally not committed. See `starter/kuairand/README.md` for download instructions and [`docs/BASELINE_REPRODUCTION.md`](docs/BASELINE_REPRODUCTION.md) for the Phase 1 reproduction record.

## Status

- [x] Track selected
- [x] Official starter kit preserved
- [x] Freeze Research Agent vs Evolution Controller responsibility split
- [x] Reproduce official FM baseline
- [x] Define exact experiment schema and registry
- [ ] Implement sequential autonomous research loop
- [ ] Add evolutionary branching
- [ ] Add semantic crossover
- [ ] Run controlled benchmark experiments
- [ ] Generate final submission checkpoint and CSV
- [ ] Record 3-minute demo

## Attribution

The KuaiRand-Pure starter files under `starter/kuairand/` were provided by the TikTok TechJam 2026 organizers for Track 2.
