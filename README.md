# TikTok TechJam 2026 - Autonomous ML Research Agent for Recommender Systems

Track 2 submission workspace for TikTok TechJam 2026.

## Goal

Build an autonomous research agent that can inspect a recommender-system task, propose hypotheses, modify experiment code, run and evaluate experiments, reflect on results, and iteratively improve the model with minimal human intervention.

A planned extension is evolutionary experiment search: maintain several candidate research directions, select strong experiments, mutate promising configurations, and recombine compatible improvements while preserving diversity and compute efficiency.

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
3. Track hypothesis, code/config diff, metrics, runtime, token usage, and outcome.
4. Prefer autonomous research decisions over manual intervention.
5. Use evolutionary search only where it improves search quality - not as brute-force hyperparameter tuning.
6. Optimize for the official ranking metrics and compute efficiency.

## Quick start

Python 3.9+ is required by the official starter kit.

```bash
python starter/kuairand/baseline.py --model random
python starter/kuairand/baseline.py --model fm
```

The dataset is intentionally not committed. See `starter/kuairand/README.md` for the official download instructions.

## Status

- [x] Track selected
- [x] Official starter kit preserved
- [ ] Reproduce official FM baseline
- [ ] Define experiment schema and registry
- [ ] Implement autonomous research loop
- [ ] Add evolutionary experiment search
- [ ] Run controlled benchmark experiments
- [ ] Generate final submission checkpoint and CSV
- [ ] Record 3-minute demo

## Attribution

The KuaiRand-Pure starter files under `starter/kuairand/` were provided by the TikTok TechJam 2026 organizers for Track 2.
