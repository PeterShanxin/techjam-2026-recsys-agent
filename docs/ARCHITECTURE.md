# Architecture - Working Draft

## Core loop

```text
Benchmark / current best
        |
        v
Research Agent
  - inspect evidence
  - propose hypotheses
  - choose next experiment
        |
        v
Experiment Specification
  - parent experiment(s)
  - hypothesis
  - model/loss/features/config
  - expected mechanism
        |
        v
Code / Config Mutation
        |
        v
Sandboxed Run
        |
        v
Official Evaluator
  - GAUC
  - nDCG@5
  - primary score
        |
        v
Experiment Registry
  - metrics
  - runtime
  - token usage
  - failure info
  - code/config diff
        |
        v
Reflection + Selection
        |
        +-----------------------> next iteration
```

## Evolutionary extension

Use a small population of research directions rather than brute-force search.

- Fitness: ranking score with optional efficiency penalty
- Elitism: retain best validated experiments
- Mutation: change one meaningful research dimension
- Crossover: combine compatible improvements from strong parents
- Diversity: penalize near-duplicate proposals
- Budget: cap wall-clock, model calls, and experiment count

The research agent remains responsible for proposing and justifying experiments. The evolutionary layer organizes search and preserves useful discoveries.
