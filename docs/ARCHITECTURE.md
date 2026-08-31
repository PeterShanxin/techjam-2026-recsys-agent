# Architecture - MVP Freeze

## Core decision

The system is an **LLM-guided evolutionary research system**.

The Research Agent owns semantic ML reasoning. The Evolution Controller owns deterministic search pressure and resource discipline.

This separation is deliberate:

- The LLM is better than hand-written genetic operators at proposing meaningful, hypothesis-driven changes.
- Deterministic selection, elitism, diversity and budget rules make the search reproducible, auditable and less prone to LLM mode collapse.
- The evolutionary layer must not become brute-force AutoML.

## Responsibility split

### Research Agent - semantic researcher

Owns:

- Inspecting benchmark evidence and prior experiments
- Forming explicit hypotheses
- Proposing experiments with a mechanism/rationale
- Proposing semantic mutation of model, loss, features, training or evaluation strategy
- Proposing crossover only when two successful changes appear compatible
- Reflecting on metric changes and failures
- Recommending whether to continue, branch, combine or abandon a research direction

The Research Agent does **not** directly decide population survival or bypass the experiment runner/evaluator.

### Evolution Controller - deterministic search manager

Owns:

- Fitness calculation from official ranking metrics, with optional efficiency penalty
- Population bookkeeping and lineage
- Elitism / preservation of validated best experiments
- Selection of candidate parents / branches
- Diversity and duplicate suppression
- Experiment, token, wall-clock and compute budgets
- Deterministic guardrails for rejecting invalid or clearly dominated runs

The Evolution Controller does **not** invent research hypotheses. It provides selected evidence and candidate parents back to the Research Agent.

## Core loop

```text
Benchmark / current best
        |
        v
Research Agent
  - inspect evidence
  - propose hypothesis
  - propose semantic mutation/crossover
        |
        v
ExperimentSpec
  - parent experiment(s)
  - hypothesis
  - model/loss/features/config
  - expected mechanism
        |
        v
ExperimentRunner
  - isolated run
  - resource limits
  - capture logs/artifacts
        |
        v
Official KuaiRand Evaluator
  - GAUC
  - nDCG@5
  - primary score
        |
        v
ExperimentResult
        |
        v
Experiment Registry
  - metrics
  - runtime
  - token usage
  - failure info
  - code/config diff
  - lineage
        |
        v
Evolution Controller
  - fitness
  - selection
  - elitism
  - diversity
  - budget
        |
        v
selected research state / parents
        |
        +-----------------------> Research Agent
```

## Evolutionary semantics

Traditional GA terminology maps to this project as follows:

| Evolution concept | Project representation |
| --- | --- |
| Genome | `ExperimentSpec` |
| Individual | One executed experiment |
| Population | Small set of active research branches |
| Fitness | Official ranking score plus optional efficiency penalty |
| Mutation | LLM-proposed, hypothesis-driven semantic change |
| Crossover | LLM-proposed combination of compatible successful mechanisms |
| Selection | Deterministic controller rule |
| Elitism | Preserve best validated checkpoints |
| Diversity | Suppress near-duplicate experiments and branch collapse |
| Generation | One bounded batch/round of research experiments |

Mutation should normally change one meaningful research dimension so attribution remains interpretable.

Crossover is **conditional, not mandatory**. The Research Agent should combine parents only when their improvements are plausibly compatible and sufficiently independent.

## Incremental implementation plan

### V1 - Sequential autonomous research

```text
best -> propose -> run -> evaluate -> reflect -> keep/reject -> repeat
```

Goal: prove the autonomous loop works before adding branching.

### V2 - Evolutionary branching

Maintain a small population, initially target 3-4 active branches.

```text
        baseline
       /   |    \
   branch A B   C
      |      \
      D       E
```

Use explicit fitness, elitism, diversity and budgets.

### V3 - Semantic crossover

When two branches show compatible improvements, ask the Research Agent to propose a combined experiment with an explicit rationale.

Example:

```text
BPR loss improvement       sequence-model improvement
          \                   /
           \                 /
        semantic crossover proposal
                    |
             BPR + sequence
```

## Guardrails

1. Keep the official KuaiRand evaluator immutable.
2. All experiments must execute through one stable `ExperimentRunner` boundary.
3. Every experiment must have a machine-readable `ExperimentSpec` and `ExperimentResult`.
4. Preserve lineage, code/config diff, metrics, runtime, token usage and failure information.
5. Keep population sizes small. Do not disguise brute-force hyperparameter search as evolution.
6. Prefer scientifically meaningful mutations over random parameter perturbations.
7. Track manual interventions explicitly because autonomy is part of the Track 2 evaluation story.
8. Optimize first for a reliable end-to-end loop and measurable benchmark improvement, then add sophistication.

## Phase 2 interfaces (implemented)

See [`docs/EXPERIMENT_HARNESS.md`](EXPERIMENT_HARNESS.md) for the full contract.

### ExperimentSpec

Frozen dataclass in `research_agent.experiments`. Model-agnostic: model/loss/width live in `parameters` or the candidate file. `experiment_id` is run identity. `spec_hash` is the execution fingerprint (schema, implementation, parameters, seed, split). Parents: 0 baseline / 1 mutation / many crossover.

### ExperimentRunner

`ExperimentRunner.run(spec) -> ExperimentResult`. Isolated `runs/<experiment_id>/`, subprocess candidate, timeout, score validation, then organizer `evaluate(user_ids, labels, scores)` only.

### ExperimentResult

Statuses: `success`, `failed`, `timeout`, `invalid`. Success carries official `GAUC` / `nDCG@5` / `primary`. Decisions are not stored on the result.

### ExperimentRegistry

SQLite primitives: persist spec/result, spec-hash lookup, lineage/ancestry, decision (`pending` / `accepted` / `rejected`), validation elite, first-parent rollback target. No fitness policy.

### Split policy

Default and autonomous-search split is `valid`. Test requires `allow_test_split` or `allow_test=True`. Elite ranking never uses test.

### Phase 3 sequential Research Agent (implemented)

See [`docs/RESEARCH_AGENT.md`](RESEARCH_AGENT.md).

Gemini owns structured research calls. The Phase 2 runner/registry/evaluator stay deterministic. Repair calls are bounded and counted.

### Phase 4 Evolution Controller (implemented)

See [`docs/EVOLUTION.md`](EVOLUTION.md).

Deterministic population, fitness, elitism, diversity, budgets, and semantic mutation/crossover via the Research Agent. Not brute-force AutoML.

