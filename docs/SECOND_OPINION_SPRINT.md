# Second-opinion audit and sprint 2

Separate from the P0 sprint in [`PERFORMANCE_SPRINT.md`](PERFORMANCE_SPRINT.md). That
document stands unchanged; nothing here rewrites it.

Machine-readable evidence: [`evidence/sprint2_autonomous_sprint.json`](evidence/sprint2_autonomous_sprint.json).
Validation only. Test was never loaded, scored or inspected. Evaluator SHA-256 (LF)
`ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de`, unchanged.

## 1. Independent audit findings

An independent review of the P0 sprint measured the following on the validation split.
These are diagnostic findings about the *benchmark and the search process*. None of the
parameter values found here were given to the research agent.

**The metric could not resolve what the project was reporting.** Bootstrapping validation
users gives an absolute primary standard deviation of **0.00216**, roughly the official
convergence epsilon. Paired against a shared baseline the standard deviation is about
**0.0005**. The P0 winner's +8.8e-6 is about 2% of that. The whole pipeline's +0.00085
over `fm-root` does not clear a paired 95% interval (CI [-0.00021, +0.00187], P=0.934).

**Both metrics are invariant to per-user monotone transforms.** GAUC and nDCG@5 are
computed strictly inside one user's impression list, so global sigmoid, affine rescaling
and per-user z-scoring leave the primary bitwise unchanged (verified). Per-user constants
cannot change within-user order, which makes the FM's first-order user weight, its global
bias, and every user-constant catalog column inert unless crossed.

**The additive-residual family is closed.** Blending train-derived count features on top
of the frozen elite, with the blend weight chosen on validation (an optimistic upper
bound), gave zero for item rate, author rate, video-by-tab, author-by-tab, user-author
affinity, user-tab, duration-bucket rate, item support, user duration preference and
video age; below +1.2e-4 for tab, tag and hour-of-day. These features are individually
informative and informationally redundant with the FM.

**Duration re-bucketing is closed.** `long_view` is ~98% a deterministic function of
`play_time_ms` and `duration_ms` with a threshold near 18 s, but four alternative duration
encodings, including one with an explicit 18000 ms knot, all landed inside noise.

**Validation is a short-list problem.** Mean 5.58 impressions per user, median 4. 30.3% of
users are all-negative and 11.9% all-positive, so 42% of nDCG@5 weight cannot be moved by
any model. Users with 6-20 impressions carry 63.5% of GAUC weight and 65% of the
realizable nDCG@5 gain. GAUC and nDCG@5 favour the same users.

### Three blind spots in the search process

1. **Regularization and the training objective were unreachable.** `l2` was pinned at
   1e-6 in the root parameters, absent from every dead-end and promising-category list,
   and the operator prompt discouraged touching hyperparameters without evidence. The FM's
   validation primary peaks around epoch 7 and early-stops at 11 while training loss keeps
   falling - an overfitting signature nobody could act on.
2. **`parent + alpha * residual` was a dominant degenerate strategy.** Five of seven P0
   offspring tuned `alpha` on validation with `alpha = 0` inside the grid. That shape
   cannot score below its parent, so fitness rewarded it unconditionally.
   `_classify_validity` only compared source text, so all of them counted as
   `hypothesis_tested`.
3. **Diversity was inert.** Seven of eight P0 proposals left `research_family`,
   `mechanism_tags` and `changed_axes` blank. The semantic signature collapsed to
   `("other", (), ())`, which `duplicate_reason` short-circuits, and `_crossover_parents`
   needs two distinct signatures - so duplicate suppression and crossover both silently
   stopped working.

Every prior ranking-objective attempt also failed on **runtime**, not on evidence: the
pairwise candidates built O(pairs) SGD loops over ~1.9M sampled pairs per epoch and hit
the timeout. That is an implementation failure and never counted as science.

## 2. System changes

| Change | Blind spot | Files |
| --- | --- | --- |
| `heavily_searched_axes`, `underexplored_axes`, `validation_noise`, `audit_findings` in `ResearchState` | 1 | `agent/constants.py`, `agent/state.py` |
| Near-identity rejection on within-user ordering | 2 | `evolution/identity.py`, `evolution/controller.py`, `evolution/config.py` |
| `research_family` / `mechanism_tags` / `changed_axes` required, `"other"` refused | 3 | `agent/proposal.py` |
| Mutation parent avoids drawing every offspring from one family | 3 | `evolution/controller.py` |
| Vectorized within-user grouping + gradient-driven FM | runtime | `lab/ranking.py`, `lab/capabilities.py` |
| `scripts/paired_bootstrap.py` | evidence standard | `scripts/` |

The axis lists name **axes, not settings**. A test asserts that no measured value from the
audit leaks into anything the agent reads. `GradientFM` applies Adam to a caller-supplied
`dL/dlogit` and contains no loss function; a test asserts its public surface is exactly
`logits / predict / apply / state / load_state`. A full 1.14M-row epoch now costs 1.7 s,
so an objective-swapped FM is as cheap as the pointwise one.

Deliberately **not** changed: the SplitSafeStore object API (13.1 s construction, 1.14M
dataclasses) - a real deterrent, but rewriting it was scope the sprint did not need; the
fitness function; population/elite sizes; and the organizer dead-end and promising-category
lists, which are provenance.

## 3. Autonomous sprint

Session `rs-20260831T062638Z-939b7000`. `gemini-3.6-flash`, thinking medium, population 4,
elites 2, up to 4 generations and 8 new evaluations, 600 s per experiment, 4800 s wall.
Priors `fm-root`, `fm-ensemble-3seed`, `final-swa7-ensemble` (the frozen 0.6023186).

Stop reason **converged** (stagnation 3), not budget exhaustion. 7 evaluations used of 8.
46 min wall, 415,105 tokens, 12 LLM calls (9 research, 3 repair), 0 GPU, **0 manual
interventions**.

| ID | Operator | Family | Changed axes | Primary | vs frozen |
| --- | --- | --- | --- | --- | --- |
| 001 | mutation | ranking_loss | training_objective | 0.6000769 | -0.002242 |
| 004 | mutation | training_data_selection | training_data_selection | 0.6023929 | +0.000074 |
| 005 | crossover | training_data_selection | - | failed (subprocess) | - |
| 006 | crossover | ensemble | ensembling, training_data_selection | 0.6024617 | +0.000143 |
| 007 | crossover | ensemble | + regularization | 0.6025183 | +0.000200 |
| **008** | **crossover** | **ensemble** | **ensembling, data selection, regularization** | **0.6029037** | **+0.000585** |
| 009 | crossover | ensemble | + model_capacity | 0.6027744 | +0.000456 |

The agent reached both named blind spots on its own. 001 used `GradientFM` and
`user_groups` to train a joint pointwise-BCE + within-user listwise softmax objective; it
executed cleanly at 2.0 s/epoch and lost, which is the first genuine negative result on
ranking objectives in this project rather than another timeout. 007 introduced
tier-adaptive L2 unprompted, and 009 independently rediscovered the organizer's finding
that varying embedding dimension does not help.

No near-identity rejections fired: every child produced a genuinely different ranking.

### Best candidate

`rs-20260831T062638Z-939b7000-008`, crossover of 007 x 006. Eight FM members with top-2
checkpoint SWA each, probability mean, split across three tiers differing in both train-row
selection and L2: 2x strict (users with >=3 impressions and mixed labels, l2=1e-4), 2x
moderate (>=2 impressions and mixed labels, l2=1e-5), 4x full split (l2=1e-6).

Frozen at [`tiered_ensemble_scorer.py`](../src/research_agent/recommenders/tiered_ensemble_scorer.py),
spec [`tiered_ensemble_valid.json`](../configs/experiments/tiered_ensemble_valid.json).

Mechanism verified executing: stdout shows 8 distinct members with three L2 values, two
filtered row counts (1,129,780 and 1,130,240 of 1,141,112) and different early-stop epochs.

## 4. Evidence

| Check | Result |
| --- | --- |
| Type A rerun, same seed | 0.6029037, bitwise identical |
| Repo copy | 0.6029037, bitwise identical, 174 s |
| Type B seed 7 | 0.6029838 |
| Type B seed 2024 | 0.6031161 |

All three seeds beat the frozen elite; the worst is +0.000585. Seed 42 is kept because it
is the agent's own choice - picking seed 2024 after seeing the sweep would be selection
across replicates.

Paired user bootstrap, 2000 reps, seed 7 (`scripts/paired_bootstrap.py`):

| Comparison | Delta | 95% CI | P(delta>0) |
| --- | --- | --- | --- |
| 008 vs frozen SWA7 | +0.0005851 | [-0.0003427, +0.0015007] | 0.888 |
| 008 vs fm-root | +0.0014350 | [+0.0002577, +0.0026169] | **0.990** |
| frozen SWA7 vs fm-root (reference) | +0.0008499 | [-0.0002086, +0.0018664] | 0.934 |

008 is the first candidate in this project that is significantly better than the FM root
under a paired bootstrap. Against the incumbent it is better at every seed tried but its
interval still includes zero. **No statistical significance is claimed against the
incumbent.**

Runtime: 184 s versus the incumbent's 383 s, for 8 members versus 7.

## 5. Finalization

Sprint-2 candidate 008 was accepted as the Phase 5 submission candidate under the project's
standing rule (validation primary), frozen as `final-tiered-ensemble`, and run on **test once**
for the official CSV. `submit.py --check` passed on 170,588 rows.

`final-swa7-ensemble` is retained unchanged as historical evidence and stays runnable with
`scripts/run_final_candidate.py --split valid --legacy-swa7`. Nothing in
[`PERFORMANCE_SPRINT.md`](PERFORMANCE_SPRINT.md) or the Phase 4 evidence was rewritten.

### The test result did not confirm the validation gain

| Candidate | Valid primary | Test primary |
| --- | --- | --- |
| `final-tiered-ensemble` (submitted) | **0.6029037** | **0.5963754** |
| `final-swa7-ensemble` (superseded) | 0.6023186 | 0.5963862 |

The +0.00059 validation margin became **−0.0000109** on test. Paired user bootstrap on test
(observation only, 2000 reps, seed 7): 95% CI [−0.0007454, +0.0007482], P(Δ>0) = **0.515**.
The two candidates are statistically indistinguishable on test.

We kept the validation-selected candidate. The alternative — swapping back after seeing the
test number — is test-driven selection, which is the failure mode this whole system exists to
prevent. Both candidates lose ~0.006 from validation to test, which is a property of the split.

### Cost

Like-for-like, both as committed repo files through the same harness:

| Split | tiered | swa7 | Δ |
| --- | --- | --- | --- |
| valid | **173.9s** | 225.0s | −22.7% |
| test | **208.6s** | 216.7s | −3.7% |

Cheaper despite one more member (8 vs 7), because four members train on filtered rows and all
early-stop sooner. The live-elite figures (183.7s vs 383.4s) ran under different load and are
not a fair head-to-head.

### One wording correction

`starter/kuairand/baseline.py` `FM.predict` returns **logits**, not probabilities. Both frozen
scorers therefore average raw FM scores, not probabilities, and their docstrings say
"probability" incorrectly. No score or model changed - only the description. The scorer files
are left byte-identical so the `source_fingerprint` values recorded in `runs/*/result.json`
stay valid; the correction is recorded as `aggregation_space_correction` in
[`evidence/canonical_benchmark.json`](evidence/canonical_benchmark.json).
