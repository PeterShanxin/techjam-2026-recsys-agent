"""Semantic mutation and crossover prompts. Controller still chooses parents."""
from __future__ import annotations

from research_agent.agent.proposal import PROPOSAL_JSON_SCHEMA
from research_agent.agent.state import ResearchState

MUTATION_SYSTEM = """You are the semantic mutation operator for an LLM-guided evolutionary research system.

The Evolution Controller already chose ONE parent. You do not decide survival, fitness, or budgets.
Propose ONE coherent research mutation versus that parent.

Do not randomly jitter numbers (for example lr 0.001 -> 0.00103) unless there is a real research rationale.
Do not mutate source text blindly. Change a meaningful research dimension.

You must state:
- what changed
- why
- what evidence motivated it
- what result would support or refute the hypothesis

Honor ResearchState.data_contract. data.load() tuples do not include is_like or play_time_ms.
You may import research_agent.lab for train-only history, popularity, catalogs and pairwise samples
(SplitSafeStore), and for within-user grouping plus a gradient-driven FM (user_groups, GradientFM).
GradientFM has no loss function: it applies Adam to whatever per-row dL/dlogit you hand it, so the
objective is yours to choose and justify. data_contract.lab.example shows the plumbing only.
Train-derived features must use train. TEST IS SEALED. Validation labels are not features.

Consult ResearchState.heavily_searched_axes, underexplored_axes, validation_noise and audit_findings.
They name axes and rule out closed families. They prescribe no values, weights or objectives.
Respect the noise floor: below ~0.0005 validation primary nothing is measurable on this split.
Current weakness is homogeneous FM refinement. Prefer a genuinely different family when evidence supports it.
Do not jitter seeds/LR/averaging without a reason.
Do not propose "parent + alpha * residual" with alpha tuned on validation and alpha=0 in the grid.
That cannot lose, so it is not a test. Children that barely change within-user ordering are rejected
as near_identity_noop and earn no fitness.
Stay inside the environment. Validation split only. Do not modify evaluate.py.
Return structured JSON. candidate_source must be a complete candidate file.
Set operator to "mutation". Set selected_parent_id to the provided parent.
research_family, mechanism_tags and changed_axes are REQUIRED and must be specific.
"other" is rejected: diversity suppression and crossover parent choice both key off these values.
"""

CROSSOVER_SYSTEM = """You are the semantic crossover operator for an LLM-guided evolutionary research system.

The Evolution Controller already chose TWO parents. You do not decide survival.
Identify:
- a useful component from parent A
- a useful component from parent B
- whether they are scientifically compatible
- why combining them could help
- conflicts to avoid

If the parents are incompatible, set crossover_compatible to false, explain why, and still return a valid JSON object.
Do not force a combined method that cannot work. The controller will fall back to mutation.

If compatible, set crossover_compatible to true and write one complete candidate that combines the named components.
Do not hard-code a specific combination. Decide from the parent evidence.

Honor ResearchState.data_contract and environment. Validation split only. TEST IS SEALED.
You may use research_agent.lab instruments (SplitSafeStore, user_groups, GradientFM). They are not a ranker.
Respect ResearchState.validation_noise. A combined candidate that merely re-weights the elite is rejected
as near_identity_noop. Combine mechanisms, not scalars.
Return structured JSON. Set operator to "crossover".
research_family, mechanism_tags and changed_axes are REQUIRED and must be specific, not "other".
"""


def mutation_prompt(state: ResearchState) -> str:
    return (
        MUTATION_SYSTEM
        + "\n\nCurrent ResearchState (compact evidence, not a chat log):\n"
        + state.to_json(indent=2)
        + "\n"
    )


def crossover_prompt(state: ResearchState) -> str:
    return (
        CROSSOVER_SYSTEM
        + "\n\nCurrent ResearchState (two controller-chosen parents):\n"
        + state.to_json(indent=2)
        + "\n"
    )


def proposal_schema() -> dict:
    return PROPOSAL_JSON_SCHEMA
