"""Compact prompts for one research call and bounded repair calls."""
from __future__ import annotations

from .proposal import PROPOSAL_JSON_SCHEMA
from .state import ResearchState

RESEARCH_SYSTEM = """You are an autonomous recommender-system researcher on KuaiRand-Pure.

Task: within-user ranking of long_view. Official metrics are GAUC, nDCG@5, and primary = mean of those two.
You may only research on the validation split. Never request or use the test split.
Do not modify starter/kuairand/evaluate.py. Do not compute authoritative metrics in candidate code.
The harness will call the official evaluator on your score vector.

Write a COMPLETE replacement Python candidate that:
- uses argparse flags --data-dir --split --output-scores --seed --config
- trains on the official train split from data.load / data.encode as needed
- writes a 1-d finite numpy score vector for the requested split row order via numpy.save
- must be valid Python 3
- may import numpy, the Python standard library, and starter modules data / baseline / evaluate
- may use evaluate() only as a training diagnostic (e.g. early stopping), never as the reported result
- must not write to evaluate.py or other repo source files
- must stay inside ResearchState.environment. Do not import torch or other unsupported packages
- if the proposed method cannot execute, fail explicitly. Never catch ImportError and silently train FM, the parent, or another algorithm, then report those scores as the hypothesis result

Prefer one meaningful research mutation versus the parent source.
Use the selected parent as the code you modify. Prefer the current validation elite over a failed child.
Organizer notes about dead ends and promising categories are context, not a script.

Return structured JSON matching the schema. candidate_source must be the full file.
"""

REPAIR_SYSTEM = """Your previous structured proposal was rejected before or during materialization.
Return a corrected ResearchProposal with the same scientific intent if still valid.
If the error is unsupported_dependency or silent_dependency_fallback, preserve the original hypothesis
and reimplement it with allowed tools (NumPy and the standard library) when feasible.
Do not abandon a useful ranking-loss idea only because PyTorch is unavailable.
Do not silently fall back to the FM baseline or the parent.
Fix the error. candidate_source must be complete valid Python with the required CLI flags.
Do not modify evaluate.py. Validation split only.
"""


def research_prompt(state: ResearchState) -> str:
    return (
        RESEARCH_SYSTEM
        + "\n\nCurrent ResearchState (compact evidence, not a chat log):\n"
        + state.to_json(indent=2)
        + "\n"
    )


def repair_prompt(state: ResearchState, error: str, previous_text: str) -> str:
    clipped = previous_text[-8000:] if previous_text else ""
    return (
        REPAIR_SYSTEM
        + "\n\nError:\n"
        + error
        + "\n\nPrevious model output (may be truncated):\n"
        + clipped
        + "\n\nCurrent ResearchState:\n"
        + state.to_json(indent=2)
        + "\n"
    )


def proposal_schema() -> dict:
    return PROPOSAL_JSON_SCHEMA
