"""Research Agent: sequential Gemini-backed research loop."""

from .accounting import ResourceLedger
from .constants import DEFAULT_RESEARCH_MODEL, DEFAULT_THINKING_LEVEL, FM_ROOT_ID
from .fm_root import fm_root_spec
from .loop import ResearchAgent, ResearchRun
from .proposal import ProposalError, ResearchProposal
from .state import ResearchState, build_research_state
from .trace import ResearchTrace
from .workspace import CandidateWorkspace

__all__ = [
    "DEFAULT_RESEARCH_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "FM_ROOT_ID",
    "CandidateWorkspace",
    "ProposalError",
    "ResearchAgent",
    "ResearchProposal",
    "ResearchRun",
    "ResearchState",
    "ResearchTrace",
    "ResourceLedger",
    "build_research_state",
    "fm_root_spec",
]
