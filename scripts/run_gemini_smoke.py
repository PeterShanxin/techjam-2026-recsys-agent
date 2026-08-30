"""Minimal live Gemini Interactions REST smoke. Fails if GEMINI_API_KEY is unset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_agent.agent.proposal import PROPOSAL_JSON_SCHEMA, ResearchProposal
from research_agent.llm import (
    GeminiProvider,
    LLMAuthError,
    LLMConfigError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequest,
    LLMTransientError,
)
from research_agent.llm.credentials import resolve_gemini_api_key
from research_agent.llm.secrets import assert_no_secrets, redact_text
from research_agent.llm.types import DEFAULT_MODEL, DEFAULT_THINKING_LEVEL

SMOKE_PROMPT = """Return one ResearchProposal JSON object.
selected_parent_id must be fm-root.
timeout_seconds may be omitted.
candidate_source must be a complete short Python file that:
- uses argparse flags --data-dir --split --output-scores --seed --config
- writes a 1-d numpy zeros vector of length 1 with numpy.save
Keep every other string field one sentence.
"""


def main(argv: list[str] | None = None) -> int:
    try:
        resolve_gemini_api_key(ROOT)
    except LLMConfigError as exc:
        print(redact_text(str(exc)), file=sys.stderr)
        return 2
    provider = GeminiProvider(model=DEFAULT_MODEL, repo_root=ROOT)
    request = LLMRequest(
        prompt=SMOKE_PROMPT,
        response_schema=PROPOSAL_JSON_SCHEMA,
        model=DEFAULT_MODEL,
        thinking_level=DEFAULT_THINKING_LEVEL,
        purpose="smoke",
    )
    try:
        response = provider.generate(request)
    except (
        LLMConfigError,
        LLMAuthError,
        LLMProtocolError,
        LLMRateLimitError,
        LLMTransientError,
    ) as exc:
        print(redact_text(str(exc)), file=sys.stderr)
        return 2
    payload = {
        "parsed_keys": sorted(response.parsed.keys()) if response.parsed else None,
        "usage": response.usage.to_dict(),
        "status": response.usage.status,
        "thinking_level": response.usage.thinking_level,
        "model": response.usage.model,
    }
    assert_no_secrets(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if response.parsed is None or response.usage.status != "success":
        return 1
    try:
        ResearchProposal.from_dict(response.parsed)
    except Exception as exc:
        print(redact_text(f"proposal validation failed: {exc}"), file=sys.stderr)
        return 1
    if response.usage.input_tokens is None and response.usage.total_tokens is None:
        print("warning: no first-party token counts", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
