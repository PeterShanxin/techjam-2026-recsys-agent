"""Minimal live Gemini Developer API smoke. Zero-cost skip if GEMINI_API_KEY is unset."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from research_agent.llm import GeminiProvider, LLMConfigError, LLMRequest
from research_agent.llm.secrets import assert_no_secrets, redact_text
from research_agent.llm.types import DEFAULT_MODEL, DEFAULT_THINKING_LEVEL

SMOKE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ok": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["ok", "note"],
}


def main(argv: list[str] | None = None) -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is not set; not faking a live smoke call.")
        return 2
    provider = GeminiProvider(model=DEFAULT_MODEL)
    request = LLMRequest(
        prompt='Return JSON with ok=true and note="phase3-smoke".',
        response_schema=SMOKE_SCHEMA,
        model=DEFAULT_MODEL,
        thinking_level=DEFAULT_THINKING_LEVEL,
        purpose="smoke",
    )
    try:
        response = provider.generate(request)
    except LLMConfigError as exc:
        print(redact_text(str(exc)), file=sys.stderr)
        return 2
    payload = {
        "parsed": response.parsed,
        "usage": response.usage.to_dict(),
        "status": response.usage.status,
    }
    assert_no_secrets(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if response.parsed is None or response.usage.status != "success":
        return 1
    if response.usage.input_tokens is None and response.usage.total_tokens is None:
        print("warning: no first-party token counts in usage_metadata", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
