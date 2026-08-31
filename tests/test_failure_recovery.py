"""Failure-recovery matrix claims are backed by tests. Zero API spend."""
from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.llm import GeminiProvider, LLMAuthError, LLMConfigError
from research_agent.llm.credentials import resolve_gemini_api_key
from research_agent.llm.types import LLMRequest


SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


class ScriptedTransport:
    def __init__(self, responses: list[tuple[int, dict]]):
        self.responses = list(responses)
        self.calls: list[tuple] = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append((url, payload, dict(headers), timeout))
        if not self.responses:
            raise AssertionError("transport script exhausted")
        return self.responses.pop(0)


def _request() -> LLMRequest:
    return LLMRequest(
        prompt="Return JSON.",
        response_schema=SCHEMA,
        model="gemini-3.6-flash",
        thinking_level="medium",
        purpose="research",
    )


def test_forbidden_auth_is_stop_not_retry(monkeypatch):
    monkeypatch.setattr("research_agent.llm.gemini.time.sleep", lambda _s: None)
    transport = ScriptedTransport([(403, {"error": {"message": "denied"}})])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    with pytest.raises(LLMAuthError, match="authentication"):
        provider.generate(_request())
    assert len(transport.calls) == 1


def test_missing_key_is_config_error(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        resolve_gemini_api_key(tmp_path)
