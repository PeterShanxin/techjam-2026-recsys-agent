"""LLM usage records and fake/gemini providers. No paid API calls."""
from __future__ import annotations

import os
from types import SimpleNamespace

from research_agent.llm import (
    FakeProvider,
    GeminiProvider,
    LLMRequest,
    redact_text,
    usage_from_api_metadata,
)


SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}


def _request(**kwargs) -> LLMRequest:
    payload = {
        "prompt": "Return JSON.",
        "response_schema": SCHEMA,
        "model": "gemini-3.7-flash",
        "thinking_level": "medium",
        "purpose": "research",
    }
    payload.update(kwargs)
    return LLMRequest(**payload)


def test_usage_from_api_metadata_does_not_estimate_missing_fields():
    usage, raw = usage_from_api_metadata(
        {"prompt_token_count": 10, "candidates_token_count": 4},
        provider="gemini",
        model="gemini-3.7-flash",
        thinking_level="medium",
        purpose="research",
        status="success",
        latency_seconds=0.2,
    )
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.thinking_tokens is None
    assert usage.cached_tokens is None
    assert usage.total_tokens is None
    assert usage.thinking_level == "medium"
    assert "prompt_token_count" in raw


def test_fake_provider_records_usage_and_thinking_level():
    provider = FakeProvider(
        script=[{"ok": True, "usage": {"prompt_token_count": 3, "total_token_count": 9}}],
        model="fake-model",
    )
    response = provider.generate(_request(thinking_level="low", purpose="smoke"))
    assert response.parsed == {"ok": True}
    assert response.usage.provider == "fake"
    assert response.usage.thinking_level == "low"
    assert response.usage.purpose == "smoke"
    assert response.usage.input_tokens == 3
    assert response.usage.total_tokens == 9
    assert response.usage.status == "success"
    assert provider.calls[0].thinking_level == "low"


def test_redact_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestSecretKeyValue0000000000000")
    leaked = "header AIzaSyTestSecretKeyValue0000000000000 footer"
    assert "AIzaSyTestSecretKeyValue0000000000000" not in redact_text(leaked)
    assert "[REDACTED]" in redact_text(leaked)


def test_gemini_provider_with_injected_client_captures_usage():
    usage_meta = SimpleNamespace(
        prompt_token_count=11,
        candidates_token_count=7,
        thoughts_token_count=5,
        cached_content_token_count=2,
        total_token_count=23,
    )
    raw = SimpleNamespace(
        text='{"ok": true, "note": "hi"}',
        parsed={"ok": True, "note": "hi"},
        usage_metadata=usage_meta,
    )

    class DummyModels:
        def generate_content(self, model, contents, config):
            assert model == "gemini-3.7-flash"
            return raw

    client = SimpleNamespace(models=DummyModels())
    provider = GeminiProvider(model="gemini-3.7-flash", client=client)
    response = provider.generate(_request())
    assert response.parsed == {"ok": True, "note": "hi"}
    assert response.usage.model == "gemini-3.7-flash"
    assert response.usage.thinking_level == "medium"
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.thinking_tokens == 5
    assert response.usage.cached_tokens == 2
    assert response.usage.total_tokens == 23
    assert response.usage.status == "success"


def test_gemini_provider_does_not_persist_key_in_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDoNotLeakThisKeyValue000000000")
    raw = SimpleNamespace(
        text='{"ok": true}',
        parsed={"ok": True},
        usage_metadata=SimpleNamespace(total_token_count=1),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_kw: raw)
    )
    response = GeminiProvider(client=client).generate(_request())
    dumped = str(response.to_dict())
    assert "AIzaSyDoNotLeakThisKeyValue000000000" not in dumped
    assert os.environ["GEMINI_API_KEY"] not in dumped
