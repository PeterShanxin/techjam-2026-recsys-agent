"""LLM usage records and Gemini REST provider. No paid API calls."""
from __future__ import annotations

import os

import pytest

from research_agent.llm import (
    FakeProvider,
    GeminiProvider,
    LLMAuthError,
    LLMConfigError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequest,
    LLMTransientError,
    build_interaction_payload,
    extract_model_output_text,
    redact_text,
    resolve_gemini_api_key,
    usage_from_api_metadata,
)
from research_agent.llm.gemini import INTERACTIONS_URL


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


def _completed(text: str, usage: dict | None = None) -> dict:
    return {
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": text}],
            }
        ],
        "usage": usage
        or {
            "total_input_tokens": 11,
            "total_output_tokens": 7,
            "total_thought_tokens": 5,
            "total_cached_tokens": 2,
            "total_tokens": 25,
            "total_tool_use_tokens": 0,
        },
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


def test_usage_from_api_metadata_does_not_estimate_missing_fields():
    usage, raw = usage_from_api_metadata(
        {"total_input_tokens": 10, "total_output_tokens": 4},
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
    assert "total_input_tokens" in raw


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


def test_process_env_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-process")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    assert resolve_gemini_api_key(tmp_path) == "from-process"


def test_dotenv_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file-only\n", encoding="utf-8")
    assert resolve_gemini_api_key(tmp_path) == "from-file-only"


def test_missing_key_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        resolve_gemini_api_key(tmp_path)
    with pytest.raises(LLMConfigError):
        GeminiProvider(repo_root=tmp_path)


def test_interaction_payload_uses_current_schema():
    req = _request(thinking_level="medium")
    payload = build_interaction_payload(req, model="gemini-3.7-flash")
    assert payload["model"] == "gemini-3.7-flash"
    assert payload["store"] is False
    assert payload["generation_config"]["thinking_level"] == "medium"
    assert payload["response_format"]["type"] == "text"
    assert payload["response_format"]["schema"] == SCHEMA
    assert "outputs" not in payload
    assert "response_mime_type" not in payload
    assert "temperature" not in payload.get("generation_config", {})


def test_gemini_rest_parses_steps_and_usage(monkeypatch):
    transport = ScriptedTransport([(200, _completed('{"ok": true}'))])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    response = provider.generate(_request())
    url, payload, headers, _timeout = transport.calls[0]
    assert url == INTERACTIONS_URL
    assert headers["x-goog-api-key"] == "test-key"
    assert headers["Api-Revision"] == "2026-05-20"
    assert payload["generation_config"]["thinking_level"] == "medium"
    assert response.parsed == {"ok": True}
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.thinking_tokens == 5
    assert response.usage.cached_tokens == 2
    assert response.usage.total_tokens == 25
    dumped = str(response.to_dict())
    assert "test-key" not in dumped


def test_high_thinking_repair_request():
    transport = ScriptedTransport([(200, _completed('{"ok": true}'))])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    provider.generate(_request(thinking_level="high", purpose="repair"))
    _url, payload, _headers, _timeout = transport.calls[0]
    assert payload["generation_config"]["thinking_level"] == "high"


def test_auth_error_does_not_echo_key():
    secret = "AIzaSyDoNotLeakThisKeyValue000000000"
    transport = ScriptedTransport([(401, {"error": {"message": "denied"}})])
    provider = GeminiProvider(transport=transport, api_key=secret)
    with pytest.raises(LLMAuthError, match="authentication") as exc:
        provider.generate(_request())
    assert secret not in str(exc.value)


def test_incomplete_status_is_protocol_error():
    body = _completed('{"ok": true}')
    body["status"] = "failed"
    transport = ScriptedTransport([(200, body)])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    with pytest.raises(LLMProtocolError, match="completed"):
        provider.generate(_request())


def test_missing_status_still_parses_steps():
    body = _completed('{"ok": true}')
    del body["status"]
    transport = ScriptedTransport([(200, body)])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    response = provider.generate(_request())
    assert response.parsed == {"ok": True}


def test_tool_use_tokens_captured_when_present():
    usage, raw = usage_from_api_metadata(
        {
            "total_input_tokens": 3,
            "total_output_tokens": 2,
            "total_thought_tokens": 1,
            "total_cached_tokens": 0,
            "total_tokens": 6,
            "total_tool_use_tokens": 4,
        },
        provider="gemini",
        model="gemini-3.7-flash",
        thinking_level="medium",
        purpose="research",
        status="success",
        latency_seconds=0.1,
    )
    assert usage.tool_use_tokens == 4
    assert raw["total_tool_use_tokens"] == 4


def test_rate_limit_after_retries(monkeypatch):
    monkeypatch.setattr("research_agent.llm.gemini.time.sleep", lambda _s: None)
    transport = ScriptedTransport([(429, {}), (429, {}), (429, {})])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    with pytest.raises(LLMRateLimitError):
        provider.generate(_request())
    assert len(transport.calls) == 3


def test_transient_server_error(monkeypatch):
    monkeypatch.setattr("research_agent.llm.gemini.time.sleep", lambda _s: None)
    transport = ScriptedTransport(
        [
            (500, {"error": {"code": "api_error", "message": "high demand"}}),
            (500, {"error": {"code": "api_error", "message": "high demand"}}),
            (500, {"error": {"code": "api_error", "message": "high demand"}}),
        ]
    )
    provider = GeminiProvider(transport=transport, api_key="test-key")
    with pytest.raises(LLMTransientError, match="high demand"):
        provider.generate(_request())


def test_malformed_json_http_raises_protocol():
    def bad_transport(url, payload, headers, timeout):
        raise LLMProtocolError("Gemini HTTP response was not JSON")

    provider = GeminiProvider(transport=bad_transport, api_key="test-key")
    with pytest.raises(LLMProtocolError, match="not JSON"):
        provider.generate(_request())


def test_malformed_model_json_is_invalid_not_silent_empty():
    transport = ScriptedTransport([(200, _completed("not-json"))])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    response = provider.generate(_request())
    assert response.parsed is None
    assert response.usage.status == "invalid"
    assert response.text == "not-json"


def test_missing_steps_is_protocol_error():
    transport = ScriptedTransport([(200, {"status": "completed", "usage": {}})])
    provider = GeminiProvider(transport=transport, api_key="test-key")
    with pytest.raises(LLMProtocolError, match="steps"):
        provider.generate(_request())


def test_extract_model_output_ignores_legacy_outputs():
    body = _completed('{"ok": true}')
    body["outputs"] = [{"text": "legacy"}]
    assert extract_model_output_text(body) == '{"ok": true}'


def test_gemini_provider_does_not_persist_key_in_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyDoNotLeakThisKeyValue000000000")
    transport = ScriptedTransport([(200, _completed('{"ok": true}'))])
    response = GeminiProvider(transport=transport, api_key=os.environ["GEMINI_API_KEY"]).generate(
        _request()
    )
    dumped = str(response.to_dict())
    assert "AIzaSyDoNotLeakThisKeyValue000000000" not in dumped
    assert os.environ["GEMINI_API_KEY"] not in dumped
