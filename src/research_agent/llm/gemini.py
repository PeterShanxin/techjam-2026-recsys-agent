"""Gemini Developer API provider. google-genai SDK only. No other vendors."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from .secrets import assert_no_secrets, redact_text, sanitize
from .types import (
    DEFAULT_MODEL,
    LLMConfigError,
    LLMRequest,
    LLMResponse,
    usage_from_api_metadata,
)

API_KEY_ENV = "GEMINI_API_KEY"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client

    def generate(self, request: LLMRequest) -> LLMResponse:
        assert_no_secrets(request.prompt)
        model = request.model or self.model
        client = self._get_client()
        config = self._generate_config(request)
        started = time.perf_counter()
        try:
            raw = client.models.generate_content(
                model=model,
                contents=request.prompt,
                config=config,
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            usage, raw_usage = usage_from_api_metadata(
                getattr(exc, "usage_metadata", None),
                provider=self.name,
                model=model,
                thinking_level=request.thinking_level,
                purpose=request.purpose,
                status="error",
                latency_seconds=latency,
                error=redact_text(str(exc)),
            )
            return LLMResponse(text="", parsed=None, usage=usage, raw_usage=raw_usage)

        latency = time.perf_counter() - started
        text = redact_text(_response_text(raw))
        parsed = _parse_payload(raw, text)
        status = "success" if parsed is not None else "invalid"
        error = None if parsed is not None else "structured output missing or not valid JSON"
        usage, raw_usage = usage_from_api_metadata(
            getattr(raw, "usage_metadata", None),
            provider=self.name,
            model=model,
            thinking_level=request.thinking_level,
            purpose=request.purpose,
            status=status,
            latency_seconds=latency,
            error=error,
        )
        response = LLMResponse(
            text=text,
            parsed=sanitize(parsed) if parsed is not None else None,
            usage=usage,
            raw_usage=sanitize(raw_usage),
        )
        assert_no_secrets(response.to_dict())
        return response

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise LLMConfigError(f"{API_KEY_ENV} is not set")
        genai, _types = _import_genai()
        return genai.Client(api_key=key)

    def _generate_config(self, request: LLMRequest) -> Any:
        try:
            _genai, types = _import_genai()
        except LLMConfigError:
            return {
                "thinking_config": {"thinking_level": request.thinking_level},
                "response_mime_type": "application/json",
                "response_schema": request.response_schema,
            }
        thinking = _thinking_config(types, request.thinking_level)
        return types.GenerateContentConfig(
            thinking_config=thinking,
            response_mime_type="application/json",
            response_schema=request.response_schema,
        )


def _import_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMConfigError(
            "google-genai is not installed. Install the llm extra: pip install google-genai"
        ) from exc
    return genai, types


def _thinking_config(types: Any, level: str) -> Any:
    enum = getattr(types, "ThinkingLevel", None)
    thinking_level: Any = level
    if enum is not None:
        for attr in (level.upper(), level):
            if hasattr(enum, attr):
                thinking_level = getattr(enum, attr)
                break
    return types.ThinkingConfig(thinking_level=thinking_level)


def _response_text(raw: Any) -> str:
    text = getattr(raw, "text", None)
    if text:
        return str(text)
    parsed = getattr(raw, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return json.dumps(parsed.model_dump(), ensure_ascii=True)
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=True)
    return ""


def _parse_payload(raw: Any, text: str) -> dict[str, Any] | None:
    parsed = getattr(raw, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            dumped = parsed.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if isinstance(parsed, dict):
            return dict(parsed)
    if not text.strip():
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
