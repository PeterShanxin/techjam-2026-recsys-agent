"""Scripted provider for tests. Spends zero API money."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable

from .secrets import redact_text, sanitize
from .types import LLMRequest, LLMResponse, UsageRecord, usage_from_api_metadata

ScriptItem = dict[str, Any] | LLMResponse | Exception | Callable[[LLMRequest], Any]


class FakeProvider:
    """Queue of proposal dicts / responses / exceptions. Deterministic usage."""

    name = "fake"

    def __init__(
        self,
        script: Iterable[ScriptItem] | None = None,
        *,
        model: str = "fake-model",
        default_usage: dict[str, int] | None = None,
    ) -> None:
        self.model = model
        self._script = list(script or [])
        self.calls: list[LLMRequest] = []
        self.responses: list[LLMResponse] = []
        self.default_usage = default_usage or {
            "prompt_token_count": 100,
            "candidates_token_count": 50,
            "thoughts_token_count": 20,
            "cached_content_token_count": 0,
            "total_token_count": 170,
        }

    def push(self, item: ScriptItem) -> None:
        self._script.append(item)

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        started = time.perf_counter()
        if not self._script:
            raise IndexError("FakeProvider script exhausted")
        item = self._script.pop(0)
        if callable(item) and not isinstance(item, LLMResponse):
            item = item(request)
        if isinstance(item, Exception):
            latency = time.perf_counter() - started
            usage, raw = usage_from_api_metadata(
                {},
                provider=self.name,
                model=request.model or self.model,
                thinking_level=request.thinking_level,
                purpose=request.purpose,
                status="error",
                latency_seconds=latency,
                error=redact_text(str(item)),
            )
            response = LLMResponse(text="", parsed=None, usage=usage, raw_usage=raw)
            self.responses.append(response)
            raise item
        if isinstance(item, LLMResponse):
            self.responses.append(item)
            return item

        parsed = dict(item)
        text = json.dumps(parsed, ensure_ascii=True)
        usage_meta = dict(self.default_usage)
        if "usage" in parsed:
            usage_meta.update(parsed.pop("usage"))
        latency = time.perf_counter() - started
        usage, raw = usage_from_api_metadata(
            usage_meta,
            provider=self.name,
            model=request.model or self.model,
            thinking_level=request.thinking_level,
            purpose=request.purpose,
            status="success",
            latency_seconds=latency,
        )
        response = LLMResponse(
            text=text,
            parsed=sanitize(parsed),
            usage=usage,
            raw_usage=raw,
        )
        self.responses.append(response)
        return response
