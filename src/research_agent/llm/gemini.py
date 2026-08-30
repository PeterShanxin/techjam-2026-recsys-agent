"""Gemini Interactions REST provider. No google-genai SDK. urllib only."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .credentials import resolve_gemini_api_key
from .secrets import assert_no_secrets, redact_text, sanitize
from .types import (
    DEFAULT_MODEL,
    LLMAuthError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    usage_from_api_metadata,
)

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
DEFAULT_HTTP_TIMEOUT = 300.0
MAX_TRANSPORT_RETRIES = 3

Transport = Callable[[str, dict[str, Any], dict[str, str], float], tuple[int, Any]]


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        repo_root: Any | None = None,
        transport: Transport | None = None,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT,
    ) -> None:
        self.model = model
        self._transport = transport or default_transport
        self.timeout_seconds = timeout_seconds
        self.transport_retries = 0
        if transport is None:
            self._api_key = api_key or resolve_gemini_api_key(repo_root)
        else:
            self._api_key = api_key or "test-key"

    def generate(self, request: LLMRequest) -> LLMResponse:
        assert_no_secrets(request.prompt)
        model = request.model or self.model
        payload = build_interaction_payload(request, model=model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
            "Api-Revision": API_REVISION,
        }
        started = time.perf_counter()
        status, body, retries = self._post_with_retries(payload, headers)
        self.transport_retries += retries
        latency = time.perf_counter() - started
        usage_meta = _usage_meta(body)
        error_detail = _http_error_detail(body)
        if status in (401, 403):
            raise LLMAuthError("Gemini authentication failed")
        if status == 429:
            raise LLMRateLimitError(
                "Gemini rate limited" + (f": {error_detail}" if error_detail else "")
            )
        if status >= 500:
            raise LLMTransientError(
                f"Gemini upstream error HTTP {status}"
                + (f": {error_detail}" if error_detail else "")
            )
        if status != 200:
            raise LLMProtocolError(f"Gemini HTTP {status}")
        text = extract_model_output_text(body)
        parsed = _parse_json_object(text)
        status_name = "success" if parsed is not None else "invalid"
        error = None if parsed is not None else "structured output missing or not valid JSON"
        usage, raw_usage = usage_from_api_metadata(
            usage_meta,
            provider=self.name,
            model=model,
            thinking_level=request.thinking_level,
            purpose=request.purpose,
            status=status_name,
            latency_seconds=latency,
            error=error,
        )
        response = LLMResponse(
            text=redact_text(text),
            parsed=sanitize(parsed) if parsed is not None else None,
            usage=usage,
            raw_usage=sanitize(raw_usage),
        )
        assert_no_secrets(response.to_dict())
        return response

    def _post_with_retries(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[int, dict[str, Any], int]:
        retries = 0
        last_status = 0
        last_body: Any = {}
        for attempt in range(MAX_TRANSPORT_RETRIES):
            try:
                status, body = self._transport(
                    INTERACTIONS_URL, payload, headers, self.timeout_seconds
                )
            except LLMTransientError:
                retries += 1
                if attempt + 1 >= MAX_TRANSPORT_RETRIES:
                    raise
                time.sleep(_backoff(attempt))
                continue
            last_status, last_body = status, body if isinstance(body, dict) else {}
            if status in (429,) or status >= 500:
                retries += 1
                if attempt + 1 >= MAX_TRANSPORT_RETRIES:
                    return status, last_body, retries
                time.sleep(_backoff(attempt))
                continue
            return status, last_body, retries
        return last_status, last_body, retries


def build_interaction_payload(request: LLMRequest, *, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": request.prompt,
        "store": False,
        "generation_config": {"thinking_level": request.thinking_level},
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": request.response_schema,
        },
    }


def extract_model_output_text(body: Any) -> str:
    payload = _interaction_body(body)
    status = payload.get("status")
    if status is not None and status != "completed":
        raise LLMProtocolError(
            f"Gemini interaction status is {status!r}, expected 'completed'"
        )
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise LLMProtocolError("Gemini response missing steps array")
    texts: list[str] = []
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for part in step.get("content") or []:
            if isinstance(part, str) and part.strip():
                texts.append(part)
                continue
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))
    if not texts:
        raise LLMProtocolError("Gemini response has no model_output text")
    return "\n".join(texts)


def _interaction_body(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise LLMProtocolError("Gemini response is not a JSON object")
    inner = body.get("interaction")
    if isinstance(inner, dict) and "steps" not in body:
        return inner
    return body


def _usage_meta(body: Any) -> dict[str, Any]:
    payload = body if isinstance(body, dict) else {}
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return usage
    inner = payload.get("interaction")
    if isinstance(inner, dict) and isinstance(inner.get("usage"), dict):
        return inner["usage"]
    return {}


def default_transport(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
        raw = redact_text(raw)
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"error": "non-json HTTP error body"}
        return status, parsed if isinstance(parsed, dict) else {"error": "non-object error body"}
    except urllib.error.URLError as exc:
        raise LLMTransientError(redact_text(str(exc.reason if getattr(exc, "reason", None) else exc))) from exc
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise LLMProtocolError("Gemini HTTP response was not JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMProtocolError("Gemini HTTP JSON was not an object")
    return status, parsed


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text or not str(text).strip():
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _backoff(attempt: int) -> float:
    return min(30.0, 2.0 * (2 ** attempt))


def _http_error_detail(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("status") or error.get("code")
        return redact_text(str(message)[:300]) if message else ""
    if isinstance(error, str):
        return redact_text(error[:300])
    return ""
