"""LLM request/response records. Token counts come from the provider, never estimates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

THINKING_LEVELS = ("low", "medium", "high")
DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_THINKING_LEVEL = "medium"
DEFAULT_PROVIDER = "gemini"
PURPOSES = ("research", "repair", "smoke", "mutation", "crossover")


class LLMError(RuntimeError):
    """Provider or configuration failure."""


class LLMConfigError(LLMError):
    """Missing credential or invalid provider config. Fail fast. Not repairable."""


class LLMAuthError(LLMError):
    """401/403. Stop live research. Never echo credentials."""


class LLMRateLimitError(LLMError):
    """429 after bounded transport retries."""


class LLMTransientError(LLMError):
    """Network / 5xx after bounded transport retries."""


class LLMProtocolError(LLMError):
    """HTTP or envelope shape does not match the current Interactions contract."""


def normalize_thinking_level(level: str) -> str:
    value = str(level).strip().lower()
    if value not in THINKING_LEVELS:
        raise LLMConfigError(
            f"thinking_level must be one of {THINKING_LEVELS}, got {level!r}"
        )
    return value


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    thinking_level: str
    purpose: str
    status: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    tool_use_tokens: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "thinking_level": self.thinking_level,
            "purpose": self.purpose,
            "status": self.status,
            "latency_seconds": float(self.latency_seconds),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "tool_use_tokens": self.tool_use_tokens,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UsageRecord":
        return cls(
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            thinking_level=str(data.get("thinking_level", "")),
            purpose=str(data.get("purpose", "research")),
            status=str(data.get("status", "unknown")),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            input_tokens=_opt_int(data.get("input_tokens")),
            output_tokens=_opt_int(data.get("output_tokens")),
            thinking_tokens=_opt_int(data.get("thinking_tokens")),
            cached_tokens=_opt_int(data.get("cached_tokens")),
            total_tokens=_opt_int(data.get("total_tokens")),
            tool_use_tokens=_opt_int(data.get("tool_use_tokens")),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    response_schema: dict[str, Any]
    model: str = DEFAULT_MODEL
    thinking_level: str = DEFAULT_THINKING_LEVEL
    purpose: str = "research"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "thinking_level", normalize_thinking_level(self.thinking_level))
        if self.purpose not in PURPOSES:
            raise LLMConfigError(f"purpose must be one of {PURPOSES}, got {self.purpose!r}")
        if not self.prompt.strip():
            raise LLMConfigError("prompt must be non-empty")
        if not isinstance(self.response_schema, dict) or not self.response_schema:
            raise LLMConfigError("response_schema must be a non-empty dict")


@dataclass(frozen=True)
class LLMResponse:
    text: str
    parsed: dict[str, Any] | None
    usage: UsageRecord
    raw_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "parsed": self.parsed,
            "usage": self.usage.to_dict(),
            "raw_usage": dict(self.raw_usage),
        }


def usage_from_api_metadata(
    meta: Any,
    *,
    provider: str,
    model: str,
    thinking_level: str,
    purpose: str,
    status: str,
    latency_seconds: float,
    error: str | None = None,
) -> tuple[UsageRecord, dict[str, Any]]:
    """Map first-party usage metadata. Missing fields stay None. No estimates."""
    raw = _metadata_to_dict(meta)
    return (
        UsageRecord(
            provider=provider,
            model=model,
            thinking_level=thinking_level,
            purpose=purpose,
            status=status,
            latency_seconds=latency_seconds,
            input_tokens=_first_int(
                raw,
                (
                    "total_input_tokens",
                    "prompt_token_count",
                    "promptTokenCount",
                    "input_tokens",
                ),
            ),
            output_tokens=_first_int(
                raw,
                (
                    "total_output_tokens",
                    "candidates_token_count",
                    "candidatesTokenCount",
                    "output_tokens",
                ),
            ),
            thinking_tokens=_first_int(
                raw,
                (
                    "total_thought_tokens",
                    "thoughts_token_count",
                    "thoughtsTokenCount",
                    "thinking_tokens",
                ),
            ),
            cached_tokens=_first_int(
                raw,
                (
                    "total_cached_tokens",
                    "cached_content_token_count",
                    "cachedContentTokenCount",
                    "cached_tokens",
                ),
            ),
            total_tokens=_first_int(
                raw, ("total_tokens", "total_token_count", "totalTokenCount")
            ),
            tool_use_tokens=_first_int(raw, ("total_tool_use_tokens", "tool_use_tokens")),
            error=error,
        ),
        raw,
    )


def _metadata_to_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta)
    out: dict[str, Any] = {}
    for name in (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "total_token_count",
        "promptTokenCount",
        "candidatesTokenCount",
        "thoughtsTokenCount",
        "cachedContentTokenCount",
        "totalTokenCount",
        "total_input_tokens",
        "total_output_tokens",
        "total_thought_tokens",
        "total_cached_tokens",
        "total_tokens",
        "total_tool_use_tokens",
    ):
        if hasattr(meta, name):
            out[name] = getattr(meta, name)
    if not out and hasattr(meta, "model_dump"):
        dumped = meta.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return out


def _first_int(data: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in data and data[key] is not None:
            return _opt_int(data[key])
    return None


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
