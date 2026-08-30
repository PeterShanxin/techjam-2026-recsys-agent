"""LLM provider surface. Production: Gemini. Tests: FakeProvider."""

from .fake import FakeProvider
from .gemini import GeminiProvider
from .protocol import LLMProvider
from .secrets import redact_text, sanitize
from .types import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    LLMConfigError,
    LLMError,
    LLMRequest,
    LLMResponse,
    UsageRecord,
    usage_from_api_metadata,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "FakeProvider",
    "GeminiProvider",
    "LLMConfigError",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "UsageRecord",
    "redact_text",
    "sanitize",
    "usage_from_api_metadata",
]
