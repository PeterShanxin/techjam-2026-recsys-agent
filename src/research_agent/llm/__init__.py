"""LLM provider surface. Production: Gemini REST. Tests: FakeProvider."""

from .credentials import load_repo_dotenv, resolve_gemini_api_key
from .fake import FakeProvider
from .gemini import GeminiProvider, build_interaction_payload, extract_model_output_text
from .protocol import LLMProvider
from .secrets import redact_text, sanitize
from .types import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_LEVEL,
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    UsageRecord,
    usage_from_api_metadata,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_LEVEL",
    "FakeProvider",
    "GeminiProvider",
    "LLMAuthError",
    "LLMConfigError",
    "LLMError",
    "LLMProtocolError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMTransientError",
    "UsageRecord",
    "build_interaction_payload",
    "extract_model_output_text",
    "load_repo_dotenv",
    "redact_text",
    "resolve_gemini_api_key",
    "sanitize",
    "usage_from_api_metadata",
]
