"""Minimal provider protocol. Not a multi-provider framework."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import LLMRequest, LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    """One method. Swap the production Gemini class later without a registry."""

    name: str

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return structured text plus first-party usage. Never log credentials."""
        ...
