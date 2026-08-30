"""Keep API credentials out of logs, traces, prompts, and experiment metadata."""
from __future__ import annotations

import os
import re
from typing import Any

_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{20,}")
_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
REDACTED = "[REDACTED]"


def active_secrets() -> tuple[str, ...]:
    found = []
    for name in _ENV_NAMES:
        value = os.environ.get(name)
        if value:
            found.append(value)
    return tuple(found)


def redact_text(text: str | None) -> str:
    if not text:
        return ""
    out = str(text)
    for secret in active_secrets():
        if secret:
            out = out.replace(secret, REDACTED)
    out = _GOOGLE_KEY_RE.sub(REDACTED, out)
    return out


def sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items() if not _is_secret_key(k)}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def assert_no_secrets(payload: Any) -> None:
    text = repr(payload)
    for secret in active_secrets():
        if secret and secret in text:
            raise ValueError("refusing to persist a payload that contains an API key")
    if _GOOGLE_KEY_RE.search(text):
        raise ValueError("refusing to persist a payload that looks like a Google API key")


def _is_secret_key(key: Any) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in ("api_key", "apikey", "gemini_api_key", "google_api_key"))
