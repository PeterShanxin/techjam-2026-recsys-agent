"""Repo-root .env loading. Process environment always wins."""
from __future__ import annotations

import os
from pathlib import Path

from .types import LLMConfigError

API_KEY_ENV = "GEMINI_API_KEY"


def load_repo_dotenv(repo_root: Path, *, override: bool = False) -> None:
    """Load KEY=VALUE lines from repo-root .env. Never logs values."""
    path = Path(repo_root) / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def resolve_gemini_api_key(repo_root: Path | None = None) -> str:
    existing = os.environ.get(API_KEY_ENV)
    if existing and existing.strip():
        return existing.strip()
    if repo_root is not None:
        load_repo_dotenv(Path(repo_root), override=False)
        loaded = os.environ.get(API_KEY_ENV)
        if loaded and loaded.strip():
            return loaded.strip()
    raise LLMConfigError(
        f"{API_KEY_ENV} is not set in the process environment or repo-root .env"
    )
