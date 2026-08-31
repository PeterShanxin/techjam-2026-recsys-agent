"""Compact runtime capabilities for ResearchState. Derived from this machine and project deps."""
from __future__ import annotations

import importlib.util
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from research_agent.llm.secrets import sanitize

# Common research packages the agent might propose. Not an allowlist.
_PROBE_UNSUPPORTED = (
    "torch",
    "pandas",
    "sklearn",
    "tensorflow",
    "jax",
    "lightgbm",
    "xgboost",
)

STARTER_MODULES = ("data", "baseline", "evaluate")

_ENV_RULE = (
    "Generated experiments must use only this environment "
    "(Python stdlib, listed allowed third-party packages, and starter modules data/baseline/evaluate). "
    "If the proposed method cannot execute, fail explicitly. "
    "Do not silently fall back to the FM baseline, the selected parent, or another algorithm "
    "and emit those scores as evidence for the claimed hypothesis. "
    "Experiments run write-confined: the only writable location is the directory of "
    "--output-scores (plus the process temp dir), and subprocess, multiprocessing, "
    "ctypes, and network access are unavailable. Write scores with a single process."
)


@dataclass(frozen=True)
class EnvironmentCapabilities:
    python_version: str
    platform_name: str
    architecture: str
    allowed_third_party: tuple[str, ...]
    unsupported_or_unavailable: tuple[str, ...]
    starter_modules: tuple[str, ...] = STARTER_MODULES
    rule: str = _ENV_RULE

    def to_dict(self) -> dict[str, Any]:
        return sanitize(
            {
                "python_version": self.python_version,
                "platform": self.platform_name,
                "architecture": self.architecture,
                "allowed_third_party": list(self.allowed_third_party),
                "unsupported_or_unavailable": list(self.unsupported_or_unavailable),
                "starter_modules": list(self.starter_modules),
                "rule": self.rule,
            }
        )


def discover_environment(repo_root: Path | None = None) -> EnvironmentCapabilities:
    declared = _declared_third_party(repo_root)
    if not declared:
        declared = ("numpy",) if _can_import("numpy") else ()
    allowed = tuple(name for name in declared if _can_import(name))
    unsupported = tuple(name for name in _PROBE_UNSUPPORTED if name not in allowed)
    return EnvironmentCapabilities(
        python_version=platform.python_version(),
        platform_name=platform.system(),
        architecture=platform.machine(),
        allowed_third_party=allowed,
        unsupported_or_unavailable=unsupported,
    )


def format_preflight_repair_message(
    error: str,
    *,
    hypothesis: str | None,
    environment: EnvironmentCapabilities | None = None,
) -> str:
    env = environment or discover_environment()
    env_line = (
        f"Available environment: python={env.python_version} "
        f"{env.platform_name}/{env.architecture}; "
        f"allowed_third_party={list(env.allowed_third_party)}; "
        f"unsupported_or_unavailable={list(env.unsupported_or_unavailable)}."
    )
    hypo = (hypothesis or "").strip() or "(not parsed)"
    preserve = (
        "Preserve the original hypothesis. Implement it using NumPy and the standard library "
        "when feasible. Do not change the research question. "
        "Do not silently fall back to the FM baseline or parent."
    )
    return (
        f"{error}\n"
        f"{env_line}\n"
        f"Original hypothesis: {hypo}\n"
        f"{preserve}"
    )


def _can_import(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _declared_third_party(repo_root: Path | None) -> tuple[str, ...]:
    if repo_root is None:
        return ()
    root = Path(repo_root)
    names: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        names.extend(_names_from_pyproject(pyproject.read_text(encoding="utf-8")))
    requirements = root / "requirements.txt"
    if requirements.is_file():
        names.extend(_names_from_requirements(requirements.read_text(encoding="utf-8")))
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _names_from_pyproject(text: str) -> list[str]:
    names: list[str] = []
    in_deps = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("dependencies") and "=" in line and "[" in line:
            in_deps = True
            inline = line.split("[", 1)[1]
            names.extend(_dep_names_from_fragment(inline))
            if "]" in inline:
                in_deps = False
            continue
        if in_deps:
            if line.startswith("]"):
                in_deps = False
                continue
            names.extend(_dep_names_from_fragment(line))
    return names


def _names_from_requirements(text: str) -> list[str]:
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.extend(_dep_names_from_fragment(line))
    return names


def _dep_names_from_fragment(fragment: str) -> Iterable[str]:
    for token in fragment.split(","):
        token = token.strip().strip("[]\"'")
        if not token or token.startswith("#"):
            continue
        name = re.split(r"[<>=~!;\\[]", token, maxsplit=1)[0].strip()
        if name:
            yield name
