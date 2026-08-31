"""Minimal, fail-closed environment for untrusted candidate subprocesses.

Generated candidate code is untrusted. It is built from an empty mapping and
receives only variables on ``CANDIDATE_ENV_ALLOWLIST`` -- never a copy of the
parent environment. Anything the agent process holds (Gemini/OpenAI/Anthropic
keys, GitHub tokens, cloud credentials, database URLs) is absent by
construction rather than by blocklist.
"""
from __future__ import annotations

import os
import re
from typing import Mapping

# Variables a candidate genuinely needs to start Python, load NumPy's native
# libraries, and produce deterministic output. Every entry is a name whose
# value is a path, locale, or CPU descriptor -- never a credential.
_COMMON_ALLOWLIST = (
    "PATH",  # dynamic loader search path for NumPy's native libraries
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
)

# Windows refuses to start CPython without SystemRoot; the rest keep
# platform/CPU introspection and file lookups working. USERPROFILE, APPDATA,
# LOCALAPPDATA and USERNAME are deliberately excluded: they point at
# per-user credential stores and are not needed under ``-s``.
_WINDOWS_ALLOWLIST = (
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "OS",
)

# Thread-count pins. Passed through when the parent sets them because they
# change floating-point reduction order, and therefore reproducibility.
_DETERMINISM_ALLOWLIST = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

CANDIDATE_ENV_ALLOWLIST: tuple[str, ...] = tuple(
    sorted({*_COMMON_ALLOWLIST, *_WINDOWS_ALLOWLIST, *_DETERMINISM_ALLOWLIST})
)

# Names the runner sets itself; never inherited from the parent.
_RUNNER_CONTROLLED = ("PYTHONPATH", "PYTHONHASHSEED", "TMPDIR", "TEMP", "TMP")

# Belt-and-braces: refuse to forward anything whose *name* looks like a
# credential, even if a future edit adds it to the allowlist.
_SECRET_NAME_RE = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|SESSION|COOKIE|AUTH|"
    r"BEARER|PRIVATE|SIGNATURE|LICENSE|DSN|CONNECTION_?STRING)",
    re.IGNORECASE,
)
_SECRET_NAME_EXEMPT = frozenset({"PYTHONHASHSEED"})


class CandidateEnvError(ValueError):
    """Refused to build a candidate environment."""


def is_secret_like_name(name: str) -> bool:
    """True when an environment variable name reads like a credential."""
    upper = str(name).upper()
    if upper in _SECRET_NAME_EXEMPT:
        return False
    return bool(_SECRET_NAME_RE.search(upper))


def build_candidate_env(
    *,
    pythonpath: str,
    temp_dir: str,
    parent_env: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for an untrusted candidate subprocess.

    Starts from ``{}`` and adds only allowlisted names, so a variable is
    forwarded when it is explicitly permitted -- not when it fails to match a
    blocklist.
    """
    source = os.environ if parent_env is None else parent_env
    env: dict[str, str] = {}
    for name in CANDIDATE_ENV_ALLOWLIST:
        value = source.get(name)
        if value is None:
            continue
        if is_secret_like_name(name):  # pragma: no cover - allowlist is curated
            raise CandidateEnvError(f"allowlisted name looks like a credential: {name}")
        env[name] = str(value)

    env["PYTHONPATH"] = str(pythonpath)
    # Pinned so dict/set iteration over strings is stable across attempts.
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    # Keep candidate temp files inside the attempt directory.
    for name in ("TMPDIR", "TEMP", "TMP"):
        env[name] = str(temp_dir)

    for name, value in (extra or {}).items():
        key = str(name)
        if is_secret_like_name(key):
            raise CandidateEnvError(f"refusing to pass credential-like variable: {key}")
        if key in _RUNNER_CONTROLLED:
            raise CandidateEnvError(f"{key} is runner-controlled and cannot be overridden")
        env[key] = str(value)
    return env
