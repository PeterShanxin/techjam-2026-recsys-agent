"""Source, config, and environment fingerprints. Not a security sandbox."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_bytes, sha256_file, sha256_text


def config_fingerprint(parameters: dict[str, Any]) -> str:
    return sha256_text(canonical_json(parameters))


def source_fingerprint(paths: Iterable[Path]) -> str:
    records = []
    for path in paths:
        resolved = Path(path)
        records.append(
            {
                "path": resolved.as_posix(),
                "sha256": sha256_file(resolved) if resolved.is_file() else None,
                "missing": not resolved.is_file(),
            }
        )
    records.sort(key=lambda item: item["path"])
    return sha256_text(canonical_json(records))


def evaluate_py_sha256(evaluate_py: Path) -> str:
    data = evaluate_py.read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(data)


def git_metadata(repo_root: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "git_dirty": None}
    return {"git_head": head, "git_dirty": bool(porcelain.strip())}


def environment_metadata(
    *,
    repo_root: Path,
    entrypoint: Path,
    evaluate_py: Path,
    source_fp: str,
    config_fp: str,
) -> dict[str, Any]:
    meta = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "entrypoint": entrypoint.as_posix(),
        "source_fingerprint": source_fp,
        "config_fingerprint": config_fp,
        "evaluate_py": evaluate_py.as_posix(),
        "evaluate_py_sha256": evaluate_py_sha256(evaluate_py) if evaluate_py.is_file() else None,
        "cwd": Path.cwd().as_posix(),
        "pid": os.getpid(),
    }
    try:
        import numpy as np

        meta["numpy"] = np.__version__
    except ImportError:
        meta["numpy"] = None
    meta.update(git_metadata(repo_root))
    return meta
