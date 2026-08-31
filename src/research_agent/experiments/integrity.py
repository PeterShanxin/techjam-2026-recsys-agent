"""Content integrity for evaluator, starter, and reference assets.

The runner snapshots every protected file before a candidate runs and again
after it exits. Any added, removed, or modified file invalidates the attempt.
Every protected file is hashed in full, every attempt. There is no
metadata-keyed shortcut: size and mtime are attacker-controlled, and this
check is the enforced property, so it does not get to be approximate. It does
not care *how* a mutation happened, only that the bytes changed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

# Build products the parent process legitimately creates while importing the
# starter modules. Excluded so they cannot raise a false integrity alarm.
IGNORED_DIR_NAMES = frozenset({"__pycache__", ".git", ".pytest_cache", ".mypy_cache"})
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd~"})

_CHUNK = 1 << 20


@dataclass(frozen=True)
class ProtectedManifest:
    """SHA-256 of every protected file, keyed by POSIX path."""

    digests: Mapping[str, str]

    @property
    def digest(self) -> str:
        """Single hash over the whole manifest, for cheap equality and logging."""
        joined = "\n".join(f"{path}:{self.digests[path]}" for path in sorted(self.digests))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.digests)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_protected_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw in roots:
        root = Path(raw)
        if not root.exists():
            continue
        resolved = root.resolve()
        if resolved.is_file():
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
            continue
        for path in sorted(resolved.rglob("*")):
            if any(part in IGNORED_DIR_NAMES for part in path.parts):
                continue
            if path.suffix in IGNORED_SUFFIXES:
                continue
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def build_manifest(roots: Iterable[Path]) -> ProtectedManifest:
    """Hash every file under ``roots`` in full. Missing roots contribute nothing.

    Overlapping roots are fine: files are de-duplicated by resolved path, so
    the dataset living inside ``starter/`` is hashed once, not twice.
    """
    digests: dict[str, str] = {}
    for path in _iter_protected_files(roots):
        key = path.as_posix()
        try:
            digests[key] = sha256_path(path)
        except OSError as exc:  # unreadable file is itself a change worth surfacing
            digests[key] = f"unreadable:{exc.__class__.__name__}"
    return ProtectedManifest(digests=digests)


def diff_manifests(before: ProtectedManifest, after: ProtectedManifest) -> dict[str, str]:
    """Map path -> 'modified' | 'removed' | 'added'. Empty when nothing changed."""
    changes: dict[str, str] = {}
    for path, digest in before.digests.items():
        found = after.digests.get(path)
        if found is None:
            changes[path] = "removed"
        elif found != digest:
            changes[path] = "modified"
    for path in after.digests:
        if path not in before.digests:
            changes[path] = "added"
    return changes
