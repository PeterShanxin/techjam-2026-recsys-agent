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

IGNORED_DIR_NAMES = frozenset({".git", ".pytest_cache", ".mypy_cache"})
# Bytecode the parent legitimately creates as it imports its own modules
# during a session. Skipped by default so it cannot raise a false alarm, but
# see ``include_bytecode``: for the starter tree it is hashed, because a
# planted .pyc whose header matches the real source would be picked up by the
# *next* parent process even though this one already bound the good module.
BYTECODE_DIR_NAMES = frozenset({"__pycache__"})
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


def _iter_protected_files(
    roots: Iterable[Path], include_bytecode: bool = False
) -> Iterable[Path]:
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
        try:
            children = sorted(resolved.rglob("*"))
        except OSError:
            # An unwalkable root is itself a change worth surfacing, not a
            # reason to crash the run that was about to be validated.
            yield resolved
            continue
        for path in children:
            if any(part in IGNORED_DIR_NAMES for part in path.parts):
                continue
            is_bytecode = (
                path.suffix in IGNORED_SUFFIXES
                or any(part in BYTECODE_DIR_NAMES for part in path.parts)
            )
            if is_bytecode and not include_bytecode:
                continue
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def build_manifest(
    roots: Iterable[Path], *, include_bytecode: bool = False
) -> ProtectedManifest:
    """Hash every file under ``roots`` in full. Missing roots contribute nothing.

    Overlapping roots are fine: files are de-duplicated by resolved path, so
    the dataset living inside ``starter/`` is hashed once, not twice.

    Never raises for an unreadable path. The caller is usually deciding
    whether to invalidate a run, and an exception there would crash the run
    instead of failing it, so problems are recorded as digest values.
    """
    digests: dict[str, str] = {}
    for path in _iter_protected_files(roots, include_bytecode):
        key = path.as_posix()
        try:
            digests[key] = sha256_path(path)
        except OSError as exc:  # unreadable file is itself a change worth surfacing
            digests[key] = f"unreadable:{exc.__class__.__name__}"
    return ProtectedManifest(digests=digests)


def merge_manifests(*manifests: ProtectedManifest) -> ProtectedManifest:
    digests: dict[str, str] = {}
    for manifest in manifests:
        digests.update(manifest.digests)
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
