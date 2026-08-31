"""Content integrity for evaluator, starter, and reference assets.

The runner snapshots every protected file before a candidate runs and again
after it exits. Any added, removed, or modified file invalidates the attempt.
This is the backstop behind the write boundary in ``candidate_guard``: it does
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


def _is_excluded(path: Path, excluded: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in excluded)


def _iter_protected_files(
    roots: Iterable[Path], excluded: tuple[Path, ...] = ()
) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw in roots:
        root = Path(raw)
        if not root.exists():
            continue
        resolved = root.resolve()
        if _is_excluded(resolved, excluded):
            continue
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
            if path in seen or _is_excluded(path, excluded):
                continue
            seen.add(path)
            yield path


def build_manifest(
    roots: Iterable[Path],
    *,
    cache: dict[str, tuple[int, int, str]] | None = None,
    exclude: Iterable[Path] = (),
) -> ProtectedManifest:
    """Hash every file under ``roots``. Missing roots contribute nothing.

    ``cache`` maps path -> (size, mtime_ns, digest) and lets bulk reference
    data (the KuaiRand CSVs) skip re-hashing when its metadata is untouched.
    Pass it only for large read-only trees: the evaluator and source assets,
    which are the score-critical surface, are always hashed in full. Forging
    size and mtime requires ``os.utime``, which the candidate guard denies
    outside the sandbox.

    ``exclude`` drops a subtree from the walk. The runner needs it because the
    KuaiRand dataset lives *inside* ``starter/``, and would otherwise be
    hashed once uncached as part of the source tree and again via ``cache``.
    """
    excluded = tuple(Path(p).resolve() for p in exclude)
    digests: dict[str, str] = {}
    for path in _iter_protected_files(roots, excluded):
        key = path.as_posix()
        try:
            if cache is not None:
                stat = path.stat()
                cached = cache.get(key)
                if cached is not None and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
                    digests[key] = cached[2]
                    continue
                digest = sha256_path(path)
                cache[key] = (stat.st_size, stat.st_mtime_ns, digest)
                digests[key] = digest
                continue
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
