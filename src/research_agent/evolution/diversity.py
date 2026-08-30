"""Lightweight structured diversity. No embeddings."""
from __future__ import annotations

from typing import Iterable

from .types import PopulationMember


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().replace("-", "_").split()) or "other"


def semantic_signature(
    research_family: str,
    mechanism_tags: Iterable[str],
    changed_axes: Iterable[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    family = _norm(research_family)
    tags = tuple(sorted({_norm(item) for item in mechanism_tags if str(item).strip()}))
    axes = tuple(sorted({_norm(item) for item in changed_axes if str(item).strip()}))
    return (family, tags, axes)


def member_signature(member: PopulationMember) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    return semantic_signature(member.research_family, member.mechanism_tags, member.changed_axes)


def duplicate_reason(
    candidate: PopulationMember,
    existing: Iterable[PopulationMember],
) -> str | None:
    known = list(existing)
    if candidate.spec_hash:
        for item in known:
            if item.spec_hash and item.spec_hash == candidate.spec_hash:
                return "spec_hash"
    if candidate.source_fingerprint:
        for item in known:
            if item.source_fingerprint and item.source_fingerprint == candidate.source_fingerprint:
                return "source_fingerprint"
    sig = member_signature(candidate)
    if sig[0] == "other" and not sig[1] and not sig[2]:
        return None
    for item in known:
        if member_signature(item) == sig:
            return "semantic_signature"
    return None
