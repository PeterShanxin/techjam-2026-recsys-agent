"""Exact and semantic duplicate suppression. Zero API spend."""
from __future__ import annotations

from evolution_helpers import make_member
from research_agent.evolution.diversity import duplicate_reason, semantic_signature


def test_semantic_signature_is_structured_not_embedding():
    sig = semantic_signature("Ensemble", ["Bagging", "fm"], ["ensembling", "Ensembling"])
    assert sig == ("ensemble", ("bagging", "fm"), ("ensembling",))


def test_exact_duplicate_by_spec_hash_and_fingerprint():
    existing = [
        make_member(experiment_id="old", spec_hash="abc", source_fingerprint="fp1"),
    ]
    same_hash = make_member(experiment_id="new", spec_hash="abc", source_fingerprint="other")
    same_fp = make_member(experiment_id="new2", spec_hash="zzz", source_fingerprint="fp1")
    unique = make_member(
        experiment_id="new3",
        spec_hash="zzz",
        source_fingerprint="fp9",
        research_family="temporal",
        mechanism_tags=("hourmin",),
        changed_axes=("temporal_signal",),
    )
    assert duplicate_reason(same_hash, existing) == "spec_hash"
    assert duplicate_reason(same_fp, existing) == "source_fingerprint"
    assert duplicate_reason(unique, existing) is None


def test_semantic_duplicate_uses_family_tags_and_axes():
    existing = [
        make_member(
            experiment_id="old",
            spec_hash="h1",
            source_fingerprint="fp1",
            research_family="ensemble",
            mechanism_tags=("bagging",),
            changed_axes=("ensembling",),
        )
    ]
    twin = make_member(
        experiment_id="new",
        spec_hash="h2",
        source_fingerprint="fp2",
        research_family="ensemble",
        mechanism_tags=("bagging",),
        changed_axes=("ensembling",),
    )
    different = make_member(
        experiment_id="loss",
        spec_hash="h3",
        source_fingerprint="fp3",
        research_family="ranking_loss",
        mechanism_tags=("bpr",),
        changed_axes=("objective",),
    )
    assert duplicate_reason(twin, existing) == "semantic_signature"
    assert duplicate_reason(different, existing) is None
