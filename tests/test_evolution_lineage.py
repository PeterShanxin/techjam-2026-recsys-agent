"""Session-scoped lineage export. Registry stays global source of truth."""
from __future__ import annotations

from pathlib import Path

from experiment_helpers import make_spec
from research_agent.experiments import ExperimentRegistry
from research_agent.evolution.lineage import (
    format_lineage,
    lineage_forest,
    scoped_lineage_ids,
    session_lineage_forest,
)


def _registry(tmp_path: Path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "registry.sqlite")


def test_session_scoped_lineage_excludes_unrelated_history(tmp_path: Path):
    registry = _registry(tmp_path)
    session = "rs-live-aaaa"
    specs = [
        make_spec(experiment_id="random-valid-seed0", origin="baseline"),
        make_spec(experiment_id="fm-root", origin="baseline"),
        make_spec(
            experiment_id="fm-ensemble-3seed",
            origin="mutation",
            parent_ids=("fm-root",),
            parameters={"num_models": 3},
        ),
        make_spec(
            experiment_id="rs-old-bbbb-001",
            origin="mutation",
            parent_ids=("fm-root",),
            parameters={"old": True},
        ),
        make_spec(
            experiment_id=f"{session}-001",
            origin="mutation",
            parent_ids=("fm-ensemble-3seed",),
            parameters={"n": 1},
        ),
        make_spec(
            experiment_id=f"{session}-003",
            origin="mutation",
            parent_ids=("fm-ensemble-3seed",),
            parameters={"n": 3},
        ),
        make_spec(
            experiment_id=f"{session}-004",
            origin="mutation",
            parent_ids=(f"{session}-003",),
            parameters={"n": 4},
        ),
        make_spec(
            experiment_id=f"{session}-002",
            origin="mutation",
            parent_ids=("fm-root",),
            parameters={"n": 2},
        ),
        make_spec(
            experiment_id=f"{session}-005",
            origin="crossover",
            parent_ids=(f"{session}-004", "fm-ensemble-3seed"),
            parameters={"n": 5},
        ),
        make_spec(
            experiment_id=f"{session}-006",
            origin="crossover",
            parent_ids=(f"{session}-004", "fm-ensemble-3seed"),
            parameters={"n": 6},
        ),
    ]
    for spec in specs:
        registry.insert_spec(spec)

    included = scoped_lineage_ids(registry, session)
    assert "random-valid-seed0" not in included
    assert "rs-old-bbbb-001" not in included
    assert "fm-root" in included
    assert "fm-ensemble-3seed" in included
    assert f"{session}-004" in included
    assert f"{session}-005" in included

    tree = format_lineage(session_lineage_forest(registry, session))
    assert "random-valid-seed0" not in tree
    assert "rs-old-bbbb-001" not in tree
    assert tree.startswith("fm-root\n")
    assert "└── fm-ensemble-3seed" in tree or "├── fm-ensemble-3seed" in tree
    assert f"{session}-005 crossover({session}-004, fm-ensemble-3seed)" in tree
    assert f"{session}-006 crossover({session}-004, fm-ensemble-3seed)" in tree
    assert f"└── {session}-004" in tree or f"├── {session}-004" in tree
    assert f"{session}-002" in tree
    fm_pos = tree.index("fm-root")
    ens_pos = tree.index("fm-ensemble-3seed")
    child_pos = tree.index(f"{session}-003")
    assert fm_pos < ens_pos < child_pos

    full = format_lineage(lineage_forest(registry))
    assert "random-valid-seed0" in full


def test_scoped_lineage_keeps_seed_ancestors_without_session_prefix(tmp_path: Path):
    registry = _registry(tmp_path)
    session = "rs-sess-cccc"
    registry.insert_spec(make_spec(experiment_id="fm-root", origin="baseline"))
    registry.insert_spec(
        make_spec(
            experiment_id="fm-ensemble-3seed",
            origin="mutation",
            parent_ids=("fm-root",),
            parameters={"num_models": 3},
        )
    )
    included = scoped_lineage_ids(registry, session, extra_ids=("fm-root", "fm-ensemble-3seed"))
    assert included == {"fm-root", "fm-ensemble-3seed"}
    tree = format_lineage(session_lineage_forest(registry, session, extra_ids=("fm-root", "fm-ensemble-3seed")))
    assert "└── fm-ensemble-3seed" in tree or "├── fm-ensemble-3seed" in tree
