"""Lineage forest from the experiment registry. No second database."""
from __future__ import annotations

from typing import Any, Iterable

from research_agent.experiments import ExperimentRegistry


def scoped_lineage_ids(
    registry: ExperimentRegistry,
    session_id: str,
    extra_ids: Iterable[str] = (),
) -> set[str]:
    """Session experiments plus ancestors needed to explain them.

    Filtering is presentation/query only. The registry is unchanged.
    """
    prefix = f"{session_id}-"
    core = {str(item) for item in extra_ids if item}
    for experiment_id in registry.iter_ids():
        text = str(experiment_id)
        if text == session_id or text.startswith(prefix):
            core.add(text)
    included: set[str] = set()
    stack = list(core)
    while stack:
        current = stack.pop()
        if current in included:
            continue
        included.add(current)
        entry = registry.peek(current)
        if entry is None:
            continue
        for parent_id in registry.parents(current):
            if parent_id not in included:
                stack.append(parent_id)
    return included


def lineage_forest(
    registry: ExperimentRegistry,
    include_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = None if include_ids is None else {str(item) for item in include_ids}
    ids = [str(item) for item in registry.iter_ids() if allowed is None or str(item) in allowed]
    kids: dict[str, list[str]] = {experiment_id: [] for experiment_id in ids}
    for experiment_id in ids:
        for parent_id in registry.parents(experiment_id):
            if allowed is not None and parent_id not in allowed:
                continue
            kids.setdefault(parent_id, []).append(experiment_id)
    seen: set[str] = set()

    def node(experiment_id: str) -> dict[str, Any]:
        seen.add(experiment_id)
        entry = registry.peek(experiment_id)
        origin = "" if entry is None else entry.spec.origin
        parents = registry.parents(experiment_id) if entry is not None else []
        if allowed is not None:
            parents = [item for item in parents if item in allowed]
        label = experiment_id
        if origin == "crossover" and len(parents) >= 2:
            label = f"{experiment_id} crossover({', '.join(parents)})"
        children = []
        for child in kids.get(experiment_id, []):
            if child in seen:
                continue
            # Attach crossover children to the first parent only.
            child_parents = registry.parents(child)
            if allowed is not None:
                child_parents = [item for item in child_parents if item in allowed]
            if len(child_parents) >= 2 and child_parents[0] != experiment_id:
                continue
            children.append(node(child))
        return {
            "experiment_id": experiment_id,
            "label": label,
            "origin": origin,
            "parent_ids": parents,
            "children": children,
        }

    roots = [experiment_id for experiment_id in ids if not registry.parents(experiment_id)]
    if allowed is not None:
        roots = [
            experiment_id
            for experiment_id in ids
            if not [item for item in registry.parents(experiment_id) if item in allowed]
        ]
    forest = [node(experiment_id) for experiment_id in roots]
    leftover = [experiment_id for experiment_id in ids if experiment_id not in seen]
    forest.extend(node(experiment_id) for experiment_id in leftover)
    return forest


def session_lineage_forest(
    registry: ExperimentRegistry,
    session_id: str,
    extra_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    return lineage_forest(
        registry,
        include_ids=scoped_lineage_ids(registry, session_id, extra_ids),
    )


def format_lineage(forest: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    def walk(node: dict[str, Any], prefix: str, is_last: bool, *, is_root: bool) -> None:
        label = str(node.get("label") or node["experiment_id"])
        if is_root:
            lines.append(label)
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "│   ")
        children = list(node.get("children") or [])
        for index, child in enumerate(children):
            walk(child, child_prefix, index == len(children) - 1, is_root=False)

    for index, root in enumerate(forest):
        walk(root, "", True, is_root=True)
        if index != len(forest) - 1:
            lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")
