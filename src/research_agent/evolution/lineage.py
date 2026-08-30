"""Lineage forest from the experiment registry. No second database."""
from __future__ import annotations

from typing import Any

from research_agent.experiments import ExperimentRegistry


def lineage_forest(registry: ExperimentRegistry) -> list[dict[str, Any]]:
    ids = [str(item) for item in registry.iter_ids()]
    kids: dict[str, list[str]] = {experiment_id: [] for experiment_id in ids}
    for experiment_id in ids:
        for parent_id in registry.parents(experiment_id):
            kids.setdefault(parent_id, []).append(experiment_id)
    seen: set[str] = set()

    def node(experiment_id: str) -> dict[str, Any]:
        seen.add(experiment_id)
        entry = registry.peek(experiment_id)
        origin = "" if entry is None else entry.spec.origin
        parents = registry.parents(experiment_id) if entry is not None else []
        label = experiment_id
        if origin == "crossover" and len(parents) >= 2:
            label = f"{experiment_id} crossover({', '.join(parents)})"
        children = []
        for child in kids.get(experiment_id, []):
            if child in seen:
                continue
            # Attach crossover children to the first parent only.
            child_parents = registry.parents(child)
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
    forest = [node(experiment_id) for experiment_id in roots]
    leftover = [experiment_id for experiment_id in ids if experiment_id not in seen]
    forest.extend(node(experiment_id) for experiment_id in leftover)
    return forest


def format_lineage(forest: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    def walk(node: dict[str, Any], prefix: str, is_last: bool) -> None:
        connector = "" if prefix == "" else ("└── " if is_last else "├── ")
        lines.append(f"{prefix}{connector}{node.get('label') or node['experiment_id']}")
        child_prefix = prefix + ("" if prefix == "" else ("    " if is_last else "│   "))
        children = list(node.get("children") or [])
        for index, child in enumerate(children):
            walk(child, child_prefix, index == len(children) - 1)

    for index, root in enumerate(forest):
        walk(root, "", True)
        if index != len(forest) - 1:
            lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")
