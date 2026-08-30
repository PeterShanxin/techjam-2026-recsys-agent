"""Isolated generated-candidate workspace. Never writes into starter/ or src/ during a run."""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agent.experiments.canonical import sha256_text
from research_agent.experiments.fingerprint import source_fingerprint
from research_agent.experiments.spec import ExperimentSpec, ImplementationRef

from .constants import CANDIDATE_FILENAME
from .safety import SafetyError, validate_candidate_source


@dataclass(frozen=True)
class MaterializedCandidate:
    experiment_id: str
    dest: Path
    source: str
    fingerprint: str
    diff_vs_parent: str
    implementation: ImplementationRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "path": self.dest.as_posix(),
            "source_fingerprint": self.fingerprint,
            "diff_vs_parent": self.diff_vs_parent,
            "implementation": self.implementation.to_dict(),
        }


class CandidateWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dest_for(self, experiment_id: str) -> Path:
        return self.root / experiment_id / CANDIDATE_FILENAME

    def materialize(
        self,
        *,
        experiment_id: str,
        source: str,
        parent_source: str,
        repo_root: Path,
    ) -> MaterializedCandidate:
        dest = self.dest_for(experiment_id)
        if dest.exists():
            raise SafetyError(
                f"refusing to overwrite existing candidate for {experiment_id}: {dest}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        validate_candidate_source(source, dest, self.root)
        dest.write_text(source, encoding="utf-8")
        written = dest.read_text(encoding="utf-8")
        if written != source:
            raise SafetyError("generated source round-trip mismatch")
        fingerprint = source_fingerprint([dest])
        dest_resolved = dest.resolve()
        try:
            rel = dest_resolved.relative_to(Path(repo_root).resolve()).as_posix()
            implementation = ImplementationRef(entrypoint=rel, source_root=None)
        except ValueError:
            implementation = ImplementationRef(entrypoint=dest_resolved.as_posix(), source_root=None)
        return MaterializedCandidate(
            experiment_id=experiment_id,
            dest=dest,
            source=source,
            fingerprint=fingerprint,
            diff_vs_parent=unified_diff(parent_source, source, from_name="parent.py", to_name=CANDIDATE_FILENAME),
            implementation=implementation,
        )

    def load_parent_source(self, spec: ExperimentSpec, repo_root: Path) -> str:
        path = _resolve_entrypoint(spec, repo_root)
        if not path.is_file():
            raise SafetyError(f"parent source not found: {path}")
        return path.read_text(encoding="utf-8")


def unified_diff(parent: str, child: str, *, from_name: str, to_name: str) -> str:
    return "".join(
        difflib.unified_diff(
            parent.splitlines(keepends=True),
            child.splitlines(keepends=True),
            fromfile=from_name,
            tofile=to_name,
            n=3,
        )
    )


def _resolve_entrypoint(spec: ExperimentSpec, repo_root: Path) -> Path:
    raw = Path(spec.implementation.entrypoint)
    if raw.is_absolute():
        return raw
    root = spec.implementation.source_root
    base = Path(root) if root else Path(repo_root)
    if not base.is_absolute():
        base = Path(repo_root) / base
    return (base / raw).resolve()
