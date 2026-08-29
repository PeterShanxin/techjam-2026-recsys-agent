"""Model-agnostic experiment request.

Experiment identity (experiment_id) is separate from the execution
fingerprint (spec_hash). Two reruns may share a hash and differ in id.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json, sha256_text
from .errors import SpecError
from .splits import DEFAULT_EVALUATION_SPLIT, normalize_evaluation_split

SCHEMA_VERSION = "1"
ORIGINS = ("baseline", "manual", "mutation", "crossover")
EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ImplementationRef:
    """Where candidate code lives. Not a model-family field."""

    entrypoint: str
    source_root: str | None = None
    extra_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entrypoint or not str(self.entrypoint).strip():
            raise SpecError("implementation.entrypoint is required")
        object.__setattr__(self, "entrypoint", str(self.entrypoint).replace("\\", "/"))
        if self.source_root:
            object.__setattr__(self, "source_root", str(self.source_root).replace("\\", "/"))
        extras = tuple(str(p).replace("\\", "/") for p in self.extra_paths)
        object.__setattr__(self, "extra_paths", extras)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entrypoint": self.entrypoint,
            "source_root": self.source_root,
            "extra_paths": list(self.extra_paths),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ImplementationRef":
        extras = data.get("extra_paths") or ()
        return cls(
            entrypoint=str(data["entrypoint"]),
            source_root=data.get("source_root"),
            extra_paths=tuple(extras),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    implementation: ImplementationRef
    hypothesis: str = ""
    rationale: str = ""
    origin: str = "manual"
    parent_ids: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    evaluation_split: str = DEFAULT_EVALUATION_SPLIT
    timeout_seconds: float = 600.0
    allow_test_split: bool = False
    tags: tuple[str, ...] = ()
    notes: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SpecError(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        if not EXPERIMENT_ID_RE.fullmatch(self.experiment_id):
            raise SpecError(
                "experiment_id must match [A-Za-z0-9][A-Za-z0-9._-]* "
                f"(got {self.experiment_id!r})"
            )
        if self.origin not in ORIGINS:
            raise SpecError(f"origin must be one of {ORIGINS}, got {self.origin!r}")
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(self, "tags", tuple(str(t) for t in self.tags))
        object.__setattr__(
            self, "evaluation_split", normalize_evaluation_split(self.evaluation_split)
        )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SpecError("seed must be an int")
        if self.timeout_seconds <= 0:
            raise SpecError("timeout_seconds must be positive")
        if not isinstance(self.parameters, dict):
            raise SpecError("parameters must be a dict")
        object.__setattr__(self, "parameters", json.loads(canonical_json(self.parameters)))
        _validate_parents(self.origin, self.parent_ids)

    @property
    def spec_hash(self) -> str:
        return sha256_text(canonical_json(self.fingerprint_payload()))

    def fingerprint_payload(self) -> dict[str, Any]:
        """Execution fingerprint only. Identity and prose are excluded."""
        return {
            "evaluation_split": self.evaluation_split,
            "implementation": self.implementation.to_dict(),
            "parameters": self.parameters,
            "schema_version": self.schema_version,
            "seed": self.seed,
        }

    def test_opt_in(self) -> bool:
        return bool(self.allow_test_split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "spec_hash": self.spec_hash,
            "parent_ids": list(self.parent_ids),
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "origin": self.origin,
            "implementation": self.implementation.to_dict(),
            "parameters": self.parameters,
            "seed": self.seed,
            "evaluation_split": self.evaluation_split,
            "timeout_seconds": self.timeout_seconds,
            "allow_test_split": self.allow_test_split,
            "tags": list(self.tags),
            "notes": self.notes,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True) + (
            "\n" if indent is not None else ""
        )

    def write_json(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        if "implementation" not in data:
            raise SpecError("implementation is required")
        spec = cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            experiment_id=str(data["experiment_id"]),
            parent_ids=tuple(data.get("parent_ids") or ()),
            hypothesis=str(data.get("hypothesis", "")),
            rationale=str(data.get("rationale", "")),
            origin=str(data.get("origin", "manual")),
            implementation=ImplementationRef.from_dict(data["implementation"]),
            parameters=dict(data.get("parameters") or {}),
            seed=int(data.get("seed", 0)),
            evaluation_split=str(data.get("evaluation_split", DEFAULT_EVALUATION_SPLIT)),
            timeout_seconds=float(data.get("timeout_seconds", 600.0)),
            allow_test_split=bool(data.get("allow_test_split", False)),
            tags=tuple(data.get("tags") or ()),
            notes=str(data.get("notes", "")),
        )
        stored = data.get("spec_hash")
        if stored and stored != spec.spec_hash:
            raise SpecError(
                f"stored spec_hash {stored!r} does not match recomputed {spec.spec_hash!r}"
            )
        return spec

    @classmethod
    def from_json(cls, text: str) -> "ExperimentSpec":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_path(cls, path: Path) -> "ExperimentSpec":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _validate_parents(origin: str, parent_ids: tuple[str, ...]) -> None:
    if any(not pid or not str(pid).strip() for pid in parent_ids):
        raise SpecError("parent_ids must not contain empty values")
    if len(parent_ids) != len(set(parent_ids)):
        raise SpecError("parent_ids must be unique")
    for pid in parent_ids:
        if not EXPERIMENT_ID_RE.fullmatch(pid):
            raise SpecError(f"invalid parent id {pid!r}")
    if origin == "baseline" and parent_ids:
        raise SpecError("baseline origin requires zero parents")
    if origin == "mutation" and len(parent_ids) != 1:
        raise SpecError("mutation origin requires exactly one parent")
    if origin == "crossover" and len(parent_ids) < 2:
        raise SpecError("crossover origin requires at least two parents")
