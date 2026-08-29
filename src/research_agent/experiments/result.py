"""Raw experiment outcome. Selection decisions live in the registry, not here."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

STATUSES = ("success", "failed", "timeout", "invalid")
RESULT_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class Metrics:
    gauc: float
    ndcg_at_5: float
    primary: float
    users: int | None = None
    rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "GAUC": float(self.gauc),
            "nDCG@5": float(self.ndcg_at_5),
            "primary": float(self.primary),
        }
        if self.users is not None:
            payload["users"] = int(self.users)
        if self.rows is not None:
            payload["rows"] = int(self.rows)
        return payload

    @classmethod
    def from_official(cls, official: Mapping[str, Any]) -> "Metrics":
        return cls(
            gauc=float(official["GAUC"]),
            ndcg_at_5=float(official["nDCG@5"]),
            primary=float(official["primary"]),
            users=int(official["users"]) if "users" in official else None,
            rows=int(official["rows"]) if "rows" in official else None,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Metrics":
        gauc = data.get("GAUC", data.get("gauc"))
        ndcg = data.get("nDCG@5", data.get("ndcg_at_5"))
        primary = data.get("primary")
        if gauc is None or ndcg is None or primary is None:
            raise ValueError("metrics require GAUC, nDCG@5, and primary")
        return cls(
            gauc=float(gauc),
            ndcg_at_5=float(ndcg),
            primary=float(primary),
            users=data.get("users"),
            rows=data.get("rows"),
        )


@dataclass(frozen=True)
class FailureInfo:
    kind: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": self.message, "details": dict(self.details)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "FailureInfo | None":
        if not data:
            return None
        return cls(
            kind=str(data.get("kind", "unknown")),
            message=str(data.get("message", "")),
            details=dict(data.get("details") or {}),
        )


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: str
    evaluation_split: str
    seed: int
    spec_hash: str
    wall_seconds: float
    return_code: int | None
    run_dir: str
    stdout_path: str
    stderr_path: str
    scores_path: str | None = None
    metrics: Metrics | None = None
    source_fingerprint: str = ""
    config_fingerprint: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    failure: FailureInfo | None = None
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.status == "success" and self.metrics is None:
            raise ValueError("successful result requires metrics")
        if self.status != "success" and self.metrics is not None:
            raise ValueError("metrics are only stored on successful results")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "evaluation_split": self.evaluation_split,
            "seed": self.seed,
            "spec_hash": self.spec_hash,
            "wall_seconds": float(self.wall_seconds),
            "return_code": self.return_code,
            "run_dir": self.run_dir,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "scores_path": self.scores_path,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "source_fingerprint": self.source_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "environment": dict(self.environment),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True) + (
            "\n" if indent is not None else ""
        )

    def write_json(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentResult":
        metrics = data.get("metrics")
        return cls(
            schema_version=str(data.get("schema_version", RESULT_SCHEMA_VERSION)),
            experiment_id=str(data["experiment_id"]),
            status=str(data["status"]),
            evaluation_split=str(data["evaluation_split"]),
            seed=int(data["seed"]),
            spec_hash=str(data.get("spec_hash", "")),
            wall_seconds=float(data.get("wall_seconds", 0.0)),
            return_code=data.get("return_code"),
            run_dir=str(data.get("run_dir", "")),
            stdout_path=str(data.get("stdout_path", "")),
            stderr_path=str(data.get("stderr_path", "")),
            scores_path=data.get("scores_path"),
            metrics=None if metrics is None else Metrics.from_dict(metrics),
            source_fingerprint=str(data.get("source_fingerprint", "")),
            config_fingerprint=str(data.get("config_fingerprint", "")),
            environment=dict(data.get("environment") or {}),
            failure=FailureInfo.from_dict(data.get("failure")),
        )
