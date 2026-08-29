"""SQLite experiment registry. Deterministic primitives only — no fitness policy."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import RegistryError
from .result import ExperimentResult
from .spec import ExperimentSpec
from .splits import RESEARCH_SPLIT

DECISIONS = ("pending", "accepted", "rejected")
SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    spec_hash TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    origin TEXT NOT NULL,
    evaluation_split TEXT NOT NULL,
    seed INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parents (
    experiment_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, parent_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    evaluation_split TEXT NOT NULL,
    primary_score REAL,
    gauc REAL,
    ndcg_at_5 REAL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    experiment_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_experiments_spec_hash ON experiments(spec_hash);
CREATE INDEX IF NOT EXISTS idx_results_status ON results(status);
CREATE INDEX IF NOT EXISTS idx_results_split_primary
    ON results(evaluation_split, status, primary_score);
"""


@dataclass(frozen=True)
class RegistryEntry:
    spec: ExperimentSpec
    result: ExperimentResult | None
    decision: str
    created_at: str


class ExperimentRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def insert_spec(self, spec: ExperimentSpec) -> RegistryEntry:
        existing = self._conn.execute(
            "SELECT spec_hash FROM experiments WHERE experiment_id = ?",
            (spec.experiment_id,),
        ).fetchone()
        if existing:
            if existing["spec_hash"] != spec.spec_hash:
                raise RegistryError(
                    f"experiment_id {spec.experiment_id!r} already exists with a different spec_hash"
                )
            return self.get(spec.experiment_id)
        now = _utc_now()
        self._conn.execute(
            """
            INSERT INTO experiments (
                experiment_id, spec_hash, spec_json, origin, evaluation_split, seed, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.experiment_id,
                spec.spec_hash,
                spec.to_json(indent=None),
                spec.origin,
                spec.evaluation_split,
                spec.seed,
                now,
            ),
        )
        for position, parent_id in enumerate(spec.parent_ids):
            self._conn.execute(
                "INSERT INTO parents (experiment_id, parent_id, position) VALUES (?, ?, ?)",
                (spec.experiment_id, parent_id, position),
            )
        self._conn.execute(
            "INSERT INTO decisions (experiment_id, decision, updated_at) VALUES (?, 'pending', ?)",
            (spec.experiment_id, now),
        )
        self._conn.commit()
        return self.get(spec.experiment_id)

    def upsert_result(self, result: ExperimentResult) -> RegistryEntry:
        if self._conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?",
            (result.experiment_id,),
        ).fetchone() is None:
            raise RegistryError(
                f"cannot store result for unknown experiment_id {result.experiment_id!r}"
            )
        now = _utc_now()
        primary = gauc = ndcg = None
        if result.metrics is not None:
            primary = float(result.metrics.primary)
            gauc = float(result.metrics.gauc)
            ndcg = float(result.metrics.ndcg_at_5)
        self._conn.execute(
            """
            INSERT INTO results (
                experiment_id, status, evaluation_split, primary_score, gauc, ndcg_at_5,
                result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                status = excluded.status,
                evaluation_split = excluded.evaluation_split,
                primary_score = excluded.primary_score,
                gauc = excluded.gauc,
                ndcg_at_5 = excluded.ndcg_at_5,
                result_json = excluded.result_json,
                created_at = excluded.created_at
            """,
            (
                result.experiment_id,
                result.status,
                result.evaluation_split,
                primary,
                gauc,
                ndcg,
                result.to_json(indent=None),
                now,
            ),
        )
        self._conn.commit()
        return self.get(result.experiment_id)

    def get(self, experiment_id: str) -> RegistryEntry:
        row = self._conn.execute(
            "SELECT spec_json, created_at FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise RegistryError(f"unknown experiment_id {experiment_id!r}")
        spec = ExperimentSpec.from_json(row["spec_json"])
        result_row = self._conn.execute(
            "SELECT result_json FROM results WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        decision_row = self._conn.execute(
            "SELECT decision FROM decisions WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return RegistryEntry(
            spec=spec,
            result=None if result_row is None else ExperimentResult.from_dict(
                json.loads(result_row["result_json"])
            ),
            decision=decision_row["decision"] if decision_row else "pending",
            created_at=row["created_at"],
        )

    def get_spec(self, experiment_id: str) -> ExperimentSpec:
        return self.get(experiment_id).spec

    def get_result(self, experiment_id: str) -> ExperimentResult | None:
        return self.get(experiment_id).result

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

    def query_by_status(self, status: str) -> list[RegistryEntry]:
        rows = self._conn.execute(
            "SELECT experiment_id FROM results WHERE status = ? ORDER BY created_at, experiment_id",
            (status,),
        ).fetchall()
        return [self.get(row["experiment_id"]) for row in rows]

    def successful(self, *, evaluation_split: str = RESEARCH_SPLIT) -> list[RegistryEntry]:
        rows = self._conn.execute(
            """
            SELECT experiment_id FROM results
            WHERE status = 'success' AND evaluation_split = ?
            ORDER BY created_at, experiment_id
            """,
            (evaluation_split,),
        ).fetchall()
        return [self.get(row["experiment_id"]) for row in rows]

    def rank_validation(self) -> list[RegistryEntry]:
        """Successful validation experiments, primary desc. Test is excluded."""
        rows = self._conn.execute(
            """
            SELECT r.experiment_id
            FROM results r
            JOIN experiments e ON e.experiment_id = r.experiment_id
            JOIN decisions d ON d.experiment_id = r.experiment_id
            WHERE r.status = 'success'
              AND r.evaluation_split = ?
              AND d.decision != 'rejected'
            ORDER BY r.primary_score DESC, e.created_at ASC, e.experiment_id ASC
            """,
            (RESEARCH_SPLIT,),
        ).fetchall()
        return [self.get(row["experiment_id"]) for row in rows]

    def find_by_spec_hash(self, spec_hash: str) -> list[RegistryEntry]:
        rows = self._conn.execute(
            "SELECT experiment_id FROM experiments WHERE spec_hash = ? ORDER BY created_at, experiment_id",
            (spec_hash,),
        ).fetchall()
        return [self.get(row["experiment_id"]) for row in rows]

    def parents(self, experiment_id: str) -> list[str]:
        self.get(experiment_id)
        rows = self._conn.execute(
            "SELECT parent_id FROM parents WHERE experiment_id = ? ORDER BY position",
            (experiment_id,),
        ).fetchall()
        return [row["parent_id"] for row in rows]

    def children(self, experiment_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT experiment_id FROM parents WHERE parent_id = ? ORDER BY experiment_id",
            (experiment_id,),
        ).fetchall()
        return [row["experiment_id"] for row in rows]

    def ancestry(self, experiment_id: str) -> list[str]:
        """Closest-first unique ancestors. Cycle-safe."""
        self.get(experiment_id)
        seen: set[str] = set()
        ordered: list[str] = []
        queue = list(self.parents(experiment_id))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            ordered.append(current)
            if self._exists(current):
                queue.extend(self.parents(current))
        return ordered

    def mark_decision(self, experiment_id: str, decision: str) -> str:
        if decision not in DECISIONS:
            raise RegistryError(f"decision must be one of {DECISIONS}, got {decision!r}")
        self.get(experiment_id)
        self._conn.execute(
            "UPDATE decisions SET decision = ?, updated_at = ? WHERE experiment_id = ?",
            (decision, _utc_now(), experiment_id),
        )
        self._conn.commit()
        return decision

    def get_decision(self, experiment_id: str) -> str:
        return self.get(experiment_id).decision

    def elite(self) -> RegistryEntry | None:
        """Highest successful validation primary. Test never participates."""
        ranked = self.rank_validation()
        return ranked[0] if ranked else None

    def rollback_target(self, experiment_id: str) -> str | None:
        """First parent, if any. Phase 4 can choose a smarter checkpoint."""
        parents = self.parents(experiment_id)
        return parents[0] if parents else None

    def iter_ids(self) -> Iterable[str]:
        rows = self._conn.execute(
            "SELECT experiment_id FROM experiments ORDER BY created_at, experiment_id"
        ).fetchall()
        return [row["experiment_id"] for row in rows]

    def _exists(self, experiment_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            is not None
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
