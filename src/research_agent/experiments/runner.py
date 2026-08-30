"""Stable experiment runner.

Filesystem isolation + subprocess + timeout. Not a security sandbox.

Identity (experiment_id) is immutable once registered. Each execution
attempt evaluates only score artifacts created by that attempt.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from research_agent.evaluation.official import (
    EVALUATE_PY,
    REPO_ROOT,
    STARTER,
    official_evaluate,
    official_load,
    split_labels,
)

from .canonical import canonical_json
from .errors import ExperimentIdCollision, ForbiddenTestSplit, SpecError
from .fingerprint import config_fingerprint, environment_metadata, source_fingerprint
from .registry import ExperimentRegistry, RegistryEntry
from .result import ExperimentResult, FailureInfo, Metrics
from .spec import ExperimentSpec
from .splits import assert_split_allowed

SCORES_NAME = "scores.npy"
PUBLISHED_EXECUTION = ("scores.npy", "stdout.log", "stderr.log", "metadata.json")


class ExperimentRunner:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        runs_dir: Path | None = None,
        registry: ExperimentRegistry | None = None,
        data_dir: Path | None = None,
        allow_test: bool = False,
        python_executable: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.runs_dir = Path(runs_dir) if runs_dir else self.repo_root / "runs"
        self.allow_test = allow_test
        self.python_executable = python_executable or sys.executable
        self.data_dir = Path(data_dir) if data_dir else _default_data_dir()
        registry_path = self.runs_dir / "registry.sqlite"
        self.registry = registry if registry is not None else ExperimentRegistry(registry_path)

    def run(self, spec: ExperimentSpec, *, allow_test: bool | None = None) -> ExperimentResult:
        allow = self.allow_test if allow_test is None else allow_test
        allow = allow or spec.test_opt_in()

        existing = self.registry.peek(spec.experiment_id)
        if existing is not None and existing.spec.spec_hash != spec.spec_hash:
            return self._collision_result(spec, existing)

        try:
            self.registry.insert_spec(spec)
        except ExperimentIdCollision:
            current = self.registry.peek(spec.experiment_id)
            if current is None:
                raise
            return self._collision_result(spec, current)

        registered = self.registry.get_spec(spec.experiment_id)
        run_dir = self.runs_dir / registered.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        registered.write_json(run_dir / "spec.json")
        config_path = run_dir / "config.json"
        config_path.write_text(canonical_json(registered.parameters) + "\n", encoding="utf-8")

        attempt_id = uuid.uuid4().hex
        attempt_dir = run_dir / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True)
        _clear_published_execution(run_dir)

        started = time.perf_counter()
        try:
            assert_split_allowed(registered.evaluation_split, allow)
            if not self.data_dir.is_dir():
                raise SpecError(f"data dir not found: {self.data_dir}")
            entrypoint = self._resolve_entrypoint(registered)
            if not entrypoint.is_file():
                raise SpecError(f"entrypoint not found: {entrypoint}")
            source_paths = [entrypoint, *self._resolve_extra_paths(registered)]
            source_fp = source_fingerprint(source_paths)
            config_fp = config_fingerprint(registered.parameters)
            metadata = environment_metadata(
                repo_root=self.repo_root,
                entrypoint=entrypoint,
                evaluate_py=EVALUATE_PY,
                source_fp=source_fp,
                config_fp=config_fp,
            )
            metadata["attempt_id"] = attempt_id
            (attempt_dir / "metadata.json").write_text(
                canonical_json(metadata) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            kind = "test_split" if isinstance(exc, ForbiddenTestSplit) else "spec"
            return self._finish(
                registered,
                run_dir,
                attempt_dir,
                status="invalid",
                wall=time.perf_counter() - started,
                return_code=None,
                attempt_scores=None,
                metrics=None,
                source_fp="",
                config_fp=config_fingerprint(registered.parameters),
                failure=FailureInfo(kind, str(exc)),
                entrypoint=self._resolve_entrypoint(registered),
            )

        scores_path = attempt_dir / SCORES_NAME
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        command = [
            self.python_executable,
            str(entrypoint),
            "--data-dir",
            str(self.data_dir),
            "--split",
            registered.evaluation_split,
            "--output-scores",
            str(scores_path),
            "--seed",
            str(registered.seed),
            "--config",
            str(config_path),
        ]
        env = os.environ.copy()
        pythonpath = [
            str(self.repo_root / "src"),
            str(STARTER),
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=float(registered.timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _write_text(stdout_path, exc.stdout)
            _write_text(stderr_path, exc.stderr)
            return self._finish(
                registered,
                run_dir,
                attempt_dir,
                status="timeout",
                wall=time.perf_counter() - started,
                return_code=None,
                attempt_scores=None,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=FailureInfo(
                    "timeout",
                    f"candidate exceeded timeout_seconds={registered.timeout_seconds}",
                    {"command": command, "attempt_id": attempt_id},
                ),
                entrypoint=entrypoint,
                environment=metadata,
            )

        _write_text(stdout_path, completed.stdout)
        _write_text(stderr_path, completed.stderr)
        wall = time.perf_counter() - started

        if completed.returncode != 0:
            return self._finish(
                registered,
                run_dir,
                attempt_dir,
                status="failed",
                wall=wall,
                return_code=completed.returncode,
                attempt_scores=None,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=FailureInfo(
                    "subprocess",
                    f"candidate exited with return code {completed.returncode}",
                    {"return_code": completed.returncode, "attempt_id": attempt_id},
                ),
                entrypoint=entrypoint,
                environment=metadata,
            )

        loaded = self._load_attempt_scores(scores_path, attempt_dir)
        if isinstance(loaded, FailureInfo):
            return self._finish(
                registered,
                run_dir,
                attempt_dir,
                status="invalid",
                wall=wall,
                return_code=completed.returncode,
                attempt_scores=None,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=loaded,
                entrypoint=entrypoint,
                environment=metadata,
            )

        try:
            splits = official_load(self.data_dir)
            rows = splits[registered.evaluation_split]
            expected = len(rows)
            check = _validate_scores(loaded, expected)
            if check is not None:
                return self._finish(
                    registered,
                    run_dir,
                    attempt_dir,
                    status="invalid",
                    wall=wall,
                    return_code=completed.returncode,
                    attempt_scores=scores_path,
                    metrics=None,
                    source_fp=source_fp,
                    config_fp=config_fp,
                    failure=check,
                    entrypoint=entrypoint,
                    environment=metadata,
                )
            user_ids, labels = split_labels(rows)
            official = official_evaluate(user_ids, labels, loaded)
            metrics = Metrics.from_official(official)
        except Exception as exc:
            return self._finish(
                registered,
                run_dir,
                attempt_dir,
                status="invalid",
                wall=wall,
                return_code=completed.returncode,
                attempt_scores=scores_path,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=FailureInfo("evaluate", str(exc), {"attempt_id": attempt_id}),
                entrypoint=entrypoint,
                environment=metadata,
            )

        return self._finish(
            registered,
            run_dir,
            attempt_dir,
            status="success",
            wall=wall,
            return_code=completed.returncode,
            attempt_scores=scores_path,
            metrics=metrics,
            source_fp=source_fp,
            config_fp=config_fp,
            failure=None,
            entrypoint=entrypoint,
            environment=metadata,
        )

    def _resolve_entrypoint(self, spec: ExperimentSpec) -> Path:
        raw = Path(spec.implementation.entrypoint)
        if raw.is_absolute():
            return raw
        root = spec.implementation.source_root
        base = Path(root) if root else self.repo_root
        if not base.is_absolute():
            base = self.repo_root / base
        return (base / raw).resolve()

    def _resolve_extra_paths(self, spec: ExperimentSpec) -> list[Path]:
        paths = []
        for item in spec.implementation.extra_paths:
            raw = Path(item)
            paths.append(raw if raw.is_absolute() else (self.repo_root / raw).resolve())
        return paths

    def _load_attempt_scores(self, path: Path, attempt_dir: Path) -> np.ndarray | FailureInfo:
        """Load scores only if this attempt created them. Never read published leftovers."""
        if not path.is_file():
            return FailureInfo("missing_scores", f"score artifact not found: {path}")
        try:
            resolved = path.resolve()
            if resolved.parent != attempt_dir.resolve():
                return FailureInfo(
                    "stale_scores",
                    "refusing to evaluate scores outside this attempt directory",
                    {"path": str(resolved), "attempt_dir": str(attempt_dir)},
                )
        except OSError as exc:
            return FailureInfo("invalid_scores", f"could not resolve score path: {exc}")
        try:
            return np.load(resolved)
        except Exception as exc:
            return FailureInfo("invalid_scores", f"could not load scores: {exc}")

    def _collision_result(
        self, spec: ExperimentSpec, existing: RegistryEntry
    ) -> ExperimentResult:
        run_dir = self.runs_dir / spec.experiment_id
        prior = existing.result
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            status="invalid",
            evaluation_split=spec.evaluation_split,
            seed=spec.seed,
            spec_hash=spec.spec_hash,
            wall_seconds=0.0,
            return_code=None,
            run_dir=str(run_dir),
            stdout_path=prior.stdout_path if prior else "",
            stderr_path=prior.stderr_path if prior else "",
            scores_path=None,
            metrics=None,
            source_fingerprint="",
            config_fingerprint=config_fingerprint(spec.parameters),
            environment={"existing_spec_hash": existing.spec.spec_hash},
            failure=FailureInfo(
                "id_collision",
                (
                    f"experiment_id {spec.experiment_id!r} is already registered "
                    f"with spec_hash {existing.spec.spec_hash}; "
                    f"refusing {spec.spec_hash}"
                ),
                {
                    "existing_spec_hash": existing.spec.spec_hash,
                    "incoming_spec_hash": spec.spec_hash,
                },
            ),
        )

    def _finish(
        self,
        spec: ExperimentSpec,
        run_dir: Path,
        attempt_dir: Path,
        *,
        status: str,
        wall: float,
        return_code: int | None,
        attempt_scores: Path | None,
        metrics: Metrics | None,
        source_fp: str,
        config_fp: str,
        failure: FailureInfo | None,
        entrypoint: Path,
        environment: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        env = environment or environment_metadata(
            repo_root=self.repo_root,
            entrypoint=entrypoint,
            evaluate_py=EVALUATE_PY,
            source_fp=source_fp,
            config_fp=config_fp,
        )
        env = dict(env)
        env["attempt_id"] = attempt_dir.name
        published_scores = None
        if (
            status == "success"
            and attempt_scores is not None
            and attempt_scores.is_file()
        ):
            published_scores = str(run_dir / SCORES_NAME)
        result = ExperimentResult(
            experiment_id=spec.experiment_id,
            status=status,
            evaluation_split=spec.evaluation_split,
            seed=spec.seed,
            spec_hash=spec.spec_hash,
            wall_seconds=wall,
            return_code=return_code,
            run_dir=str(run_dir),
            stdout_path=str(run_dir / "stdout.log"),
            stderr_path=str(run_dir / "stderr.log"),
            scores_path=published_scores,
            metrics=metrics,
            source_fingerprint=source_fp,
            config_fingerprint=config_fp,
            environment=env,
            failure=failure,
        )
        result.write_json(attempt_dir / "result.json")
        result.write_json(run_dir / "result.json")
        _publish_attempt(run_dir, attempt_dir, attempt_scores if status == "success" else None)
        self.registry.upsert_result(result)
        return result


def _clear_published_execution(run_dir: Path) -> None:
    for name in PUBLISHED_EXECUTION:
        path = run_dir / name
        if path.is_file():
            path.unlink()


def _publish_attempt(
    run_dir: Path,
    attempt_dir: Path,
    attempt_scores: Path | None,
) -> None:
    for name in ("stdout.log", "stderr.log", "metadata.json"):
        src = attempt_dir / name
        if src.is_file():
            shutil.copy2(src, run_dir / name)
    if attempt_scores is None:
        return
    if not attempt_scores.is_file():
        return
    if attempt_scores.resolve().parent != attempt_dir.resolve():
        return
    shutil.copy2(attempt_scores, run_dir / SCORES_NAME)


def _validate_scores(scores: np.ndarray, expected: int) -> FailureInfo | None:
    array = np.asarray(scores)
    if array.ndim != 1:
        return FailureInfo(
            "wrong_shape",
            f"scores must be 1-d, got shape {array.shape}",
            {"shape": list(array.shape)},
        )
    if array.size != expected:
        return FailureInfo(
            "wrong_length",
            f"scores length {array.size} != expected {expected}",
            {"length": int(array.size), "expected": expected},
        )
    if not np.isfinite(array.astype(np.float64)).all():
        return FailureInfo("non_finite", "scores must be finite (NaN/Inf rejected)")
    return None


def _write_text(path: Path, payload: str | bytes | None) -> None:
    if payload is None:
        text = ""
    elif isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="replace")
    else:
        text = payload
    path.write_text(text, encoding="utf-8")


def _default_data_dir() -> Path:
    env = os.environ.get("KUAI_RAND_DATA_DIR")
    if env:
        return Path(env)
    return STARTER / "KuaiRand-Pure" / "data"
