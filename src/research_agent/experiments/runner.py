"""Stable experiment runner.

Filesystem isolation + subprocess + timeout. Not a security sandbox.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
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
from .errors import ForbiddenTestSplit, SpecError
from .fingerprint import config_fingerprint, environment_metadata, source_fingerprint
from .registry import ExperimentRegistry
from .result import ExperimentResult, FailureInfo, Metrics
from .spec import ExperimentSpec
from .splits import assert_split_allowed

SCORES_NAME = "scores.npy"


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
        run_dir = self.runs_dir / spec.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir / "spec.json"
        spec.write_json(spec_path)
        config_path = run_dir / "config.json"
        config_path.write_text(canonical_json(spec.parameters) + "\n", encoding="utf-8")

        try:
            self.registry.insert_spec(spec)
        except Exception as exc:
            result = self._finish(
                spec,
                run_dir,
                status="invalid",
                wall=0.0,
                return_code=None,
                scores_path=None,
                metrics=None,
                source_fp="",
                config_fp=config_fingerprint(spec.parameters),
                failure=FailureInfo("registry", str(exc)),
                entrypoint=self._resolve_entrypoint(spec),
            )
            return result

        started = time.perf_counter()
        try:
            assert_split_allowed(spec.evaluation_split, allow)
            if not self.data_dir.is_dir():
                raise SpecError(f"data dir not found: {self.data_dir}")
            entrypoint = self._resolve_entrypoint(spec)
            if not entrypoint.is_file():
                raise SpecError(f"entrypoint not found: {entrypoint}")
            source_paths = [entrypoint, *self._resolve_extra_paths(spec)]
            source_fp = source_fingerprint(source_paths)
            config_fp = config_fingerprint(spec.parameters)
            metadata = environment_metadata(
                repo_root=self.repo_root,
                entrypoint=entrypoint,
                evaluate_py=EVALUATE_PY,
                source_fp=source_fp,
                config_fp=config_fp,
            )
            (run_dir / "metadata.json").write_text(
                canonical_json(metadata) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            status = "invalid"
            kind = "test_split" if isinstance(exc, ForbiddenTestSplit) else "spec"
            result = self._finish(
                spec,
                run_dir,
                status=status,
                wall=time.perf_counter() - started,
                return_code=None,
                scores_path=None,
                metrics=None,
                source_fp="",
                config_fp=config_fingerprint(spec.parameters),
                failure=FailureInfo(kind, str(exc)),
                entrypoint=self._resolve_entrypoint(spec),
            )
            return result

        scores_path = run_dir / SCORES_NAME
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        command = [
            self.python_executable,
            str(entrypoint),
            "--data-dir",
            str(self.data_dir),
            "--split",
            spec.evaluation_split,
            "--output-scores",
            str(scores_path),
            "--seed",
            str(spec.seed),
            "--config",
            str(config_path),
        ]
        env = os.environ.copy()
        pythonpath = [
            str(self.repo_root / "src"),
            str(STARTER),
        ]
        existing = env.get("PYTHONPATH")
        if existing:
            pythonpath.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=float(spec.timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _write_text(stdout_path, exc.stdout)
            _write_text(stderr_path, exc.stderr)
            result = self._finish(
                spec,
                run_dir,
                status="timeout",
                wall=time.perf_counter() - started,
                return_code=None,
                scores_path=None,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=FailureInfo(
                    "timeout",
                    f"candidate exceeded timeout_seconds={spec.timeout_seconds}",
                    {"command": command},
                ),
                entrypoint=entrypoint,
                environment=metadata,
            )
            return result

        _write_text(stdout_path, completed.stdout)
        _write_text(stderr_path, completed.stderr)
        wall = time.perf_counter() - started

        if completed.returncode != 0:
            result = self._finish(
                spec,
                run_dir,
                status="failed",
                wall=wall,
                return_code=completed.returncode,
                scores_path=None,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=FailureInfo(
                    "subprocess",
                    f"candidate exited with return code {completed.returncode}",
                    {"return_code": completed.returncode},
                ),
                entrypoint=entrypoint,
                environment=metadata,
            )
            return result

        loaded = self._load_scores(scores_path)
        if isinstance(loaded, FailureInfo):
            result = self._finish(
                spec,
                run_dir,
                status="invalid",
                wall=wall,
                return_code=completed.returncode,
                scores_path=str(scores_path) if scores_path.exists() else None,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=loaded,
                entrypoint=entrypoint,
                environment=metadata,
            )
            return result

        try:
            splits = official_load(self.data_dir)
            rows = splits[spec.evaluation_split]
            expected = len(rows)
            check = _validate_scores(loaded, expected)
            if check is not None:
                result = self._finish(
                    spec,
                    run_dir,
                    status="invalid",
                    wall=wall,
                    return_code=completed.returncode,
                    scores_path=str(scores_path),
                    metrics=None,
                    source_fp=source_fp,
                    config_fp=config_fp,
                    failure=check,
                    entrypoint=entrypoint,
                    environment=metadata,
                )
                return result
            user_ids, labels = split_labels(rows)
            official = official_evaluate(user_ids, labels, loaded)
            metrics = Metrics.from_official(official)
        except Exception as exc:
            result = self._finish(
                spec,
                run_dir,
                status="invalid",
                wall=wall,
                return_code=completed.returncode,
                scores_path=str(scores_path),
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=FailureInfo("evaluate", str(exc)),
                entrypoint=entrypoint,
                environment=metadata,
            )
            return result

        result = self._finish(
            spec,
            run_dir,
            status="success",
            wall=wall,
            return_code=completed.returncode,
            scores_path=str(scores_path),
            metrics=metrics,
            source_fp=source_fp,
            config_fp=config_fp,
            failure=None,
            entrypoint=entrypoint,
            environment=metadata,
        )
        return result

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

    def _load_scores(self, path: Path) -> np.ndarray | FailureInfo:
        if not path.is_file():
            return FailureInfo("missing_scores", f"score artifact not found: {path}")
        try:
            return np.load(path)
        except Exception as exc:
            return FailureInfo("invalid_scores", f"could not load scores: {exc}")

    def _finish(
        self,
        spec: ExperimentSpec,
        run_dir: Path,
        *,
        status: str,
        wall: float,
        return_code: int | None,
        scores_path: str | None,
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
            scores_path=scores_path,
            metrics=metrics,
            source_fingerprint=source_fp,
            config_fingerprint=config_fp,
            environment=env,
            failure=failure,
        )
        result.write_json(run_dir / "result.json")
        try:
            self.registry.upsert_result(result)
        except Exception:
            pass
        return result


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
