"""Stable experiment runner.

Subprocess + timeout + result integrity. **Not a security sandbox**: generated
candidate code runs as ordinary Python with the privileges of this process,
and nothing here confines what it can read or write on the host.

What this module does guarantee is that a candidate cannot quietly *change the
answer*. The evaluator, starter, source and dataset assets are hashed in this
parent process before and after every attempt against a baseline taken once
per session, and any drift invalidates the attempt before its scores are
read. The evaluation rows are read once per session from a just-verified
tree, so scoring never re-reads labels a surviving process could have
rewritten. The official evaluator is bound before the candidate runs, so a
later source or bytecode change cannot steer scoring. Candidate subprocesses get a
minimal allowlisted environment rather than a copy of this one, so the agent's
API credentials are not handed to generated code.

Real isolation of untrusted code requires an OS-level boundary (separate
low-privilege user, container, or seccomp) and is tracked as follow-up.

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
    ensure_starter_on_path,
    official_evaluate,
    official_load,
    split_labels,
)

from .candidate_env import build_candidate_env
from .canonical import canonical_json
from .errors import ExperimentIdCollision, ForbiddenTestSplit, SpecError
from .fingerprint import config_fingerprint, environment_metadata, source_fingerprint
from .integrity import ProtectedManifest, build_manifest, diff_manifests, merge_manifests
from .registry import ExperimentRegistry, RegistryEntry
from .result import ExperimentResult, FailureInfo, Metrics
from .spec import ExperimentSpec
from .splits import assert_split_allowed

SCORES_NAME = "scores.npy"
PUBLISHED_EXECUTION = ("scores.npy", "stdout.log", "stderr.log", "metadata.json")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _IntegrityDrift(Exception):
    """Protected assets already differ from the session baseline."""

    def __init__(self, failure: FailureInfo) -> None:
        super().__init__(failure.message)
        self.failure = failure


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
        protected_paths: list[Path] | None = None,
    ) -> None:
        # Resolved up front: the candidate subprocess runs with its own cwd,
        # so a relative path here would resolve against the wrong directory.
        self.repo_root = (Path(repo_root) if repo_root else REPO_ROOT).resolve()
        self.runs_dir = (Path(runs_dir) if runs_dir else self.repo_root / "runs").resolve()
        self.allow_test = allow_test
        self.python_executable = python_executable or sys.executable
        self.data_dir = (Path(data_dir) if data_dir else _default_data_dir()).resolve()
        registry_path = self.runs_dir / "registry.sqlite"
        self.registry = registry if registry is not None else ExperimentRegistry(registry_path)
        # Score-critical assets: always hashed in full, every attempt.
        self.protected_paths = (
            [Path(p) for p in protected_paths]
            if protected_paths is not None
            else [STARTER, PACKAGE_ROOT, self.repo_root / "starter"]
        )
        self._baseline: ProtectedManifest | None = None
        self._compromised: FailureInfo | None = None
        # Evaluation rows, read once per session. See _load_splits_once.
        self._splits: dict | None = None
        _bind_official_modules()

    def protected_manifest(self) -> ProtectedManifest:
        """SHA-256 of every asset a candidate must not change.

        Includes the dataset: mutated labels would change the metric just as
        surely as a mutated evaluator. Hashed in full each attempt rather than
        keyed on size and mtime, both of which a candidate can set.

        Starter bytecode is hashed too. This process already bound `evaluate`
        and `data`, so a planted .pyc cannot steer *this* run -- but one whose
        header matches the real source would be loaded by the next parent
        process, and nothing else would notice.
        """
        return merge_manifests(
            build_manifest([*self.protected_paths, self.data_dir]),
            build_manifest([STARTER, self.repo_root / "starter"], include_bytecode=True),
        )

    def _baseline_manifest(self) -> ProtectedManifest:
        """The one snapshot, taken before the first attempt of this session.

        Deliberately never refreshed. Re-snapshotting before each attempt would
        let a mutation that survived one run become the accepted baseline for
        the next, so a tampered evaluator would score every later experiment
        without another integrity failure.
        """
        if self._baseline is None:
            self._baseline = self.protected_manifest()
        return self._baseline

    def _load_splits_once(self) -> dict:
        """Read the evaluation rows once per session, then serve from memory.

        Hashing the dataset and reading it are two operations, so on their own
        they are not an atomic snapshot: a helper outliving an earlier attempt
        could rewrite the CSVs in between. Loading exactly once removes the
        window instead of narrowing it. The single read happens during the
        first attempt, before any candidate in this session has run, so there
        is no candidate process in existence to race it; every later attempt
        scores from this copy and never touches the disk.

        Safe because the dataset is a protected asset: if it changes mid
        session the integrity check invalidates the run rather than expecting
        the cache to notice.
        """
        if self._splits is None:
            self._splits = official_load(self.data_dir)
        return self._splits

    def _integrity_failure(
        self, before: ProtectedManifest, attempt_id: str
    ) -> FailureInfo | None:
        """Compare protected assets against the session baseline."""
        if self._compromised is not None:
            return self._compromised
        changes = diff_manifests(before, self.protected_manifest())
        if not changes:
            return None
        failure = FailureInfo(
            "integrity",
            (
                "protected evaluator/starter/reference assets changed during this "
                f"attempt; refusing to score it. changed={sorted(changes)}"
            ),
            {"changes": changes, "attempt_id": attempt_id},
        )
        # Latched: the tree stays dirty until a human restores it, so every
        # later attempt in this session fails too instead of re-baselining.
        self._compromised = failure
        return failure

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
        pre_run_manifest = self._baseline_manifest()
        try:
            assert_split_allowed(registered.evaluation_split, allow)
            if not self.data_dir.is_dir():
                raise SpecError(f"data dir not found: {self.data_dir}")
            # Verify *before* reading anything. subprocess.run waits only on
            # the direct child, so a helper outliving an earlier attempt could
            # rewrite the dataset between attempts; without this check the
            # next attempt would load those labels and only hash afterwards,
            # by which time the bytes could be restored.
            drift = self._integrity_failure(pre_run_manifest, attempt_id)
            if drift is not None:
                raise _IntegrityDrift(drift)
            # Must succeed before the subprocess starts: after that, a lazy
            # resolve could pick up a source or bytecode file left behind.
            _bind_official_modules(strict=True)
            splits = self._load_splits_once()
            if registered.evaluation_split not in splits:
                raise SpecError(f"unknown evaluation split: {registered.evaluation_split}")
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
            metadata["protected_manifest_sha256"] = pre_run_manifest.digest
            metadata["protected_asset_count"] = len(pre_run_manifest)
            (attempt_dir / "metadata.json").write_text(
                canonical_json(metadata) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            if isinstance(exc, _IntegrityDrift):
                failure = exc.failure
            else:
                kind = "test_split" if isinstance(exc, ForbiddenTestSplit) else "spec"
                failure = FailureInfo(kind, str(exc))
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
                failure=failure,
                entrypoint=self._resolve_entrypoint(registered),
            )

        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        # Candidate artifacts live under out/, its working directory and temp
        # files under work/ and tmp/. metadata.json and result.json stay
        # directly in the attempt directory, written only by this process, so
        # provenance is not interleaved with candidate output.
        out_dir = attempt_dir / "out"
        work_dir = attempt_dir / "work"
        temp_dir = attempt_dir / "tmp"
        for path in (out_dir, work_dir, temp_dir):
            path.mkdir(parents=True, exist_ok=True)
        scores_path = out_dir / SCORES_NAME
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
        pythonpath = os.pathsep.join([str(self.repo_root / "src"), str(STARTER)])
        env = build_candidate_env(pythonpath=pythonpath, temp_dir=str(temp_dir))

        try:
            completed = subprocess.run(
                command,
                cwd=str(work_dir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(registered.timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _write_text(stdout_path, exc.stdout)
            _write_text(stderr_path, exc.stderr)
            timeout_failure = self._integrity_failure(pre_run_manifest, attempt_id) or FailureInfo(
                "timeout",
                f"candidate exceeded timeout_seconds={registered.timeout_seconds}",
                {"command": command, "attempt_id": attempt_id},
            )
            return self._finish(
                registered,
                run_dir,
                attempt_dir,
                status="invalid" if timeout_failure.kind == "integrity" else "timeout",
                wall=time.perf_counter() - started,
                return_code=None,
                attempt_scores=None,
                metrics=None,
                source_fp=source_fp,
                config_fp=config_fp,
                failure=timeout_failure,
                entrypoint=entrypoint,
                environment=metadata,
            )

        _write_text(stdout_path, completed.stdout)
        _write_text(stderr_path, completed.stderr)
        wall = time.perf_counter() - started

        # Checked before the scores are read, so a run that changed a
        # protected asset can never reach the evaluator or publish a metric.
        integrity_failure = self._integrity_failure(pre_run_manifest, attempt_id)
        if integrity_failure is not None:
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
                failure=integrity_failure,
                entrypoint=entrypoint,
                environment=metadata,
            )

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

        loaded = self._load_attempt_scores(scores_path, out_dir)
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

    def _load_attempt_scores(self, path: Path, out_dir: Path) -> np.ndarray | FailureInfo:
        """Load scores only if this attempt created them. Never read published leftovers."""
        if not path.is_file():
            return FailureInfo("missing_scores", f"score artifact not found: {path}")
        try:
            resolved = path.resolve()
            if resolved.parent != out_dir.resolve():
                return FailureInfo(
                    "stale_scores",
                    "refusing to evaluate scores outside this attempt output directory",
                    {"path": str(resolved), "out_dir": str(out_dir)},
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


def _bind_official_modules(*, strict: bool = False) -> None:
    """Import the official evaluator and loader before any candidate runs.

    Once bound in sys.modules they cannot be re-resolved, so neither a replaced
    source file nor a planted ``__pycache__/*.pyc`` (which the integrity walk
    skips, because this process legitimately creates bytecode there) can change
    how this process scores anything.

    ``strict`` refuses to continue when binding did not happen. Swallowing the
    failure would leave ``official_evaluate`` to resolve ``evaluate`` lazily
    after the candidate exits, which is the window this closes.
    """
    try:
        ensure_starter_on_path()
        import data  # noqa: F401
        import evaluate  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise SpecError(f"official evaluator/loader could not be bound: {exc}") from exc
        return
    if strict and ("evaluate" not in sys.modules or "data" not in sys.modules):
        raise SpecError("official evaluator/loader did not bind before the candidate ran")


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
    if attempt_scores.resolve().parent != (attempt_dir / "out").resolve():
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
