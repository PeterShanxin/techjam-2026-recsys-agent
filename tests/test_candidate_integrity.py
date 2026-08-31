"""Result integrity for untrusted candidate code, and a sanitized environment.

Scope note: generated code runs as ordinary Python with this process's
privileges. Nothing here confines what a candidate can read or write on the
host, and none of these tests claim otherwise. What they pin is narrower and
enforceable: a candidate that mutates a protected asset cannot get a score for
that run, and the agent's credentials are not handed to the subprocess.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from conftest import EVALUATE_PY
from experiment_helpers import make_spec, write_mini_dataset
from research_agent.experiments import ExperimentRunner, ImplementationRef
from research_agent.experiments.candidate_env import (
    CandidateEnvError,
    build_candidate_env,
    is_secret_like_name,
)
from research_agent.experiments.integrity import build_manifest, diff_manifests

PLANTED_SECRETS = {
    "GEMINI_API_KEY": "AIzaPLANTEDGEMINI0123456789abcdefghij",
    "OPENAI_API_KEY": "sk-proj-PLANTEDOPENAI0123456789",
    "ANTHROPIC_API_KEY": "sk-ant-PLANTEDANTHROPIC0123456789",
    "GITHUB_TOKEN": "ghp_PLANTEDGITHUB0123456789abcd",
    "AWS_SECRET_ACCESS_KEY": "PLANTEDAWSSECRET0123456789abcdefghij",
    "DATABASE_URL": "postgres://user:PLANTEDPASSWORD@db.internal:5432/prod",
}

CANDIDATE_TEMPLATE = '''
import argparse
import json
from pathlib import Path

import numpy as np
from data import load


def side_effect(cfg):
{body}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output-scores", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg["output_scores"] = args.output_scores
    side_effect(cfg)
    rows = load(args.data_dir)[args.split]
    np.save(args.output_scores, np.random.default_rng(args.seed).random(len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _indent(body: str) -> str:
    return "\n".join(
        "    " + line if line.strip() else line for line in body.strip("\n").split("\n")
    )


def build_candidate(path: Path, body: str) -> Path:
    path.write_text(CANDIDATE_TEMPLATE.format(body=_indent(body)), encoding="utf-8")
    return path


@pytest.fixture
def harness(tmp_path):
    """A runner whose protected assets are a disposable copy of the real ones."""
    data_dir = write_mini_dataset(tmp_path)
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "evaluate.py").write_bytes(EVALUATE_PY.read_bytes())
    (protected / "reference.json").write_text('{"primary": 0.6015}', encoding="utf-8")
    runner = ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
        protected_paths=[protected],
    )
    return runner, tmp_path, protected


def run_candidate(harness, body, *, experiment_id, params=None):
    runner, tmp_path, protected = harness
    candidate = build_candidate(tmp_path / f"{experiment_id}.py", body)
    payload = {"protected": str(protected)}
    payload.update(params or {})
    return runner.run(
        make_spec(
            experiment_id=experiment_id,
            implementation=ImplementationRef(entrypoint=str(candidate)),
            parameters=payload,
            timeout_seconds=120.0,
        )
    )


def attempt_dir_of(runner: ExperimentRunner, experiment_id: str) -> Path:
    attempts = sorted((runner.runs_dir / experiment_id / "attempts").iterdir())
    assert attempts, "no attempt directory recorded"
    return attempts[-1]


# --------------------------------------------------------------------------
# Sanitized candidate environment
# --------------------------------------------------------------------------


def test_env_is_built_from_an_allowlist_not_a_blocklist():
    parent = {
        "PATH": "/usr/bin",
        "SYSTEMROOT": r"C:\Windows",
        "UNRELATED_INTERNAL_HOSTNAME": "db.internal",
        **PLANTED_SECRETS,
    }
    env = build_candidate_env(pythonpath="/src", temp_dir="/tmp/x", parent_env=parent)
    assert env["PATH"] == "/usr/bin"
    # Not a secret, but not allowlisted either: absent by construction.
    assert "UNRELATED_INTERNAL_HOSTNAME" not in env
    for name, value in PLANTED_SECRETS.items():
        assert name not in env
        assert value not in env.values()


def test_env_omits_user_home_and_profile():
    parent = {"HOME": "/home/me", "USERPROFILE": r"C:\Users\me", "APPDATA": r"C:\Users\me\AppData"}
    env = build_candidate_env(pythonpath="/src", temp_dir="/tmp/x", parent_env=parent)
    assert not {"HOME", "USERPROFILE", "APPDATA"} & set(env)


def test_env_pins_reproducibility_controls():
    env = build_candidate_env(pythonpath="/src", temp_dir="/tmp/x", parent_env={})
    assert env["PYTHONHASHSEED"] == "0"
    assert env["TMPDIR"] == env["TEMP"] == env["TMP"] == "/tmp/x"


def test_env_forwards_thread_pins_that_change_float_reduction_order():
    parent = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "2"}
    env = build_candidate_env(pythonpath="/src", temp_dir="/tmp/x", parent_env=parent)
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "2"


@pytest.mark.parametrize(
    "name", ["MY_API_KEY", "svc_token", "DB_PASSWORD", "SESSION_COOKIE", "X_PRIVATE_KEY"]
)
def test_extra_env_rejects_credential_like_names(name):
    assert is_secret_like_name(name)
    with pytest.raises(CandidateEnvError):
        build_candidate_env(
            pythonpath="/src", temp_dir="/tmp/x", parent_env={}, extra={name: "v"}
        )


def test_extra_env_cannot_override_runner_controlled_names():
    with pytest.raises(CandidateEnvError):
        build_candidate_env(
            pythonpath="/src", temp_dir="/tmp/x", parent_env={}, extra={"PYTHONPATH": "/evil"}
        )


def test_candidate_subprocess_does_not_receive_parent_credentials(harness, monkeypatch):
    runner, _, _ = harness
    for name, value in PLANTED_SECRETS.items():
        monkeypatch.setenv(name, value)
    body = """
    import os as _o
    dump = Path(cfg["output_scores"]).with_name("env_dump.json")
    dump.write_text(json.dumps(dict(_o.environ)), encoding="utf-8")
    """
    result = run_candidate(harness, body, experiment_id="env-check")
    assert result.status == "success"

    dumps = list(attempt_dir_of(runner, "env-check").rglob("env_dump.json"))
    assert dumps, "candidate did not write its environment dump"
    seen = json.loads(dumps[0].read_text(encoding="utf-8"))
    blob = json.dumps(seen)
    for name, value in PLANTED_SECRETS.items():
        assert name not in seen, f"{name} reached the candidate"
        assert value not in blob


# --------------------------------------------------------------------------
# Protected-asset integrity
# --------------------------------------------------------------------------


def test_mutating_a_protected_asset_invalidates_the_run(harness):
    """The enforced property: tampering cannot produce a score.

    The candidate really does overwrite the file -- nothing stops it -- and the
    run is discarded because the hashes moved.
    """
    runner, _, protected = harness
    body = """
    target = Path(cfg["protected"]) / ("eval" + "uate.py")
    target.write_text("def evaluate(*a, **k):\\n    return {'primary': 1.0}\\n", encoding="utf-8")
    """
    before = build_manifest([protected])
    result = run_candidate(harness, body, experiment_id="tamper")
    after = build_manifest([protected])

    assert diff_manifests(before, after), "test is void: the candidate did not mutate anything"
    assert result.status == "invalid"
    assert result.failure is not None and result.failure.kind == "integrity"
    assert result.metrics is None
    assert result.scores_path is None
    changed = result.failure.details["changes"]
    assert any(path.endswith("evaluate.py") for path in changed)


def test_integrity_violation_is_latched_across_runs(harness):
    """A dirty tree must not become the next run's clean baseline.

    Re-snapshotting per attempt would let a mutation that survived one run be
    accepted as normal by the next, so a poisoned evaluator would score freely.
    """
    runner, tmp_path, protected = harness
    tamper = """
    target = Path(cfg["protected"]) / ("eval" + "uate.py")
    target.write_text("POISONED\\n", encoding="utf-8")
    """
    first = run_candidate(harness, tamper, experiment_id="latch-first")
    second = run_candidate(harness, "pass", experiment_id="latch-second")

    assert first.status == "invalid" and first.failure.kind == "integrity"
    assert (protected / "evaluate.py").read_text(encoding="utf-8").strip() == "POISONED"
    assert second.status == "invalid", "poisoned tree was re-baselined and scored"
    assert second.failure.kind == "integrity"
    assert second.metrics is None and second.scores_path is None


def test_clean_candidate_still_scores_normally(harness):
    """Positive control: legitimate experiments are unaffected."""
    runner, _, protected = harness
    body = """
    scratch = Path(cfg["output_scores"]).with_name("scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "checkpoint.npy").write_bytes(b"ok")
    """
    before = build_manifest([protected])
    result = run_candidate(harness, body, experiment_id="clean")
    assert result.status == "success"
    assert result.metrics is not None
    assert result.scores_path is not None
    assert diff_manifests(before, build_manifest([protected])) == {}


def test_candidate_artifacts_are_separated_from_parent_provenance(harness):
    """Scores land under out/; metadata.json and result.json stay parent-owned."""
    runner, _, _ = harness
    result = run_candidate(harness, "pass", experiment_id="layout")
    assert result.status == "success"

    attempt = attempt_dir_of(runner, "layout")
    assert (attempt / "out" / "scores.npy").is_file()
    assert (attempt / "metadata.json").is_file()
    assert (attempt / "result.json").is_file()
    assert not (attempt / "out" / "metadata.json").exists()

    metadata = json.loads((attempt / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["protected_manifest_sha256"]) == 64
    assert metadata["protected_asset_count"] >= 2


def test_default_protected_set_covers_the_assets_that_decide_scores(tmp_path):
    data_dir = write_mini_dataset(tmp_path)
    runner = ExperimentRunner(runs_dir=tmp_path / "runs", data_dir=data_dir)
    tracked = set(runner.protected_manifest().digests)

    assert EVALUATE_PY.resolve().as_posix() in tracked
    for name in ("data.py", "baseline.py"):
        assert any(path.endswith(f"/kuairand/{name}") for path in tracked)
    assert any(path.endswith("/research_agent/evaluation/official.py") for path in tracked)
    assert any(path.endswith("/research_agent/experiments/runner.py") for path in tracked)
    # Build products must not create phantom integrity failures.
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in tracked)


def test_manifest_detects_modified_removed_and_added_files(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "a.py").write_text("a\n", encoding="utf-8")
    (root / "b.py").write_text("b\n", encoding="utf-8")
    before = build_manifest([root])

    (root / "a.py").write_text("mutated\n", encoding="utf-8")
    (root / "b.py").unlink()
    (root / "c.py").write_text("new\n", encoding="utf-8")
    changes = diff_manifests(before, build_manifest([root]))

    assert changes[(root / "a.py").resolve().as_posix()] == "modified"
    assert changes[(root / "b.py").resolve().as_posix()] == "removed"
    assert changes[(root / "c.py").resolve().as_posix()] == "added"


def test_dataset_is_covered_once_and_hashed_in_full(tmp_path):
    """The dataset lives under starter/; it must be covered exactly once.

    Hashed in full rather than keyed on size and mtime: both are values a
    candidate can set, and this check is the enforced property.
    """
    starter_like = tmp_path / "starter"
    (starter_like / "kuairand").mkdir(parents=True)
    (starter_like / "kuairand" / "evaluate.py").write_text("x = 1\n", encoding="utf-8")
    data_dir = starter_like / "kuairand" / "KuaiRand-Pure" / "data"
    data_dir.mkdir(parents=True)
    log = data_dir / "log.csv"
    log.write_text("a,b\n1,2\n", encoding="utf-8")

    runner = ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
        protected_paths=[starter_like],
    )
    digests = runner.protected_manifest().digests
    assert any(path.endswith("/kuairand/evaluate.py") for path in digests)
    matches = [path for path in digests if path.endswith("/KuaiRand-Pure/data/log.csv")]
    assert len(matches) == 1, f"dataset covered {len(matches)} times: {matches}"

    # A same-size edit with the original timestamps restored is still caught,
    # which a metadata-keyed cache would have missed.
    stat = log.stat()
    before = runner.protected_manifest()
    log.write_text("a,b\n9,9\n", encoding="utf-8")
    os.utime(log, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert log.stat().st_size == stat.st_size
    assert diff_manifests(before, runner.protected_manifest())


# --------------------------------------------------------------------------
# Evaluator binding
# --------------------------------------------------------------------------


def test_official_evaluator_is_bound_before_any_candidate_runs(tmp_path, monkeypatch):
    """Blocks planted bytecode: __pycache__ is skipped by the integrity walk.

    Once `evaluate` is resolved in sys.modules the parent cannot be steered to
    a different implementation, whether by a replaced source file or a
    hash-based .pyc that loads without checking its source.
    """
    monkeypatch.delitem(sys.modules, "evaluate", raising=False)
    monkeypatch.delitem(sys.modules, "data", raising=False)
    assert "evaluate" not in sys.modules

    ExperimentRunner(runs_dir=tmp_path / "runs", data_dir=write_mini_dataset(tmp_path))
    assert "evaluate" in sys.modules, "evaluator was not bound at runner construction"
    assert "data" in sys.modules


def test_run_fails_closed_when_the_evaluator_cannot_be_bound(tmp_path, monkeypatch):
    """Binding must not fail open.

    A swallowed ImportError would leave `official_evaluate` to resolve
    `evaluate` lazily after the candidate exits -- the window binding early
    was added to close.
    """
    import research_agent.experiments.runner as runner_mod

    data_dir = write_mini_dataset(tmp_path)
    candidate = build_candidate(tmp_path / "quiet.py", "pass")
    runner = ExperimentRunner(
        repo_root=tmp_path, runs_dir=tmp_path / "runs", data_dir=data_dir
    )

    def refuse(*args, **kwargs):
        if kwargs.get("strict"):
            raise runner_mod.SpecError("simulated: starter modules unavailable")

    monkeypatch.setattr(runner_mod, "_bind_official_modules", refuse)
    result = runner.run(
        make_spec(
            experiment_id="bind-fail",
            implementation=ImplementationRef(entrypoint=str(candidate)),
            parameters={"n": 1},
        )
    )
    assert result.status == "invalid"
    assert result.metrics is None
    assert "could not be bound" in result.failure.message or "simulated" in result.failure.message


def test_advisory_lint_is_not_relied_on_as_containment():
    """The AST check is documented as lint; a split string walks past it.

    Pinned so nobody re-promotes it to a security control: the enforced check
    is the integrity manifest, exercised by the tampering test above.
    """
    from research_agent.agent.safety import SafetyError, assert_no_evaluator_tampering

    obvious = 'open("starter/kuairand/evaluate.py", "w").write("x")\n'
    with pytest.raises(SafetyError):
        assert_no_evaluator_tampering(obvious)

    evasive = 'open("starter/kuairand/" + "eval" + "uate.py", "w").write("x")\n'
    assert_no_evaluator_tampering(evasive)  # advisory only: does not raise
