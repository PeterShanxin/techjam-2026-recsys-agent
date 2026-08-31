"""Adversarial tests for the untrusted-candidate sandbox (issues #15 and #16).

These exercise the runtime boundary, not the AST lint in ``agent/safety.py``.
Every attack here is written so that source-level pattern matching cannot see
it: names are split and rejoined, calls go through ``getattr``/``exec``, and
some run in a spawned interpreter.
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
from research_agent.experiments.candidate_guard import VIOLATION_MARKER
from research_agent.experiments.integrity import build_manifest, diff_manifests

PLANTED_SECRETS = {
    "GEMINI_API_KEY": "AIzaPLANTEDGEMINI0123456789abcdefghij",
    "OPENAI_API_KEY": "sk-proj-PLANTEDOPENAI0123456789",
    "ANTHROPIC_API_KEY": "sk-ant-PLANTEDANTHROPIC0123456789",
    "GITHUB_TOKEN": "ghp_PLANTEDGITHUB0123456789abcd",
    "AWS_SECRET_ACCESS_KEY": "PLANTEDAWSSECRET0123456789abcdefghij",
    "AZURE_CLIENT_SECRET": "PLANTEDAZURE0123456789",
    "DATABASE_URL": "postgres://user:PLANTEDPASSWORD@db.internal:5432/prod",
    "HF_TOKEN": "hf_PLANTEDHUGGINGFACE0123456789",
}

# A candidate that behaves normally, plus an injected attack body. The attack
# runs before scores are written so a successful attack would still produce a
# scoreable artifact -- which is exactly what must not happen.
CANDIDATE_TEMPLATE = '''
import argparse
import json
from pathlib import Path

import numpy as np
from data import load


def attack(cfg):
{attack}


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
    try:
        attack(cfg)
    except Exception as exc:
        Path(args.output_scores).with_name("attack_error.txt").write_text(
            f"{{type(exc).__name__}}: {{exc}}", encoding="utf-8"
        )
    rows = load(args.data_dir)[args.split]
    np.save(args.output_scores, np.random.default_rng(args.seed).random(len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _indent(body: str) -> str:
    return "\n".join("    " + line if line.strip() else line for line in body.strip("\n").split("\n"))


def build_candidate(path: Path, attack: str) -> Path:
    path.write_text(CANDIDATE_TEMPLATE.format(attack=_indent(attack)), encoding="utf-8")
    return path


@pytest.fixture
def sandbox(tmp_path):
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


def run_attack(sandbox, attack: str, *, experiment_id: str, params=None):
    runner, tmp_path, protected = sandbox
    candidate = build_candidate(tmp_path / f"{experiment_id}.py", attack)
    payload = {"protected": str(protected), "outside": str(tmp_path / "escaped.txt")}
    payload.update(params or {})
    before = build_manifest([protected])
    result = runner.run(
        make_spec(
            experiment_id=experiment_id,
            implementation=ImplementationRef(entrypoint=str(candidate)),
            parameters=payload,
            timeout_seconds=120.0,
        )
    )
    after = build_manifest([protected])
    return result, diff_manifests(before, after)


def attempt_dir_of(runner: ExperimentRunner, experiment_id: str) -> Path:
    attempts = sorted((runner.runs_dir / experiment_id / "attempts").iterdir())
    assert attempts, "no attempt directory recorded"
    return attempts[-1]


def candidate_output(runner: ExperimentRunner, experiment_id: str, name: str) -> Path | None:
    found = list(attempt_dir_of(runner, experiment_id).rglob(name))
    return found[0] if found else None


def violation_text(result) -> str:
    path = Path(result.stderr_path)
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def assert_blocked(result, label: str) -> None:
    """A violation ends the process, so the evidence is stderr plus a failed run.

    The guard cannot raise into candidate code: an exception carries
    `__traceback__.tb_frame`, and on 3.13+ writing that frame's f_locals
    writes through to the closure cells holding the write roots.
    """
    text = violation_text(result)
    assert VIOLATION_MARKER in text, (
        f"{label} was not blocked by the sandbox; stderr: {text[:300]!r}"
    )
    assert result.status != "success", f"{label} was denied but the attempt still succeeded"


# --------------------------------------------------------------------------
# Issue #15 -- candidate environment carries no parent secrets
# --------------------------------------------------------------------------


def test_env_is_built_from_allowlist_not_a_blocklist():
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


def test_env_never_forwards_user_home_or_profile():
    parent = {"HOME": "/home/me", "USERPROFILE": r"C:\Users\me", "APPDATA": r"C:\Users\me\AppData"}
    env = build_candidate_env(pythonpath="/src", temp_dir="/tmp/x", parent_env=parent)
    assert not {"HOME", "USERPROFILE", "APPDATA"} & set(env)


def test_env_pins_determinism_controls():
    env = build_candidate_env(pythonpath="/src", temp_dir="/tmp/x", parent_env={})
    assert env["PYTHONHASHSEED"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["TMPDIR"] == env["TEMP"] == env["TMP"] == "/tmp/x"


def test_env_forwards_thread_pins_that_affect_reproducibility():
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


def test_candidate_subprocess_cannot_read_planted_parent_secrets(sandbox, monkeypatch):
    runner, _, _ = sandbox
    for name, value in PLANTED_SECRETS.items():
        monkeypatch.setenv(name, value)
    attack = """
    dump = Path(cfg["output_scores"]).with_name("env_dump.json")
    import os as _o
    dump.write_text(json.dumps(dict(_o.environ)), encoding="utf-8")
    """
    result, _ = run_attack(sandbox, attack, experiment_id="env-exfil")
    assert result.status == "success"  # a legitimate experiment still completes

    dump_path = candidate_output(runner, "env-exfil", "env_dump.json")
    assert dump_path is not None, "candidate did not write its environment dump"
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    for name, value in PLANTED_SECRETS.items():
        assert name not in dump, f"{name} reached the candidate"
    blob = json.dumps(dump)
    for value in PLANTED_SECRETS.values():
        assert value not in blob


def test_candidate_cannot_read_repo_dotenv(sandbox):
    runner, tmp_path, _ = sandbox
    (tmp_path / ".env").write_text("GEMINI_API_KEY=AIzaONDISK0123456789\n", encoding="utf-8")
    attack = """
    target = Path(cfg["repo_root"]) / ".env"
    stolen = Path(cfg["output_scores"]).with_name("dotenv.txt")
    stolen.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    """
    result, _ = run_attack(
        sandbox, attack, experiment_id="dotenv-read", params={"repo_root": str(tmp_path)}
    )
    assert candidate_output(runner, "dotenv-read", "dotenv.txt") is None
    assert_blocked(result, "dotenv read")


# --------------------------------------------------------------------------
# Issue #16 -- protected assets survive every mutation technique
# --------------------------------------------------------------------------

# Each attack is source-obfuscated so an AST or string scanner cannot see it.
MUTATION_ATTACKS = {
    "direct_absolute_write": """
    name = "eval" + "uate" + ".p" + "y"
    (Path(cfg["protected"]) / name).write_text("def evaluate(*a, **k):\\n    return {}\\n", encoding="utf-8")
    """,
    "parent_traversal": """
    rel = Path("..") / ".." / ".." / ".." / ".." / "protected" / ("eval" + "uate.py")
    rel.write_text("x = 1\\n", encoding="utf-8")
    """,
    "os_open_low_level": """
    import os as _o
    target = str(Path(cfg["protected"]) / ("eval" + "uate.py"))
    fd = _o.open(target, _o.O_WRONLY | _o.O_TRUNC)
    _o.write(fd, b"tampered")
    _o.close(fd)
    """,
    "shutil_copy2": """
    import shutil as _s
    decoy = Path(cfg["output_scores"]).with_name("decoy.py")
    decoy.write_text("def evaluate(*a, **k):\\n    return {}\\n", encoding="utf-8")
    _s.copy2(decoy, Path(cfg["protected"]) / ("eval" + "uate.py"))
    """,
    "os_replace": """
    import os as _o
    staged = Path(cfg["output_scores"]).with_name("staged.py")
    staged.write_text("def evaluate(*a, **k):\\n    return {}\\n", encoding="utf-8")
    _o.replace(str(staged), str(Path(cfg["protected"]) / ("eval" + "uate.py")))
    """,
    "shutil_move": """
    import shutil as _s
    staged = Path(cfg["output_scores"]).with_name("moved.py")
    staged.write_text("y = 2\\n", encoding="utf-8")
    _s.move(str(staged), str(Path(cfg["protected"]) / ("eval" + "uate.py")))
    """,
    "unlink_evaluator": """
    (Path(cfg["protected"]) / ("eval" + "uate.py")).unlink()
    """,
    "symlink_escape": """
    import os as _o
    link = Path(cfg["output_scores"]).with_name("link")
    _o.symlink(cfg["protected"], str(link), target_is_directory=True)
    (link / ("eval" + "uate.py")).write_text("z = 3\\n", encoding="utf-8")
    """,
    "exec_computed_source": """
    src = "open(r'{}', 'w').write('nope')".format(str(Path(cfg["protected"]) / ("eval" + "uate.py")))
    exec(compile(src, "<computed>", "exec"), {"open": open})
    """,
    "dynamic_import_getattr": """
    mod = __import__("path" + "lib")
    writer = getattr(getattr(mod, "Path")(cfg["protected"]) / ("eval" + "uate.py"), "write_" + "text")
    writer("tampered\\n", encoding="utf-8")
    """,
    "subprocess_mutation": """
    import subprocess as _sp
    import sys as _sys
    target = str(Path(cfg["protected"]) / ("eval" + "uate.py"))
    _sp.run([_sys.executable, "-c", "open(r'%s','w').write('x')" % target], check=False)
    """,
    "ctypes_raw_syscall": """
    import ctypes as _c
    import os as _o
    _c.WinDLL("kernel32") if _o.name == "nt" else _c.CDLL(None)
    """,
    "chmod_then_write": """
    import os as _o
    target = str(Path(cfg["protected"]) / ("eval" + "uate.py"))
    _o.chmod(target, 0o777)
    open(target, "w").write("tampered")
    """,
    "rmtree_protected": """
    import shutil as _s
    _s.rmtree(cfg["protected"])
    """,
    "write_reference_json": """
    forged = '{"primary": 1.0}'
    (Path(cfg["protected"]) / ("refer" + "ence.json")).write_text(forged, encoding="utf-8")
    """,
}


@pytest.mark.parametrize("name,attack", sorted(MUTATION_ATTACKS.items()))
def test_protected_assets_survive_mutation_attempt(sandbox, name, attack):
    runner, _, _ = sandbox
    result, changes = run_attack(sandbox, attack, experiment_id=f"mut-{name}")
    assert changes == {}, f"{name} mutated protected assets: {changes}"
    assert_blocked(result, name)


def test_candidate_cannot_write_into_sibling_run_directory(sandbox):
    """The run directory publishes results; only the parent may write there."""
    runner, _, _ = sandbox
    attack = """
    published = Path(cfg["output_scores"]).parents[3] / "scores.npy"
    published.write_bytes(b"forged")
    """
    result, _ = run_attack(sandbox, attack, experiment_id="run-dir-write")
    assert_blocked(result, "run directory write")


def test_candidate_may_write_inside_its_own_attempt_directory(sandbox):
    runner, _, _ = sandbox
    attack = """
    import os as _o
    scratch = Path(cfg["output_scores"]).with_name("scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "checkpoint.npy").write_bytes(b"ok")
    _o.replace(str(scratch / "checkpoint.npy"), str(scratch / "final.npy"))
    """
    result, changes = run_attack(sandbox, attack, experiment_id="legit-write")
    assert result.status == "success"
    assert result.metrics is not None
    assert changes == {}
    assert VIOLATION_MARKER not in violation_text(result)
    assert candidate_output(runner, "legit-write", "final.npy") is not None


def test_mutation_that_evades_the_guard_still_invalidates_the_run(sandbox, monkeypatch):
    """The hash manifest is the backstop if the write boundary is ever bypassed.

    Simulated by mutating a protected asset from the parent while the candidate
    runs, which is indistinguishable to the runner from a candidate escape.
    """
    runner, tmp_path, protected = sandbox
    candidate = build_candidate(tmp_path / "quiet.py", "pass")
    real_run = ExperimentRunner.run

    def tamper_midrun(self, spec, **kwargs):
        original = self._integrity_failure

        def mutate_then_check(before, attempt_id):
            forged = "def evaluate(*a, **k):\n    return {}\n"
            (protected / "evaluate.py").write_text(forged, encoding="utf-8")
            return original(before, attempt_id)

        monkeypatch.setattr(self, "_integrity_failure", mutate_then_check)
        return real_run(self, spec, **kwargs)

    monkeypatch.setattr(ExperimentRunner, "run", tamper_midrun)
    result = runner.run(
        make_spec(
            experiment_id="integrity-backstop",
            implementation=ImplementationRef(entrypoint=str(candidate)),
            parameters={"protected": str(protected)},
        )
    )
    assert result.status == "invalid"
    assert result.failure is not None
    assert result.failure.kind == "integrity"
    assert result.metrics is None
    assert result.scores_path is None
    changed = result.failure.details["changes"]
    assert any(path.endswith("evaluate.py") for path in changed)


def test_real_evaluator_and_source_are_protected_by_default(tmp_path):
    """The default protected set covers the assets that actually decide scores."""
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


def test_integrity_manifest_detects_every_kind_of_drift(tmp_path):
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


def test_data_hash_cache_rehashes_when_contents_change(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    target = root / "log.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    cache: dict = {}
    before = build_manifest([root], cache=cache)
    assert len(cache) == 1

    target.write_text("a,b\n9,9\n", encoding="utf-8")
    os.utime(target, ns=(1_000_000_000, 2_000_000_000))
    after = build_manifest([root], cache=cache)
    assert diff_manifests(before, after)


def test_dataset_inside_starter_is_hashed_once_through_the_cache(tmp_path):
    """The KuaiRand data lives under starter/; it must not be full-hashed twice."""
    starter_like = tmp_path / "starter"
    (starter_like / "kuairand").mkdir(parents=True)
    (starter_like / "kuairand" / "evaluate.py").write_text("x = 1\n", encoding="utf-8")
    data_dir = starter_like / "kuairand" / "KuaiRand-Pure" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "log.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    runner = ExperimentRunner(
        repo_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=data_dir,
        protected_paths=[starter_like],
    )
    manifest = runner.protected_manifest()
    # Both the evaluator and the dataset are covered exactly once...
    assert any(path.endswith("/kuairand/evaluate.py") for path in manifest.digests)
    assert any(path.endswith("/KuaiRand-Pure/data/log.csv") for path in manifest.digests)
    # ...and the dataset digest came from the metadata-keyed cache.
    assert len(runner._data_hash_cache) == 1

    (data_dir / "log.csv").write_text("a,b\n9,9\n", encoding="utf-8")
    assert diff_manifests(manifest, runner.protected_manifest())


# --------------------------------------------------------------------------
# Regressions for the Cursor Bugbot / Security Agent findings on PR #17
# --------------------------------------------------------------------------

ESCAPE_ATTACKS = {
    # Security Agent: rebinding shared stdlib callables made the hook validate
    # a different path than the kernel acted on.
    "rebind_realpath": """
    import os as _o
    real = _o.path.realpath
    allowed = str(Path(cfg["output_scores"]).parent)
    _o.path.realpath = lambda p, *a, **k: allowed
    try:
        (Path(cfg["protected"]) / ("eval" + "uate.py")).write_text("PWNED", encoding="utf-8")
    finally:
        _o.path.realpath = real
    """,
    "rebind_fspath": """
    import os as _o
    real = _o.fspath
    target = Path(cfg["protected"]) / ("eval" + "uate.py")
    decoy = str(Path(cfg["output_scores"]).parent / "decoy")
    _o.fspath = lambda p: decoy
    try:
        target.write_text("PWNED", encoding="utf-8")
    finally:
        _o.fspath = real
    """,
    # Bugbot: module-level helpers were reachable through sys.modules.
    "rebind_guard_helper": """
    import sys as _s
    for name, mod in list(_s.modules.items()):
        if hasattr(mod, "SandboxViolation"):
            for attr in ("canonical", "_normalise", "_contained", "contained"):
                if hasattr(mod, attr):
                    setattr(mod, attr, lambda *a, **k: None)
    (Path(cfg["protected"]) / ("eval" + "uate.py")).write_text("PWNED", encoding="utf-8")
    """,
    # Security Agent: C-level writers that never raise an `open` event.
    "sqlite3_native_write": """
    import sqlite3 as _s
    con = _s.connect(str(Path(cfg["protected"]) / "planted.sqlite"))
    con.execute("create table t (a int)")
    con.commit()
    con.close()
    """,
    "dbm_native_write": """
    import dbm as _d
    with _d.open(str(Path(cfg["protected"]) / "planted_dbm"), "c") as db:
        db["k"] = "v"
    """,
    # Security Agent: relative paths cannot be validated under POSIX dir_fd.
    "relative_path_write": """
    import os as _o
    _o.chdir(str(Path(cfg["output_scores"]).parent))
    open("relative_escape.txt", "w").write("x")
    """,
}


# Rebinding os.fspath is the one case the guard does not answer with a
# violation: pathlib calls os.fspath itself, so the candidate redirects its
# own write into the sandbox. The hook still uses the captured original, so
# the protected tree is untouched either way -- the attack only hurts the
# attacker.
SELF_DEFEATING = frozenset({"rebind_fspath"})


@pytest.mark.parametrize("name,attack", sorted(ESCAPE_ATTACKS.items()))
def test_sandbox_escape_is_blocked(sandbox, name, attack):
    runner, _, _ = sandbox
    result, changes = run_attack(sandbox, attack, experiment_id=f"esc-{name}")
    assert changes == {}, f"{name} mutated protected assets: {changes}"
    if name not in SELF_DEFEATING:
        assert_blocked(result, name)


def test_candidate_cannot_read_files_outside_its_sandbox(sandbox):
    """Reads are allowlisted, not merely deny-listed.

    Without this, stripping secrets from the environment is theatre: the
    candidate reads them back from a credential file or /proc/<ppid>/environ.
    """
    runner, tmp_path, _ = sandbox
    creds = tmp_path / "fake_home" / "credentials"
    creds.parent.mkdir()
    creds.write_text("aws_secret_access_key = LEAKEDFROMDISK\n", encoding="utf-8")
    attack = """
    stolen = Path(cfg["creds"]).read_text(encoding="utf-8")
    Path(cfg["output_scores"]).with_name("stolen.txt").write_text(stolen, encoding="utf-8")
    """
    result, _ = run_attack(
        sandbox, attack, experiment_id="read-escape", params={"creds": str(creds)}
    )
    assert candidate_output(runner, "read-escape", "stolen.txt") is None
    assert_blocked(result, "host credential read")


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="POSIX /proc only")
def test_candidate_cannot_read_parent_process_environ(sandbox):
    runner, _, _ = sandbox
    attack = """
    import os as _o
    blob = Path("/proc/%d/environ" % _o.getppid()).read_bytes()
    Path(cfg["output_scores"]).with_name("ppid_env.bin").write_bytes(blob)
    """
    result, _ = run_attack(sandbox, attack, experiment_id="proc-environ")
    assert candidate_output(runner, "proc-environ", "ppid_env.bin") is None
    assert_blocked(result, "/proc parent environ")


def test_guard_denies_unknown_audit_events_by_default():
    """Default-deny, not deny-listing: sqlite3 and dbm write from C.

    Enumerating dangerous events cannot work -- `sqlite3.connect` raises no
    `open` event at all, and on Windows `shutil.copy2` raises only
    `_winapi.CopyFile2`.
    """
    from research_agent.experiments import candidate_guard as guard

    for event in ("sqlite3.connect", "_winapi.CopyFile2", "ctypes.dlopen",
                  "subprocess.Popen", "socket.connect", "os.symlink", "dbm.open"):
        assert event not in guard.ALLOWED_EVENTS
        assert not event.startswith(guard.ALLOWED_PREFIXES)


def test_integrity_violation_is_latched_across_runs(sandbox):
    """A dirty tree must not become the next run's clean baseline.

    Re-snapshotting per attempt let a mutation that survived one run be
    accepted as normal by the next, so a poisoned evaluator scored freely.
    """
    runner, tmp_path, protected = sandbox
    candidate = build_candidate(tmp_path / "quiet.py", "pass")

    def spec_for(name):
        return make_spec(
            experiment_id=name,
            implementation=ImplementationRef(entrypoint=str(candidate)),
            parameters={"protected": str(protected), "n": name},
        )

    original = runner._integrity_failure

    def tamper_once(before, attempt_id):
        (protected / "evaluate.py").write_text("POISONED\n", encoding="utf-8")
        runner._integrity_failure = original
        return original(before, attempt_id)

    runner._integrity_failure = tamper_once
    first = runner.run(spec_for("latch-first"))
    second = runner.run(spec_for("latch-second"))

    assert first.status == "invalid" and first.failure.kind == "integrity"
    assert second.status == "invalid", "poisoned tree was re-baselined and scored"
    assert second.failure.kind == "integrity"
    assert second.metrics is None and second.scores_path is None


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


# --------------------------------------------------------------------------
# Round-2 Cursor findings: closure reachability, native loading, fail-open bind
# --------------------------------------------------------------------------

REFLECTION_ATTACKS = {
    # gc.get_objects / get_referrers enumerate live objects, which reaches the
    # hook function; closure cells are writable, so this would let a candidate
    # swap the write roots out from under the guard.
    "gc_reach_hook_closure": """
    import gc as _gc
    hooks = [o for o in _gc.get_objects()
             if getattr(o, "__name__", None) == "hook" and getattr(o, "__closure__", None)]
    Path(cfg["output_scores"]).with_name("hooks.txt").write_text(str(len(hooks)), encoding="utf-8")
    for fn in hooks:
        for cell in fn.__closure__:
            try:
                cell.cell_contents = ()
            except (ValueError, TypeError):
                pass
    """,
    # Frame introspection is the other route to the same cells: on 3.13+
    # frame.f_locals writes through to the real locals.
    "frame_reach_hook_closure": """
    import sys as _s
    _s.setprofile(lambda *a: None)
    """,
    "getframe_walk": """
    import sys as _s
    _s._getframe(0)
    """,
}


@pytest.mark.parametrize("name,attack", sorted(REFLECTION_ATTACKS.items()))
def test_hook_closure_is_unreachable(sandbox, name, attack):
    """The guard's captured locals must not be reachable, let alone writable."""
    runner, _, _ = sandbox
    result, changes = run_attack(sandbox, attack, experiment_id=f"refl-{name}")
    assert changes == {}
    assert_blocked(result, name)


NATIVE_ATTACKS = {
    # Write a shared library into the sandbox, put the sandbox on sys.path,
    # import it. Native constructors then run fopen/write with no Python
    # audit event at all, bypassing every check above.
    "native_via_syspath": """
    import sys as _s
    out = Path(cfg["output_scores"]).parent
    dst = out / ("evilmod" + cfg["suffix"])
    with open(cfg["donor"], "rb") as fh:
        blob = fh.read()
    with open(dst, "wb") as fh:
        fh.write(blob)
    _s.path.insert(0, str(out))
    import evilmod
    """,
    # Same, without sys.path: a direct ExtensionFileLoader. The import event
    # populates `filename` on this route.
    "native_via_direct_loader": """
    import importlib.machinery as _m
    import importlib.util as _u
    out = Path(cfg["output_scores"]).parent
    dst = out / "blob_no_extension"
    with open(cfg["donor"], "rb") as fh:
        blob = fh.read()
    with open(dst, "wb") as fh:
        fh.write(blob)
    loader = _m.ExtensionFileLoader("blobmod", str(dst))
    spec = _u.spec_from_file_location("blobmod", str(dst), loader=loader)
    _u.module_from_spec(spec)
    """,
    "add_dll_directory_on_sandbox": """
    import os as _o
    _o.add_dll_directory(str(Path(cfg["output_scores"]).parent))
    """,
}


def _native_donor():
    suffix = ".pyd" if os.name == "nt" else ".so"
    for base in (Path(sys.prefix) / "DLLs", Path(sys.prefix) / "lib-dynload"):
        if base.is_dir():
            found = sorted(base.glob(f"*{suffix}"))
            if found:
                return str(found[0]), suffix
    return None, suffix


@pytest.mark.parametrize("name,attack", sorted(NATIVE_ATTACKS.items()))
def test_native_code_cannot_be_loaded_from_the_sandbox(sandbox, name, attack):
    """The sandbox is writable or importable, never both."""
    runner, _, _ = sandbox
    donor, suffix = _native_donor()
    if donor is None and "native" in name:
        pytest.skip("no native extension module available to copy")
    result, changes = run_attack(
        sandbox, attack, experiment_id=f"nat-{name}",
        params={"donor": donor or "", "suffix": suffix},
    )
    assert changes == {}
    assert_blocked(result, name)


def test_gc_and_frame_events_are_not_allowlisted():
    from research_agent.experiments import candidate_guard as guard

    for event in ("gc.get_objects", "gc.get_referrers", "sys._getframe",
                  "sys._current_frames", "sys.settrace", "sys.setprofile"):
        assert event not in guard.ALLOWED_EVENTS
        assert not event.startswith(guard.ALLOWED_PREFIXES), event


def test_run_fails_closed_when_the_evaluator_cannot_be_bound(tmp_path, monkeypatch):
    """Binding must not fail open.

    A swallowed ImportError would leave `official_evaluate` to resolve
    `evaluate` lazily after the candidate exits -- the exact window that
    binding early was added to close.
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


# --------------------------------------------------------------------------
# Round-3 Cursor findings: relative import origins, subinterpreters,
# and traceback frames reaching the hook's closure
# --------------------------------------------------------------------------

ROUND3_ATTACKS = {
    # canonical() returns None for a relative path. The import gate used to
    # *skip* on None while `open` denied on it, and the runner sets cwd to a
    # write root -- so a relative origin walked straight through.
    "relative_import_origin": """
    import importlib.machinery as _m
    import importlib.util as _u
    import os as _o
    out = Path(cfg["output_scores"]).parent
    dst = out / "blob_rel"
    with open(cfg["donor"], "rb") as fh:
        blob = fh.read()
    with open(dst, "wb") as fh:
        fh.write(blob)
    _o.chdir(str(out))
    loader = _m.ExtensionFileLoader("relmod", "blob_rel")
    spec = _u.spec_from_file_location("relmod", "blob_rel", loader=loader)
    _u.module_from_spec(spec)
    """,
    "relative_syspath_entry": """
    import os as _o
    import sys as _s
    _o.chdir(str(Path(cfg["output_scores"]).parent))
    _s.path.insert(0, "./")
    import evilmod
    """,
    # Audit hooks are per-interpreter and are not copied into a new one, so
    # anything run there would have no boundary at all.
    "subinterpreter_escape": """
    try:
        import _interpreters as _i
    except ImportError:
        import _xxsubinterpreters as _i
    interp = _i.create()
    _i.run_string(interp, "open(r'%s', 'w').write('x')" % cfg["escape_target"])
    """,
}


@pytest.mark.parametrize("name,attack", sorted(ROUND3_ATTACKS.items()))
def test_round3_escapes_are_blocked(sandbox, name, attack):
    runner, tmp_path, _ = sandbox
    donor, suffix = _native_donor()
    if donor is None:
        donor = ""
    result, changes = run_attack(
        sandbox, attack, experiment_id=f"r3-{name}",
        params={
            "donor": donor,
            "suffix": suffix,
            "escape_target": str(tmp_path / "subinterp_escape.txt"),
        },
    )
    assert changes == {}
    assert not (tmp_path / "subinterp_escape.txt").exists()
    assert_blocked(result, name)


def test_violation_does_not_hand_a_traceback_back_to_the_candidate(sandbox):
    """A raised violation would leak the hook's frame.

    `SandboxViolation.__traceback__.tb_frame` is the hook's own frame, and on
    3.13+ writing that frame's f_locals writes through to the closure cells
    holding the write roots, read roots and allow list. So the guard exits the
    process instead of raising: the except branch below must never run.
    """
    runner, _, protected = sandbox
    attack = """
    import sys as _s
    try:
        open(str(Path(cfg["protected"]) / ("eval" + "uate.py")), "w").write("x")
    except BaseException as exc:
        frames = []
        tb = getattr(exc, "__traceback__", None)
        while tb is not None:
            frames.append(sorted(tb.tb_frame.f_locals))
            tb = tb.tb_next
        Path(cfg["output_scores"]).with_name("caught.txt").write_text(
            repr(frames), encoding="utf-8")
    """
    result, changes = run_attack(sandbox, attack, experiment_id="tb-reflect")
    assert changes == {}
    assert candidate_output(runner, "tb-reflect", "caught.txt") is None, (
        "candidate caught the violation and could read the hook's frame locals"
    )
    assert_blocked(result, "traceback reflection")


def test_subinterpreter_and_native_namespaces_are_not_allowlisted():
    from research_agent.experiments import candidate_guard as guard

    for event in ("cpython.PyInterpreterState_New", "cpython.run_stdin",
                  "ctypes.dlopen", "sqlite3.connect"):
        assert event not in guard.ALLOWED_EVENTS
        assert not event.startswith(guard.ALLOWED_PREFIXES), event


# --------------------------------------------------------------------------
# Round-4 Cursor finding: re-entry into the hook via candidate-controlled
# objects and rebound stdlib attributes
# --------------------------------------------------------------------------


def test_denial_path_cannot_be_hijacked_by_rebinding_stderr(sandbox):
    """`sys.stderr.write` is assignable, so it must be bound at install time.

    A late-bound lookup would run candidate code with the hook frame live on
    the stack, and `f_back` from there reaches the closure cells holding the
    roots -- including `_exit`, which could be nopped so the denied operation
    proceeds.
    """
    runner, _, protected = sandbox
    attack = """
    import sys as _s
    marker = Path(cfg["output_scores"]).with_name("hijacked.txt")

    def evil(*a, **k):
        marker.write_text("stderr.write hijacked", encoding="utf-8")
        return 0

    _s.stderr.write = evil
    open(str(Path(cfg["protected"]) / ("eval" + "uate.py")), "w").write("x")
    """
    result, changes = run_attack(sandbox, attack, experiment_id="hijack-stderr")
    assert changes == {}
    assert candidate_output(runner, "hijack-stderr", "hijacked.txt") is None
    assert_blocked(result, "stderr hijack")


def test_hook_never_calls_repr_on_candidate_objects(sandbox):
    """`shutil.*` events carry the caller's own objects, unlike `open`.

    CPython normalises paths to str before raising `open`/`os.*`, but a
    Python-level `sys.audit("shutil.copyfile", src, dst)` passes whatever was
    handed in -- so a `__repr__` or `__fspath__` in a denial message would
    execute candidate code inside the hook.
    """
    runner, _, protected = sandbox
    attack = """
    import shutil as _sh
    marker = Path(cfg["output_scores"]).with_name("reentered.txt")

    class Evil:
        def __fspath__(self):
            marker.write_text("__fspath__ ran inside the hook", encoding="utf-8")
            return str(Path(cfg["protected"]) / ("eval" + "uate.py"))

        def __repr__(self):
            marker.write_text("__repr__ ran inside the hook", encoding="utf-8")
            return "innocent"

    _sh.copyfile(str(Path(cfg["output_scores"])), Evil())
    """
    result, changes = run_attack(sandbox, attack, experiment_id="repr-reentry")
    assert changes == {}
    assert candidate_output(runner, "repr-reentry", "reentered.txt") is None, (
        "candidate code ran inside the audit hook"
    )
    assert_blocked(result, "repr re-entry")


def test_interpreter_teardown_events_do_not_fail_a_clean_run():
    """Denying all of cpython.* would kill every run during finalisation."""
    from research_agent.experiments import candidate_guard as guard

    assert "cpython.PyInterpreterState_Clear" in guard.ALLOWED_EVENTS
    assert "cpython.PyInterpreterState_Delete" in guard.ALLOWED_EVENTS
    assert "cpython.PyInterpreterState_New" not in guard.ALLOWED_EVENTS
    assert not "cpython.PyInterpreterState_New".startswith(guard.ALLOWED_PREFIXES)


def test_testcapi_subinterpreter_route_is_denied(sandbox):
    """_testcapi.run_in_subinterp is the subinterpreter capability by another name."""
    runner, tmp_path, _ = sandbox
    attack = """
    import _testcapi as _t
    _t.run_in_subinterp("open(r'%s', 'w').write('x')" % cfg["escape_target"])
    """
    result, changes = run_attack(
        sandbox, attack, experiment_id="testcapi-subinterp",
        params={"escape_target": str(tmp_path / "testcapi_escape.txt")},
    )
    assert changes == {}
    assert not (tmp_path / "testcapi_escape.txt").exists()
    assert_blocked(result, "_testcapi subinterpreter")


def test_str_subclass_cannot_re_enter_the_hook_through_concatenation(sandbox):
    """A str subclass with __radd__ runs before str.__add__ on `literal + value`."""
    runner, _, _ = sandbox
    attack = """
    marker = Path(cfg["output_scores"]).with_name("radd.txt")

    class Evil(str):
        def __radd__(self, other):
            marker.write_text("__radd__ ran inside the hook", encoding="utf-8")
            return other + str.__str__(self)

    target = Evil(str(Path(cfg["protected"]) / ("eval" + "uate.py")))
    open(target, "w").write("x")
    """
    result, changes = run_attack(sandbox, attack, experiment_id="radd-reentry")
    assert changes == {}
    assert candidate_output(runner, "radd-reentry", "radd.txt") is None, (
        "candidate __radd__ ran inside the audit hook"
    )
    assert_blocked(result, "str subclass re-entry")
