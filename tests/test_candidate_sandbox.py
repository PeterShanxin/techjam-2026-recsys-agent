"""Adversarial tests for the untrusted-candidate sandbox (issues #15 and #16).

These exercise the runtime boundary, not the AST lint in ``agent/safety.py``.
Every attack here is written so that source-level pattern matching cannot see
it: names are split and rejoined, calls go through ``getattr``/``exec``, and
some run in a spawned interpreter.
"""
from __future__ import annotations

import json
import os
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


def attack_error(runner: ExperimentRunner, experiment_id: str) -> str:
    markers = list(attempt_dir_of(runner, experiment_id).rglob("attack_error.txt"))
    return markers[0].read_text(encoding="utf-8") if markers else ""


def candidate_output(runner: ExperimentRunner, experiment_id: str, name: str) -> Path | None:
    found = list(attempt_dir_of(runner, experiment_id).rglob(name))
    return found[0] if found else None


def assert_blocked(error: str, label: str) -> None:
    """SandboxViolation subclasses PermissionError; the candidate sees either name."""
    assert "SandboxViolation" in error or "PermissionError" in error, (
        f"{label} was not blocked by the sandbox; candidate saw: {error!r}"
    )


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
    assert result.status == "success"
    assert candidate_output(runner, "dotenv-read", "dotenv.txt") is None
    assert_blocked(attack_error(runner, "dotenv-read"), "dotenv read")


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
    assert_blocked(attack_error(runner, f"mut-{name}"), name)


def test_candidate_cannot_write_into_sibling_run_directory(sandbox):
    """The run directory publishes results; only the parent may write there."""
    runner, _, _ = sandbox
    attack = """
    published = Path(cfg["output_scores"]).parents[3] / "scores.npy"
    published.write_bytes(b"forged")
    """
    result, _ = run_attack(sandbox, attack, experiment_id="run-dir-write")
    assert result.status == "success"
    assert_blocked(attack_error(runner, "run-dir-write"), "run directory write")


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
    assert attack_error(runner, "legit-write") == ""
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
