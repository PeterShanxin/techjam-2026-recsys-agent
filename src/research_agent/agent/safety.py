"""Generated-code checks. Controlled research workspace, not a security sandbox."""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
import sysconfig
from pathlib import Path

from .constants import CANDIDATE_FILENAME
from .environment import EnvironmentCapabilities, discover_environment

CLI_FLAGS = ("--data-dir", "--split", "--output-scores", "--seed", "--config")
WRITE_FUNCS = {"write_text", "write_bytes", "write", "replace", "unlink", "move", "copy", "copy2"}
_IMPORT_ERROR_NAMES = {"ImportError", "ModuleNotFoundError"}


class SafetyError(ValueError):
    """Generated candidate failed a workspace/syntax/contract check."""


def syntax_check(source: str) -> ast.AST:
    try:
        tree = ast.parse(source, filename=CANDIDATE_FILENAME)
    except SyntaxError as exc:
        raise SafetyError(f"syntax error: {exc}") from exc
    try:
        compile(source, CANDIDATE_FILENAME, "exec")
    except SyntaxError as exc:
        raise SafetyError(f"compile error: {exc}") from exc
    return tree


def assert_cli_contract(source: str, tree: ast.AST | None = None) -> None:
    tree = tree or syntax_check(source)
    found = set(_string_constants(tree))
    missing = [flag for flag in CLI_FLAGS if flag not in found]
    if missing:
        raise SafetyError(f"candidate CLI missing flags: {missing}")


def assert_workspace_path(dest: Path, workspace_root: Path) -> None:
    try:
        dest_resolved = dest.resolve()
        root_resolved = workspace_root.resolve()
        dest_resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise SafetyError(f"generated path escapes workspace: {dest}") from exc
    if dest_resolved.name != CANDIDATE_FILENAME:
        raise SafetyError(f"generated file must be named {CANDIDATE_FILENAME}")


def assert_no_evaluator_tampering(source: str, tree: ast.AST | None = None) -> None:
    tree = tree or syntax_check(source)
    if _writes_forbidden_path(tree, source):
        raise SafetyError("candidate appears to modify starter/kuairand/evaluate.py")


def assert_no_silent_dependency_fallback(
    tree: ast.AST, environment: EnvironmentCapabilities
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not _catches_import_error(node):
            continue
        for stmt in node.body:
            for name in _import_toplevel_names(stmt):
                if not is_allowed_import(name, environment):
                    raise SafetyError(f"silent_dependency_fallback: {name}")


def assert_allowed_imports(tree: ast.AST, environment: EnvironmentCapabilities) -> None:
    for node in ast.walk(tree):
        for name in _import_toplevel_names(node):
            if not is_allowed_import(name, environment):
                raise SafetyError(f"unsupported_dependency: {name}")


def is_allowed_import(name: str, environment: EnvironmentCapabilities) -> bool:
    if not name:
        return True
    if name in environment.starter_modules or name in environment.allowed_third_party:
        return True
    return _is_stdlib(name)


def validate_candidate_source(
    source: str,
    dest: Path,
    workspace_root: Path,
    *,
    environment: EnvironmentCapabilities | None = None,
) -> ast.AST:
    if not source or not source.strip():
        raise SafetyError("candidate_source is empty")
    assert_workspace_path(dest, workspace_root)
    tree = syntax_check(source)
    assert_cli_contract(source, tree)
    assert_no_evaluator_tampering(source, tree)
    env = environment or discover_environment()
    assert_no_silent_dependency_fallback(tree, env)
    assert_allowed_imports(tree, env)
    return tree


def _string_constants(tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def _writes_forbidden_path(tree: ast.AST, source: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name not in WRITE_FUNCS and name != "open":
            continue
        blob = " ".join(_subtree_strings(node)).replace("\\", "/")
        if "evaluate.py" in blob:
            return True
    if re.search(r"open\([^)]*evaluate\.py[^)]*,\s*['\"]w", source):
        return True
    return False


def _subtree_strings(node: ast.AST) -> list[str]:
    out = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            out.append(child.value)
    return out


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _import_toplevel_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [_root_module(alias.name) for alias in node.names if alias.name]
    if isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            return []
        if node.module:
            return [_root_module(node.module)]
        return []
    return []


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:
            return True
        if _handler_mentions_import_error(handler.type):
            return True
    return False


def _handler_mentions_import_error(exc: ast.AST) -> bool:
    if isinstance(exc, ast.Name):
        return exc.id in _IMPORT_ERROR_NAMES
    if isinstance(exc, ast.Attribute):
        return exc.attr in _IMPORT_ERROR_NAMES
    if isinstance(exc, ast.Tuple):
        return any(_handler_mentions_import_error(elt) for elt in exc.elts)
    return False


def _is_stdlib(name: str) -> bool:
    if name in sys.builtin_module_names:
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None:
        return name in stdlib_names
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
    if spec is None:
        return False
    origin = spec.origin
    if origin in ("built-in", "frozen"):
        return True
    if not origin:
        return False
    path = Path(origin).resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    try:
        path.relative_to(stdlib)
    except ValueError:
        return False
    return "site-packages" not in path.parts and "dist-packages" not in path.parts


