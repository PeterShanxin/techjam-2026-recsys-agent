"""Generated-code checks. Controlled research workspace, not a security sandbox."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .constants import CANDIDATE_FILENAME

CLI_FLAGS = ("--data-dir", "--split", "--output-scores", "--seed", "--config")
WRITE_FUNCS = {"write_text", "write_bytes", "write", "replace", "unlink", "move", "copy", "copy2"}


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


def validate_candidate_source(source: str, dest: Path, workspace_root: Path) -> ast.AST:
    if not source or not source.strip():
        raise SafetyError("candidate_source is empty")
    assert_workspace_path(dest, workspace_root)
    tree = syntax_check(source)
    assert_cli_contract(source, tree)
    assert_no_evaluator_tampering(source, tree)
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


