"""Compact KuaiRand data capabilities. Derived from starter data.py, not a row dump."""
from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from research_agent.lab.capabilities import lab_contract_dict
from research_agent.llm.secrets import sanitize

# Organizer README lists these log columns. data.load() does not copy them into tuples.
_ORGANIZER_AUX_LOG_FIELDS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "play_time_ms",
    "hourmin",
    "time_ms",
    "is_hate",
    "profile_stay_time",
    "comment_stay_time",
)

_RAW_LOG_MARKERS = (
    "log_standard",
    "log_random",
    "user_features_pure",
    "video_features_basic",
)

_LOAD_TUPLE_FALLBACK = (
    {"index": 0, "name": "date", "type": "int", "meaning": "YYYYMMDD from the log row"},
    {"index": 1, "name": "user_id", "type": "str", "meaning": "user identifier"},
    {"index": 2, "name": "video_id", "type": "str", "meaning": "video identifier"},
    {
        "index": 3,
        "name": "author_id",
        "type": "str",
        "meaning": "joined from video_features_basic_pure.csv; UNK if missing",
    },
    {"index": 4, "name": "tab", "type": "str", "meaning": "recommendation tab"},
    {"index": 5, "name": "duration_ms", "type": "float", "meaning": "video duration in milliseconds"},
    {
        "index": 6,
        "name": "long_view",
        "type": "int",
        "meaning": "official binary target from LABEL long_view",
    },
)

_CONTRACT_RULE = (
    "A proposal that claims mechanism X must actually have the data required to execute X. "
    "data.load() returns 7-tuples; it does not expose is_like, play_time_ms, or other aux log columns. "
    "To use those columns, read raw CSVs or call research_agent.lab train-aux / history APIs. "
    "Train-derived features must come from train. Validation labels are evaluator-only. "
    "TEST IS SEALED. Do not dump raw rows into prompts. Official target is long_view. "
    "Evaluation is within-user ranking with GAUC, nDCG@5, and primary = mean of those two."
)

_LAB_AUX_APIS = ("get_user_history", "train_aux", "train_events")
_LAB_CONTEXT_APIS = ("inference_rows", "get_user_history", "train_events")
_LAB_CONTEXT_FIELDS = frozenset(("hourmin", "time_ms"))


class DataContractError(ValueError):
    """Proposal claims a data field the supported loader does not provide."""


@dataclass(frozen=True)
class DataContract:
    load_tuple_fields: tuple[dict[str, Any], ...]
    encode_fields: tuple[str, ...]
    official_target: str
    splits: dict[str, list[int]]
    available_via_load: tuple[str, ...]
    not_available_via_load: tuple[str, ...]
    raw_files: tuple[dict[str, Any], ...]
    starter_data_py: str
    rule: str = _CONTRACT_RULE

    def to_dict(self) -> dict[str, Any]:
        return sanitize(
            {
                "load": {
                    "module": "starter/kuairand/data.py function load()",
                    "returns": "dict[split_name] -> list of tuples in official date windows",
                    "tuple_length": len(self.load_tuple_fields),
                    "tuple_fields": [dict(item) for item in self.load_tuple_fields],
                },
                "encode": {
                    "module": "starter/kuairand/data.py function encode()",
                    "returns": (
                        "per split (X, y, users); X int32 (N, n_fields), "
                        "y float32 long_view, users user_id list"
                    ),
                    "fields": list(self.encode_fields),
                    "y": "tuple field long_view (index 6)",
                    "users": "tuple field user_id (index 1)",
                },
                "official_target": self.official_target,
                "evaluation": {
                    "task": "within-user ranking",
                    "metrics": ["GAUC", "nDCG@5"],
                    "primary": "mean(GAUC, nDCG@5)",
                    "research_split": "valid",
                },
                "splits": {
                    name: {"date_lo": lo, "date_hi": hi} for name, (lo, hi) in self.splits.items()
                },
                "available_via_load": list(self.available_via_load),
                "not_available_via_load": list(self.not_available_via_load),
                "raw_files": [dict(item) for item in self.raw_files],
                "starter_data_py": self.starter_data_py,
                "lab": lab_contract_dict(),
                "test_sealed": True,
                "rule": self.rule,
            }
        )


def discover_data_contract(
    repo_root: Path | None = None,
    data_dir: Path | None = None,
) -> DataContract:
    data_py = _locate_data_py(repo_root)
    source = data_py.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(data_py))
    label = _assign_string(tree, "LABEL") or "long_view"
    encode_fields = tuple(
        _assign_string_list(tree, "FIELDS")
        or ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
    )
    splits = _assign_splits(tree) or {
        "train": [20220408, 20220421],
        "valid": [20220422, 20220428],
        "test": [20220429, 20220508],
    }
    tuple_fields = _load_tuple_fields_from_source(source) or tuple(
        dict(item) for item in _LOAD_TUPLE_FALLBACK
    )
    available = tuple(item["name"] for item in tuple_fields)
    raw_files = _raw_file_inventory(data_dir)
    raw_columns: set[str] = set()
    for item in raw_files:
        raw_columns.update(item.get("columns") or [])
    not_available: list[str] = []
    seen: set[str] = set()
    for name in list(_ORGANIZER_AUX_LOG_FIELDS) + sorted(raw_columns):
        if name in available or name in seen:
            continue
        seen.add(name)
        not_available.append(name)
    rel = "starter/kuairand/data.py"
    try:
        if repo_root is not None:
            rel = data_py.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        rel = data_py.as_posix()
    return DataContract(
        load_tuple_fields=tuple(tuple_fields),
        encode_fields=encode_fields,
        official_target=label,
        splits={k: list(v) for k, v in splits.items()},
        available_via_load=available,
        not_available_via_load=tuple(not_available),
        raw_files=tuple(raw_files),
        starter_data_py=rel,
    )


def claimed_unavailable_fields(
    *,
    proposal: Mapping[str, Any] | None,
    source: str,
    contract: DataContract,
) -> tuple[str, ...]:
    unavailable = {name.lower(): name for name in contract.not_available_via_load}
    texts: list[str] = [source or ""]
    required: list[str] = []
    if proposal is not None:
        for key in (
            "hypothesis",
            "rationale",
            "expected_mechanism",
            "mutation_summary",
            "what_changed",
            "observation",
        ):
            value = proposal.get(key) if isinstance(proposal, Mapping) else getattr(proposal, key, "")
            if value:
                texts.append(str(value))
        raw_required = (
            proposal.get("required_data_fields")
            if isinstance(proposal, Mapping)
            else getattr(proposal, "required_data_fields", ())
        )
        if raw_required:
            required.extend(str(item) for item in raw_required)
    blob = "\n".join(texts)
    hits: list[str] = []
    seen: set[str] = set()
    for name in required:
        key = name.lower()
        if key in unavailable and name not in seen:
            seen.add(unavailable[key])
            hits.append(unavailable[key])
    for key, original in unavailable.items():
        if original in seen:
            continue
        if _mentions_field(blob, original):
            seen.add(original)
            hits.append(original)
    if not hits:
        return ()
    if _reads_claimed_fields_from_raw_csv(source, tuple(hits)):
        return ()
    if _lab_exposes_claimed_fields(source, tuple(hits)):
        return ()
    return tuple(hits)


def validate_proposal_data_claims(proposal: Any, contract: DataContract) -> None:
    source = getattr(proposal, "candidate_source", "") or ""
    mapping = proposal.to_dict() if hasattr(proposal, "to_dict") else dict(proposal)
    missing = claimed_unavailable_fields(proposal=mapping, source=source, contract=contract)
    if missing:
        raise DataContractError(
            f"unavailable_data_field: {list(missing)}. "
            "data.load() tuples do not include these columns."
        )


def format_data_contract_repair_message(
    *,
    fields: Iterable[str],
    hypothesis: str | None,
    contract: DataContract | None = None,
) -> str:
    contract = contract or discover_data_contract()
    available = ", ".join(contract.available_via_load)
    hypo = (hypothesis or "").strip() or "(not parsed)"
    return (
        f"unavailable_data_field: {list(fields)}. "
        f"data.load() returns 7-tuples with fields [{available}]. "
        "Those aux columns are not present on the loader tuples. "
        "Either read the raw log CSVs explicitly (log_standard_*.csv) or pick a mechanism "
        "that uses available loader fields. "
        f"Original hypothesis: {hypo}\n"
        "Preserve the original hypothesis if it can be implemented by reading the raw files; "
        "otherwise reimplement with available loader fields. "
        "Do not silently train binary long_view FM and treat matching FM metrics as evidence."
    )


def _locate_data_py(repo_root: Path | None) -> Path:
    candidates = []
    if repo_root is not None:
        candidates.append(Path(repo_root) / "starter" / "kuairand" / "data.py")
    here = Path(__file__).resolve()
    candidates.append(here.parents[3] / "starter" / "kuairand" / "data.py")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("starter/kuairand/data.py not found for data contract discovery")


def _assign_string(tree: ast.AST, name: str) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return None


def _assign_string_list(tree: ast.AST, name: str) -> list[str] | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.List):
                        out = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out.append(elt.value)
                        return out
    return None


def _assign_splits(tree: ast.AST) -> dict[str, list[int]] | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SPLITS":
                    if not isinstance(node.value, ast.Dict):
                        return None
                    out: dict[str, list[int]] = {}
                    for key, value in zip(node.value.keys, node.value.values):
                        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                            continue
                        if isinstance(value, ast.Tuple) and len(value.elts) == 2:
                            nums = []
                            for elt in value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                                    nums.append(int(elt.value))
                            if len(nums) == 2:
                                out[key.value] = nums
                    return out or None
    return None


def _load_tuple_fields_from_source(source: str) -> tuple[dict[str, Any], ...] | None:
    needed = ("r['date']", "r['user_id']", "r['video_id']", "r['tab']", "r['duration_ms']")
    if not all(token in source for token in needed):
        return None
    if "vid2author" not in source or "LABEL" not in source:
        return None
    return tuple(dict(item) for item in _LOAD_TUPLE_FALLBACK)


def _raw_file_inventory(data_dir: Path | None) -> list[dict[str, Any]]:
    catalog = [
        {
            "file": "log_standard_4_08_to_4_21_pure.csv",
            "used_by_load": True,
            "note": "Interaction log. load() copies date, user_id, video_id, tab, duration_ms, long_view only.",
        },
        {
            "file": "log_standard_4_22_to_5_08_pure.csv",
            "used_by_load": True,
            "note": "Interaction log covering valid and test date windows.",
        },
        {
            "file": "video_features_basic_pure.csv",
            "used_by_load": True,
            "note": "load() joins author_id only. Other video columns are unused by the default loader.",
        },
        {
            "file": "user_features_pure.csv",
            "used_by_load": False,
            "note": "Not read by data.load(). Candidates may open it explicitly.",
        },
        {
            "file": "log_random_4_22_to_5_08_pure.csv",
            "used_by_load": False,
            "note": "Random-exposure log. Not read by data.load().",
        },
        {
            "file": "video_features_statistic_pure.csv",
            "used_by_load": False,
            "leakage_risk": "high",
            "note": "Unscoped catalog engagement counts. Date window unknown. Do not use as default popularity.",
        },
    ]
    if data_dir is None:
        return catalog
    root = Path(data_dir)
    if not root.is_dir():
        return catalog
    out: list[dict[str, Any]] = []
    used = {
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
        "video_features_basic_pure.csv",
    }
    known = {item["file"]: item for item in catalog}
    for path in sorted(root.glob("*.csv")):
        columns = _csv_header(path)
        base = dict(known.get(path.name, {"file": path.name, "used_by_load": path.name in used}))
        base["columns"] = columns[:24]
        base["n_header_columns"] = len(columns)
        out.append(base)
    return out or catalog


def _csv_header(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            row = next(reader, [])
        return [item.strip() for item in row if item.strip()]
    except OSError:
        return []


def _mentions_field(text: str, name: str) -> bool:
    if not name:
        return False
    if f"'{name}'" in text or f'"{name}"' in text:
        return True
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


def _reads_claimed_fields_from_raw_csv(source: str, fields: tuple[str, ...]) -> bool:
    """Filename mentions alone are not enough; data.load() also names those CSVs."""
    if not fields:
        return False
    if "DictReader" not in source and "csv.reader" not in source:
        return False
    lowered = source.replace("\\", "/")
    if not any(marker in lowered for marker in _RAW_LOG_MARKERS):
        return False
    return all(_mentions_field(source, name) for name in fields)


def _lab_exposes_claimed_fields(source: str, fields: tuple[str, ...]) -> bool:
    if not fields:
        return False
    if "SplitSafeStore" not in source and "research_agent.lab" not in source:
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            called.add(func.attr)
        elif isinstance(func, ast.Name):
            called.add(func.id)
    aux_fields = tuple(name for name in fields if name not in _LAB_CONTEXT_FIELDS)
    context_fields = tuple(name for name in fields if name in _LAB_CONTEXT_FIELDS)
    if aux_fields and not called.intersection(_LAB_AUX_APIS):
        return False
    if context_fields and not called.intersection(_LAB_CONTEXT_APIS):
        return False
    if not aux_fields and not context_fields:
        return False
    return all(_mentions_field(source, name) for name in fields)
