"""Shared paths for official KuaiRand starter tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter" / "kuairand"
DEFAULT_DATA_DIR = STARTER / "KuaiRand-Pure" / "data"
EVALUATE_PY = STARTER / "evaluate.py"
# SHA-256 of the committed LF blob. Normalize CRLF so Windows autocrlf checkouts match.
EVALUATE_SHA256 = "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"


def evaluate_py_canonical_bytes() -> bytes:
    return EVALUATE_PY.read_bytes().replace(b"\r\n", b"\n")

if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))


def resolve_data_dir() -> Path:
    env = os.environ.get("KUAI_RAND_DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


@pytest.fixture(scope="session")
def kuairand_splits():
    data_dir = resolve_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"KuaiRand-Pure data not present at {data_dir}")
    from data import load

    return load(str(data_dir))
