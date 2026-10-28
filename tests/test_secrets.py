"""Fail if any API key from .env appears in any project file other than .env."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}


def _secrets() -> list[str]:
    values = dotenv_values(ROOT / ".env")
    return [v for k, v in values.items() if k.endswith("_KEY") and v]


def _project_files() -> list[Path]:
    return [
        f for f in ROOT.rglob("*")
        if f.is_file() and f.name != ".env" and not SKIP_DIRS.intersection(f.relative_to(ROOT).parts)
    ]


def test_no_key_outside_env() -> None:
    secrets = _secrets()
    if not secrets:
        pytest.skip("no keys in .env")
    leaks = []
    for f in _project_files():
        content = f.read_bytes()
        if any(s.encode() in content for s in secrets):
            leaks.append(str(f.relative_to(ROOT)))
    assert not leaks, f"API key found in: {leaks}"


def test_env_is_gitignored() -> None:
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines
