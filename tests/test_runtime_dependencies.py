from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fetch_worker_dependency_is_direct() -> None:
    """Model fetching must not rely on an optional ML extra pulling Hub in."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {
        entry.split("[", 1)[0].split(";", 1)[0].split(">", 1)[0].lower()
        for entry in project["dependencies"]
    }
    assert "huggingface-hub" in names
