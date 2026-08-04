"""Loader units against a temp manual tree, plus the dev-checkout fallback."""

from pathlib import Path

import pytest

from warlock.studio.manual import loader
from warlock.studio.state import AppState, ManualState


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "00-index.md").write_text("# Warlock Studio Manual\n", encoding="utf-8")
    (tmp_path / "01-overview.md").write_text("# Overview\n\nHello.\n", encoding="utf-8")
    (tmp_path / "09-installation.md").write_text("# Installation\n", encoding="utf-8")
    (tmp_path / "12-architecture.md").write_text("# Architecture\n", encoding="utf-8")
    return tmp_path


def test_chapters_sorted_titled_and_grouped(tree: Path):
    chapters = loader.chapters(root=tree)
    assert [c.key for c in chapters] == [
        "00-index", "01-overview", "09-installation", "12-architecture",
    ]
    by_key = {c.key: c for c in chapters}
    assert by_key["01-overview"].title == "Overview"
    assert by_key["00-index"].part == ""
    assert by_key["01-overview"].part == "Using Warlock Studio"
    assert by_key["09-installation"].part == "Setup & operations"
    assert by_key["12-architecture"].part == "Architecture"


def test_load_reads_the_file(tree: Path):
    assert "Hello." in loader.load("01-overview", root=tree)


def test_load_rejects_unknown_key(tree: Path):
    with pytest.raises(KeyError):
        loader.load("99-nope", root=tree)
    with pytest.raises(KeyError):
        loader.load("../../etc/passwd", root=tree)


def test_dev_checkout_fallback_finds_repo_docs():
    # In this checkout warlock/manual is not packaged, so manual_dir() must
    # resolve to <repo>/docs/manual.
    assert loader.manual_dir().name == "manual"
    assert loader.manual_dir().parent.name == "docs"


def test_manual_state_open_at():
    """Navigation only. Whether the manual is on screen is ``state.mode``, so
    open_at deliberately does not carry a second visibility flag."""
    state = AppState()
    assert isinstance(state.manual, ManualState)
    state.manual.open_at("03-generating-meshes", "exports")
    assert not hasattr(state.manual, "open")
    assert state.manual.chapter == "03-generating-meshes"
    assert state.manual.pending_anchor == "exports"
