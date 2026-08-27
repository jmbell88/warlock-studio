"""Loader units against a temp manual tree, plus the dev-checkout fallback."""

from pathlib import Path

import pytest

from warlock.studio.manual import loader
from warlock.studio.state import AppState, ManualState


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "00-index.md").write_text("# Warlock Studio Manual\n", encoding="utf-8")
    (tmp_path / "20-overview.md").write_text("# Overview\n\nHello.\n", encoding="utf-8")
    (tmp_path / "39-installation.md").write_text("# Installation\n", encoding="utf-8")
    (tmp_path / "43-architecture.md").write_text("# Architecture\n", encoding="utf-8")
    return tmp_path


def test_chapters_sorted_titled_and_grouped(tree: Path):
    chapters = loader.chapters(root=tree)
    assert [c.key for c in chapters] == [
        "00-index", "20-overview", "39-installation", "43-architecture",
    ]
    by_key = {c.key: c for c in chapters}
    assert by_key["20-overview"].title == "Overview"
    assert by_key["00-index"].part == ""
    assert by_key["20-overview"].part == "Using Warlock Studio"
    assert by_key["39-installation"].part == "Setup & operations"
    assert by_key["43-architecture"].part == "Architecture"


def test_every_real_chapter_lands_in_a_part():
    """Against the real manual, not the synthetic tree: a chapter outside
    every PARTS range renders in the TOC as a bare "Contents" entry, which is
    the fallback meant only for 00-index."""
    orphans = [c.key for c in loader.chapters() if c.number > 0 and not c.part]
    assert orphans == []


def test_load_reads_the_file(tree: Path):
    assert "Hello." in loader.load("20-overview", root=tree)


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
    """Navigation only: ``open_at`` says *where*, never *whether*.

    The visibility flag beside it is new (the UI redesign, wave 3) and this test is
    the record of the inversion: while the Manual was a mode, a flag here would
    have been a second answer to a question ``state.mode`` already answered.
    There is no mode to disagree with now, and ``open_at`` still leaves it
    alone -- ``render.open_at`` is what raises the overlay, so a caller that
    only wants to *pre-position* the reader (a deep link followed while it is
    already up) does not force it open.
    """
    state = AppState()
    assert isinstance(state.manual, ManualState)
    assert state.manual.open is False
    state.manual.open_at("23-generating-meshes", "exports")
    assert state.manual.open is False
    assert state.manual.chapter == "23-generating-meshes"
    assert state.manual.pending_anchor == "exports"
