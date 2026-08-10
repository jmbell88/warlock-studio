"""The changelog is read and parsed once per process, per path (C3).

Home draws the news block every frame, and ``entries`` was a ``read_text``
plus a full regex parse per call -- two of each per frame, because ``current``
reads through ``entries`` too. The file ships inside the wheel and cannot
change under a running process, so once per resolved path is the right
lifetime. ``parse`` stays pure and uncached: it takes a string, and the parse
tests feed it fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock import changelog

FIXTURE = "## 0.2.0 - 2026-01-02\n\n- Did a thing.\n"
OTHER = "## 9.9.9\n\n- A different file entirely.\n"


@pytest.fixture(autouse=True)
def _fresh_cache():
    changelog._CACHE.clear()
    yield
    changelog._CACHE.clear()


@pytest.fixture
def reads(monkeypatch):
    """Every path ``read_text`` touched, in order."""
    counted: list[Path] = []
    real = Path.read_text

    def wrapper(self: Path, *args, **kwargs):
        counted.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", wrapper)
    return counted


def test_entries_reads_the_file_at_most_once(tmp_path, reads):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(FIXTURE, encoding="utf-8")
    first = changelog.entries(path)
    for _ in range(3):
        assert changelog.entries(path) == first
    assert reads == [path]


def test_current_reads_through_the_same_cache(tmp_path, reads):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(FIXTURE, encoding="utf-8")
    changelog.entries(path)
    assert changelog.current("0.2.0", path).version == "0.2.0"
    assert changelog.current("9.9.9", path).version == "0.2.0"
    assert reads == [path]


def test_the_cache_is_keyed_per_path(tmp_path, reads):
    """A different path is a different file: the lifetime argument is about one
    shipped copy, not about every changelog anybody ever names."""
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text(FIXTURE, encoding="utf-8")
    b.write_text(OTHER, encoding="utf-8")
    assert changelog.entries(a)[0].version == "0.2.0"
    assert changelog.entries(b)[0].version == "9.9.9"
    assert changelog.entries(a)[0].version == "0.2.0"
    assert reads == [a, b]
