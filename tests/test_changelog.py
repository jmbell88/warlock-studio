"""The changelog reader Home's "What's new" block draws.

Pure and stdlib-only, so all of this runs without a window. The one test that
is not about parsing is the lockstep assertion at the bottom: the file is
hand-written, which is exactly why a release bump can leave it behind.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from warlock import changelog

REPO = Path(__file__).resolve().parents[1]

FIXTURE = """\
# Changelog

Some preamble that is neither a heading nor a bullet.

## 0.2.0 — 2026-01-02

- Did a thing.
- Did another thing.

## 0.1.0

* An older **thing**, written with a star,
  hard-wrapped onto a second line.

| a table | that is ignored |
|---|---|
"""


def test_it_parses_headings_dates_and_bullets():
    releases = changelog.parse(FIXTURE)
    assert [r.version for r in releases] == ["0.2.0", "0.1.0"]
    assert releases[0].date == "2026-01-02"
    assert releases[0].bullets == ("Did a thing.", "Did another thing.")
    # A dateless heading is a release with no date, not a failure to parse one.
    assert releases[1].date == ""
    # The continuation is folded in and the emphasis markers are gone: the file
    # is hard-wrapped like everything else here, and imgui draws one weight.
    assert releases[1].bullets == (
        "An older thing, written with a star, hard-wrapped onto a second line.",
    )


def test_anything_that_is_not_a_heading_or_a_bullet_is_ignored():
    """Forgiving in one direction only. The file has to be able to carry a
    preamble and a table without this becoming a Markdown implementation."""
    assert changelog.parse("just prose\n\nmore prose\n") == []


def test_unindented_prose_after_a_bullet_is_not_folded_into_it():
    """Indentation in the *raw* line decides. A paragraph the file is allowed to
    carry, folded into the bullet above it, is how a note becomes a release
    note nobody wrote."""
    releases = changelog.parse("## 1.0.0\n\n- a bullet\n\nA paragraph.\n")
    assert releases[0].bullets == ("a bullet",)


def test_a_bullet_before_any_heading_belongs_to_no_release():
    """It is dropped rather than attached to the first heading that comes
    along: guessing which release an orphan belongs to is how a line ends up
    advertised under a version that never shipped."""
    releases = changelog.parse("- orphan\n\n## 1.0.0\n\n- real\n")
    assert len(releases) == 1
    assert releases[0].bullets == ("real",)


def test_a_missing_file_is_an_empty_list_rather_than_an_exception(tmp_path):
    """``[]`` for every way of having nothing to say -- Home draws this, and a
    screen the app opens on may not be the thing that fails to start."""
    assert changelog.entries(tmp_path / "nope.md") == []
    assert changelog.current("0.0.1", tmp_path / "nope.md") is None


def test_current_prefers_the_named_version_and_falls_back_to_the_newest(tmp_path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(FIXTURE, encoding="utf-8")
    assert changelog.current("0.1.0", path).version == "0.1.0"
    # A dev checkout routinely runs a version the file has not been written for
    # yet, and "what changed recently" is still the right answer there.
    assert changelog.current("9.9.9", path).version == "0.2.0"


def test_the_shipped_file_parses_and_leads_with_this_version():
    """The lockstep assertion. ``pyproject.toml`` is the manifest the version
    lives in, and nothing derives this file from it, so a bump that did not
    touch the changelog would ship a build whose "What's new" is the previous
    release's."""
    version = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    version = version["project"]["version"]
    releases = changelog.entries(REPO / "CHANGELOG.md")
    assert releases, "the repo's CHANGELOG.md must parse to at least one release"
    assert releases[0].version == version
    assert releases[0].bullets


def test_the_reader_finds_the_repo_copy_in_a_dev_checkout():
    """``changelog_path`` mirrors ``manual.loader.manual_dir``: packaged copy
    first, repo root as the dev fallback. Either answer is a real file here."""
    assert changelog.changelog_path().is_file()
    assert changelog.entries()


def test_it_imports_nothing_from_the_app():
    """Pure in the ``vram.py``/``memlog.py`` sense: no ``service``, no
    ``queue``, no ``studio``."""
    source = (REPO / "src" / "warlock" / "changelog.py").read_text(encoding="utf-8")
    for banned in ("service", "queue", "studio"):
        assert f"import {banned}" not in source
        assert f"from .{banned}" not in source
