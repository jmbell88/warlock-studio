"""Links *into* the manual from outside it, which nothing else was checking.

``tests/manual/test_docs.py`` walks the manual's own links and anchors and is
strict about them -- but it only ever opens files under ``docs/manual/``. Every
link that points *at* the manual from the repo root or from a measurement
document was therefore unchecked, and both kinds had rotted the same way: the
manual has been renumbered, and `README.md` and `LEFTOVERS.md` were still
sending readers to ``docs/manual/14-configuration.md``, which is the *shortcuts*
chapter. LEFTOVERS presented that one as a citation it had already corrected,
which is the sharpest version of the problem: a correction is not exempt from
the rule it enforces.

So this is the outward half of the same gate. A renumbering now fails here
rather than silently pointing a reader at the wrong chapter, and it fails at the
file that carries the stale link rather than inside the manual.

Anchors are checked as well as filenames, because a chapter can survive a
renumbering with its headings rewritten -- ``#optional-image-models-and-style-loras``
is exactly such a link, from a measurement document, and it is the kind that
lands the reader on the right page and the wrong part of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "manual"

# The files outside docs/manual/ that are allowed to link into it. Named rather
# than globbed: a glob would quietly stop covering a file that was renamed, and
# the whole point here is that a link nobody walks is a link that rots.
SOURCES = (
    ROOT / "README.md",
    ROOT / "LEFTOVERS.md",
    ROOT / "CLAUDE.md",
    *sorted((ROOT / "docs" / "measurements").glob("*.md")),
)

# Two shapes, and the second is the one that caused this test to exist.
#
# ``LINK`` is an inline markdown link. Reference-style links are used nowhere in
# these files and adding one should be a deliberate act.
#
# ``MENTION`` is a bare backticked path. LEFTOVERS' broken
# ``docs/manual/14-configuration.md`` was one of these, not a link -- so a
# checker that only walked ``[](...)`` would have missed the very citation the
# file presents as already corrected. A path a reader is expected to open is a
# link whether or not it has brackets round it.
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
MENTION = re.compile(r"`(docs/manual/[0-9A-Za-z._/-]+\.md(?:#[\w-]+)?)`")


def _slug(heading: str) -> str:
    """GitHub's anchor slug, which is also what the in-app manual generates."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text)


def _anchors(path: Path) -> set[str]:
    return {
        _slug(line.lstrip("#"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    }


def _manual_links() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for source in SOURCES:
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        targets = [t for t in LINK.findall(text) if "manual/" in t]
        # A backticked mention is repo-root-relative wherever it appears, so it
        # is rewritten to be relative to the file that carries it before
        # resolution -- ``_resolve`` joins against ``source.parent``.
        for mention in MENTION.findall(text):
            rel = (ROOT / mention.split("#")[0]).resolve()
            anchor = mention.partition("#")[2]
            spelled = rel.as_posix() + (f"#{anchor}" if anchor else "")
            targets.append(spelled)
        found.extend((source, t) for t in dict.fromkeys(targets))
    return found


def _resolve(source: Path, target: Path | str) -> tuple[Path, str]:
    """A link target -> (the file it names, the anchor it names or "")."""
    raw, _, anchor = str(target).partition("#")
    return (source.parent / raw).resolve(), anchor


def test_the_sources_actually_carry_manual_links():
    """A guard on the guard: if the regex or the filter ever stops matching,
    every assertion below passes vacuously and says nothing."""
    links = _manual_links()
    assert len(links) >= 3, f"expected several links into docs/manual/, found {links}"


@pytest.mark.parametrize("source,target", _manual_links(), ids=str)
def test_a_link_into_the_manual_names_a_chapter_that_exists(source: Path, target: str):
    path, _anchor = _resolve(source, target)
    assert path.exists(), (
        f"{source.name} links to {target}, which does not exist. "
        f"The manual is numbered 00-index..21-extending -- check whether the "
        f"chapter has been renumbered."
    )
    assert path.parent == MANUAL.resolve(), (
        f"{source.name} links to {target}, which is outside docs/manual/"
    )


@pytest.mark.parametrize("source,target", _manual_links(), ids=str)
def test_a_link_into_the_manual_names_an_anchor_that_exists(source: Path, target: str):
    path, anchor = _resolve(source, target)
    if not anchor or not path.exists():
        pytest.skip("no anchor, or the filename check already covers it")
    assert anchor in _anchors(path), (
        f"{source.name} links to {target}, but {path.name} has no heading "
        f"slugging to '{anchor}'. Its headings slug to: {sorted(_anchors(path))}"
    )
