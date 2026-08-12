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
# ``docs/TODO.md`` sat here until 2026-08-11, when the roadmap file was deleted
# outright (`de87838`; there is no roadmap file now -- see docs/INVARIANTS.md).
# Its entry is gone rather than commented into the tuple, because a declared
# source that does not exist used to be *silent*: ``_manual_links`` skipped a
# missing path, so the link count simply fell, and the global-count guard below
# was slack enough to absorb it. That is the exact silent-shrink failure this
# file's own comments warn about, and it is why
# ``test_every_declared_source_exists`` now fails on the missing file itself
# rather than leaving the shrink to be inferred from a total.
SOURCES = (
    ROOT / "README.md",
    ROOT / "CLAUDE.md",
    # The invariants reference CLAUDE.md's detail moved into on 2026-08-10. It
    # carries no manual mentions yet, but it is exactly the file that will grow
    # them, and the rule here is that sources are named, not globbed.
    ROOT / "docs" / "INVARIANTS.md",
    # The optional-model catalogue the README's download sections moved into on
    # 2026-08-10. It points readers at the guidance-panel chapter, so its links
    # rot with a renumbering exactly the way README.md's do.
    ROOT / "docs" / "MODELS.md",
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
# A mention is resolved relative to the file that carries it, exactly as a link
# is. It used to be treated as repo-root-relative, which was indistinguishable
# from correct only while every source lived at the repo root: `docs/LEFTOVERS.md`
# moved into `docs/` on 2026-08-09 and correctly re-spelled its citation as
# `manual/16-configuration.md`, which the old root-anchored pattern then stopped
# matching altogether -- a silent loss of coverage rather than a failure, which is
# what the guard-on-the-guard below exists to turn into a failure.
#
# The optional prefix is what keeps both spellings honest: `README.md` and
# `CLAUDE.md` are at the root and say `docs/manual/...`, a measurement document
# says `../manual/...`, and each resolves against its own parent to the same
# place. Anchoring on the opening backtick is load-bearing -- it is why
# `tests/manual/test_docs.py` is not mistaken for a manual chapter.
MENTION = re.compile(r"`((?:\.\./|docs/)?manual/[0-9A-Za-z._/-]+\.md(?:#[\w-]+)?)`")


# The outward half again, one level up: a citation of a *non-manual* document.
#
# The manual checks above only ever look at ``manual/...`` targets, so a source
# could cite `docs/CAMERA.md` -- a file that never existed in any of this repo's
# commits -- and three documents did, for months. The same shape produced live
# citations of `../LEFTOVERS.md`, which did exist and was deleted.
#
# Those two cases want opposite treatment, so the rule is: a cited document must
# either exist, or the citing line must *say* it is gone. Anything containing
# "deleted" on the same line is taken as that admission -- which is what the
# annotated LEFTOVERS cites and INVARIANTS' roll-call of executed plans already
# say in prose. A dead citation with no such word is a reader sent to nowhere.
#
# ``memory`` is the second exemption and a different kind: CLAUDE.md and
# INVARIANTS.md both point at `warlock-stack.md`, which is a note in the agent's
# own memory store rather than a file in this repo. A line that says "memory" is
# citing that store, and there is nothing here to resolve it against.
DEAD_CITE_MARKER = "deleted"
EXTERNAL_STORE_MARKER = "memory"
DOC_MENTION = re.compile(r"`((?:\.\./|\./|docs/)?[0-9A-Za-z._/-]+\.md)`")


def _doc_citations() -> list[tuple[Path, int, str]]:
    """(source, 1-based line, target) for every backticked or linked ``.md``
    citation that is not a manual chapter and not marked as deleted."""
    found: list[tuple[Path, int, str]] = []
    for source in SOURCES:
        if not source.exists():
            continue
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            if DEAD_CITE_MARKER in lowered or EXTERNAL_STORE_MARKER in lowered:
                continue
            targets = list(DOC_MENTION.findall(line))
            targets += [t for t in LINK.findall(line) if t.endswith(".md")]
            for target in dict.fromkeys(targets):
                if "manual/" in target:
                    continue  # the tests above own these
                found.append((source, number, target))
    return found


def test_the_document_citation_sweep_finds_something():
    """The same guard-on-the-guard the manual sweep gets, for the same reason:
    if ``DOC_MENTION`` or either exemption ever widens far enough to match
    nothing, every case below passes vacuously and the sweep says nothing."""
    cites = _doc_citations()
    assert len(cites) >= 10, f"expected many .md citations across SOURCES, found {cites}"


@pytest.mark.parametrize(
    "source,line,target",
    _doc_citations(),
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_a_cited_document_exists_or_says_it_is_gone(source: Path, line: int, target: str):
    path, _anchor = _resolve(source, target)
    if not path.exists():
        path = (ROOT / target.lstrip("./")).resolve()
    assert path.exists(), (
        f"{source.name}:{line} cites `{target}`, which does not exist. Either fix "
        f"the path, or -- if the document was deleted -- say so on the same line "
        f"(the word '{DEAD_CITE_MARKER}' is the marker) so a reader knows to look "
        f"in git history instead of hunting for a file that is not there."
    )


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
        # No rewriting: a mention resolves against its own file's parent, which
        # is what ``_resolve`` already does for a link. One rule for both shapes.
        targets.extend(MENTION.findall(text))
        found.extend((source, t) for t in dict.fromkeys(targets))
    return found


def _resolve(source: Path, target: Path | str) -> tuple[Path, str]:
    """A link target -> (the file it names, the anchor it names or "")."""
    raw, _, anchor = str(target).partition("#")
    return (source.parent / raw).resolve(), anchor


# The named sources that are expected to point readers into the manual. Not
# every declared source is here, and the difference is deliberate:
# ``docs/INVARIANTS.md`` and ``CLAUDE.md`` are declared because they are exactly
# the files that will grow manual links, but neither carries one today, and
# asserting one would be asserting a wish rather than a fact. The measurement
# documents are globbed and each is free to carry none. Everything in this set
# has links today, so a file that loses its last one is a regression rather
# than an edit.
LINKED_SOURCES = (
    ROOT / "README.md",
    ROOT / "docs" / "MODELS.md",
)


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.name)
def test_every_declared_source_exists(source: Path):
    """A declared source that is missing must fail *here*, not shrink coverage.

    ``docs/TODO.md`` was deleted on 2026-08-11 while still named in ``SOURCES``,
    and nothing failed: ``_manual_links`` skips a path that does not exist, so
    the only symptom was a lower total that the count guard below still passed.
    A source is either present and walked, or removed from the tuple on purpose.
    """
    assert source.exists(), (
        f"{source} is declared in SOURCES but does not exist. Either restore it "
        f"or delete its entry -- a named source that silently vanishes is how "
        f"this file stops covering what it claims to cover."
    )


def test_the_sources_actually_carry_manual_links():
    """A guard on the guard: if the regex or the filter ever stops matching,
    every assertion below passes vacuously and says nothing."""
    links = _manual_links()
    assert len(links) >= 3, f"expected several links into docs/manual/, found {links}"


@pytest.mark.parametrize("source", LINKED_SOURCES, ids=lambda p: p.name)
def test_each_linking_source_still_carries_its_own_links(source: Path):
    """Per-source, because a global count hides a file going quiet.

    The total above is satisfied by three links from one file; this is what
    notices that README stopped linking into the manual while MODELS.md grew
    two more.
    """
    mine = [target for src, target in _manual_links() if src == source]
    assert mine, f"{source.name} carries no link or mention of docs/manual/ any more"


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
