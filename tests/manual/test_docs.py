"""The manual's own gate: every chapter parses, every link and anchor resolves.

This is the test that makes the parser's strictness useful -- a chapter that
drifts outside the subset, or a link that dangles, fails CI rather than
rendering wrong in the app or on GitHub.
"""

import re
from pathlib import Path

from warlock.studio.manual import loader, parser

EXPECTED_KEYS = [
    "00-index",
    "01-overview",
    "02-home",
    "03-generating-references",
    "04-generating-meshes",
    "05-rigging-and-posing",
    "06-poser",
    "07-sprite-sheets",
    "08-inker",
    "09-inker-animation",
    "10-clay",
    "11-plotter",
    "12-packwright",
    "13-library-and-jobs",
    "14-profiles",
    "15-review",
    "16-shortcuts",
    "17-installation",
    "18-configuration",
    "19-app-settings",
    "20-troubleshooting",
    "21-architecture",
    "22-pipelines",
    "23-extending",
]


def _all_blocks():
    return {c.key: parser.parse(loader.load(c.key)) for c in loader.chapters()}


def _spans(blocks):
    for block in blocks:
        if isinstance(block, parser.Paragraph | parser.ListItem):
            yield from block.spans
        elif isinstance(block, parser.Table):
            for row in (*block.header, *(cell for r in block.rows for cell in r)):
                yield from row


def test_every_expected_chapter_exists_and_parses():
    assert [c.key for c in loader.chapters()] == EXPECTED_KEYS
    _all_blocks()  # parse() raising is the failure


def test_every_chapter_opens_with_a_single_h1():
    for key, blocks in _all_blocks().items():
        h1s = [b for b in blocks if isinstance(b, parser.Heading) and b.level == 1]
        assert len(h1s) == 1, f"{key}: exactly one H1 required"
        assert isinstance(blocks[0], parser.Heading), f"{key}: must open with its H1"


def test_cross_links_resolve():
    anchors = {
        key: {b.anchor for b in blocks if isinstance(b, parser.Heading)}
        for key, blocks in _all_blocks().items()
    }
    for key, blocks in _all_blocks().items():
        for span in _spans(blocks):
            if span.kind != "link" or span.target.startswith(("http://", "https://")):
                continue
            page, _, fragment = span.target.partition("#")
            target_key = page.removesuffix(".md") if page else key
            assert target_key in anchors, f"{key}: link to unknown chapter {span.target}"
            if fragment:
                assert fragment in anchors[target_key], (
                    f"{key}: link to missing anchor {span.target}"
                )


def test_index_links_every_chapter():
    text = loader.load("00-index")
    linked = set(re.findall(r"\((\d\d-[\w-]+)\.md", text))
    assert linked == set(EXPECTED_KEYS) - {"00-index"}


def test_index_sections_match_the_loaders_parts():
    """A chapter's number decides its order *and* its part, in both directions.

    CLAUDE.md has claimed this gate covers grouping for as long as it has
    claimed the number decides the part, but ``test_index_links_every_chapter``
    only ever asserted linkage -- so the index could list a chapter under one
    heading while ``loader.PARTS`` filed it under another, which is exactly what
    chapter 14 did (index: Part I; PARTS: "Setup & operations"). Asserting the
    index's own ``##`` sections against PARTS closes that direction.
    """
    text = loader.load("00-index")
    index: dict[str, list[int]] = {}
    section: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            index.setdefault(section, [])
        elif section is not None:
            found = re.search(r"\((\d\d)-[\w-]+\.md", line)
            if found:
                index[section].append(int(found.group(1)))

    last = max(int(c.key[:2]) for c in loader.chapters())
    expected = {label: [n for n in rng if n <= last] for label, rng in loader.PARTS}
    assert index == expected, (
        "docs/manual/00-index.md's sections and loader.PARTS disagree about "
        "which part a chapter belongs to. Move the index entry or widen the "
        "range -- but not only one of them."
    )


def test_help_targets_resolve():
    from warlock.studio.manual.targets import HELP_TARGETS

    assert HELP_TARGETS, "the context-help map must not be empty"
    anchors = {
        key: {b.anchor for b in blocks if isinstance(b, parser.Heading)}
        for key, blocks in _all_blocks().items()
    }
    for pane, (chapter, anchor) in HELP_TARGETS.items():
        assert chapter in anchors, f"{pane}: unknown chapter {chapter}"
        if anchor is not None:
            assert anchor in anchors[chapter], f"{pane}: missing anchor {chapter}#{anchor}"


def test_no_prose_line_runs_past_the_wrap_column():
    """Source lines stop where the renderer does.

    ``render.MAX_LINE_CHARS`` is the column the app wraps prose at, so a source
    line longer than that is one nobody reads at either end. Deliberately
    *prose only*: a table row and a list item cannot be split across source
    lines in this subset -- the parser joins a paragraph's lines with a space
    but starts a new block on anything else -- so a rule over them would be a
    rule about rendering, not about readability. Code fences are verbatim.
    """
    limit = 120
    for chapter in loader.chapters():
        fenced = False
        for number, line in enumerate(loader.load(chapter.key).split("\n"), 1):
            if line.startswith("```"):
                fenced = not fenced
                continue
            stripped = line.lstrip()
            if fenced or stripped.startswith(("|", "-", "#")) or re.match(r"\d+\. ", stripped):
                continue
            assert len(line) <= limit, (
                f"{chapter.key}:{number}: {len(line)} characters; wrap prose at {limit}"
            )


def test_help_button_call_sites_match_help_targets():
    """Guards the HELP_TARGETS <-> pane call-site seam.

    render.help_button silently no-ops on an unknown key, so a typo'd key in a
    pane would drop its (?) button with no test failure anywhere else. Scan
    every pane source file for the literal keys passed to help_button and
    check the set against HELP_TARGETS exactly, in both directions: a pane key
    with no HELP_TARGETS entry is a dead button, and a HELP_TARGETS entry with
    no call site is dead data.
    """
    from warlock.studio.manual.targets import HELP_TARGETS

    studio_dir = Path(__file__).resolve().parents[2] / "src/warlock/studio"
    pattern = re.compile(r'help_button\(\s*ctx\s*,\s*"([^"]+)"\s*\)')
    found: set[str] = set()
    # main.py joins the scan because Review's workspace panes are drawn there
    # rather than in panes/ -- its (?) would otherwise be invisible to this
    # test in both directions.
    for path in [*(studio_dir / "panes").glob("*.py"), studio_dir / "main.py"]:
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert found == HELP_TARGETS.keys()
