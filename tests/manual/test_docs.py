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
    "01-before-you-begin",
    "02-your-first-asset",
    "03-finding-your-work",
    "04-judging-what-you-made",
    "05-drawing",
    "06-animating",
    "07-modelling",
    "08-rigging-and-posing",
    "09-building-a-map",
    "10-packing-an-atlas",
    "11-a-character-sprite-sheet",
    "12-tuning-what-you-get",
    "13-putting-it-in-a-game",
    "14-making-a-soundtrack",
    "20-overview",
    "21-home",
    "22-generating-references",
    "23-generating-meshes",
    "24-the-3d-viewport",
    "25-rigging-and-posing",
    "26-poser",
    "27-sprite-sheets",
    "28-inker",
    "29-inker-animation",
    "30-clay",
    "31-plotter",
    "32-packwright",
    "33-troupe",
    "34-sirens",
    "35-library-and-jobs",
    "36-profiles",
    "37-review",
    "38-shortcuts",
    "39-installation",
    "40-configuration",
    "41-app-settings",
    "42-troubleshooting",
    "43-architecture",
    "44-pipelines",
    "45-extending",
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


def test_every_image_resolves():
    """The link test's sibling. A screenshot is generated rather than written,
    so the failure mode is a chapter referring to a capture nobody ran --
    which degrades to its alt text on screen and is therefore silent."""
    root = loader.manual_dir()
    for key, blocks in _all_blocks().items():
        for block in blocks:
            if not isinstance(block, parser.Image):
                continue
            assert (root / block.path).is_file(), (
                f"{key}: no image at {block.path}"
            )
            assert block.alt.strip(), f"{key}: {block.path} has no alt text"


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

    The expectation is built from the chapters that *exist*, not from the whole
    of each range. It used to be ``[n for n in rng if n <= last]``, which is the
    same thing only while every part is densely filled -- and the Tutorials part
    reserves 01-19 for a series that grows a chapter at a time, so it is not.
    Read the other way, the old form asserted that a chapter exists for every
    number below the highest one, which is a claim about the manual's size that
    this test was never meant to make; a part with nothing in it needs no
    heading, and drops out.
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

    grouped: dict[str, list[int]] = {label: [] for label, _ in loader.PARTS}
    for chapter in loader.chapters():
        if chapter.part:
            grouped[chapter.part].append(int(chapter.key[:2]))
    expected = {label: sorted(nums) for label, nums in grouped.items() if nums}
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
    # ``help_button_inline`` counts too: it is the same button and the same
    # target map, placed where the cursor is rather than right-aligned, and a
    # scan that missed it would call the viewport's (?) dead data.
    pattern = re.compile(r'help_button(?:_inline)?\(\s*ctx\s*,\s*"([^"]+)"\s*\)')
    found: set[str] = set()
    # main.py joins the scan because Review's workspace panes are drawn there
    # rather than in panes/ -- its (?) would otherwise be invisible to this
    # test in both directions.
    for path in [*(studio_dir / "panes").glob("*.py"), studio_dir / "main.py"]:
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert found == HELP_TARGETS.keys()


# --- the mode count ---------------------------------------------------------
#
# Prose, and so self-correcting by nothing. ``docs/INVARIANTS.md`` -- the
# authoritative file -- said "Ten modes" and enumerated nine plus Settings, with
# Troupe missing, while CLAUDE.md, README.md and ``modes.py`` all said eleven;
# the manual's own mode list was missing Troupe too, despite carrying a chapter
# about it. Both are pinned to the one list that decides.

_COUNT_WORDS = {
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
}


def _root():
    return Path(__file__).resolve().parents[2]


def test_both_documents_state_the_mode_count_the_rail_actually_draws():
    from warlock.studio import modes

    want = _COUNT_WORDS[len(modes.MODES)]
    invariants = (_root() / "docs" / "INVARIANTS.md").read_text(encoding="utf-8")
    overview = (_root() / "docs" / "manual" / "20-overview.md").read_text(encoding="utf-8")
    assert f"**{want.capitalize()} modes and one rail" in invariants
    assert f"chooses between {want} modes" in overview


def test_every_mode_is_named_in_the_manuals_own_list_of_them():
    """The count agreeing is not the same as the list agreeing: a chapter can
    exist for a mode the overview never mentions, which is how Troupe came to
    be documented and unlisted at the same time."""
    from warlock.studio import modes

    overview = (_root() / "docs" / "manual" / "20-overview.md").read_text(encoding="utf-8")
    section = overview.split("## The modes", 1)[1].split("\n## ", 1)[0]
    missing = [label for _key, label, _icon in modes.MODES if f"**{label}.**" not in section]
    assert not missing, f"the overview's mode list does not name {missing}"
