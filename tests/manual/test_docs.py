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
    "02-generating-references",
    "03-generating-meshes",
    "04-rigging-and-posing",
    "05-sprite-sheets",
    "06-paint",
    "07-library-and-jobs",
    "08-shortcuts",
    "09-installation",
    "10-configuration",
    "11-troubleshooting",
    "12-architecture",
    "13-pipelines",
    "14-extending",
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

    panes_dir = Path(__file__).resolve().parents[2] / "src/warlock/studio/panes"
    pattern = re.compile(r'help_button\(\s*ctx\s*,\s*"([^"]+)"\s*\)')
    found: set[str] = set()
    for path in panes_dir.glob("*.py"):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert found == HELP_TARGETS.keys()
