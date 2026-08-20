"""Parser units: the subset parses, everything else raises."""

import pytest

from warlock.studio.manual import parser
from warlock.studio.manual.parser import (
    CodeBlock,
    Heading,
    ListItem,
    ManualSyntaxError,
    Paragraph,
    Span,
    Table,
)


def test_slugify_matches_github_style():
    assert parser.slugify("Triangle budget") == "triangle-budget"
    assert parser.slugify("What `doctor` checks") == "what-doctor-checks"
    assert parser.slugify("VRAM modes (exclusive)") == "vram-modes-exclusive"


def test_heading_levels_and_anchor():
    blocks = parser.parse("# Title\n\n## Sub section\n")
    assert blocks == [
        Heading(1, "Title", "title"),
        Heading(2, "Sub section", "sub-section"),
    ]


def test_paragraph_joins_lines_and_parses_spans():
    blocks = parser.parse("Plain **bold** and *soft* and `code`\nnext line.\n")
    [para] = blocks
    assert isinstance(para, Paragraph)
    assert para.spans == (
        Span("text", "Plain "),
        Span("bold", "bold"),
        Span("text", " and "),
        Span("italic", "soft"),
        Span("text", " and "),
        Span("code", "code"),
        Span("text", " next line."),
    )


def test_link_span_carries_target():
    [para] = parser.parse("See [Rigging](06-rigging-and-posing.md#templates).\n")
    assert Span("link", "Rigging", "06-rigging-and-posing.md#templates") in para.spans


def test_code_block():
    blocks = parser.parse("```powershell\nuv run warlock\n```\n")
    assert blocks == [CodeBlock("uv run warlock", "powershell")]


def test_lists_keep_marker_depth_and_order():
    blocks = parser.parse("- one\n  - nested\n1. first\n")
    assert blocks == [
        ListItem(0, False, "-", (Span("text", "one"),)),
        ListItem(1, False, "-", (Span("text", "nested"),)),
        ListItem(0, True, "1.", (Span("text", "first"),)),
    ]


def test_table():
    blocks = parser.parse("| Key | Effect |\n|---|---|\n| `a` | does a |\n")
    [table] = blocks
    assert isinstance(table, Table)
    assert table.header == ((Span("text", "Key"),), (Span("text", "Effect"),))
    assert table.rows == (((Span("code", "a"),), (Span("text", "does a"),)),)


@pytest.mark.parametrize(
    "source, fragment",
    [
        ("![](x.png)\n", "alt text"),
        ("text ![alt](x.png) more\n", "line of its own"),
        ("a <div> b\n", "subset"),
        ("> quoted\n", "blockquote"),
        ("unclosed **bold\n", "unclosed"),
        ("unclosed `code\n", "unclosed"),
        ("[text](no-close\n", "link"),
        ("```py\nnever closed\n", "fence"),
        ("| a | b |\n| c | d |\n", "separator"),
        ("   - three-space indent\n", "indent"),
    ],
)
def test_violations_raise(source, fragment):
    with pytest.raises(ManualSyntaxError) as err:
        parser.parse(source)
    assert fragment in str(err.value)


def test_error_carries_line_number():
    with pytest.raises(ManualSyntaxError) as err:
        parser.parse("fine\n\n> bad\n")
    assert err.value.line_no == 3


# --- the search reads the manual, not its table of contents -------------------


def _chapter(key: str = "09-inker", title: str = "Inker"):
    from types import SimpleNamespace

    return SimpleNamespace(key=key, title=title, part="Editors")


def test_a_phrase_that_appears_only_in_prose_finds_its_chapter(monkeypatch):
    """The search used to walk headings only, which makes a manual searchable
    by its table of contents and no more: "gltfpack", "prompt_hash" and
    "WARLOCK_VRAM_BUDGET" are each named in a paragraph and in no heading
    anywhere, so the three strings a reader is most likely to arrive with found
    nothing at all."""
    from warlock.studio.manual import loader, render

    blocks = render._blocks("09-inker")
    monkeypatch.setattr(render, "_blocks", lambda key: blocks)
    prose = " ".join(loader.block_text(b) for b in blocks if not _is_heading(b))
    word = next(w for w in prose.split() if len(w) > 9 and w.isalpha())
    headings = " ".join(loader.block_text(b) for b in blocks if _is_heading(b))
    assert word.lower() not in headings.lower(), "pick a word the headings do not carry"
    assert render._matches(_chapter(), word.lower()) is True


def _is_heading(block) -> bool:
    from warlock.studio.manual import parser

    return isinstance(block, parser.Heading)


def test_every_block_type_contributes_its_text():
    """One function over all five, so a block type added later is searchable by
    having been added rather than by somebody remembering this file.

    It lives in ``loader`` rather than ``render`` because the chapter search and
    the TOC tree's section search both read it -- see ``loader.block_text``.
    """
    from warlock.studio.manual import loader, parser

    span = parser.Span(kind="text", text="findable")
    cases = [
        parser.Heading(level=2, text="findable", anchor="findable"),
        parser.CodeBlock(text="findable --flag", lang="text"),
        parser.Paragraph(spans=(span,)),
        parser.ListItem(depth=0, ordered=False, marker="-", spans=(span,)),
        parser.Table(header=((span,),), rows=(((span,),),)),
    ]
    for block in cases:
        assert "findable" in loader.block_text(block), type(block).__name__


# --- images -------------------------------------------------------------------


def test_an_image_on_its_own_line_is_a_block():
    """The manual could describe a control it could not show you, which for a
    chapter about a toolbar is the wrong half to have."""
    blocks = parser.parse("![The Inker toolbar](img/08-toolbar.png)\n")
    assert blocks == [parser.Image("The Inker toolbar", "img/08-toolbar.png")]


def test_an_image_breaks_the_paragraph_around_it():
    blocks = parser.parse("before\n![shot](a.png)\nafter\n")
    kinds = [type(b).__name__ for b in blocks]
    assert kinds == ["Paragraph", "Image", "Paragraph"]


def test_html_is_still_outside_the_subset():
    """Only images left the forbidden set."""
    with pytest.raises(ManualSyntaxError) as err:
        parser.parse("a <div> b\n")
    assert "HTML" in str(err.value)
