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
    [para] = parser.parse("See [Rigging](05-rigging-and-posing.md#templates).\n")
    assert Span("link", "Rigging", "05-rigging-and-posing.md#templates") in para.spans


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
        ("![alt](x.png)\n", "subset"),
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
