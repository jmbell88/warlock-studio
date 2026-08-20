"""The manual's markdown subset, parsed to typed blocks.

Pure on purpose -- no imgui, no filesystem -- so every rule about what the
manual may contain is assertable headlessly, the same bargain the paint
engine makes. The parser is strict: a construct outside the subset raises
rather than rendering wrong, which is the mechanism that keeps the files
honest on GitHub and in-app alike.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class ManualSyntaxError(ValueError):
    def __init__(self, line_no: int, message: str) -> None:
        super().__init__(f"line {line_no}: {message}")
        self.line_no = line_no


@dataclass(frozen=True)
class Span:
    kind: str  # text | bold | italic | code | link
    text: str
    target: str = ""


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    anchor: str


@dataclass(frozen=True)
class Paragraph:
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class CodeBlock:
    text: str
    lang: str


@dataclass(frozen=True)
class ListItem:
    depth: int
    ordered: bool
    marker: str  # the literal "-" or "3." -- render shows source numbering
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class Image:
    """``![alt](path)`` on a line of its own.

    ``path`` is relative to the chapter, and resolving it is the renderer's
    job -- the parser is a pure function of text and has no filesystem. Alt
    text is required rather than optional: it is what the block degrades to
    when the file is missing, which is the state a fresh checkout of the
    docs is in until the screenshots are generated.
    """

    alt: str
    path: str


@dataclass(frozen=True)
class Table:
    header: tuple[tuple[Span, ...], ...]
    rows: tuple[tuple[tuple[Span, ...], ...], ...]


Block = Heading | Paragraph | CodeBlock | Image | ListItem | Table


def slugify(text: str) -> str:
    """GitHub-style anchor: lowercase, punctuation dropped, spaces to hyphens."""
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"\s+", "-", slug)


_TOKEN = re.compile(r"(\*\*|\*|`|\[)")


def parse_spans(text: str, line_no: int) -> tuple[Span, ...]:
    spans: list[Span] = []

    def emit(kind: str, value: str, target: str = "") -> None:
        if value:
            spans.append(Span(kind, value, target))

    i = 0
    while i < len(text):
        match = _TOKEN.search(text, i)
        if match is None:
            emit("text", text[i:])
            break
        emit("text", text[i : match.start()])
        token, j = match.group(1), match.end()
        if token == "`":
            end = text.find("`", j)
            if end < 0:
                raise ManualSyntaxError(line_no, "unclosed ` code span")
            emit("code", text[j:end])
            i = end + 1
        elif token == "**":
            end = text.find("**", j)
            if end < 0:
                raise ManualSyntaxError(line_no, "unclosed ** bold")
            emit("bold", text[j:end])
            i = end + 2
        elif token == "*":
            end = text.find("*", j)
            if end < 0:
                raise ManualSyntaxError(line_no, "unclosed * italic")
            emit("italic", text[j:end])
            i = end + 1
        else:  # [
            close = text.find("](", j)
            end = text.find(")", close + 2) if close >= 0 else -1
            if close < 0 or end < 0:
                raise ManualSyntaxError(line_no, "malformed link")
            emit("link", text[j:close], text[close + 2 : end])
            i = end + 1
    return tuple(spans)


_FENCE = re.compile(r"^```(\w*)\s*$")
_HEADING = re.compile(r"^(#{1,4}) +(.+)$")
_ULIST = re.compile(r"^( *)- +(.*)$")
_OLIST = re.compile(r"^( *)(\d+\.) +(.*)$")
_SEP = re.compile(r"^\|(?: *:?-{3,}:? *\|)+ *$")
# Inline HTML only. Images used to be here too -- the manual could describe
# a control it could not show you, which for a chapter about a *toolbar* is
# the wrong half to have. They are a block of their own now, and still only
# as a whole line: an image inside a sentence is a layout problem the manual
# renderer has no answer for.
_FORBIDDEN = re.compile(r"<[A-Za-z!/]")
_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")


def parse(text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = text.split("\n")
    para: list[str] = []
    para_start = 0

    def flush() -> None:
        nonlocal para
        if para:
            blocks.append(Paragraph(parse_spans(" ".join(para), para_start)))
            para = []

    i = 0
    while i < len(lines):
        line, no = lines[i], i + 1
        fence = _FENCE.match(line)
        if fence:
            flush()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ManualSyntaxError(no, "unclosed code fence")
            blocks.append(CodeBlock("\n".join(body), fence.group(1)))
            i += 1
            continue
        image = _IMAGE.match(line.strip())
        if image:
            flush()
            alt, path = image.group(1).strip(), image.group(2).strip()
            if not alt:
                raise ManualSyntaxError(no, "an image needs alt text")
            blocks.append(Image(alt, path))
            i += 1
            continue
        if "![" in line:
            raise ManualSyntaxError(no, "an image needs a line of its own")
        if _FORBIDDEN.search(line):
            raise ManualSyntaxError(no, "HTML is outside the manual subset")
        if line.lstrip().startswith(">"):
            raise ManualSyntaxError(no, "blockquotes are outside the manual subset")
        if not line.strip():
            flush()
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            title = heading.group(2).strip()
            blocks.append(Heading(len(heading.group(1)), title, slugify(title)))
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush()
            table, i = _parse_table(lines, i)
            blocks.append(table)
            continue
        ordered = _OLIST.match(line)
        unordered = _ULIST.match(line)
        if ordered or unordered:
            flush()
            if ordered:
                indent, marker, rest = (
                    ordered.group(1),
                    ordered.group(2),
                    ordered.group(3),
                )
            else:
                indent, marker, rest = unordered.group(1), "-", unordered.group(2)
            if len(indent) % 2:
                raise ManualSyntaxError(
                    no, "list indent must be a multiple of two spaces"
                )
            blocks.append(
                ListItem(len(indent) // 2, bool(ordered), marker, parse_spans(rest, no))
            )
            i += 1
            continue
        if not para:
            para_start = no
        para.append(line.strip())
        i += 1
    flush()
    return blocks


def _cells(line: str, no: int) -> tuple[tuple[Span, ...], ...]:
    inner = line.strip().strip("|")
    return tuple(parse_spans(cell.strip(), no) for cell in inner.split("|"))


def _parse_table(lines: list[str], i: int) -> tuple[Table, int]:
    start = i
    raw: list[tuple[str, int]] = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        raw.append((lines[i], i + 1))
        i += 1
    if len(raw) < 2 or not _SEP.match(raw[1][0].strip()):
        raise ManualSyntaxError(start + 1, "table needs a |---| separator row")
    header = _cells(*raw[0])
    rows = tuple(_cells(text, no) for text, no in raw[2:])
    for text, no in raw[2:]:
        if len(_cells(text, no)) != len(header):
            raise ManualSyntaxError(
                no, "table row has a different column count than the header"
            )
    return Table(header, rows), i
