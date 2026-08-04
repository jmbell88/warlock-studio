"""The imgui half of the manual: the window, the TOC, and drawn blocks.

Everything above this module (parser, loader, targets) is pure; this file is
the only one allowed to touch imgui, mirroring the Inker engine / inker_mode
split. Blocks are cached parsed per chapter -- the files cannot change while
the app runs.
"""

from __future__ import annotations

import logging
from typing import Any

from imgui_bundle import imgui

from .. import fonts, icons, theme, widgets
from ..tokens import sp
from . import loader, parser
from .targets import HELP_TARGETS

log = logging.getLogger(__name__)

_blocks_cache: dict[str, list[parser.Block]] = {}
_chapters_cache: list[loader.Chapter] | None = None


def _toc() -> list[loader.Chapter]:
    global _chapters_cache
    if _chapters_cache is None:
        _chapters_cache = loader.chapters()
    return _chapters_cache


def _blocks(key: str) -> list[parser.Block]:
    if key not in _blocks_cache:
        try:
            _blocks_cache[key] = parser.parse(loader.load(key))
        except Exception:
            log.exception("could not load manual chapter %s", key)
            _blocks_cache[key] = [
                parser.Paragraph((parser.Span("text", "This chapter could not be loaded."),))
            ]
    return _blocks_cache[key]


def help_button(ctx: Any, pane: str) -> None:
    """The (?) a pane shows; opens the manual at that pane's chapter."""
    target = HELP_TARGETS.get(pane)
    if target is None:
        return
    offset = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - sp(26)
    imgui.same_line(max(offset, 0.0))
    if widgets.icon_button(f"{icons.INFO}##help-{pane}", "Open the manual section"):
        ctx.state.manual.open_at(*target)


def draw_window(ctx: Any) -> None:
    ms = ctx.state.manual
    if not ms.open:
        return
    imgui.set_next_window_size((sp(940), sp(640)), imgui.Cond_.first_use_ever.value)
    expanded, still_open = imgui.begin("Manual###manual", True)
    ms.open = bool(still_open)
    if not expanded or not ms.open:
        imgui.end()
        return
    if imgui.begin_child("manual-toc", (sp(240), 0), imgui.ChildFlags_.borders.value):
        _draw_toc(ms)
    imgui.end_child()
    imgui.same_line()
    if imgui.begin_child("manual-page", (0, 0)):
        _draw_chapter(ctx, ms)
    imgui.end_child()
    imgui.end()


def _draw_toc(ms: Any) -> None:
    imgui.set_next_item_width(-1)
    changed, ms.search = imgui.input_text_with_hint(
        "##manual-search", "Search chapters...", ms.search
    )
    needle = ms.search.strip().lower()
    part = None
    for chapter in _toc():
        if needle and not _matches(chapter, needle):
            continue
        if chapter.part != part and chapter.part:
            widgets.section(chapter.part)
        part = chapter.part
        label = chapter.title if chapter.part else "Contents"
        if imgui.selectable(f"{label}##{chapter.key}", ms.chapter == chapter.key)[0]:
            ms.open_at(chapter.key)


def _matches(chapter: loader.Chapter, needle: str) -> bool:
    if needle in chapter.title.lower():
        return True
    return any(
        needle in b.text.lower()
        for b in _blocks(chapter.key)
        if isinstance(b, parser.Heading)
    )


def _draw_chapter(ctx: Any, ms: Any) -> None:
    anchor = ms.pending_anchor
    for index, block in enumerate(_blocks(ms.chapter)):
        _draw_block(ctx, ms, block, index, anchor)
    if ms.pending_anchor == anchor:
        ms.pending_anchor = None
    imgui.dummy((0, sp(8)))
    imgui.separator()
    toc = _toc()
    idx = next((i for i, c in enumerate(toc) if c.key == ms.chapter), 0)
    if idx > 0 and imgui.button(f"{icons.ARROW_LEFT} {toc[idx - 1].title}"):
        ms.open_at(toc[idx - 1].key)
    if idx + 1 < len(toc):
        if idx > 0:
            imgui.same_line()
        if imgui.button(f"{toc[idx + 1].title} {icons.CHEVRON_RIGHT}"):
            ms.open_at(toc[idx + 1].key)


def _draw_block(ctx: Any, ms: Any, block: parser.Block, index: int, anchor: str | None) -> None:
    if isinstance(block, parser.Heading):
        imgui.dummy((0, sp(6)))
        if block.level <= 2:
            with fonts.push(imgui, fonts.SEMIBOLD, 22 if block.level == 1 else 17):
                imgui.text_wrapped(block.text)
        else:
            widgets.section(block.text)
        if anchor == block.anchor:
            imgui.set_scroll_here_y(0.1)
        if block.level <= 2:
            imgui.separator()
    elif isinstance(block, parser.Paragraph):
        _draw_spans(ctx, ms, block.spans)
        imgui.dummy((0, sp(4)))
    elif isinstance(block, parser.CodeBlock):
        rows = block.text.count("\n") + 1
        height = imgui.get_text_line_height_with_spacing() * rows + sp(12)
        imgui.input_text_multiline(
            f"##code-{index}", block.text, (-1, height),
            imgui.InputTextFlags_.read_only.value,
        )
    elif isinstance(block, parser.ListItem):
        imgui.dummy((sp(14) * (block.depth + 1), 0))
        imgui.same_line()
        widgets.muted(block.marker if block.ordered else "-")
        imgui.same_line()
        imgui.begin_group()
        _draw_spans(ctx, ms, block.spans)
        imgui.end_group()
    elif isinstance(block, parser.Table):
        flags = imgui.TableFlags_.borders.value | imgui.TableFlags_.row_bg.value
        if imgui.begin_table(f"##table-{index}", len(block.header), flags):
            for cell in block.header:
                imgui.table_next_column()
                with fonts.label(imgui):
                    _draw_spans(ctx, ms, cell)
            for row in block.rows:
                for cell in row:
                    imgui.table_next_column()
                    _draw_spans(ctx, ms, cell)
            imgui.end_table()


def _draw_spans(ctx: Any, ms: Any, spans: tuple[parser.Span, ...]) -> None:
    """Word-level wrapping so one line can mix styles.

    imgui's text_wrapped wraps a single style run; a paragraph here changes
    colour and font mid-line, so the wrap decision has to be per word.
    """
    max_x = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
    started = False
    for span in spans:
        pieces = span.text.split(" ")
        for n, piece in enumerate(pieces):
            word = piece + (" " if n < len(pieces) - 1 else "")
            if not piece and not word:
                continue
            if started:
                imgui.same_line(0.0, 0.0)
                if imgui.get_cursor_pos_x() + imgui.calc_text_size(piece).x > max_x:
                    imgui.new_line()
            _draw_word(ctx, ms, span, word)
            started = True


def _draw_word(ctx: Any, ms: Any, span: parser.Span, word: str) -> None:
    if span.kind == "bold":
        with fonts.label(imgui):
            imgui.text(word)
    elif span.kind == "italic":
        widgets.text_colored(theme.MUTED, word)
    elif span.kind == "code":
        widgets.text_colored(theme.ACCENT, word)
    elif span.kind == "link":
        widgets.text_colored(theme.ACCENT, word)
        rect_min, rect_max = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        imgui.get_window_draw_list().add_line(
            (rect_min.x, rect_max.y), (rect_max.x, rect_max.y),
            imgui.get_color_u32(theme.rgba(theme.ACCENT)),
        )
        if imgui.is_item_hovered():
            imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
        if imgui.is_item_clicked():
            _follow(ctx, ms, span.target)
    else:
        imgui.text(word)


def _follow(ctx: Any, ms: Any, target: str) -> None:
    if target.startswith(("http://", "https://")):
        # Offline app: never launch a browser. The URL lands on the clipboard.
        imgui.set_clipboard_text(target)
        ctx.toast("Link copied to the clipboard.")
        return
    page, _, anchor = target.partition("#")
    key = page.removesuffix(".md") if page else ms.chapter
    ms.open_at(key, anchor or None)
