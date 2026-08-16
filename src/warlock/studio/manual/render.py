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

from .. import fonts, icons, theme, tokens, widgets
from ..tokens import sp
from . import loader, parser
from .targets import HELP_TARGETS, TROUBLESHOOTING

log = logging.getLogger(__name__)

_blocks_cache: dict[str, list[parser.Block]] = {}
_chapters_cache: list[loader.Chapter] | None = None

# Prose stops being readable long before a maximised window runs out of room:
# a 2560px page wraps a paragraph at something like 300 characters, which the
# eye cannot track back to the next line. Every measure in this module goes
# through _measure() so the body, the headings, the code blocks and the tables
# all stop at the same column.
MAX_LINE_CHARS = 120


def _measure() -> float:
    """The wrap width: the pane, or 120 characters, whichever is narrower."""
    return min(
        imgui.get_content_region_avail().x,
        imgui.calc_text_size("0").x * MAX_LINE_CHARS,
    )


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
    """The (?) a pane shows; opens the manual overlay at that pane's chapter.

    Name, signature and call-site shape are unchanged from when this switched
    modes, which is the whole of how the overlay reached forty-odd panes in one
    edit -- and why both ``tests/manual`` coverage gates went on passing.
    """
    target = HELP_TARGETS.get(pane)
    if target is None:
        return
    offset = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - sp(26)
    imgui.same_line(max(offset, 0.0))
    if widgets.icon_button(f"{icons.INFO}##help-{pane}", "Open the manual section"):
        open_at(ctx, target)


def open_at(ctx: Any, target: tuple[str, str | None]) -> None:
    """Raise the manual over the current screen, at ``target``.

    This used to be ``set_mode(ctx.state, "manual")``, and the change is the
    point rather than a refactor of it (the UI redesign, wave 3): help is consulted
    *about the pane you are on*, and a mode switch took that pane away to show
    it. The reader then had the answer and no longer had the question -- which
    is worst exactly where the (?) is most useful, beside a control whose
    setting somebody was in the middle of choosing.
    """
    ctx.state.manual.open = True
    ctx.state.manual.open_at(*target)


def troubleshooting_button(ctx: Any, label: str = "Troubleshooting") -> bool:
    """A small button onto chapter 12. -> whether it was pressed, so a caller
    that wants to close a popup first can."""
    if not imgui.small_button(label):
        return False
    open_at(ctx, TROUBLESHOOTING)
    return True


# How wide the overlay is allowed to get, and how much of the window's height
# it takes. The width is the same argument ``MAX_LINE_CHARS`` makes one level
# up -- prose stops being readable long before a maximised window runs out of
# room -- with the TOC's 240 added to it, so the *page* half still lands near
# the measure the renderer wraps at. The height is a fraction rather than a
# figure because what has to stay visible is the app *behind* it: an overlay
# that fills the window is a mode with extra steps.
OVERLAY_MAX_W = 1040.0
OVERLAY_HEIGHT = 0.8
# The margin kept clear on each side when the window is too narrow for the
# maximum. Two gutters of the host window's own padding.
OVERLAY_MARGIN = 80.0

# Whether the overlay was up last frame, so ``popover_enter`` can be told which
# frame is its first. Module state for the same reason ``_blocks_cache`` is:
# there is exactly one manual and it is drawn from exactly one place.
_was_open = [False]


def close(ctx: Any) -> None:
    ctx.state.manual.open = False


def toggle(ctx: Any) -> None:
    """What F1 does. A toggle rather than an open, because the key that raises
    a reference should be the key that puts it away."""
    ctx.state.manual.open = not ctx.state.manual.open


def draw_overlay(ctx: Any) -> None:
    """The manual over whatever is on screen, rather than instead of it.

    Drawn from ``App._overlays`` and *before* the command palette, so Ctrl+K
    still floats above this -- the palette is how you leave anywhere, including
    here.

    A plain window rather than a modal popup, and that is load-bearing twice
    over: a modal would take the one popup slot imgui gives a frame (the
    palette's, and every confirm's), and it would dim the screen behind it,
    which is the half of "this is a mode" the overlay exists not to be. It is
    still frosted and shadowed like every other floating surface, so it reads
    as being *over* the app rather than as a pane inside it.
    """
    ms = ctx.state.manual
    if not ms.open:
        _was_open[0] = False
        return
    appearing = not _was_open[0]
    _was_open[0] = True

    viewport = imgui.get_main_viewport()
    alpha, rise = widgets.popover_enter("manual", appearing)
    width = min(viewport.work_size.x - sp(OVERLAY_MARGIN), sp(OVERLAY_MAX_W))
    height = viewport.work_size.y * OVERLAY_HEIGHT
    imgui.set_next_window_pos(
        (
            viewport.work_pos.x + viewport.work_size.x * 0.5,
            viewport.work_pos.y + viewport.work_size.y * 0.5 + rise,
        ),
        imgui.Cond_.always.value,
        (0.5, 0.5),
    )
    imgui.set_next_window_size((width, height))
    # The palette's recipe exactly: the background is cleared before ``begin``
    # paints it and drawn back below, as a blur when there is one and as the
    # solid fill when there is not.
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    opened = imgui.begin(
        "##manual-overlay",
        None,
        imgui.WindowFlags_.no_title_bar.value
        | imgui.WindowFlags_.no_move.value
        | imgui.WindowFlags_.no_resize.value
        | imgui.WindowFlags_.no_collapse.value
        | imgui.WindowFlags_.no_saved_settings.value,
    )
    widgets.pop_surface_rounding()
    if opened:
        widgets.window_shadow("overlay", radius=radius)
        if frosted:
            widgets.window_backdrop(radius=radius)
        _overlay_header(ctx)
        draw_body(ctx)
    imgui.end()
    imgui.pop_style_var()


def _overlay_header(ctx: Any) -> None:
    """The title and the way out, on one line above the split."""
    widgets.pane_title("Manual")
    imgui.same_line()
    close_w = imgui.get_frame_height()
    imgui.set_cursor_pos_x(
        max(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - close_w, 0.0)
    )
    if widgets.icon_button(f"{icons.CIRCLE_X}##manual-close", "Close (Esc)", borderless=True):
        close(ctx)


def draw_body(ctx: Any) -> None:
    """The manual's two children, filling whatever host is current.

    Unchanged by the move off a mode and onto an overlay (the UI redesign, wave 3):
    it fills the region it is given, so the region being a window rather than
    the host is not its business. What *did* go is the claim in the old
    docstring that there is no visibility flag -- see :class:`state.ManualState`.
    """
    ms = ctx.state.manual
    _warm_blocks()
    if imgui.begin_child("manual-toc", (sp(240), 0), imgui.ChildFlags_.borders.value):
        _draw_toc(ms)
    imgui.end_child()
    imgui.same_line()
    # The TOC is bordered and so gets window padding for free; the page is not,
    # and a borderless child pads by zero -- which is what put the prose flush
    # against the divider.
    if imgui.begin_child(
        "manual-page", (0, 0), imgui.ChildFlags_.always_use_window_padding.value
    ):
        # Deliberately *not* inside a ``widgets.section_blocks`` scope. The
        # blocks group controls in a panel; a heading here names a passage of
        # prose, and tinting every one of them would turn a chapter into a
        # stack of cards -- the manual is a document, and a document's headings
        # are separated by their own typography.
        _draw_chapter(ctx, ms)
    imgui.end_child()


def _draw_toc(ms: Any) -> None:
    imgui.set_next_item_width(-1)
    changed, ms.search = imgui.input_text_with_hint(
        "##manual-search", "Search chapters...", ms.search
    )
    needle = ms.search.strip().lower()
    part = None
    found = 0
    for chapter in _toc():
        if needle and not _matches(chapter, needle):
            continue
        found += 1
        if chapter.part != part and chapter.part:
            widgets.section(chapter.part)
        part = chapter.part
        label = chapter.title if chapter.part else "Contents"
        if imgui.selectable(f"{label}##{chapter.key}", ms.chapter == chapter.key)[0]:
            ms.open_at(chapter.key)
    if needle and not found:
        # H73. A search that matches nothing used to empty the whole table of
        # contents, which reads as the manual having vanished rather than as a
        # query having missed -- and the search box is above the list, so there
        # was nothing on screen to say a filter was on.
        widgets.empty_state(
            icons.SEARCH,
            "No chapter matches.",
            "Titles, headings, body text, tables and commands are all searched.",
        )


#: How many chapters ``_warm_blocks`` parses per frame. Small enough that the
#: work is invisible in a frame budget, large enough that the whole manual is
#: cached within a handful of frames -- long before anyone can switch into the
#: mode and reach the search box.
_WARM_PER_FRAME = 3


def _warm_blocks() -> None:
    """Parse a few uncached chapters, spread over the frames the mode is open.

    ``_matches`` searches headings, so the first keystroke in the search box
    used to parse *every* chapter at once, on the frame thread -- a read and a
    parse per file, in the one place the app is meant to feel like a document
    viewer. Doing it while the reader is looking at the contents list costs
    nothing anyone can see, and by the time the box is focused the cache is
    hot. Bounded per frame rather than done in one go for the same reason it
    was moved at all.

    Deliberately not a TaskRunner job: ``_blocks_cache`` is a plain dict read
    by the frame thread every frame with no lock, and the three entry points
    into this mode would each have to remember to schedule it. Cheap, ordinary
    frame work is the smaller thing.
    """
    warmed = 0
    for chapter in _toc():
        if warmed >= _WARM_PER_FRAME:
            return
        if chapter.key not in _blocks_cache:
            _blocks(chapter.key)
            warmed += 1


def _block_text(block: parser.Block) -> str:
    """Everything readable in one block, as a flat string.

    Every block type, because the search used to see only headings and a
    manual searched by heading is a manual searched by its table of contents:
    "gltfpack", "prompt_hash" and "WARLOCK_VRAM_BUDGET" are each named in one
    paragraph and in no heading anywhere, so the three things a reader is most
    likely to arrive with found nothing.

    Code blocks are included deliberately. A command line is exactly the kind
    of string somebody pastes into a search box, and it is the one kind that
    never appears in prose.
    """
    if isinstance(block, parser.Heading | parser.CodeBlock):
        return block.text
    if isinstance(block, parser.Paragraph | parser.ListItem):
        return " ".join(span.text for span in block.spans)
    if isinstance(block, parser.Table):
        cells = [*block.header, *(cell for row in block.rows for cell in row)]
        return " ".join(span.text for cell in cells for span in cell)
    return ""


def _matches(chapter: loader.Chapter, needle: str) -> bool:
    if needle in chapter.title.lower():
        return True
    # The blocks are already cached and warmed a few chapters per frame, so
    # widening this from headings to every block costs one more pass over
    # structures that are in memory either way.
    return any(needle in _block_text(b).lower() for b in _blocks(chapter.key))


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
            # text_wrapped wraps at the content region, which is the one measure
            # this module does not use; bracket it so a heading breaks on the
            # same column the prose under it does.
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + _measure())
            # A chapter title and a section under it, from the ramp rather than
            # from two literals: 22 and 17 were one file's private type scale,
            # near enough to TEXT_HEADING and TEXT_TITLE that nothing was
            # bought by their being different.
            #
            # The chapter title is the page's *one loud thing* (UX.md Phase 2),
            # so it takes display size: a manual page is the one screen in the
            # app that is entirely prose, and at heading size its title was the
            # same weight as the four ``##`` sections under it.
            size = tokens.TEXT_DISPLAY if block.level == 1 else tokens.TEXT_HEADING
            with fonts.push(imgui, fonts.SEMIBOLD, size):
                imgui.text_wrapped(block.text)
            imgui.pop_text_wrap_pos()
        else:
            widgets.section(block.text)
        if anchor == block.anchor:
            imgui.set_scroll_here_y(0.1)
        if block.level <= 2:
            # Space rather than a rule (UX.md Phase 2), the same argument
            # ``widgets.section`` makes: a heading followed by a hairline
            # followed by a paragraph is three horizontal things where two
            # would do, and it is the ruling that made the manual read as a
            # settings dialog rather than as a page.
            imgui.dummy((0, sp(tokens.SP_2)))
    elif isinstance(block, parser.Paragraph):
        _draw_spans(ctx, ms, block.spans)
        imgui.dummy((0, sp(4)))
    elif isinstance(block, parser.CodeBlock):
        rows = block.text.count("\n") + 1
        height = imgui.get_text_line_height_with_spacing() * rows + sp(12)
        imgui.input_text_multiline(
            f"##code-{index}", block.text, (_measure(), height),
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
        if imgui.begin_table(f"##table-{index}", len(block.header), flags, (_measure(), 0.0)):
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
    max_x = imgui.get_cursor_pos_x() + _measure()
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
