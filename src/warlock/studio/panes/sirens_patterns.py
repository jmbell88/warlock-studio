"""Sirens' centre pane: the pattern grid, and the mode's heartbeat.

**Drawn with the draw list, not with widgets.** A 64-row pattern over five
channels is 1,600 cells, and one imgui widget per cell is 1,600 ids, 1,600
hit-tests and a layout pass per frame -- for text that is never interactive
individually. The grid is text and rectangles on one draw list, and *one*
invisible button over the whole thing turns a click into a caret move. That is
``plotter_canvas``'s argument at a different scale: what a canvas spends its
effort not building is a widget tree.

**Only the visible rows are drawn.** A pattern can be 256 rows and the pane can
show forty of them; drawing the rest is text the clipper would throw away
anyway, and at eight open tabs it is the frame budget.

**This pane is the pump.** There is no per-mode update hook, so the thing that
draws every frame is what re-arms the renderer -- ``sirens_mode.pump`` here,
the ``motion.py`` idiom, and the same job ``packwright_preview`` does for the
atlas. The flag is cleared inside ``request_render`` on an accepted submit, so
calling this every frame is free when nothing changed.

**No help button.** It is the canvas, not a panel -- ``plotter_canvas``'s rule,
and ``tests/manual/test_coverage.py`` lists both by name.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, sirens_mode, theme, widgets
from ..sirens import document as D
from ..sirens import notes
from ..tokens import sp

#: One row's height and one column-group's width, in design pixels. Both are
#: measured from the monospace text they hold rather than chosen: a cell is
#: ``"C-4 01 .. 000"`` and a group narrower than that clips the effect column,
#: which is the one nobody notices is missing.
ROW_H = 16.0
GUTTER_W = 34.0
CHANNEL_W = 116.0

#: The five columns' character widths within a group, in the document's order.
#: ``document.COLUMNS`` worth of entries, asserted by a test rather than by a
#: comment -- a sixth column added to the engine must widen the group here or
#: the grid silently stops drawing it.
COLUMN_CHARS: tuple[int, ...] = (3, 2, 2, 1, 2)

#: Every byte a cell can hold, formatted once. Three of the five columns are a
#: byte, and a visible grid is up to ``visible x channels`` cells *per frame* --
#: so this is three ``f"{n:02X}"`` calls per cell that never had to happen.
_HEX: tuple[str, ...] = tuple(f"{value:02X}" for value in range(256))

#: ``{(text, font_size): advance}``. The grid measures every column of every
#: visible cell, twice where the caret is, and the strings come from a tiny
#: fixed vocabulary -- 256 byte spellings, the note names, the effect letters --
#: so the measurement is the same answer over and over. Keyed on the font size
#: as well as the text because ``sp`` scales with the display and a cached
#: advance from another scale would misplace every caret. Bounded by that
#: vocabulary; nothing here is user text.
_ADVANCE: dict[tuple[str, float], float] = {}


def draw(ctx: Any) -> None:

    state = sirens_mode.ensure(ctx)
    tab = state.active
    # The pump, before anything can return: a frame that drew the empty state
    # is still a frame, and a tab whose render failed must keep being offered
    # the chance to succeed.
    sirens_mode.pump(ctx)

    _tabs(ctx, state)
    if tab is None:
        _empty(ctx)
        return

    # Before the grid, not after: the grid sizes its row count from the content
    # region still available, so a strip drawn under it would be a row past the
    # bottom of the pane.
    _toolbar(ctx, state)
    pattern = sirens_mode.caret_pattern(ctx, tab)
    if pattern is None:
        widgets.muted("This song has no patterns. Add one from the Order panel.")
        return

    _headers(ctx, state, tab, pattern)
    _grid(ctx, state, tab, pattern)


def _headers(ctx: Any, state: Any, tab: Any, pattern: Any) -> None:
    """One button per channel, over the column it belongs to.

    **The channels had no names on screen at all**, which is the plainest thing
    missing from this pane: five columns of dots, and the only way to find out
    which one was the noise channel was to type into it (the 2026-09-02 review,
    section 8). The button carries the name, its state says whether the mix is
    playing it, and a click mutes -- with solo on the right-hand button, since
    the two are used together and a modifier-click is a control nobody finds.
    """
    from imgui_bundle import imgui

    channels = list(tab.doc.channels)[: pattern.channels]
    if not channels:
        return
    chan_w = sp(CHANNEL_W)
    imgui.dummy((sp(GUTTER_W) - sp(4), 1))
    for index, channel in enumerate(channels):
        imgui.same_line()
        muted, soloed, audible = sirens_mode.channel_state(ctx, channel.uid)
        name = channel.name or f"{channel.kind.capitalize()} {index + 1}"
        # Two buttons in one column's width: the name (mute) and an S (solo).
        label = name if audible else f"{icons.SLASH} {name}"
        if widgets.disabled_button(
            f"{label}###sirens-mute-{channel.uid}",
            True,
            (chan_w - sp(26), 0),
            tooltip=(
                f"{name} -- {channel.kind}. Click to "
                f"{'unmute' if muted else 'mute'} it in the mix."
                + ("" if audible or muted else " Another channel is soloed.")
            ),
        ):
            sirens_mode.toggle_mute(ctx, channel.uid, tab)
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.CIRCLE if soloed else 'S'}###sirens-solo-{channel.uid}",
            True,
            (sp(20), 0),
            tooltip=(
                "Stop soloing this channel."
                if soloed
                else "Play this channel alone. Solo wins over every mute."
            ),
        ):
            sirens_mode.toggle_solo(ctx, channel.uid, tab)


def _tabs(ctx: Any, state: Any) -> None:
    from imgui_bundle import imgui

    if not state.docs:
        return
    flags = (
        imgui.TabBarFlags_.reorderable.value
        | imgui.TabBarFlags_.auto_select_new_tabs.value
    )
    if imgui.begin_tab_bar("sirens-tabs", flags):
        for tab in list(state.docs):
            # imgui's own dot, not a ``"* "`` prefix -- see ``inker_canvas``.
            item_flags = imgui.TabItemFlags_.unsaved_document.value if tab.dirty else 0
            opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
            if opened:
                state.activate(tab.uid)
                imgui.end_tab_item()
            if not keep:
                sirens_mode.close_tab(ctx, tab.uid)
        imgui.end_tab_bar()


def _empty(ctx: Any) -> None:
    from pathlib import Path

    widgets.nothing_open(
        "Start a song or open a .wsng.",
        [
            ("New song", lambda: sirens_mode.new_document(ctx)),
            ("Open a file...", lambda: sirens_mode.ask_open(ctx)),
        ],
        recent_paths=sirens_mode.recent_paths(ctx),
        on_open=lambda path: sirens_mode.open_path(ctx, Path(path)),
    )


def _cell_text(cells: Any, row: int, channel: int) -> tuple[str, ...]:
    """One cell's five columns, as the strings the grid draws.

    ``"..."`` for an empty note and ``".."`` for an empty byte, which is the
    tracker convention and is not decoration: a run of dots is how the eye
    finds the rows where *something* happens, and a blank there makes a pattern
    unreadable at a glance.
    """
    from ..sirens import synth

    note = int(cells[row, channel, D.NOTE])
    instrument = int(cells[row, channel, D.INSTRUMENT])
    volume = int(cells[row, channel, D.VOLUME])
    effect = int(cells[row, channel, D.EFFECT])
    param = int(cells[row, channel, D.PARAM])
    letter = synth.EFFECT_NAMES.get(effect, (".", ""))[0] if effect >= 0 else "."
    return (
        notes.name(note) if note != notes.EMPTY else "...",
        _HEX[instrument] if instrument >= 0 else "..",
        _HEX[volume] if volume >= 0 else "..",
        letter,
        _HEX[param] if param >= 0 else "..",
    )


def _caret_span(column: int, digit: int, part: str) -> tuple[int, int]:
    """Which characters of a cell's text the caret rings, as ``(start, count)``.

    A column typed one nibble at a time gets a caret over the **nibble**,
    because two-digit entry is otherwise invisible: the first key changes one
    character of the cell and nothing anywhere says a second key is still owed,
    so an entry interrupted by an arrow key looks exactly like an entry that
    finished. The columns taken in a single keystroke -- the note, the volume,
    the effect letter -- keep the whole-cell caret, since they have no
    sub-position to show and a caret narrower than the value it is over would
    be pointing at half a thing.
    """
    if sirens_mode.COLUMN_DIGITS[column] > 1 and 0 <= digit < len(part):
        return (digit, 1)
    return (0, len(part))


def column_at(dx: float, widths: Any, gap: float) -> int:
    """Which of a cell's five columns a click ``dx`` into its channel is on.

    ``widths`` is the drawn advance of each column's text, in order, and ``gap``
    is the space after each -- the same two numbers the draw loop steps ``cx``
    by, so the answer is the column under the pixel rather than a second
    opinion about the layout.

    A click in the gap after a column takes that column: the space belongs to
    the value on its left the way a tracker's does, and a caret that refused to
    move because the press landed one pixel wide of a glyph is a control that
    works most of the time. Left of the first column and right of the last both
    clamp, for the same reason the row does.

    Pure, and here rather than inline in the click branch, because "click on
    ``Fxx``, type, get a note" was the whole defect: the press moved the row and
    the channel and left the column where it was (the 2026-09-02 review,
    section 8).
    """
    edge = 0.0
    for column, width in enumerate(widths):
        edge += float(width) + float(gap)
        if dx < edge:
            return column
    return max(0, len(tuple(widths)) - 1)


def _advance(imgui: Any, text: str) -> float:
    """``calc_text_size(text).x``, memoised. See :data:`_ADVANCE`."""
    key = (text, float(imgui.get_font_size()))
    got = _ADVANCE.get(key)
    if got is None:
        got = float(imgui.calc_text_size(text).x)
        _ADVANCE[key] = got
    return got


def _grid(ctx: Any, state: Any, tab: Any, pattern: Any) -> None:
    from imgui_bundle import imgui

    cells = pattern.cells
    row_h = sp(ROW_H)
    gutter = sp(GUTTER_W)
    chan_w = sp(CHANNEL_W)
    origin = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail()
    draw_list = imgui.get_window_draw_list()

    text = imgui.get_color_u32(theme.rgba(theme.TEXT))
    muted = imgui.get_color_u32(theme.rgba(theme.MUTED))
    accent = imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.35))
    beat = imgui.get_color_u32(theme.rgba(theme.ELEV_1))
    caret = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    block = imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.18))

    visible = max(1, int(avail.y // row_h))
    playhead = sirens_mode.playhead_row(ctx, tab)
    # Which row sits at the top. Centred on the caret -- or on the playhead
    # while following, which is what ``SirensState.follow`` buys: a playhead
    # that scrolls off the pane within a bar is a playhead nobody watches.
    focus = playhead if (state.follow and playhead is not None) else state.row
    top = max(0, min(int(focus) - visible // 2, max(0, pattern.rows - visible)))
    selection = state.selection(tab)

    for index in range(visible):
        row = top + index
        if row >= pattern.rows:
            break
        y = origin.y + index * row_h
        if row % D.ROWS_PER_BEAT == 0:
            # The beat stripe. Without it a 64-row pattern is an undifferen-
            # tiated column of dots and counting to the downbeat is manual.
            draw_list.add_rect_filled(
                (origin.x, y), (origin.x + avail.x, y + row_h), beat
            )
        if playhead is not None and row == playhead:
            draw_list.add_rect_filled(
                (origin.x, y), (origin.x + avail.x, y + row_h), accent
            )
        draw_list.add_text((origin.x, y), muted, f"{row:03d}")
        for channel in range(pattern.channels):
            x = origin.x + gutter + channel * chan_w
            if x > origin.x + avail.x:
                break
            if selection is not None:
                srow, schan, srows, schans = selection
                if srow <= row < srow + srows and schan <= channel < schan + schans:
                    draw_list.add_rect_filled(
                        (x, y), (x + chan_w, y + row_h), block
                    )
            parts = _cell_text(cells, row, channel)
            cx = x
            for column, part in enumerate(parts):
                colour = text if part[0] not in "." else muted
                draw_list.add_text((cx, y), colour, part)
                if row == state.row and channel == state.channel and column == state.column:
                    start, count = _caret_span(column, state.digit, part)
                    lead = _advance(imgui, part[:start]) if start else 0.0
                    width = _advance(imgui, part[start : start + count])
                    # ``add_rect`` is (p_min, p_max, col, rounding, thickness,
                    # flags), and the thickness comes *before* the flags. The
                    # other order type-errors, and only on the frames that draw
                    # a caret -- which is every frame with a grid on screen, and
                    # which nothing caught until the panes were drawn under a
                    # test (``tests/test_sirens_panes_smoke.py``).
                    draw_list.add_rect(
                        (cx + lead - 1, y),
                        (cx + lead + width + 1, y + row_h),
                        caret,
                        0.0,
                        1.5,
                    )
                cx += _advance(imgui, part) + sp(6)

    # One invisible button over the whole grid, which is what makes a click a
    # caret move without 1,600 widget ids. Sized to the region rather than to
    # the content so a click below the last row still lands here rather than
    # falling through to the window.
    imgui.invisible_button("sirens-grid", (max(avail.x, 1.0), max(avail.y, 1.0)))
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mouse = imgui.get_mouse_pos()
        row = top + int((mouse.y - origin.y) // row_h)
        channel = int((mouse.x - origin.x - gutter) // chan_w)
        if mouse.x >= origin.x + gutter:
            # The column too, measured off the cell that was actually drawn:
            # a press that moved the row and the channel and left the column
            # alone meant clicking on ``Fxx``, typing, and getting a note.
            column = state.column
            if 0 <= row < pattern.rows and 0 <= channel < pattern.channels:
                parts = _cell_text(cells, row, channel)
                column = column_at(
                    mouse.x - (origin.x + gutter + channel * chan_w),
                    [_advance(imgui, part) for part in parts],
                    sp(6),
                )
            sirens_mode.set_caret(ctx, row=row, channel=channel, column=column)


def _toolbar(ctx: Any, state: Any) -> None:
    """The strip over the grid: what the caret is, and where it is going.

    Not a pane of its own, because none of it is a *setting* -- it is the
    caret's own state, and putting it in a sidebar would mean reading one
    column to find out what the other column will do with the next keystroke.
    """
    from imgui_bundle import imgui

    if state.active is None:
        return
    imgui.set_next_item_width(sp(90))
    changed, value = controls.slider_int("Octave", state.octave, 0, 9)
    if changed:
        state.octave = int(value)
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed, value = controls.slider_int("Step", state.step, 0, 16)
    if changed:
        state.step = int(value)
    imgui.same_line()
    changed, value = controls.checkbox("Follow", state.follow)
    if changed:
        state.follow = bool(value)
    imgui.same_line()
    widgets.muted(f"{icons.AUDIO_WAVEFORM} row {state.row:03d}")
    # **What the grid is editing, said where the editing happens.** Adding a
    # sound effect repoints this grid at the effect's own pattern, and the
    # panel that did it is in another column; a reader who typed into the grid
    # afterwards had nothing on this surface telling them which of the two
    # documents-within-the-document they were changing.
    label = sirens_mode.caret_pattern_label(ctx)
    if label:
        imgui.same_line()
        widgets.muted(f"|  {label}")
