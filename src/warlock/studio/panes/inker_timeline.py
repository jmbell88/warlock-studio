"""The animation timeline: frames across, tracks down.

A strip along the bottom of the centre column, drawn **only** when the document
is animated -- a still document's layout is byte-for-byte what it always was,
which is what keeps "the timeline is opt-in" true of the pixels and not just of
the data model.

Two things about this pane are load-bearing rather than incidental.

It is where playback is **ticked**, because there is no per-mode update hook in
this app and a pane's ``draw`` is the only thing that runs every frame. That is
the ``motion.py`` idiom applied to something with more at stake, so the tick is
a single call into ``inker_mode`` and every rule about *what* a tick does lives
there, next to the pure ``animation.advance`` it delegates the arithmetic to.

And every control here is gated on ``tab.busy`` rather than on ``tab.saving``.
A save is encoding the layer stack off-thread; playback is walking the playhead
through frames. Both make restructuring the grid unsafe, and asking one question
is what stops the third reason being added in one place and forgotten in nine.

Cell thumbnails are deliberately not here. They are the obvious next thing and
they are a per-cel texture on a grid that can be fifty wide, which is the
``StripRender`` problem again -- worth doing on a budget, not worth doing by
accident.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import inker_mode, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp

#: The strip's height in design pixels, reserved out of the centre column.
STRIP_H = 150.0

CELL = 20.0
GUTTER = 2.0
TRACK_LABEL_W = 96.0


def _u32(value: int, alpha: float = 1.0) -> int:
    return imgui.color_convert_float4_to_u32(theme.rgba(value, alpha))


def draw(ctx: Any) -> None:
    state = ctx.state.inker
    tab = None if state is None else state.active
    if tab is None or tab.doc.anim is None:
        return
    _tick(tab)
    _transport(ctx, tab)
    imgui.separator()
    _grid(ctx, tab)


def _tick(tab: Any) -> None:
    """Advance playback by the time since the last drawn frame.

    ``delta_time`` and not a wall clock: it is already the number imgui uses to
    animate everything else on screen, so a stalled frame stalls the clip by the
    same amount it stalls every other animation rather than by a different one.
    """
    if tab.playing:
        inker_mode.tick_playback(tab, imgui.get_io().delta_time * 1000.0)


def _transport(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    anim = doc.anim
    state = ctx.state.inker
    index = tab.play_index if tab.playing else anim.current

    if widgets.disabled_button("|<", not tab.busy, (sp(28), 0)):
        doc.set_current_frame(0)
    imgui.same_line()
    if widgets.disabled_button("<", not tab.busy, (sp(28), 0)):
        inker_mode.step_frame(ctx, -1, tab)
    imgui.same_line()
    if widgets.disabled_button("Stop" if tab.playing else "Play", not tab.saving, (sp(48), 0)):
        inker_mode.toggle_play(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button(">", not tab.busy, (sp(28), 0)):
        inker_mode.step_frame(ctx, 1, tab)
    imgui.same_line()
    if widgets.disabled_button(">|", not tab.busy, (sp(28), 0)):
        doc.set_current_frame(len(anim.frames) - 1)

    imgui.same_line()
    imgui.text(f"{index + 1}/{len(anim.frames)}")
    imgui.same_line()
    imgui.text(f"{anim.duration_ms()} ms")

    imgui.same_line()
    imgui.begin_disabled(tab.busy)
    if imgui.button("+ Frame"):
        doc.add_frame()
    imgui.same_line()
    if imgui.button("+ Copy"):
        doc.add_frame(copy=True)
    imgui.same_line()
    if imgui.button("+ Link"):
        doc.add_frame(link=True)
    imgui.same_line()
    if widgets.disabled_button("Delete", len(anim.frames) > 1):
        doc.remove_frame()
    imgui.end_disabled()

    imgui.same_line()
    changed, value = widgets.toggle("Onion", state.onion, tag="inker-onion")
    if changed:
        state.onion = value
    imgui.same_line()
    if widgets.disabled_button("Export sheet", not tab.busy):
        inker_mode.export_sheet(ctx, tab)
    widgets.help_marker(
        "Writes a packed PNG of every frame plus a JSON sidecar naming the cells,"
        " their durations and any tags."
    )
    imgui.same_line()
    manual_render.help_button(ctx, "inker-timeline")

    # The frame the counter above is naming, which during playback is the one
    # going past rather than the one the playhead will come back to. Read-only
    # while it moves: an edit box whose value changes ten times a second is not
    # something a user can type into, and ``tick_playback`` deliberately does
    # not move ``anim.current``, so a write here would land on a frame that is
    # not the one on screen.
    imgui.set_next_item_width(sp(90))
    imgui.begin_disabled(tab.busy)
    changed, value = imgui.input_int("ms", anim.frames[index].duration_ms, 10, 50)
    if changed:
        doc.set_frame_duration(index, value)
    imgui.end_disabled()

    _onion_controls(state)


#: How many neighbours either side onion skinning will draw. A ceiling rather
#: than an open int because each one is a full-canvas texture and a draw: the
#: number is a working preference, not a budget the user should be able to spend
#: the frame time on by typing.
MAX_ONION = 5


def _onion_controls(state: Any) -> None:
    """Depth and fade for onion skinning, shown only while it is on.

    The fields have been read by the canvas since onion skinning landed and
    fixed at 1/1/0.35 because nothing wrote them -- which is the wrong default
    for a two-frame cycle and for a twelve-frame walk in opposite directions.
    They are app-level settings, like the toggle beside them and every other
    tool setting, because how far back a user wants to see is a property of how
    they work rather than of the drawing.

    On the second row, after the duration box, rather than after the toggle:
    the transport row is already full, and a ``same_line`` past the panel edge
    does not wrap, it hides the control.
    """
    if not state.onion:
        return
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    changed, value = imgui.input_int("back", state.onion_before, 1, 1)
    if changed:
        state.onion_before = max(0, min(int(value), MAX_ONION))
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    changed, value = imgui.input_int("ahead", state.onion_after, 1, 1)
    if changed:
        state.onion_after = max(0, min(int(value), MAX_ONION))
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed, alpha = imgui.slider_float("fade", state.onion_alpha, 0.05, 1.0, "%.2f")
    if changed:
        state.onion_alpha = min(1.0, max(0.05, float(alpha)))


def _grid(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    anim = doc.anim
    cell, gutter = sp(CELL), sp(GUTTER)
    if not imgui.begin_child("inker-timeline-grid", (0, 0), 0):
        imgui.end_child()
        return

    _frame_headers(ctx, tab, cell, gutter)
    # Top first, like the layers panel: the engine's list is painter's order and
    # the timeline reads the same way down the page that the stack does.
    for index in range(len(anim.tracks) - 1, -1, -1):
        _track_row(ctx, tab, index, cell, gutter)
    _tag_row(ctx, tab, cell, gutter)

    imgui.end_child()


def _frame_headers(ctx: Any, tab: Any, cell: float, gutter: float) -> None:
    anim = tab.doc.anim
    playing = tab.play_index if tab.playing else anim.current
    imgui.dummy((sp(TRACK_LABEL_W), cell))
    for index, frame in enumerate(anim.frames):
        imgui.same_line(0.0, gutter)
        imgui.push_id(f"fh{frame.uid}")
        current = index == playing
        if current:
            imgui.push_style_color(imgui.Col_.button.value, theme.rgba(theme.ACCENT, 0.9))
        # Every tenth frame numbered: a fifty-frame clip with a number in every
        # 20px cell is a wall of digits, and a tick every ten is how a ruler
        # solves the same problem.
        label = str(index + 1) if index % 10 == 0 or current else "."
        if imgui.button(label, (cell, cell)) and not tab.busy:
            tab.doc.set_current_frame(index)
        if current:
            imgui.pop_style_color()
        _frame_menu(tab, index)
        imgui.pop_id()


def _frame_menu(tab: Any, index: int) -> None:
    doc = tab.doc
    if not imgui.begin_popup_context_item(f"framemenu{index}"):
        return
    imgui.begin_disabled(tab.busy)
    if imgui.menu_item_simple("Insert before"):
        doc.add_frame(index)
    if imgui.menu_item_simple("Duplicate (copied)"):
        doc.set_current_frame(index)
        doc.add_frame(index + 1, copy=True)
    if imgui.menu_item_simple("Duplicate (linked)"):
        doc.set_current_frame(index)
        doc.add_frame(index + 1, link=True)
    imgui.separator()
    # Disabled at the ends rather than clicked-and-ignored: an enabled item that
    # does nothing reads as a bug in the move, not as "there is nowhere to go".
    last = len(doc.anim.frames) - 1
    imgui.begin_disabled(index <= 0)
    if imgui.menu_item_simple("Move left"):
        doc.move_frame(index, index - 1)
    imgui.end_disabled()
    imgui.begin_disabled(index >= last)
    if imgui.menu_item_simple("Move right"):
        doc.move_frame(index, index + 1)
    imgui.end_disabled()
    imgui.separator()
    if imgui.menu_item_simple("Delete"):
        doc.remove_frame(index)
    imgui.separator()
    # A one-frame span, renamed and stretched from the tag's own menu below.
    # The alternative -- a modal asking for a name and a range up front -- is
    # three answers for something the user is about to look at and adjust
    # anyway, and there is no frame-range selection for it to read.
    if imgui.menu_item_simple("New tag here"):
        doc.add_tag(f"tag {len(doc.anim.tags) + 1}", index)
    imgui.end_disabled()
    imgui.end_popup()


def _track_row(ctx: Any, tab: Any, track_index: int, cell: float, gutter: float) -> None:
    doc = tab.doc
    anim = doc.anim
    track = anim.tracks[track_index]
    active_track = track_index == doc.stack.active_index

    imgui.push_id(f"tr{track.uid}")
    if active_track:
        widgets.text_colored(theme.ACCENT, track.name[:14])
    elif not track.visible:
        widgets.muted(track.name[:14])
    else:
        imgui.text(track.name[:14])
    imgui.same_line(sp(TRACK_LABEL_W))

    for frame_index, frame in enumerate(anim.frames):
        if frame_index:
            imgui.same_line(0.0, gutter)
        imgui.push_id(frame.uid)
        _cell(tab, track, frame, track_index, frame_index, cell)
        imgui.pop_id()
    imgui.pop_id()


def _cell(tab: Any, track: Any, frame: Any, ti: int, fi: int, cell: float) -> None:
    doc = tab.doc
    anim = doc.anim
    layer = anim.cel(track.uid, frame.uid)
    linked = layer is not None and anim.is_linked(track.uid, frame.uid)
    # Three states, three glyphs, and colour on top rather than instead: a grid
    # that only differs by hue is unreadable to a chunk of people, which is the
    # argument STATUS_GLYPHS already makes elsewhere in this app.
    if layer is None:
        label, colour, alpha = "", theme.PANEL, 0.5
    elif linked:
        label, colour, alpha = "=", theme.ACCENT, 0.55
    else:
        label, colour, alpha = "*", theme.ACCENT, 0.9
    imgui.push_style_color(imgui.Col_.button.value, theme.rgba(colour, alpha))
    if imgui.button(label, (cell, cell)) and not tab.busy:
        doc.set_active_layer(ti)
        doc.set_current_frame(fi)
    imgui.pop_style_color()
    _cell_menu(tab, ti, fi, layer is not None, linked)


def _cell_menu(tab: Any, ti: int, fi: int, has_cel: bool, linked: bool) -> None:
    doc = tab.doc
    if not imgui.begin_popup_context_item("celmenu"):
        return
    imgui.begin_disabled(tab.busy)
    imgui.begin_disabled(fi <= 0)
    if imgui.menu_item_simple("Link to previous frame"):
        doc.link_cel(fi - 1, track_index=ti, frame_index=fi)
    imgui.end_disabled()
    if linked and imgui.menu_item_simple("Unlink"):
        doc.unlink_cel(track_index=ti, frame_index=fi)
    if has_cel and imgui.menu_item_simple("Clear"):
        doc.clear_cel(track_index=ti, frame_index=fi)
    imgui.end_disabled()
    imgui.end_popup()


def _tag_row(ctx: Any, tab: Any, cell: float, gutter: float) -> None:
    """Tags as a band under the grid, drawn rather than laid out.

    A row of widgets would need one per frame to keep the columns aligned; a
    single line per tag with the draw list needs the column arithmetic once and
    lets a tag be any length without inventing a widget for a span.

    The name doubles as the handle: right-click it for the menu, and a rename
    swaps it for an input in place. Renaming inline rather than in a modal is
    what lets the ends be set from the playhead in the same menu -- a dialog
    would have to own the whole tag, and the whole tag is a name plus two
    numbers the user is picking by looking at the grid behind it.
    """
    anim = tab.doc.anim
    if not anim.tags:
        return
    state = ctx.state.inker
    imgui.dummy((0, sp(2)))
    draw_list = imgui.get_window_draw_list()
    for index, tag in enumerate(list(anim.tags)):
        imgui.push_id(f"tag{index}")
        imgui.dummy((sp(TRACK_LABEL_W), sp(14)))
        origin = imgui.get_item_rect_min()
        top = origin.y + sp(4)
        start = origin.x + sp(TRACK_LABEL_W) + (cell + gutter) * max(0, tag.start)
        width = (cell + gutter) * (max(tag.start, tag.end) - max(0, tag.start) + 1) - gutter
        draw_list.add_rect_filled(
            (start, top), (start + max(width, cell), top + sp(4)), _u32(theme.OK, 0.8)
        )
        imgui.same_line(sp(TRACK_LABEL_W))
        if state.tag_editing == index:
            _tag_rename(ctx, tab, index)
        else:
            widgets.muted(f"{tag.name}{'' if tag.loop else ' (once)'}")
            _tag_menu(ctx, tab, index, tag)
        imgui.pop_id()


def _tag_rename(ctx: Any, tab: Any, index: int) -> None:
    """The name field, committed on Enter or on losing focus.

    Both, because a user who clicks away has still finished typing -- and
    because leaving the field open would leave the timeline in a mode nothing
    else can get it out of. Escape is the way out that keeps the old name.
    """
    state = ctx.state.inker
    imgui.set_next_item_width(sp(140))
    # The house idiom, from ``dialogs``: focus the field while nothing else has
    # it, so opening the rename puts the caret in it without a one-shot flag.
    if not imgui.is_any_item_active():
        imgui.set_keyboard_focus_here()
    # ``enter_returns_true`` makes the flag mean *Enter*, not *changed*, while
    # the returned string is the live buffer either way -- so the buffer is
    # stored unconditionally. See the same note in ``dialogs``.
    entered, value = imgui.input_text(
        "##tagname", state.tag_name, imgui.InputTextFlags_.enter_returns_true.value
    )
    state.tag_name = value
    if imgui.is_key_pressed(imgui.Key.escape):
        state.tag_editing = -1
        return
    if entered or imgui.is_item_deactivated():
        if not tab.busy:
            tab.doc.set_tag(index, name=state.tag_name.strip() or "tag")
        state.tag_editing = -1


def _tag_menu(ctx: Any, tab: Any, index: int, tag: Any) -> None:
    doc = tab.doc
    state = ctx.state.inker
    if not imgui.begin_popup_context_item("tagmenu"):
        return
    imgui.begin_disabled(tab.busy)
    if imgui.menu_item_simple("Rename"):
        state.tag_editing = index
        state.tag_name = tag.name
    # Both ends from the playhead, which is the frame the user just clicked to
    # get here: a tag is a span of the timeline and the timeline is what they
    # are looking at, so there is nothing to type.
    if imgui.menu_item_simple(f"Start at frame {doc.anim.current + 1}"):
        doc.set_tag(index, start=doc.anim.current)
    if imgui.menu_item_simple(f"End at frame {doc.anim.current + 1}"):
        doc.set_tag(index, end=doc.anim.current)
    if imgui.menu_item_simple("Loop", "", tag.loop):
        doc.set_tag(index, loop=not tag.loop)
    imgui.separator()
    if imgui.menu_item_simple("Delete tag"):
        doc.remove_tag(index)
        state.tag_editing = -1
    imgui.end_disabled()
    imgui.end_popup()
