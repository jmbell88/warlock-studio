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

from .. import controls, icons, inker_mode, theme, toolbar, widgets
from ..inker import animation
from ..manual import render as manual_render
from ..tokens import sp

#: The strip's height in design pixels, reserved out of the centre column.
STRIP_H = 150.0

CELL = 20.0
#: What a cell grows to when thumbnails are on. Big enough for the drawing in
#: it to be recognisable and small enough that a fifty-frame clip still fits
#: across the strip at a normal window width.
THUMB_CELL = 36.0
GUTTER = 2.0
TRACK_LABEL_W = 96.0

#: How far a track's label shifts per level of group nesting. v1 of the L3
#: timeline half is indent only -- no header row, no fold arrow, unlike the
#: layers panel's ``_group_row`` -- so this is the entire feature: a plain
#: offset on the label of a track that ``doc.group_of`` says is inside one.
#: Deliberately smaller than ``inker_layers.INDENT``: the label column here is
#: a fixed ``TRACK_LABEL_W`` a name is already truncated to fit, where the
#: layers panel's column grows with the window.
GROUP_INDENT = 8.0

#: The whole-number magnifications the export combo offers. Whole numbers only,
#: because the point of the setting is that nothing is resampled -- x1.5 would
#: have to invent a rule for which source pixel a destination one comes from,
#: which is exactly what ``transform.upscale`` exists not to do.
EXPORT_SCALES = (("1", "1x"), ("2", "2x"), ("3", "3x"), ("4", "4x"), ("8", "8x"))


def _u32(value: int, alpha: float = 1.0) -> int:
    return imgui.color_convert_float4_to_u32(theme.rgba(value, alpha))


def track_depth(doc: Any, track_uid: int) -> list[int]:
    """The chain of group uids above one track, innermost first -- ``[]`` at
    root, exactly ``groups.ancestry``'s answer.

    A thin, testable wrapper: ``_track_row`` only wants the length, to decide
    how far to indent, but a pure function over ``(doc, uid)`` is what a test
    can call without a window -- the same reason ``cell_index`` beside it is
    pure.
    """
    if not doc.groups:
        return []
    from ..inker import groups as gp

    return gp.ancestry(doc.group_of, track_uid)


def cell_index(
    point: tuple[float, float],
    *,
    x0: float,
    tops: dict[int, float],
    cell: float,
    gutter: float,
    frames: int,
) -> tuple[int, int] | None:
    """``(track index, frame index)`` under a screen point, or None.

    Pure, and it is pure because the drag *has* to be geometric: a pressed
    imgui button suppresses hover on every neighbour, so ``is_item_hovered``
    stops answering the moment a marquee starts and the range would freeze at
    the cell it began on. Everything positional therefore comes in as
    arguments -- ``tops`` is the screen y of each track's row, captured while
    the row was drawn, which is also what maps the point through the grid's own
    scrolling child without this function knowing the child exists.

    Between two columns, below a row, or off either end is None rather than the
    nearest cell: a drag that snapped to the closest thing would select cells
    the cursor never touched.
    """
    if cell <= 0.0 or frames < 1:
        return None
    pitch = cell + gutter
    offset = point[0] - x0
    if offset < 0.0:
        return None
    frame = int(offset // pitch)
    if frame >= frames or offset - frame * pitch > cell:
        return None
    for track, top in tops.items():
        if top <= point[1] <= top + cell:
            return (track, frame)
    return None


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


#: The transport's five buttons, as ASCII rather than glyphs.
#:
#: ``icons.py`` is a *transcription* of lucide-static 0.525.0's codepoint
#: assignments, and its docstring forbids guessing one -- so a name the vendored
#: subset does not carry cannot simply be invented here. It has ``play`` and
#: ``square``; it has no skip-back, no skip-forward and no chevron-left, which
#: is four of the five. Two ASCII characters in a button that is already the
#: width of a glyph is the honest fallback, and it is what a video editor's
#: transport has always looked like anyway.
_STEPS = (
    ("first", "|<", "First frame"),
    ("prev", "<", "Previous frame"),
)
_STEPS_AFTER = (
    ("next", ">", "Next frame"),
    ("last", ">|", "Last frame"),
)


#: The frame-duration box, in design pixels.
#:
#: Wide because ``input_int`` with a step draws its own -/+ buttons *inside* the
#: item width: at 1.5 those two take about 110 px, so the 90 this used to be
#: left roughly a character and a half for the number. The row it is on can
#: afford it -- ``toolbar`` wraps a trailing block that no longer fits onto its
#: own line rather than clipping it -- which is the trade: a taller row on a
#: narrow strip, and a legible figure at every width.
MS_W = 128.0


def _transport(ctx: Any, tab: Any) -> None:
    """Two rows, both laid out by :mod:`~warlock.studio.toolbar`.

    This was the worst ``same_line`` chain in the app -- seventeen items across
    one row, so at 150 % the export buttons were simply not on screen. It is
    split by what the controls are *about*: the top row is the frame you are on
    (transport, the frame operations, the counter and that frame's duration),
    the bottom row is what leaves the app (the three exports, with the two view
    toggles and the export scale beside them).

    The transport is pinned, so it collapses to glyphs and stops -- a play
    button that moves into an overflow menu when the window is dragged is not a
    transport. **Delete frame** is pinned for the other half of the same rule.
    The non-buttons -- the counter, the duration box, the toggles, the scale
    combo, the (?) -- go in each row's ``trailing``, which is measured before
    the tiers are chosen and so cannot be the thing that gets clipped.
    """
    doc = tab.doc
    anim = doc.anim
    state = ctx.state.inker
    index = tab.play_index if tab.playing else anim.current

    items = [
        toolbar.Item(key, label, tooltip=tip, enabled=not tab.busy, pinned=True)
        for key, label, tip in _STEPS
    ]
    items.append(
        toolbar.Item(
            "play",
            "Stop" if tab.playing else "Play",
            icons.SQUARE if tab.playing else icons.PLAY,
            tooltip="Stop playback" if tab.playing else "Play the clip (Space)",
            enabled=not tab.saving,
            pinned=True,
        )
    )
    items += [
        toolbar.Item(key, label, tooltip=tip, enabled=not tab.busy, pinned=True)
        for key, label, tip in _STEPS_AFTER
    ]
    items += [
        toolbar.Item(
            "add", "Frame", icons.PLUS, tooltip="Add an empty frame",
            enabled=not tab.busy, priority=1,
        ),
        toolbar.Item(
            "copy", "Copy", icons.COPY, tooltip="Add a copy of this frame",
            enabled=not tab.busy, priority=1,
        ),
        toolbar.Item(
            "link", "Link",
            tooltip="Add a frame whose cels are links to this one's",
            enabled=not tab.busy, priority=1,
        ),
        toolbar.Item(
            "remove", "Delete frame", icons.TRASH,
            enabled=not tab.busy and len(anim.frames) > 1,
            reason="A clip needs at least one frame.",
            role=toolbar.ButtonRole.DESTRUCTIVE, pinned=True, priority=1,
        ),
    ]
    toolbar.toolbar(
        "inker-transport",
        items,
        lambda key: _frame_action(ctx, tab, key),
        trailing=_frame_trailing(ctx, tab, index),
    )

    toolbar.toolbar(
        "inker-timeline-out",
        [
            toolbar.Item(
                "sheet", "Export sheet", icons.GRID,
                tooltip="Writes a packed PNG of every frame plus a JSON sidecar "
                "naming the cells, their durations and any tags.",
                enabled=not tab.busy,
            ),
            toolbar.Item(
                "gif", "Export GIF", icons.FILM,
                tooltip="Writes the whole timeline as an animated GIF, looping. A "
                "GIF holds no partial transparency and times frames in hundredths "
                "of a second, so soft edges become hard ones and a duration is "
                "rounded to the nearest 10 ms.",
                enabled=not tab.busy,
            ),
            toolbar.Item(
                "pngs", "Export PNGs", icons.IMAGE,
                tooltip="Writes one numbered PNG per frame beside the name you "
                "pick -- name_0000.png, name_0001.png and so on.",
                enabled=not tab.busy,
            ),
        ],
        lambda key: _export_action(ctx, tab, key),
        trailing=_output_trailing(ctx, state),
    )
    _onion_controls(state)


def _frame_action(ctx: Any, tab: Any, key: str) -> None:
    doc = tab.doc
    anim = doc.anim
    if key == "first":
        doc.set_current_frame(0)
    elif key == "prev":
        inker_mode.step_frame(ctx, -1, tab)
    elif key == "play":
        inker_mode.toggle_play(ctx, tab)
    elif key == "next":
        inker_mode.step_frame(ctx, 1, tab)
    elif key == "last":
        doc.set_current_frame(len(anim.frames) - 1)
    elif key == "add":
        doc.add_frame()
    elif key == "copy":
        doc.add_frame(copy=True)
    elif key == "link":
        doc.add_frame(link=True)
    elif key == "remove":
        doc.remove_frame()


def _export_action(ctx: Any, tab: Any, key: str) -> None:
    if key == "sheet":
        inker_mode.export_sheet(ctx, tab)
    elif key == "gif":
        inker_mode.export_gif(ctx, tab)
    elif key == "pngs":
        inker_mode.export_pngs(ctx, tab)


def _frame_trailing(ctx: Any, tab: Any, index: int) -> tuple[float, Any]:
    """Where you are in the clip, and how long this frame lasts."""
    anim = tab.doc.anim
    # The position only. The clip's total used to be here too and it was the
    # widest thing on the app's tightest row, sitting next to a box that shows
    # this frame's duration -- two numbers in milliseconds a hand's breadth
    # apart, one of which is not about the frame you are on.
    counter = f"{index + 1}/{len(anim.frames)}"
    gap = imgui.get_style().item_spacing.x
    width = (
        imgui.calc_text_size(counter).x
        + sp(MS_W)
        + imgui.calc_text_size("ms").x
        + gap * 2
    )

    def draw_it() -> None:
        imgui.text(counter)
        imgui.same_line()
        # The frame the counter is naming, which during playback is the one
        # going past rather than the one the playhead will come back to.
        # Read-only while it moves: an edit box whose value changes ten times a
        # second is not something a user can type into, and ``tick_playback``
        # deliberately does not move ``anim.current``, so a write here would
        # land on a frame that is not the one on screen.
        imgui.set_next_item_width(sp(MS_W))
        imgui.begin_disabled(tab.busy)
        changed, value = controls.input_int("ms", anim.frames[index].duration_ms, 10, 50)
        if changed:
            tab.doc.set_frame_duration(index, value)
        imgui.end_disabled()

    return (width, draw_it)


def _output_trailing(ctx: Any, state: Any) -> tuple[float, Any]:
    """The two view toggles, the export magnification, and the (?)."""
    gap = imgui.get_style().item_spacing.x
    switch = sp(32) + sp(6)
    width = (
        switch * 2
        + imgui.calc_text_size("Onion").x
        + imgui.calc_text_size("Thumbs").x
        + sp(64)
        + sp(26)
        + gap * 4
    )

    def draw_it() -> None:
        changed, value = widgets.toggle("Onion", state.onion, tag="inker-onion")
        if changed:
            state.onion = value
        imgui.same_line()
        changed, value = widgets.toggle(
            "Thumbs", state.timeline_thumbs, tag="inker-thumbs"
        )
        if changed:
            state.timeline_thumbs = value
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Draws each cel's picture in its timeline cell, and grows the "
                "cells to fit. Linked cels share one thumbnail, so a link is "
                "visible as the same drawing in several columns."
            )
        imgui.same_line()
        scale = widgets.combo(
            "##inkerscale", str(int(state.export_scale)), list(EXPORT_SCALES), sp(64)
        )
        state.export_scale = max(1, int(scale))
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Magnifies every export by a whole number, nearest neighbour -- "
                "each pixel drawn N times and nothing resampled. The sheet "
                "sidecar is built on the scaled size, so its cells and trims "
                "describe the file that is written; sidecars bound for "
                "Packwright are not scaled."
            )
        manual_render.help_button(ctx, "inker-timeline")

    return (width, draw_it)


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

    Its own row, rather than continuing one. It used to ``same_line`` after the
    duration box because the transport row was already full and a ``same_line``
    past the panel edge does not wrap, it hides the control; now the row above
    ends in a right-aligned (?), so continuing it would put these three past
    the edge for certain.
    """
    if not state.onion:
        return
    imgui.set_next_item_width(sp(70))
    changed, value = controls.input_int("back", state.onion_before, 1, 1)
    if changed:
        state.onion_before = max(0, min(int(value), MAX_ONION))
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    changed, value = controls.input_int("ahead", state.onion_after, 1, 1)
    if changed:
        state.onion_after = max(0, min(int(value), MAX_ONION))
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed, alpha = controls.slider_float("fade", state.onion_alpha, 0.05, 1.0, "%.2f")
    if changed:
        state.onion_alpha = min(1.0, max(0.05, float(alpha)))


def _grid(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    anim = doc.anim
    state = ctx.state.inker
    cell = sp(THUMB_CELL if state.timeline_thumbs else CELL)
    gutter = sp(GUTTER)
    if not imgui.begin_child("inker-timeline-grid", (0, 0), 0):
        imgui.end_child()
        return

    _frame_headers(ctx, tab, cell, gutter)
    # Where every cell ended up, filled in as the rows draw. The marquee is
    # measured against this rather than against hover, and the numbers have to
    # be *screen* coordinates so the scrolling child maps for free.
    geom: dict[str, Any] = {
        "x0": 0.0,
        "tops": {},
        "cell": cell,
        "gutter": gutter,
        "frames": len(anim.frames),
    }
    # Top first, like the layers panel: the engine's list is painter's order and
    # the timeline reads the same way down the page that the stack does.
    for index in range(len(anim.tracks) - 1, -1, -1):
        _track_row(ctx, tab, index, cell, gutter, geom)
    _range_gesture(ctx, tab, geom)
    _range_overlay(ctx, tab, geom)
    _tag_row(ctx, tab, cell, gutter)

    imgui.end_child()


def _range_gesture(ctx: Any, tab: Any, geom: dict[str, Any]) -> None:
    """Extend the range while the mouse is held, and clear it on Escape.

    Run once after the rows rather than per cell, because the whole point of
    measuring geometrically is that the cell under the cursor is not the cell
    that owns the press.
    """
    state = ctx.state.inker
    if tab.range_sel is not None and imgui.is_key_pressed(imgui.Key.escape):
        tab.range_sel = None
        state.timeline_anchor = None
        return
    if state.timeline_anchor is None or not imgui.is_mouse_down(0):
        return
    hit = cell_index(tuple(imgui.get_mouse_pos()), **geom)
    if hit is None:
        return
    anchor_t, anchor_f = state.timeline_anchor
    track, frame = hit
    tab.range_sel = (
        min(anchor_t, track),
        max(anchor_t, track),
        min(anchor_f, frame),
        max(anchor_f, frame),
    )


def _range_overlay(ctx: Any, tab: Any, geom: dict[str, Any]) -> None:
    """One accent outline round the whole range, not a tint per cell.

    A per-cell fill would fight the three cel states the cells already use
    colour for -- empty, drawn, linked -- and the selection is one thing rather
    than n things.
    """
    rect = tab.range_sel
    if rect is None:
        return
    t0, t1, f0, f1 = rect
    tops: dict[int, float] = geom["tops"]
    cell, gutter, frames = geom["cell"], geom["gutter"], geom["frames"]
    # Clamped *here*, at use: the stored rect is allowed to name frames a
    # delete has since taken away, and trimming it on every edit would shrink
    # the user's selection under them.
    rows = [top for track, top in tops.items() if t0 <= track <= t1]
    if not rows or frames < 1:
        return
    lo = max(0, min(int(f0), frames - 1))
    hi = max(0, min(int(f1), frames - 1))
    if hi < lo:
        return
    pitch = cell + gutter
    x = geom["x0"] + pitch * lo
    imgui.get_window_draw_list().add_rect(
        (x - 1.0, min(rows) - 1.0),
        (x + pitch * (hi - lo) + cell + 1.0, max(rows) + cell + 1.0),
        _u32(theme.ACCENT),
        # rounding is the 4th positional and thickness the *5th*.
        0.0,
        sp(2),
    )


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
        if controls.button(label, (cell, cell)) and not tab.busy:
            tab.doc.set_current_frame(index)
        if current:
            imgui.pop_style_color()
        _frame_menu(tab, index)
        imgui.pop_id()


def _frame_menu(tab: Any, index: int) -> None:
    doc = tab.doc
    if not imgui.begin_popup_context_item(f"framemenu{index}"):
        return
    widgets.popup_chrome(_imgui=imgui)
    imgui.begin_disabled(tab.busy)
    if controls.menu_item_simple("Insert before"):
        doc.add_frame(index)
    if controls.menu_item_simple("Duplicate (copied)"):
        doc.set_current_frame(index)
        doc.add_frame(index + 1, copy=True)
    if controls.menu_item_simple("Duplicate (linked)"):
        doc.set_current_frame(index)
        doc.add_frame(index + 1, link=True)
    imgui.separator()
    # Disabled at the ends rather than clicked-and-ignored: an enabled item that
    # does nothing reads as a bug in the move, not as "there is nowhere to go".
    last = len(doc.anim.frames) - 1
    imgui.begin_disabled(index <= 0)
    if controls.menu_item_simple("Move left"):
        doc.move_frame(index, index - 1)
    imgui.end_disabled()
    imgui.begin_disabled(index >= last)
    if controls.menu_item_simple("Move right"):
        doc.move_frame(index, index + 1)
    imgui.end_disabled()
    imgui.separator()
    if controls.menu_item_simple("Delete"):
        doc.remove_frame(index)
    imgui.separator()
    # A one-frame span, renamed and stretched from the tag's own menu below.
    # The alternative -- a modal asking for a name and a range up front -- is
    # three answers for something the user is about to look at and adjust
    # anyway, and there is no frame-range selection for it to read.
    if controls.menu_item_simple("New tag here"):
        doc.add_tag(f"tag {len(doc.anim.tags) + 1}", index)
    imgui.end_disabled()
    imgui.end_popup()


def _track_row(
    ctx: Any,
    tab: Any,
    track_index: int,
    cell: float,
    gutter: float,
    geom: dict[str, Any] | None = None,
) -> None:
    doc = tab.doc
    anim = doc.anim
    track = anim.tracks[track_index]
    active_track = track_index == doc.stack.active_index

    imgui.push_id(f"tr{track.uid}")
    # Indent only (L3 v1): no header row and no fold, so a grouped track's row
    # is exactly like an ungrouped one except that its label is shifted right
    # by its nesting depth. ``same_line(sp(TRACK_LABEL_W))`` below is an
    # *absolute* offset from the window's own left edge, not from wherever the
    # indent left the cursor, so it puts the first cell at the same x either
    # way -- the indent cannot move a cell, only the text before it.
    depth = len(track_depth(doc, track.uid))
    if depth:
        imgui.indent(sp(GROUP_INDENT) * depth)
    if active_track:
        widgets.text_colored(theme.ACCENT, track.name[:14])
    elif not track.visible:
        widgets.muted(track.name[:14])
    else:
        imgui.text(track.name[:14])
    if depth:
        imgui.unindent(sp(GROUP_INDENT) * depth)
    imgui.same_line(sp(TRACK_LABEL_W))

    for frame_index, frame in enumerate(anim.frames):
        if frame_index:
            imgui.same_line(0.0, gutter)
        imgui.push_id(frame.uid)
        _cell(ctx, tab, track, frame, track_index, frame_index, cell, geom)
        imgui.pop_id()
    imgui.pop_id()


def _cell(
    ctx: Any,
    tab: Any,
    track: Any,
    frame: Any,
    ti: int,
    fi: int,
    cell: float,
    geom: dict[str, Any] | None = None,
) -> None:
    doc = tab.doc
    anim = doc.anim
    state = ctx.state.inker
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
    if controls.button(label, (cell, cell)) and not tab.busy:
        doc.set_active_layer(ti)
        doc.set_current_frame(fi)
    imgui.pop_style_color()
    if geom is not None:
        low = imgui.get_item_rect_min()
        geom["tops"][ti] = low.y
        if fi == 0:
            geom["x0"] = low.x
    # ``is_item_clicked`` and not the button's own return: a marquee starts on
    # the *press*, and the button answers on the release -- by which time the
    # drag is over.
    if imgui.is_item_clicked(0):
        _press(ctx, tab, ti, fi)
    elif imgui.is_item_clicked(1) and not _in_range(tab.range_sel, ti, fi):
        # A right-click outside the range moves it here first, so the menu
        # below can never act on cells that are not the ones under the cursor.
        _press(ctx, tab, ti, fi)
    if state.timeline_thumbs:
        _cel_thumb(ctx, tab, layer, cell)
    _cell_menu(ctx, tab, ti, fi, layer is not None, linked)


def _in_range(rect: tuple[int, int, int, int] | None, ti: int, fi: int) -> bool:
    if rect is None:
        return False
    t0, t1, f0, f1 = rect
    return t0 <= ti <= t1 and f0 <= fi <= f1


def _press(ctx: Any, tab: Any, ti: int, fi: int) -> None:
    """Start a range at this cell, or extend the last one to it with Shift."""
    state = ctx.state.inker
    if imgui.get_io().key_shift and state.timeline_anchor is not None:
        anchor_t, anchor_f = state.timeline_anchor
        tab.range_sel = (
            min(anchor_t, ti),
            max(anchor_t, ti),
            min(anchor_f, fi),
            max(anchor_f, fi),
        )
        return
    state.timeline_anchor = (ti, fi)
    tab.range_sel = (ti, ti, fi, fi)


def _cel_thumb(ctx: Any, tab: Any, layer: Any, cell: float) -> None:
    """Draw a cel's picture inside the button that was just laid out.

    Over the button rather than instead of it, so the three cel-state colours
    and the whole right-click surface are untouched -- and an empty slot draws
    nothing at all, which is what a placeholder should look like.

    Only asked for when the cell is actually on screen: the grid can be fifty
    wide inside a scroller a dozen cells across, and requesting a texture per
    cell would upload the other thirty-eight every frame.
    """
    from . import inker_textures

    if layer is None or ctx.viewer is None:
        return
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    if not imgui.is_rect_visible(low, high):
        return
    texture = inker_textures.cel_thumb(ctx, tab, layer, int(max(8.0, cell)))
    if texture is None:
        return
    imgui.get_window_draw_list().add_image(
        widgets.texture_ref(texture),
        (low.x + 1.0, low.y + 1.0),
        (high.x - 1.0, high.y - 1.0),
    )


def _cell_menu(
    ctx: Any, tab: Any, ti: int, fi: int, has_cel: bool, linked: bool
) -> None:
    doc = tab.doc
    if not imgui.begin_popup_context_item("celmenu"):
        return
    widgets.popup_chrome(_imgui=imgui)
    imgui.begin_disabled(tab.busy)
    imgui.begin_disabled(fi <= 0)
    if controls.menu_item_simple("Link to previous frame"):
        doc.link_cel(fi - 1, track_index=ti, frame_index=fi)
    imgui.end_disabled()
    if linked and controls.menu_item_simple("Unlink"):
        doc.unlink_cel(track_index=ti, frame_index=fi)
    if has_cel and controls.menu_item_simple("Clear"):
        doc.clear_cel(track_index=ti, frame_index=fi)
    _range_menu(ctx, tab)
    imgui.end_disabled()
    imgui.end_popup()


def _range_menu(ctx: Any, tab: Any) -> None:
    """Everything the range ops offer, as one section of the cell menu.

    **Disabled, never hidden.** A menu whose items appear and disappear with
    the selection is one the user has to re-learn every time they open it; a
    greyed row says "this exists and here is why you cannot have it", which is
    the same argument ``_frame_menu``'s move items already make at the ends of
    the timeline.
    """
    state = ctx.state.inker
    doc = tab.doc
    rect = tab.range_sel
    imgui.separator()
    widgets.muted("Range")
    imgui.begin_disabled(rect is None)
    # With no range, the corner is where the user is: the active track and the
    # playhead. Only Paste can be reached in that state, and "put it here" is
    # what it should mean.
    here = (doc.stack.active_index, doc.stack.active_index, doc.anim.current, doc.anim.current)
    t0, t1, f0, f1 = rect or here
    if controls.menu_item_simple("Copy cels"):
        state.cel_clip = doc.copy_cels(t0, t1, f0, f1)
    imgui.end_disabled()
    # Paste is the one item whose gate is the *clipboard* rather than the
    # selection: it lands at the range's corner, and with no range at all the
    # playhead and active track are the corner.
    imgui.begin_disabled(state.cel_clip is None)
    if controls.menu_item_simple("Paste cels"):
        doc.paste_cels(state.cel_clip, t0, f0)
    imgui.end_disabled()

    imgui.begin_disabled(rect is None)
    imgui.separator()
    if controls.menu_item_simple("Clear cels"):
        doc.clear_range(t0, t1, f0, f1)
    if controls.menu_item_simple("Link cels"):
        doc.link_range(t0, t1, f0, f1)
    if controls.menu_item_simple("Unlink cels"):
        doc.unlink_range(t0, t1, f0, f1)

    imgui.separator()
    if controls.menu_item_simple("Duplicate frames"):
        doc.duplicate_range(f0, f1)
    if controls.menu_item_simple("Duplicate frames (linked)"):
        doc.duplicate_range(f0, f1, link=True)
    if controls.menu_item_simple("Reverse frames"):
        doc.reverse_range(f0, f1)
    if controls.menu_item_simple("Delete frames"):
        doc.remove_range(f0, f1)

    imgui.separator()
    imgui.set_next_item_width(sp(90))
    changed, value = controls.input_int("ms##rangems", state.range_ms, 10, 50)
    if changed:
        state.range_ms = max(animation.MIN_DURATION_MS, int(value))
    if controls.menu_item_simple("Set frame durations"):
        doc.set_range_duration(f0, f1, state.range_ms)
    _range_export_items(ctx, tab, f0, f1)
    imgui.end_disabled()


def _range_export_items(ctx: Any, tab: Any, f0: int, f1: int) -> None:
    """Export just this span, as the same three files the whole clip offers.

    Frames only: the range's track bounds are about *cels*, and every export
    writes flattened frames -- a sheet of "tracks 2-3 of frames 4-9" is not a
    thing the sidecar can describe.
    """
    imgui.separator()
    if controls.menu_item_simple("Export range → sheet"):
        inker_mode.export_range(ctx, tab, "sheet", (f0, f1))
    if controls.menu_item_simple("Export range → GIF"):
        inker_mode.export_range(ctx, tab, "gif", (f0, f1))
    if controls.menu_item_simple("Export range → PNG sequence"):
        inker_mode.export_range(ctx, tab, "pngs", (f0, f1))


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
            widgets.muted(f"{tag.name}{_tag_note(tag)}")
            _tag_menu(ctx, tab, index, tag)
        imgui.pop_id()


#: How each direction is written beside a tag's name. The default one is spelt
#: as nothing at all: a forward loop is what a tag has always been, so labelling
#: it would put a word on every tag in the band to distinguish the ordinary case
#: from itself.
DIRECTION_NOTES = {"forward": "", "reverse": "reverse", "pingpong": "ping-pong"}


def _tag_note(tag: Any) -> str:
    """The parenthesised aside after a tag's name, or nothing to say."""
    repeat = int(getattr(tag, "repeat", 0) or 0)
    parts = [
        DIRECTION_NOTES.get(tag.direction, ""),
        # A repeat count *replaces* the loop note rather than joining it: it is
        # the answer to the same question -- how many times does this play --
        # and printing "once" beside "x3" would be two answers to it.
        f"x{repeat}" if repeat else ("" if tag.loop else "once"),
    ]
    said = [part for part in parts if part]
    return f" ({', '.join(said)})" if said else ""


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
    entered, value = controls.input_text(
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
    widgets.popup_chrome(_imgui=imgui)
    imgui.begin_disabled(tab.busy)
    if controls.menu_item_simple("Rename"):
        state.tag_editing = index
        state.tag_name = tag.name
    # Both ends from the playhead, which is the frame the user just clicked to
    # get here: a tag is a span of the timeline and the timeline is what they
    # are looking at, so there is nothing to type.
    if controls.menu_item_simple(f"Start at frame {doc.anim.current + 1}"):
        doc.set_tag(index, start=doc.anim.current)
    if controls.menu_item_simple(f"End at frame {doc.anim.current + 1}"):
        doc.set_tag(index, end=doc.anim.current)
    repeat = int(getattr(tag, "repeat", 0) or 0)
    # Disabled rather than hidden: a count is the more specific answer to "how
    # many times", so while one is set the flag has nothing left to decide --
    # and an enabled tick that changed nothing would read as a bug in the flag.
    imgui.begin_disabled(repeat > 0)
    if controls.menu_item_simple("Loop", "", tag.loop):
        doc.set_tag(index, loop=not tag.loop)
    imgui.end_disabled()
    # Straight onto ``set_tag``, which snapshots the whole tag list into a
    # ``TagsEdit`` -- so a repeat count is undoable for free and needs no edit
    # type of its own. 0 hands the question back to the Loop flag above.
    imgui.set_next_item_width(sp(90))
    changed, value = controls.input_int("repeat", repeat, 1, 1)
    if changed:
        doc.set_tag(index, repeat=max(0, int(value)))
    widgets.help_marker(
        "How many times this tag plays before stopping. 0 leaves it to the Loop"
        " flag. Playback stays inside the tag when the count runs out -- it does"
        " not carry on into the frames after it."
    )
    # Radio items rather than a submenu: three mutually exclusive values that
    # each fit on a line, and the tick is the answer to "which way does this
    # one go" without a hover. Straight off ``animation.DIRECTIONS`` -- a
    # hand-written list here would be a second table of the same three names.
    for key in animation.DIRECTIONS:
        if controls.menu_item_simple(key.capitalize(), "", tag.direction == key):
            doc.set_tag(index, direction=key)
    imgui.separator()
    # The tag's own span, and its own looping: a tag is the one part of the
    # timeline that already says both which frames it covers and how many times
    # they play, so exporting one needs nothing typed.
    if controls.menu_item_simple("Export tag → sheet"):
        inker_mode.export_tag(ctx, tab, "sheet", index)
    if controls.menu_item_simple("Export tag → GIF"):
        inker_mode.export_tag(ctx, tab, "gif", index)
    imgui.separator()
    if controls.menu_item_simple("Delete tag"):
        doc.remove_tag(index)
        state.tag_editing = -1
    imgui.end_disabled()
    imgui.end_popup()
