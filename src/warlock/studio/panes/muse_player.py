"""Muse's transport: the take you are listening to, drawn full width.

**Why a waveform here when the tray's card refuses one.** That card's docstring
is right and stands unedited -- "a picture of a waveform tells a listener
nothing that pressing play does not tell them better" is true of a card, which
is a thing you press. A player's waveform is a different claim: it is not a
picture but *the coordinate system the other four controls are expressed in*. A
playhead is nowhere without it, a seek is a click on it, and a loop region is
two positions in it. ``muse.waveform`` carries the same distinction where the
arithmetic lives.

**Why full width, along the bottom.** Rejected alternatives, with reasons worth
keeping: a scrub bar per card is *n* playheads for a mixer with one channel; a
left sidebar puts 240 seconds across 260 dp, which is a second per pixel, and
dragging a loop marker at that scale is a guess rather than an edit; replacing
the recipe column takes away the thing you change *while* listening to the take
you just made. Full width is ~0.15 s/px.

**No undo folding.** A reader arriving from ``sirens_envelopes`` will look for
``controls.fold_undo`` around these drags and there is none, because Muse holds
no document and there is no history to fold into. Every change here is either a
view state or a request for another job row.

Drawn through :func:`layout.pane` for ``muse_brief``'s reason: that is what
registers it in ``layout.FRAME_PANES``, which is what gives it ``guard``'s error
isolation and a slot for ``probe._pane_at`` -- without it ``/exercise-mode
muse`` reports these controls against the empty-string pane.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imgui_bundle import imgui

from .. import controls, layout, muse_io, muse_mode, sirens_audio, theme, widgets
from ..manual import render as manual_render
from ..muse import waveform
from ..muse_state import MAX_XFADE_MS
from ..tokens import sp

#: The strip's height in design pixels: the envelope, a row of transport
#: buttons, and the loop row under it.
STRIP_H = 148.0

#: How tall the waveform itself is.
GRAPH_H = 62.0

#: How wide a loop marker's grab zone is, in design pixels.
#: ``sirens_envelopes.GRIP_W``, restated rather than imported: that constant is
#: about an envelope column and this one is about a time marker, and they agree
#: today by coincidence rather than by argument.
GRIP_W = 5.0

#: Which draggable marker a gesture has hold of, or "". Module state rather than
#: ``MuseState`` for ``sirens_envelopes``' reason: imgui has one active item at
#: a time, so a second drag cannot begin before the first has let go, and a
#: field on the mode's state would imply otherwise.
_grabbed: str = ""


def should_draw(ctx: Any) -> bool:
    """Whether there is a take under the strip. -> False before the first play.

    The strip is *not* drawn merely because a take exists: it is drawn once one
    has been auditioned, because what it shows is a decoded buffer and a mode
    that reserved 148 dp for a picture it has no samples for is a mode with a
    hole in it.
    """
    one = muse_mode.player(ctx)
    return one is not None and one.pcm is not None


def draw(ctx: Any) -> None:
    """The strip. Called from ``main._muse_workspace`` below the two columns."""
    with layout.pane(
        "muse-player",
        (0.0, sp(STRIP_H)),
        layout.PaneRole.CONTENT,
        edge=layout.PaneEdge.TOP,
        title="Player",
    ):
        one = muse_mode.player(ctx)
        if one is None or one.pcm is None:
            return
        manual_render.help_button(ctx, "muse-player")
        _graph(ctx, one)
        _transport(ctx, one)
        _loop_row(ctx, one)


# --- the waveform ------------------------------------------------------------


def _graph(ctx: Any, one: Any) -> None:
    origin = imgui.get_cursor_screen_pos()
    width = max(imgui.get_content_region_avail().x, 1.0)
    height = sp(GRAPH_H)
    draw_list = imgui.get_window_draw_list()

    ground = imgui.get_color_u32(theme.rgba(theme.ELEV_1))
    region_fill = imgui.get_color_u32(theme.rgba(theme.OK, 0.12))
    edge = imgui.get_color_u32(theme.rgba(theme.EDGE))
    wave = imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.85))
    marker = imgui.get_color_u32(theme.rgba(theme.OK))
    head = imgui.get_color_u32(theme.rgba(theme.WARN))

    draw_list.add_rect_filled(
        (origin.x, origin.y), (origin.x + width, origin.y + height), ground
    )
    _region_fill(draw_list, one, origin, width, height, region_fill)

    middle = origin.y + height * 0.5
    draw_list.add_line((origin.x, middle), (origin.x + width, middle), edge, 1.0)
    _envelope(draw_list, one, origin, width, height, middle, wave)
    _markers(draw_list, one, origin, width, height, marker)
    _playhead(ctx, draw_list, one, origin, width, height, head)

    imgui.invisible_button("muse-wave", (width, height))
    _input(ctx, one, origin, width)


def _envelope(draw_list, one, origin, width, height, middle, colour) -> None:
    """One ``add_rect_filled`` per column, capped at the pane's own width.

    ~1600 calls on a wide window, which is the figure worth watching with F10;
    the fallback if it is ever too many is two ``add_polyline`` calls over the
    same two rows.
    """
    if one.env is None:
        return
    columns = int(min(max(width, 1.0), 2048))
    env = waveform.window(one.env, columns)
    half = height * 0.5
    step = width / columns
    lows = middle - np.clip(env[0], -1.0, 1.0) * half
    highs = middle - np.clip(env[1], -1.0, 1.0) * half
    for index in range(columns):
        x = origin.x + index * step
        top, bottom = min(lows[index], highs[index]), max(lows[index], highs[index])
        draw_list.add_rect_filled(
            (x, top), (x + max(step, 1.0), max(bottom, top + 1.0)), colour
        )


def _region_fill(draw_list, one, origin, width, height, colour) -> None:
    if one.loop_start is None or one.loop_end is None:
        return
    a = waveform.at(one.loop_start, one.duration, width)
    b = waveform.at(one.loop_end, one.duration, width)
    draw_list.add_rect_filled(
        (origin.x + a, origin.y), (origin.x + b, origin.y + height), colour
    )


def _markers(draw_list, one, origin, width, height, colour) -> None:
    if one.loop_start is None or one.loop_end is None:
        return
    for seconds, label in ((one.loop_start, "["), (one.loop_end, "]")):
        x = origin.x + waveform.at(seconds, one.duration, width)
        draw_list.add_line((x, origin.y), (x, origin.y + height), colour, 2.0)
        draw_list.add_text((x + 2.0, origin.y), colour, label)


def _playhead(ctx, draw_list, one, origin, width, height, colour) -> None:
    x = origin.x + waveform.at(muse_mode.position(ctx), one.duration, width)
    draw_list.add_line((x, origin.y), (x, origin.y + height), colour, 1.5)


def _input(ctx: Any, one: Any, origin: Any, width: float) -> None:
    """Click to seek; drag a marker to move it.

    **Re-played on release only.** ``muse_mode.seek`` puts a fresh buffer on the
    channel, and ``make_sound`` copies -- so a seek per mouse-move during a drag
    is a ~40 MB copy per frame. The playhead the user drags is drawn from the
    pending value; the sound catches up when they let go.
    """
    global _grabbed
    if not (imgui.is_item_active() or imgui.is_item_deactivated()):
        return
    offset = imgui.get_io().mouse_pos.x - origin.x
    seconds = waveform.seconds_at(offset, one.duration, width)

    if imgui.is_item_activated():
        _grabbed = _grip_at(one, offset, width)

    if _grabbed == "start":
        muse_mode.set_region(ctx, seconds, one.loop_end)
    elif _grabbed == "end":
        muse_mode.set_region(ctx, one.loop_start, seconds)
    elif imgui.is_item_deactivated():
        muse_mode.seek(ctx, seconds)

    if imgui.is_item_deactivated():
        _grabbed = ""


def _grip_at(one: Any, offset: float, width: float) -> str:
    """Which marker, if either, is under ``offset``. -> "start"/"end"/""."""
    if one.loop_start is None or one.loop_end is None:
        return ""
    reach = sp(GRIP_W)
    for seconds, name in ((one.loop_start, "start"), (one.loop_end, "end")):
        if abs(waveform.at(seconds, one.duration, width) - offset) <= reach:
            return name
    return ""


# --- the rows ----------------------------------------------------------------


def _transport(ctx: Any, one: Any) -> None:
    playing = muse_mode.is_playing(ctx, one.job)
    # Not primary: a transport is never a pane's commit verb (Generate is).
    if widgets.transport("muse-player", playing, shortcut=""):
        if playing:
            muse_mode.stop(ctx)
        else:
            # ``play`` itself resumes an already-decoded take from wherever
            # ``stop`` left the playhead (M07) -- the ``seek`` this used to do
            # first was papering over ``play`` always rebuilding the player
            # from a fresh disk read, which is the bug, not this button.
            muse_mode.play(ctx, one.job)
    imgui.same_line()
    widgets.secondary(f"{muse_mode.position(ctx):.1f} / {one.duration:.1f}s")

    imgui.same_line()
    # The device's level, not this mode's. See ``sirens_audio.set_volume``: one
    # reserved channel means one volume, and the two audio modes draw two views
    # of the one value rather than two values.
    imgui.set_next_item_width(sp(140.0))
    changed, level = controls.slider_float(
        "Volume##muse", sirens_audio.volume(), 0.0, 1.0
    )
    if changed:
        sirens_audio.set_volume(level)


def _loop_row(ctx: Any, one: Any) -> None:
    if one.finding:
        widgets.busy("Looking for loop points", note="A full pass over the take.")
        return

    if controls.button("Find loop points"):
        muse_mode.find_loops(ctx)
    imgui.same_line()

    has_region = one.loop_start is not None and one.loop_end is not None
    if controls.button(
        "Play the loop",
        enabled=has_region,
        reason="" if has_region else "no loop region yet",
    ):
        muse_mode.play_region(ctx)
    imgui.same_line()
    if controls.button(
        "Clear", enabled=has_region, reason="" if has_region else "no loop region yet"
    ):
        muse_mode.set_region(ctx, None, None)

    if len(one.candidates) > 1:
        imgui.same_line()
        widgets.secondary(f"{len(one.candidates)} candidates:")
        for index in range(len(one.candidates)):
            imgui.same_line()
            if controls.small_button(f"{index + 1}##muse-cand-{index}"):
                muse_mode.choose_candidate(ctx, index)

    if not has_region:
        return

    imgui.set_next_item_width(sp(180.0))
    changed, fade = controls.slider_float(
        "Crossfade (ms)##muse", float(one.xfade_ms), 0.0, MAX_XFADE_MS
    )
    if changed:
        one.xfade_ms = float(fade)
    imgui.same_line()
    if controls.button("Export the loop"):
        muse_io.export_loop(ctx, one)
    imgui.same_line()
    # **Mutually exclusive with the crossfade, and the reason is the tooltip.**
    # The samples at a crossfaded seam do not exist in the source, so an
    # ``smpl`` chunk pointing into the untouched track is a loop that clicks
    # wearing a label saying it does not. ``muse_io`` refuses it as well; this
    # is the half that stops the press being offered.
    plain = one.xfade_ms <= 0.0
    if controls.button(
        "Export with loop points",
        enabled=plain,
        reason=(
            ""
            if plain
            else "a crossfaded seam cannot be written as loop points -- those"
            " samples are not in the take"
        ),
    ):
        muse_io.export_with_points(ctx, one)


__all__ = ["GRAPH_H", "GRIP_W", "STRIP_H", "draw", "should_draw"]
