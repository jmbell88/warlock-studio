"""The two view marks over the sprite, drawn in a real imgui context.

``test_scorecard_draw``'s shape and its argument: what is asserted here is
*geometry* -- where a cross lands relative to the sprite's own top-left corner
-- and a source-text assertion cannot say anything about that. The bug this
guards is silent by construction: a marker one zoom factor out, or one drawn at
a guessed centre, looks exactly like a correct one to anybody who has not
measured it.

The claim that matters most is the last one. A sheet that records no pivot must
get **no mark**, not a mark at the middle of the cell: the marker says "this is
where the engine will put the sprite's origin", and drawn from a guess it is a
lie the user cannot tell from a measurement.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import troupe_mode
from warlock.studio.panes import troupe_preview


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    fixture there."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


CELL = 32
COLUMNS = 8
ZOOM = 4
PIVOT = (11.0, 27.0)


class _Spy:
    """The window draw list, recording every call and forwarding all of them.

    Forwarding rather than faking: the point of a real context is that imgui
    itself rejects an argument list the binding will not take, which is exactly
    the failure ``test_scorecard_draw`` was written for one pane over.
    """

    def __init__(self, real):
        self.real = real
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        target = getattr(self.real, name)

        def recorded(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return target(*args, **kwargs)

        return recorded

    def of(self, name: str) -> list[tuple]:
        return [args for called, args, _kw in self.calls if called == name]


class _Texture:
    """Enough of a moderngl texture for ``widgets.texture_ref``: there is no
    renderer in this context, so nothing is registered and only ``glo`` and
    ``size`` are ever read."""

    glo = 1
    size = (COLUMNS * CELL, COLUMNS * CELL)


def _record(*, pivot: tuple[float, float] | None = PIVOT) -> dict:
    cell: dict = {"index": 0, "x": 0, "y": 0, "w": CELL, "h": CELL}
    if pivot is not None:
        cell["pivot_x"], cell["pivot_y"] = pivot
    return {"columns": COLUMNS, "frame_size": CELL, "cells": [cell]}


def _draw(ui, monkeypatch, record, *, checker: bool, show_pivot: bool) -> _Spy:
    monkeypatch.setattr(troupe_mode, "cell_index", lambda ctx: 0)
    ctx = SimpleNamespace()
    state = SimpleNamespace(zoom=ZOOM, checker=checker, show_pivot=show_pivot)
    spy: list[_Spy] = []
    real = ui.get_window_draw_list

    def spied():
        if not spy:
            spy.append(_Spy(real()))
        return spy[0]

    ui.new_frame()
    ui.begin("host")
    # Swapped by hand rather than through ``monkeypatch``: undoing it has to
    # happen inside the frame, before ``end``, and ``monkeypatch.undo`` would
    # take the fixture's own patches with it.
    ui.get_window_draw_list = spied
    try:
        troupe_preview._sprite(ctx, state, _Texture(), record)
    finally:
        ui.get_window_draw_list = real
        ui.end()
        ui.end_frame()
    assert spy, "_sprite never asked for the window draw list"
    return spy[0]


def test_the_checkerboard_is_drawn_under_the_sprite_only_when_it_is_on(
    ui, monkeypatch
):
    """Off by default, because a pattern behind every frame is noise until the
    question is "where is the transparency" -- and on, it must be *filled
    rectangles*, which is the one thing the toggle is for."""
    off = _draw(ui, monkeypatch, _record(), checker=False, show_pivot=False)
    assert off.of("add_rect_filled") == [], "nothing paints a ground when it is off"

    on = _draw(ui, monkeypatch, _record(), checker=True, show_pivot=False)
    squares = on.of("add_rect_filled")
    assert len(squares) > 1, "a checkerboard is a ground plus its light squares"


def test_the_pivot_marker_lands_on_the_pivot_the_sidecar_records(ui, monkeypatch):
    """**The geometry, measured rather than asserted about.** The mark is
    placed at ``sprite top-left + pivot * zoom``; the checkerboard's own ground
    rectangle is what tells this test where that top-left is, so the two
    numbers come from the same draw rather than from arithmetic repeated here.

    A marker that forgot the zoom, or that used design pixels, passes every
    source-text check ever written and sits in the wrong place on screen."""
    spy = _draw(ui, monkeypatch, _record(), checker=True, show_pivot=True)
    ground = spy.of("add_rect_filled")[0]
    low = ground[0]

    circles = spy.of("add_circle")
    assert len(circles) == 1, "one ring, at the origin the engine will use"
    at = circles[0][0]
    assert at[0] == pytest.approx(low[0] + PIVOT[0] * ZOOM)
    assert at[1] == pytest.approx(low[1] + PIVOT[1] * ZOOM)
    assert len(spy.of("add_line")) == 2, "a cross is two arms through the ring"


def test_a_sheet_with_no_pivot_gets_no_marker_rather_than_a_default_one(
    ui, monkeypatch
):
    """The claim: **no pivot in the sidecar, no marker.** The obvious fallback
    is the cell's own centre-bottom -- what ``sheet.sidecar`` writes when the
    renderer measured nothing -- and drawing it would put a confident cross on
    a sheet where nobody measured anything. The toggle is on throughout, so
    this is about the record and not about the switch -- so the same switch is
    proved to draw something on the record that *does* carry a pivot, in this
    test, or "no marker" would be satisfied by a mode that draws none ever."""
    with_pivot = _draw(ui, monkeypatch, _record(), checker=False, show_pivot=True)
    assert with_pivot.of("add_circle"), "the switch is on and the sidecar says where"

    without = _draw(ui, monkeypatch, _record(pivot=None), checker=False, show_pivot=True)
    assert without.of("add_circle") == []
    assert without.of("add_line") == []
