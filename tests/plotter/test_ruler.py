"""Plotter's canvas rulers, which count cells.

Inker has had rulers since 2026-08; Plotter had none at all, so the only answer
to "how far across the map am I" was the status line's readout of the one cell
under the pointer. The bands are Inker's -- same thickness, same 1/2/5 tick
ladder, same cursor shadow, and the ladder is *imported* rather than copied so
the two cannot drift.

**The number on the band is the difference.** Inker measures pixels because a
pixel is what its user places; this measures cells, because a cell is. A band
that read "512" where the user counts thirty-two tiles would be a second
coordinate system to hold in the head, which is exactly what the cell readout
already refuses to be -- so that is what these tests assert, on the real draw
call rather than on a helper.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import inker_state, plotter_mode
from warlock.studio.panes import plotter_canvas as canvas
from warlock.studio.plotter.tilemap import MapDoc


class FakeDrawList:
    """imgui's draw list, recording instead of drawing.

    The ruler is pure output, so what it *drew* is the only thing there is to
    assert. A recording list is what lets that be asserted without a GL context
    and without a screenshot nobody would read.
    """

    def __init__(self) -> None:
        self.lines: list[tuple[tuple, tuple]] = []
        self.rects: list[tuple[tuple, tuple]] = []
        self.texts: list[tuple[tuple, str]] = []

    def add_line(self, a, b, _colour, _thickness=1.0) -> None:
        self.lines.append((tuple(a), tuple(b)))

    def add_rect_filled(self, a, b, _colour, *_rest) -> None:
        self.rects.append((tuple(a), tuple(b)))

    def add_text(self, at, _colour, text) -> None:
        self.texts.append((tuple(at), str(text)))


@pytest.fixture
def bands(monkeypatch):
    """The real ``_rulers`` over a recording list. -> ``(tab, draw)``.

    ``get_color_u32`` and ``get_mouse_pos`` are the only imgui calls the bands
    make, and both are read through a function-local import -- so they are
    patched on the module itself, as ``_drive`` does for the mouse.
    """
    from imgui_bundle import imgui

    monkeypatch.setattr(imgui, "get_color_u32", lambda _colour: 0xFFFFFFFF)
    monkeypatch.setattr(imgui, "get_mouse_pos", lambda: SimpleNamespace(x=-1.0, y=-1.0))

    doc = MapDoc(64, 64, 32, 32)
    doc.add_tile_layer("Tiles")
    tab = SimpleNamespace(
        doc=doc,
        view=inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True),
    )
    return tab, FakeDrawList()


def _draw(tab, draw, *, hovered: bool = False, region=(640.0, 640.0)):
    canvas._rulers(tab, draw, (0.0, 0.0), region, hovered=hovered)
    return draw


# --- what the numbers mean ----------------------------------------------------


def test_the_labels_count_cells_and_not_pixels(bands):
    """At 32px tiles, zoom 1 and a 640px pane, twenty cells are on screen. A
    pixel ruler would be labelling 0, 100, 200...; this one labels tiles."""
    tab, draw = bands
    _draw(tab, draw)
    values = sorted({int(text) for _at, text in draw.texts})
    assert values, "the band drew no labels at all"
    assert max(values) <= 20, f"pixel-space labels leaked through: {values}"
    # 32 screen px a cell, so the 1/2/5 ladder settles on every second cell.
    assert 0 in values and 2 in values and 10 in values


def test_a_label_sits_where_that_cell_is_drawn(bands):
    """The number and the tick have to agree with the grid under them, or the
    band is decoration."""
    tab, draw = bands
    _draw(tab, draw)
    # The top band writes at a fixed y and the left band at a fixed x; the
    # left band's own zeroth label lands on both, so it is excluded by x.
    horizontal = [(at, text) for at, text in draw.texts if at[1] == 1.0 and at[0] != 2.0]
    assert horizontal
    for at, text in horizontal:
        corner = tab.doc.cell_corner(int(text), 0)
        expected = inker_state.to_screen(tab.view, (0.0, 0.0), *corner)
        assert at[0] == pytest.approx(expected[0] + 3.0), text


def test_zooming_out_coarsens_the_step_up_the_one_two_five_ladder(bands):
    """Labels never crowd: Inker's ``ruler_step``, imported rather than
    reimplemented, so there is one ladder in the app."""
    tab, draw = bands
    tab.view.zoom = 0.05  # 1.6 screen px per cell
    _draw(tab, draw)
    values = sorted({int(text) for _at, text in draw.texts})
    gaps = {b - a for a, b in zip(values, values[1:], strict=False)}
    assert gaps, "no labels to space"
    step = gaps.pop()
    assert not gaps, "the step is uniform"
    assert step in (10, 20, 50, 100, 200, 500), step


def test_both_bands_are_drawn_with_the_corner_over_them(bands):
    tab, draw = bands
    _draw(tab, draw)
    # Top band, left band, then the corner square that hides both their ticks.
    assert len(draw.rects) == 3
    corner = draw.rects[-1]
    assert corner[0] == (0.0, 0.0)
    assert corner[1][0] == pytest.approx(corner[1][1]), "square"


def test_the_cursor_shadow_appears_only_while_the_pane_is_hovered(bands, monkeypatch):
    from imgui_bundle import imgui

    tab, draw = bands
    monkeypatch.setattr(imgui, "get_mouse_pos", lambda: SimpleNamespace(x=100.0, y=200.0))
    cold = _draw(tab, FakeDrawList(), hovered=False)
    warm = _draw(tab, draw, hovered=True)
    assert len(warm.lines) == len(cold.lines) + 2


# --- the toggle ---------------------------------------------------------------


def test_the_canvas_frame_gates_the_bands_on_the_setting():
    """``draw`` needs a whole app to run, so the one line that matters is read
    off the module. Everything downstream of it is driven for real above."""
    import inspect

    body = inspect.getsource(canvas.draw)
    assert "if state.rulers:" in body and "_rulers(tab, draw_list" in body


def test_ctrl_r_toggles_the_rulers(plotter_ctx):
    import pygame

    ctx, state = plotter_ctx
    assert state.rulers is True
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, mod=pygame.KMOD_CTRL)
    assert plotter_mode.handle_key(ctx, event) is True
    assert state.rulers is False
    plotter_mode.handle_key(ctx, event)
    assert state.rulers is True


def test_the_sidebar_offers_the_toggle_beside_the_grid():
    import inspect

    from warlock.studio.panes import plotter_tools

    body = inspect.getsource(plotter_tools._body)
    assert 'widgets.toggle("Rulers (Ctrl+R)", state.rulers)' in body
