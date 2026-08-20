"""The offset-projection (hex/staggered) grid pass in ``plotter_canvas._grid``.

Every other lattice draws its grid with ``add_line``; an offset one has no
straight line running the width of the map, so it draws one closed outline
per cell with ``add_polyline`` instead. The binding reverses imgui's own
``AddPolyline(points, num, col, flags, thickness)`` -- its signature is
``(points, col, thickness, flags)`` -- and a prior transposition
(``thickness``/``flags`` swapped) crashed the frame loop the first time a
hex map was opened, unnoticed because nothing exercised this branch. Every
other ``add_polyline`` call in the pane already had the order right; this
was the only one that did not.
"""

from __future__ import annotations

from types import SimpleNamespace

import imgui_bundle
import pytest

from warlock.studio import inker_state
from warlock.studio.panes import plotter_canvas as canvas
from warlock.studio.plotter.tilemap import MapDoc


class _ClosedValue:
    """Stands in for ``imgui.ImDrawFlags_.closed.value``. The real flag is an
    ``IntFlag``, so it *is* an ``int`` -- a test that only checked
    ``isinstance(flags, int)`` against the real enum would pass even with the
    arguments transposed. An opaque sentinel cannot be mistaken for a width."""


def _fake_flag() -> SimpleNamespace:
    return SimpleNamespace(value=_ClosedValue)


class _FakeDrawList:
    def __init__(self) -> None:
        self.polyline_calls: list[tuple] = []

    def add_line(self, *_args, **_kwargs) -> None:
        pass

    def add_polyline(self, points, col, thickness, flags) -> None:
        assert isinstance(thickness, float), (
            f"thickness must be the float line width, got {type(thickness)}"
        )
        assert flags is _ClosedValue, (
            f"flags must be the closed-outline flag, got {type(flags)}"
        )
        self.polyline_calls.append((points, col, thickness, flags))


def test_the_hex_grid_draws_one_closed_outline_per_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_imgui = SimpleNamespace(
        get_color_u32=lambda _rgba: 0xFFFFFFFF,
        ImDrawFlags_=SimpleNamespace(closed=_fake_flag()),
    )
    monkeypatch.setattr(imgui_bundle, "imgui", fake_imgui)

    doc = MapDoc(4, 4, 16, 16, projection="hexagonal")
    view = inker_state.PaintView()
    draw_list = _FakeDrawList()

    canvas._grid(draw_list, doc, view, (0.0, 0.0), (200.0, 200.0))

    assert len(draw_list.polyline_calls) == doc.width * doc.height
