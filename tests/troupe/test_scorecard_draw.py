"""The QA heatmap draws, including the ring around the selected cell.

The ring was the one thing on the scorecard nobody had drawn in a test, and it
is the one thing that only draws when a cell is *current* -- so a sheet
rendered fine, and the centre pane died the moment the heatmap had a selection
to outline. ``add_rect`` was called in pyimgui's argument order (``col,
rounding, flags, thickness``) and imgui_bundle's is ``col, rounding,
thickness, flags``, which put ``2.0`` on an integer parameter; the binding
raised ``TypeError`` before drawing anything and ``studio.guard`` logged
"troupe-centre stopped drawing" once per frame from then on (a real session on
2026-09-05, ``warlock.log`` 16:52:10 onward, after the sheet itself had
rendered at 16:51:42).

So this is a real imgui context and a real ``_scorecard`` call: nothing short
of the binding actually rejecting the arguments would have caught it, which is
why no source-text assertion appears below.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import probe, troupe_mode
from warlock.studio.panes import troupe_preview
from warlock.studio.troupe import qa


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    fixture there."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui


_DIRECTIONS = ("south", "west")
_FRAMES = 2

#: One cell per direction/frame, one of them past every bad threshold so the
#: cross arm of ``_scorecard`` is drawn too.
def _score() -> qa.SheetScore:
    cells = []
    index = 0
    for direction in _DIRECTIONS:
        for frame in range(_FRAMES):
            bad = direction == "west" and frame == 1
            cells.append(
                qa.CellScore(
                    cell=index,
                    animation="walk",
                    direction=direction,
                    frame=frame,
                    metrics={"shape_delta": 0.9 if bad else 0.01},
                    flags=("shape",) if bad else (),
                )
            )
            index += 1
    return qa.SheetScore(cells=tuple(cells), worst=("shape_delta", 0.9, 3), flagged=1)


def _movement() -> dict:
    return {
        "key": "walk",
        "frames": _FRAMES,
        "directions": [{"key": d} for d in _DIRECTIONS],
    }


def _draw(ui, monkeypatch, *, direction: str, frame: int):
    monkeypatch.setattr(troupe_mode, "scores", lambda ctx: _score())
    monkeypatch.setattr(troupe_mode, "preview_movement", lambda ctx: _movement())
    ctx = SimpleNamespace()
    state = SimpleNamespace(direction=direction, frame=frame)
    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        troupe_preview._scorecard(ctx, state)
    finally:
        ui.end()
        ui.end_frame()
    return probe.census()


def test_the_heatmap_draws_the_ring_around_the_selected_cell(ui, monkeypatch):
    """The regression: ``south``/frame 0 is on the matrix, so the ring is
    drawn, and drawing it must not raise."""
    seen = _draw(ui, monkeypatch, direction="south", frame=0)
    selected = [c for c in seen if c.kind == "heatmap_cell" and c.selected]
    assert len(selected) == 1, "exactly one cell is the current one"
    assert selected[0].label == "walk south frame 1"
    assert len(seen) == len(_DIRECTIONS) * _FRAMES, "every cell is a control"


def test_the_heatmap_draws_with_no_cell_selected(ui, monkeypatch):
    """The path that survived the bug: no cell current, so no ring. Here so a
    future failure says which of the two broke."""
    seen = _draw(ui, monkeypatch, direction="north", frame=0)
    assert seen and not any(c.selected for c in seen)
