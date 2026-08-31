"""The Animation tab: an order the user can change, and a preview that plays.

Two gaps, and they had the same shape. The tab could set a duration per frame
but the frames could only be *appended* -- moving one meant deleting every frame
after it and re-adding them in order -- and nothing ever played, so the
durations were numbers you typed and then went back to the map to see. An editor
whose output you cannot look at is a form, not an editor.

The reorder is one undo step and the preview is the same ``tileset.frame_at``
the canvas substitutes gids through, so what plays here is what plays there.
Both are driven through ``_animation_tab`` itself with a recording widget layer:
the failure worth catching is a button that is drawn and wired to nothing, and
only the real dispatch can catch it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import plotter_state
from warlock.studio.panes import plotter_tileset_editor as editor
from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.tilegrid import tileset as tileset_lib
from warlock.studio.tilegrid.tileset import TileFrame, TileMeta, Tileset


def _frames(*pairs):
    return tuple(TileFrame(local_id=i, duration_ms=ms) for i, ms in pairs)


# --- the pure halves ----------------------------------------------------------


def test_frame_at_walks_the_durations_and_wraps():
    frames = _frames((0, 100), (1, 200), (2, 50))
    assert tileset_lib.frame_at(frames, 0) == 0
    assert tileset_lib.frame_at(frames, 99) == 0
    assert tileset_lib.frame_at(frames, 100) == 1
    assert tileset_lib.frame_at(frames, 299) == 1
    assert tileset_lib.frame_at(frames, 300) == 2
    assert tileset_lib.frame_at(frames, 350) == 0, "one cycle later, back to the start"


def test_frame_at_has_no_answer_for_an_empty_animation():
    """``None`` rather than ``0``: a caller handed a zeroth frame would draw one
    that is not playing."""
    assert tileset_lib.frame_at((), 0) is None


def test_the_canvas_substitutes_gids_through_the_same_function():
    """One implementation, or the map and this editor come to disagree about
    which frame a tile is on."""
    import inspect

    from warlock.studio.panes import plotter_canvas

    assert "frame_at(" in inspect.getsource(plotter_canvas.animated_gid)


@pytest.mark.parametrize(
    ("at", "delta", "expected"),
    [
        (1, -1, (1, 0, 2)),
        (1, 1, (0, 2, 1)),
        (0, -1, (0, 1, 2)),  # off the top: unchanged
        (2, 1, (0, 1, 2)),  # off the bottom: unchanged
    ],
)
def test_moved_frame_shifts_one_frame_and_refuses_to_fall_off_either_end(at, delta, expected):
    frames = _frames((0, 10), (1, 20), (2, 30))
    moved = editor.moved_frame(frames, at, delta)
    assert tuple(frame.local_id for frame in moved) == expected


def test_a_moved_frame_takes_its_duration_with_it():
    """The reason this is a list move and not a swap of two ids."""
    frames = _frames((0, 10), (1, 20), (2, 30))
    moved = editor.moved_frame(frames, 2, -1)
    assert [(f.local_id, f.duration_ms) for f in moved] == [(0, 10), (2, 30), (1, 20)]


def test_playing_frame_is_none_until_play_is_pressed():
    state = plotter_state.PlotterState()
    frames = _frames((0, 100), (1, 100))
    assert editor.playing_frame(state, frames) is None
    state.tileset_playing = True
    state.tileset_play_at = 10.0
    assert editor.playing_frame(state, frames, clock=lambda: 10.0) == 0
    assert editor.playing_frame(state, frames, clock=lambda: 10.15) == 1
    assert editor.playing_frame(state, frames, clock=lambda: 10.25) == 0


# --- through the tab ----------------------------------------------------------


class Widgets:
    """``widgets``, recording every control and clicking exactly one.

    A stand-in rather than a real imgui frame, and the thing it stands in for is
    the *layout*, not the dispatch: which id was drawn, whether it was enabled
    and what happens when it is pressed all come from the real
    ``_animation_tab``.
    """

    def __init__(self, press: str = "") -> None:
        self.press = press
        self.icon_buttons: list[tuple[str, bool]] = []
        self.text: list[str] = []

    def icon_button(self, label, _tooltip, *, enabled=True, **_kw) -> bool:
        self.icon_buttons.append((label, enabled))
        return enabled and label == self.press

    def muted(self, text) -> None:
        self.text.append(str(text))

    def muted_wrapped(self, text) -> None:
        self.text.append(str(text))

    def thumb_placeholder(self, *_a, **_k) -> None:
        return None

    def grid_width(self, _columns) -> float:
        return 100.0

    def texture_ref(self, texture) -> object:
        return texture


class Controls:
    def __init__(self, press: str = "") -> None:
        self.press = press
        self.buttons: list[str] = []

    def button(self, label, *_a, **_k) -> bool:
        self.buttons.append(label)
        return label == self.press

    def input_int(self, _label, value, *_a, **_k):
        return False, value


@pytest.fixture
def tab_scene(monkeypatch):
    """A map with one animated tile, and the editor's imgui layer stubbed."""
    pixels = np.zeros((8, 32, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    tiles = Tileset(name="Set", tile_w=8, tile_h=8, pixels=pixels)
    doc = MapDoc(4, 4, 8, 8)
    doc.add_tile_layer("Ground")
    doc.add_tileset(tiles)
    doc.set_tile_meta(
        0, 0, TileMeta(animation=_frames((0, 100), (1, 150), (2, 200)))
    )

    monkeypatch.setattr(
        editor,
        "imgui",
        SimpleNamespace(
            push_id=lambda _v: None,
            pop_id=lambda: None,
            same_line=lambda *_a, **_k: None,
            set_next_item_width=lambda _w: None,
            image=lambda *_a, **_k: None,
            separator=lambda: None,
        ),
    )
    state = plotter_state.PlotterState()
    state.editing_tileset = 0
    state.editing_tile = 0
    tab = SimpleNamespace(doc=doc, uid="tab-1", busy=False)
    ctx = SimpleNamespace(viewer=None, state=SimpleNamespace(preview={}))
    return ctx, state, tab, doc


def _run(monkeypatch, scene, *, icon="", button=""):
    ctx, state, tab, doc = scene
    widgets = Widgets(icon)
    controls = Controls(button)
    monkeypatch.setattr(editor, "widgets", widgets)
    monkeypatch.setattr(editor, "controls", controls)
    editor._animation_tab(ctx, state, tab, doc.tilesets[0], 0)
    return widgets, controls


def _order(doc):
    return tuple(f.local_id for f in doc.tilesets[0].tileset.meta_of(0).animation)


def test_the_up_arrow_actually_reorders_the_frames(monkeypatch, tab_scene):
    """The gap, closed and driven: press the second row's Up and the document
    comes back reordered."""
    _ctx, _state, _tab, doc = tab_scene
    assert _order(doc) == (0, 1, 2)
    _run(monkeypatch, tab_scene, icon=f"{editor.icons.ARROW_UP}##tsup1")
    assert _order(doc) == (1, 0, 2)


def test_the_down_arrow_moves_the_other_way(monkeypatch, tab_scene):
    _ctx, _state, _tab, doc = tab_scene
    _run(monkeypatch, tab_scene, icon=f"{editor.icons.ARROW_DOWN}##tsdown0")
    assert _order(doc) == (1, 0, 2)


def test_a_reorder_is_one_undo_step(monkeypatch, tab_scene):
    _ctx, _state, _tab, doc = tab_scene
    depth = len(doc.history)
    _run(monkeypatch, tab_scene, icon=f"{editor.icons.ARROW_UP}##tsup2")
    assert len(doc.history) == depth + 1
    doc.undo()
    assert _order(doc) == (0, 1, 2)


def test_the_arrows_at_the_ends_are_drawn_disabled_rather_than_missing(
    monkeypatch, tab_scene
):
    """A row whose arrow vanishes is a row that jumps under the pointer; a
    disabled one says "not from here" and stays where it was."""
    widgets, _controls = _run(monkeypatch, tab_scene)
    seen = dict(widgets.icon_buttons)
    assert seen[f"{editor.icons.ARROW_UP}##tsup0"] is False
    assert seen[f"{editor.icons.ARROW_UP}##tsup1"] is True
    assert seen[f"{editor.icons.ARROW_DOWN}##tsdown2"] is False


def test_pressing_a_disabled_arrow_writes_nothing(monkeypatch, tab_scene):
    _ctx, _state, _tab, doc = tab_scene
    depth = len(doc.history)
    _run(monkeypatch, tab_scene, icon=f"{editor.icons.ARROW_UP}##tsup0")
    assert _order(doc) == (0, 1, 2)
    assert len(doc.history) == depth


def test_the_remove_button_still_works(monkeypatch, tab_scene):
    _ctx, _state, _tab, doc = tab_scene
    _run(monkeypatch, tab_scene, icon="x##tsdel1")
    assert _order(doc) == (0, 2)


# --- the preview --------------------------------------------------------------


def test_play_starts_the_preview_and_stop_ends_it(monkeypatch, tab_scene):
    _ctx, state, _tab, _doc = tab_scene
    assert state.tileset_playing is False
    _run(monkeypatch, tab_scene, button="Play##tsplay")
    assert state.tileset_playing is True
    assert state.tileset_play_at > 0.0

    _widgets, playing = _run(monkeypatch, tab_scene, button="Stop##tsplay")
    assert "Stop##tsplay" in playing.buttons, "the label says what the press does"
    assert state.tileset_playing is False
    _widgets, stopped = _run(monkeypatch, tab_scene)
    assert "Play##tsplay" in stopped.buttons, "and the next frame offers Play again"


def test_play_always_restarts_at_the_first_frame(monkeypatch, tab_scene):
    _ctx, state, _tab, doc = tab_scene
    _run(monkeypatch, tab_scene, button="Play##tsplay")
    frames = doc.tilesets[0].tileset.meta_of(0).animation
    assert editor.playing_frame(state, frames, clock=lambda: state.tileset_play_at) == 0


def test_the_preview_names_the_frame_it_is_showing(monkeypatch, tab_scene):
    import time

    _ctx, state, _tab, _doc = tab_scene
    state.tileset_playing = True
    # 120 ms in, which is the middle of the second frame's 100-250 ms window --
    # wide enough that the test's own overhead cannot land it on a neighbour.
    state.tileset_play_at = time.monotonic() - 0.12
    widgets, _controls = _run(monkeypatch, tab_scene)
    assert any("Frame 2 of 3, tile 1." in line for line in widgets.text)


def test_the_preview_says_so_when_it_is_stopped(monkeypatch, tab_scene):
    widgets, _controls = _run(monkeypatch, tab_scene)
    assert any(line.startswith("Stopped.") for line in widgets.text)


def test_the_play_button_is_offered_before_there_is_anything_to_play(
    monkeypatch, tab_scene
):
    """A control that appears only once you have guessed it should exist is a
    control nobody finds."""
    _ctx, _state, _tab, doc = tab_scene
    doc.set_tile_meta(0, 0, TileMeta())
    _widgets, controls = _run(monkeypatch, tab_scene)
    assert "Play##tsplay" in controls.buttons
