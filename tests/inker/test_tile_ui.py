"""The tile UI: the stamp tool, the behaviour toggle, the panel's verbs.

State- and spy-level throughout, ``test_canvas_input.py``'s pattern: ``_press``
reads exactly one thing off imgui -- the Alt modifier -- and reaching for a live
context takes the process down, so it is stubbed and nothing here draws. The
pane-draw code is covered by the smoke walk.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from warlock.studio import inker, inker_mode, inker_state
from warlock.studio import state as state_mod
from warlock.studio.inker.tiles import TilemapCel, materialize, strip
from warlock.studio.panes import inker_canvas, inker_tiles, inker_tools
from warlock.studio.tilegrid import gid

SIZE = (32, 32)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)


def _tile(colour, w: int = 8, h: int = 8) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _tileset(*colours, w: int = 8, h: int = 8):
    stack = [np.zeros((h, w, 4), dtype=np.uint8), *[_tile(c, w, h) for c in colours]]
    return strip(np.stack(stack, axis=0))


@pytest.fixture
def scene(monkeypatch):
    """A one-tab session with the real toast machinery and imgui's io stubbed.

    The toast doubles are the real ``AppState`` methods bound over a namespace,
    ``test_inker_refusals``'s reason: ``toast_once`` coalesces, and a collecting
    lambda would test the double instead of the dedup.
    """
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=False, key_alt=False, key_ctrl=False),
    )
    doc = inker.Document.blank(*SIZE)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState(tool="tile")
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[], preview={})
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    ctx = SimpleNamespace(state=app, toast=app.toast)
    return ctx, state, tab, app


def _sent(app) -> list[tuple[str, str]]:
    return [(t.text, t.level) for t in app.toasts]


def _tilemap(tab, *colours):
    """A tilemap layer on the tab's document, active, over an 8px tileset."""
    slot = tab.doc.add_tileset(_tileset(*(colours or (RED, BLUE))))
    cel = tab.doc.add_tilemap_layer(slot.uid)
    return slot, cel


# --- the tool's place in the toolbox ------------------------------------------


def test_the_tile_stamp_has_the_one_letter_nothing_else_holds() -> None:
    """``Y``: every mnemonic is taken, ``X`` swaps the colours and ``Z`` sits
    under Ctrl+Z, where a slipped modifier would swap a tool for an undo."""
    assert inker_mode.TOOL_KEYS["y"] == "tile"
    assert "tile" in {key for key, _label, _letter in inker_state.TOOLS}
    assert inker_state.tool_label("tile") == "Tile stamp"


def test_the_tool_is_greyed_off_a_tilemap_layer_and_live_on_one(scene) -> None:
    ctx, state, tab, app = scene
    assert inker_state.tool_reason("tile", tab.doc) == inker_state.TILE_REASON
    _tilemap(tab)
    assert inker_state.tool_reason("tile", tab.doc) == ""


def test_the_reason_is_written_once_and_the_pane_reads_it(scene) -> None:
    """The toolbox greys the button, but a shortcut key selects a tool without
    asking the panel anything -- so both doors have to say the same sentence."""
    text = Path(inker_canvas.__file__).read_text(encoding="utf-8")
    assert "tool_reason" in text
    assert "needs a tilemap layer" not in text


# --- the press --------------------------------------------------------------


def test_a_stamp_on_an_ordinary_layer_refuses_out_loud(scene) -> None:
    ctx, state, tab, app = scene
    before = tab.doc.stack.active.pixels.copy()
    inker_canvas._press(ctx, state, tab, (4.0, 4.0))
    assert state.drag_kind == ""
    assert state.tip is not None and state.tip.text == inker_state.TILE_REASON
    assert np.array_equal(tab.doc.stack.active.pixels, before)


def test_a_stamp_writes_refs_and_never_a_pixel(scene) -> None:
    ctx, state, tab, app = scene
    slot, cel = _tilemap(tab)
    state.tile_local = 1
    writes: list = []
    tab.doc.begin_stroke = lambda *a, **k: writes.append(a)  # type: ignore[assignment]

    inker_canvas._press(ctx, state, tab, (9.0, 3.0))

    assert state.drag_kind == "tile"
    assert writes == []
    assert int(cel.refs[0, 1]) == 1
    assert not cel.refs[0, 0]
    # The picture is the materialization, not something the press painted.
    assert np.array_equal(cel.pixels, materialize(cel.refs, slot.tileset, cel.size))


def test_the_flip_toggles_ride_into_the_cell(scene) -> None:
    ctx, state, tab, app = scene
    slot, cel = _tilemap(tab)
    # Both fields, which is what a click in the picker sets: a local id is an
    # id *into a tileset*, so the pick carries which one (and the clamp resets
    # it when that changes -- see the clamp tests below).
    state.tileset_uid, state.tile_local = slot.uid, 2
    state.tile_flip_h = True
    state.tile_flip_d = True

    inker_canvas._press(ctx, state, tab, (0.0, 0.0))

    assert int(cel.refs[0, 0]) == 2 | gid.FLIP_H | gid.FLIP_D


def test_tile_zero_stamps_a_bare_zero_however_the_flips_are_set(scene) -> None:
    """Every orientation of blank is blank, and a flagged zero would read back
    out of an ORA as a different word for the same picture."""
    ctx, state, tab, app = scene
    state.tile_local = 0
    state.tile_flip_v = True
    assert state.tile_gid() == 0


def test_a_drag_stamps_each_cell_once(scene) -> None:
    ctx, state, tab, app = scene
    _slot, cel = _tilemap(tab)
    state.tile_local = 1
    calls: list = []
    real = tab.doc.place_tiles

    def spy(uid, origin, patch):
        calls.append(origin)
        return real(uid, origin, patch)

    tab.doc.place_tiles = spy  # type: ignore[assignment]

    inker_canvas._press(ctx, state, tab, (0.0, 0.0))
    for x in (1.0, 3.0, 7.0):  # still inside cell (0, 0)
        inker_canvas._drag(state, tab, (x, 0.0))
    inker_canvas._drag(state, tab, (9.0, 0.0))  # cell (1, 0)
    inker_canvas._drag(state, tab, (10.0, 0.0))  # same cell again

    assert calls == [(0, 0), (1, 0)]
    assert int(cel.refs[0, 0]) == 1 and int(cel.refs[0, 1]) == 1


def test_a_stamp_off_the_canvas_writes_nothing(scene) -> None:
    ctx, state, tab, app = scene
    _slot, cel = _tilemap(tab)
    head = tab.doc.history.head
    inker_canvas._press(ctx, state, tab, (-3.0, 4.0))
    assert tab.doc.history.head == head
    assert not cel.refs.any()


def test_the_cell_under_the_cursor_is_asked_once_for_both_ends(scene) -> None:
    """``tile_cell`` is what the cursor outlines and what the press writes, so
    the outline is on screen exactly when a click would land."""
    ctx, state, tab, app = scene
    _tilemap(tab)
    assert inker_canvas.tile_cell(tab.doc, (0.0, 0.0)) == (0, 0)
    assert inker_canvas.tile_cell(tab.doc, (9.9, 17.0)) == (1, 2)
    assert inker_canvas.tile_cell(tab.doc, (-0.5, 4.0)) is None
    assert inker_canvas.tile_cell(tab.doc, (4.0, 99.0)) is None


def test_the_cursor_says_nothing_on_a_layer_the_tool_refuses(scene) -> None:
    ctx, state, tab, app = scene
    assert inker_canvas.tile_cell(tab.doc, (4.0, 4.0)) is None


# --- Manual / Auto / Stack ----------------------------------------------------


def test_switching_the_behaviour_does_not_dirty_the_document(scene) -> None:
    """It is view state: the field is never serialized, and a toolbox toggle
    that marked a drawing unsaved would be a lie about the file."""
    ctx, state, tab, app = scene
    _tilemap(tab)
    tab.mark_saved()
    head = tab.doc.history.head
    assert not tab.dirty

    for behavior in ("auto", "stack", "manual"):
        tab.doc.tile_behavior = behavior
        assert tab.doc.history.head == head
        assert not tab.dirty


def test_the_toggle_writes_the_field_and_pushes_nothing() -> None:
    """The pane's own half of the rule, read off its source: one choice control
    that assigns the field, with no document call anywhere near it."""
    source = inspect.getsource(inker_tools._tile_behavior)
    assert "doc.tile_behavior = picked" in source
    # Past the docstring, which says all this in prose. ``push_style_color`` is
    # imgui's own and is not a history push, so the check is for the *document*
    # calls a view-state toggle must never make.
    body = source.split('"""', 2)[-1]
    assert "doc.history" not in body
    assert "history.push" not in body
    assert "doc." not in body.replace("doc.tile_behavior", "").replace(
        "doc.active_tilemap_uid()", ""
    )


def test_the_three_behaviours_are_the_engines_three() -> None:
    keys = {key for key, _label, _why in inker_state.TILE_BEHAVIORS}
    assert keys == {"manual", "auto", "stack"}


# --- the Manual-mode revert notice --------------------------------------------


def _paint_gesture(ctx, state, tab, point=(2.0, 2.0)):
    state.tool = "brush"
    inker_canvas._press(ctx, state, tab, point)
    inker_canvas._release(ctx, state, tab, point)


def test_a_manual_revert_says_so_once_per_gesture(scene) -> None:
    ctx, state, tab, app = scene
    _tilemap(tab)
    tab.doc.tile_behavior = "manual"
    head = tab.doc.history.head

    _paint_gesture(ctx, state, tab)

    assert tab.doc.history.head == head, "manual mode records nothing"
    assert state.tip is not None and state.tip.text == inker_state.TILE_MANUAL_REVERTED
    # And the way out is offered by *name*: Aseprite's "Disable Snap to Grid"
    # button, over an op the menus and the keyboard have too.
    assert state.tip.remedy == "tile_auto"
    # A second stroke cannot stack a second sentence: there is one tip slot.
    _paint_gesture(ctx, state, tab)
    assert _sent(app) == []
    assert state.tip.text == inker_state.TILE_MANUAL_REVERTED


def test_a_stack_mode_stroke_says_nothing_because_it_recorded_something(scene) -> None:
    ctx, state, tab, app = scene
    _tilemap(tab)
    tab.doc.tile_behavior = "stack"
    head = tab.doc.history.head

    _paint_gesture(ctx, state, tab)

    assert tab.doc.history.head != head
    assert _sent(app) == []


def test_an_ordinary_layer_never_earns_the_notice(scene) -> None:
    ctx, state, tab, app = scene
    tab.doc.tile_behavior = "manual"
    _paint_gesture(ctx, state, tab)
    assert _sent(app) == []
    assert state.tile_head is None


def test_a_selection_drag_on_a_manual_tilemap_layer_says_nothing(scene) -> None:
    """A marquee pushes no history step either, and reading a still head after
    one would announce a revert that never happened."""
    ctx, state, tab, app = scene
    _tilemap(tab)
    tab.doc.tile_behavior = "manual"
    state.tool = "select"
    inker_canvas._press(ctx, state, tab, (2.0, 2.0))
    inker_canvas._release(ctx, state, tab, (2.0, 2.0))
    assert _sent(app) == []


def test_the_stamp_itself_never_raises_the_notice(scene) -> None:
    """It writes refs and never reaches the tileset, so there is nothing for
    Manual mode to have refused."""
    ctx, state, tab, app = scene
    _tilemap(tab)
    tab.doc.tile_behavior = "manual"
    state.tile_local = 1
    inker_canvas._press(ctx, state, tab, (0.0, 0.0))
    inker_canvas._release(ctx, state, tab, (0.0, 0.0))
    assert _sent(app) == []


# --- the picker's selection ---------------------------------------------------


def test_the_bound_tileset_wins_over_the_remembered_one(scene) -> None:
    """A stamp writes a *local* id, so a panel showing another atlas beside a
    selected tilemap layer would name a completely different tile."""
    ctx, state, tab, app = scene
    other = tab.doc.add_tileset(_tileset(GREEN))
    slot, _cel = _tilemap(tab)
    state.tileset_uid = other.uid
    assert state.picked_tileset(tab.doc) == slot.uid


def test_a_stale_uid_falls_back_instead_of_pointing_at_nothing(scene) -> None:
    ctx, state, tab, app = scene
    slot = tab.doc.add_tileset(_tileset(RED))
    state.tileset_uid = slot.uid + 12345
    assert state.picked_tileset(tab.doc) == slot.uid


def test_a_document_with_no_tileset_has_nothing_to_pick(scene) -> None:
    ctx, state, tab, app = scene
    assert state.picked_tileset(tab.doc) is None


def test_the_remembered_uid_decides_while_no_tilemap_layer_is_active(scene) -> None:
    ctx, state, tab, app = scene
    tab.doc.add_tileset(_tileset(RED))
    second = tab.doc.add_tileset(_tileset(GREEN))
    state.tileset_uid = second.uid
    assert state.picked_tileset(tab.doc) == second.uid


def test_the_gid_folds_the_three_toggles_in_one_place(scene) -> None:
    ctx, state, tab, app = scene
    state.tile_local = 5
    assert state.tile_gid() == 5
    state.tile_flip_v = True
    assert state.tile_gid() == 5 | gid.FLIP_V


# --- the verbs ----------------------------------------------------------------


def test_every_tile_verb_is_disabled_never_hidden() -> None:
    """The Wave 2 rule: a panel that quietly loses a button when a document is
    not shaped a certain way is one where the feature looks imagined."""
    # ``convert_row`` is no longer a button: the verb is a menu row, and a
    # menu row's refusal is its ``Op.reason``, which ``tests/inker/
    # test_inker_ops.py`` requires of every op that can be greyed out.
    for func in (inker_tiles._verbs,):
        source = inspect.getsource(func)
        assert "widgets.disabled_button" in source
        assert "controls.button(" not in source
        # No early return in front of the buttons: that is what "hidden" would
        # look like here.
        body = source.split('"""', 2)[-1]
        assert "return" not in body


def test_the_acquiring_verbs_live_where_they_are_always_drawn() -> None:
    """The tile panel appears only once the document has a tileset, so the two
    doors that *make* the first one are menu rows instead -- the one surface
    that is drawn whether or not the document has tiles yet."""
    from warlock.studio import inker_ops

    names = {op.name for op in inker_ops.OPS}
    assert {"convert_to_tilemap", "import_tileset"} <= names
    # Superseded by the picker, which now addresses the selected tileset.
    assert "export_tileset" not in names


def test_new_tilemap_layer_adds_a_bound_layer_and_says_so(scene) -> None:
    ctx, state, tab, app = scene
    slot = tab.doc.add_tileset(_tileset(RED))
    inker_mode.new_tilemap_layer(ctx, tab, slot.uid)
    assert isinstance(tab.doc.stack.active, TilemapCel)
    assert tab.doc.stack.active.tileset_uid == slot.uid
    assert _sent(app)[-1][1] == "success"


def test_new_tilemap_layer_with_no_tileset_refuses_out_loud(scene) -> None:
    ctx, state, tab, app = scene
    inker_mode.new_tilemap_layer(ctx, tab, None)
    assert _sent(app) == [("This document has no tileset yet.", "error")]
    assert not isinstance(tab.doc.stack.active, TilemapCel)


def test_convert_to_tilemap_cuts_the_active_layer(scene) -> None:
    ctx, state, tab, app = scene
    tab.doc.stack.active.pixels[0:8, 0:8] = np.array(RED, dtype=np.uint8)
    inker_mode.convert_to_tilemap(ctx, tab, 8, 8)
    cel = tab.doc.stack.active
    assert isinstance(cel, TilemapCel)
    assert cel.refs.shape == (4, 4)
    assert _sent(app)[-1] == ("Cut into 8 x 8 tiles.", "success")


def test_convert_to_tilemap_on_a_tilemap_layer_says_why(scene) -> None:
    ctx, state, tab, app = scene
    _tilemap(tab)
    inker_mode.convert_to_tilemap(ctx, tab, 8, 8)
    assert _sent(app)[-1][1] == "warn"


def test_convert_to_raster_keeps_the_picture_and_drops_the_cells(scene) -> None:
    ctx, state, tab, app = scene
    _slot, cel = _tilemap(tab)
    tab.doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    before = cel.pixels.copy()
    inker_mode.convert_to_raster(ctx, tab)
    back = tab.doc.stack.active
    assert not isinstance(back, TilemapCel)
    assert np.array_equal(back.pixels, before)
    assert _sent(app)[-1] == ("Converted to a plain layer.", "success")


def test_convert_to_raster_on_an_ordinary_layer_says_why(scene) -> None:
    ctx, state, tab, app = scene
    inker_mode.convert_to_raster(ctx, tab)
    assert _sent(app)[-1][1] == "warn"


# --- the picker's slicing and the pick's range --------------------------------


def test_the_picker_slices_the_atlas_through_the_tileset_not_by_hand() -> None:
    """It sliced horizontal bands -- ``local / tile_count`` down a one-column
    strip -- which is right for a tileset this editor cut and wrong for an
    imported grid ``.tsx``, which is kept exactly as it arrives and therefore
    has columns, and may have a margin and spacing too."""
    source = inspect.getsource(inker_tiles._picker)
    assert "tileset.uv(local)" in source
    assert "local / rows" not in source


def test_a_two_column_atlas_gives_tile_one_the_right_half_of_the_top_row() -> None:
    """The arithmetic the picker now delegates to, pinned on the shape that
    broke it: one column is the case that made the hand-rolled slice look
    correct."""
    from warlock.studio.tilegrid.tileset import Tileset

    grid = Tileset(
        name="grid", pixels=np.zeros((16, 16, 4), dtype=np.uint8), tile_w=8, tile_h=8
    )
    assert (grid.columns, grid.rows, grid.tile_count) == (2, 2, 4)
    assert grid.uv(1) == (0.5, 0.0, 1.0, 0.5)
    assert grid.uv(2) == (0.0, 0.5, 0.5, 1.0)


def test_the_pick_resets_when_the_tileset_under_it_changes(scene) -> None:
    ctx, state, tab, app = scene
    state.tileset_uid, state.tile_local = 7, 12
    state.clamp_tile_pick(9, 40)
    assert (state.tileset_uid, state.tile_local) == (9, 1)


def test_the_pick_resets_when_the_tileset_shrinks_under_it(scene) -> None:
    """Undoing a Stack-mode append is exactly this: the tile the user is
    holding stops existing."""
    ctx, state, tab, app = scene
    state.tileset_uid, state.tile_local = 9, 12
    state.clamp_tile_pick(9, 4)
    assert state.tile_local == 1
    # A one-tile tileset has nothing but the blank.
    state.clamp_tile_pick(9, 1)
    assert state.tile_local == 0


def test_a_pick_that_is_still_in_range_is_left_alone(scene) -> None:
    ctx, state, tab, app = scene
    state.tileset_uid, state.tile_local = 9, 12
    state.clamp_tile_pick(9, 40)
    assert state.tile_local == 12


def test_the_stamp_clamps_a_stale_pick_instead_of_raising(scene) -> None:
    """``place_tiles`` refuses an over-range id **by raising**, and a raise out
    of a press is a window down rather than a refusal a user can meet -- so the
    tool applies the panel's own clamp at the write as well."""
    ctx, state, tab, app = scene
    _slot, cel = _tilemap(tab)  # tiles 0, 1, 2
    state.tile_local = 9
    inker_canvas._press(ctx, state, tab, (0.0, 0.0))
    assert state.tile_local == 1
    assert int(cel.refs[0, 0]) == 1
