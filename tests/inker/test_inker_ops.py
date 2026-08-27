"""The Inker's op registry: one list, and every entry answerable without imgui.

The two properties worth a test here are the ones that made five scattered
lists a defect rather than a style. **Every predicate is answerable against a
real document** -- a registry whose ``enabled`` raises ``AttributeError`` on a
one-layer drawing greys nothing and crashes the frame -- and **every op that
can be refused says why**, because the keyboard is the surface where the user
cannot see that the row was grey.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from warlock.studio import inker, inker_ops, inker_state
from warlock.studio import state as state_mod

SIZE = (32, 32)


def _session():
    doc = inker.Document.blank(*SIZE)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    ctx = SimpleNamespace(state=app, toast=app.toast)
    return ctx, state, tab


def test_every_op_is_in_exactly_one_menu_and_the_menus_cover_the_registry():
    assert len(OPS_BY_NAME := {op.name: op for op in inker_ops.OPS}) == len(inker_ops.OPS)
    listed = [op.name for name in inker_ops.MENUS for op in inker_ops.menu(name)]
    assert sorted(listed) == sorted(OPS_BY_NAME)


@pytest.mark.parametrize("op", inker_ops.OPS, ids=lambda op: op.name)
def test_every_predicate_answers_against_a_real_document(op):
    _, state, tab = _session()
    assert isinstance(op.enabled(state, tab), bool)


@pytest.mark.parametrize("op", inker_ops.OPS, ids=lambda op: op.name)
def test_every_predicate_answers_with_nothing_open(op):
    """The menu strip is drawn with no document, so every predicate sees None."""

    state = inker_state.InkerState()
    assert isinstance(op.enabled(state, None), bool)


@pytest.mark.parametrize(
    "op", [op for op in inker_ops.OPS if op.checked is not None], ids=lambda op: op.name
)
def test_every_tick_predicate_answers_with_nothing_open(op):
    """Same rule as ``enabled``: the strip is drawn before a document exists."""

    state = inker_state.InkerState()
    assert isinstance(op.checked(state, None), bool)


@pytest.mark.parametrize(
    "op", [op for op in inker_ops.OPS if op.enabled is not inker_ops._always],
    ids=lambda op: op.name,
)
def test_an_op_that_can_be_refused_carries_the_sentence(op):
    assert op.reason, f"{op.name} can be greyed out and says nothing about why"


def test_a_refused_op_says_why_rather_than_doing_nothing():
    ctx, state, tab = _session()
    assert inker_ops.run(ctx, inker_ops.get("undo")) is False
    assert state.tip is not None
    assert state.tip.text == inker_ops.reason_for(
        inker_ops.get("undo"), state, tab
    ), "``reason`` may be a callable now -- ``reason_for`` is the reader"


def test_the_registry_refuses_a_duplicate_name():
    with pytest.raises(ValueError):
        inker_ops.register(inker_ops.Op("undo", "Undo", lambda ctx, tab: None))


def test_the_registry_refuses_a_menu_that_does_not_exist():
    with pytest.raises(ValueError):
        inker_ops.register(
            inker_ops.Op("nowhere", "Nowhere", lambda ctx, tab: None, menu="Filters")
        )


def test_a_key_is_looked_up_context_first():
    """Enter closes a polygon inside a gesture and plays outside one."""

    assert inker_ops.by_key("Ctrl+Z").name == "undo"
    assert inker_ops.by_key("Enter", "Normal").name == "play"
    assert inker_ops.by_key("nope") is None


@pytest.mark.parametrize(
    "name",
    [
        "select_all",
        "add_layer",
        "duplicate_layer",
        "flip_h",
        "flip_v",
        "rotate90",
        "select_layer_alpha",
        "select_colour_range",
    ],
)
def test_the_plain_document_ops_run(name):
    ctx, state, tab = _session()
    assert inker_ops.run(ctx, inker_ops.get(name)) is not False


def test_the_layer_ops_run_once_there_is_a_stack():
    ctx, state, tab = _session()
    inker_ops.run(ctx, inker_ops.get("add_layer"))
    for name in ("layer_down", "layer_up", "merge_down"):
        assert inker_ops.get(name).enabled(state, tab), name
        assert inker_ops.run(ctx, inker_ops.get(name)) is not False


def test_the_selection_ops_run_with_a_selection():
    for name in ("grow", "shrink", "border", "feather", "copy_to_layer", "copy"):
        # A fresh full selection per op: these compose, and "border then
        # feather" is a different question from "does feather run at all".
        ctx, state, tab = _session()
        inker_ops.run(ctx, inker_ops.get("select_all"))
        assert inker_ops.run(ctx, inker_ops.get(name)) is not False, name
    ctx, state, tab = _session()
    inker_ops.run(ctx, inker_ops.get("select_all"))
    assert inker_ops.run(ctx, inker_ops.get("deselect")) is not False
    assert inker_ops.get("reselect").enabled(state, tab)


def test_a_declared_parameter_is_clamped_at_the_door():
    """The dialog clamps its live fields; the key path and tests do not."""

    ctx, state, tab = _session()
    inker_ops.run(ctx, inker_ops.get("select_all"))
    # 999 px of growth on a 32 px canvas is not a refusal an op should have to
    # write for itself.
    assert inker_ops.run(ctx, inker_ops.get("grow"), steps=999) is not False


def test_a_dialog_op_asks_rather_than_opening_anything():
    ctx, state, tab = _session()
    inker_ops.run(ctx, inker_ops.get("resize"))
    assert state.pending_dialog == "inker-resize"


def test_the_view_toggles_flip_the_persisted_preference(monkeypatch):
    from warlock.studio import inker_mode

    ctx, state, tab = _session()
    written = []
    monkeypatch.setattr(inker_mode, "persist", lambda ctx: written.append(True))
    before = state.grid
    inker_ops.run(ctx, inker_ops.get("toggle_grid"))
    assert state.grid is not before and written


# --- 6.3: the selection verbs ------------------------------------------------


def test_copy_merged_takes_the_picture_and_not_one_layer_of_it():
    """An ordinary copy moves a drawing between layers; this moves a part of
    the picture between documents."""

    ctx, state, tab = _session()
    doc = tab.doc
    doc.stack.active.pixels[4:8, 4:8] = (255, 0, 0, 255)
    doc.add_layer()
    doc.stack.active.pixels[4:8, 4:8] = (0, 0, 255, 128)
    doc.invalidate_all()
    doc.select_all()
    assert inker_ops.run(ctx, inker_ops.get("copy_merged"))
    pasted = doc.clipboard.take()
    assert pasted is not None
    # The blue over the red, which is neither layer on its own.
    pixel = pasted[0][6, 6]
    assert int(pixel[0]) and int(pixel[2]), tuple(int(c) for c in pixel)


def test_the_stroke_stays_inside_the_selection():
    """An outline that grew past the edge would paint pixels the user did not
    select, which is the one thing a selection is a promise about."""

    ctx, state, tab = _session()
    doc = tab.doc
    doc.select(inker.SelectionMask.from_rect(SIZE, (8, 8, 16, 16)))
    assert inker_ops.run(ctx, inker_ops.get("stroke_selection"), width=1)
    pixels = doc.stack.active.pixels
    assert int(pixels[8, 8, 3]) > 0, "the edge is drawn"
    assert int(pixels[7, 7, 3]) == 0, "and nothing outside it is"
    assert int(pixels[12, 12, 3]) == 0, "nor the middle"


def test_fill_covers_the_selection_and_nothing_else():
    ctx, state, tab = _session()
    doc = tab.doc
    doc.select(inker.SelectionMask.from_rect(SIZE, (8, 8, 16, 16)))
    state.fg = (255, 0, 0, 255)
    assert inker_ops.run(ctx, inker_ops.get("fill_selection"))
    pixels = doc.stack.active.pixels
    assert int(pixels[12, 12, 0]) == 255
    assert int(pixels[4, 4, 3]) == 0


def test_shifting_pixels_leaves_a_hole_and_is_one_step():
    ctx, state, tab = _session()
    doc = tab.doc
    doc.stack.active.pixels[8:12, 8:12] = (255, 0, 0, 255)
    doc.select(inker.SelectionMask.from_rect(SIZE, (8, 8, 12, 12)))
    head = doc.history.head
    assert inker_ops.run(ctx, inker_ops.get("shift_selected"), dx=4, dy=0)
    pixels = doc.stack.active.pixels
    assert int(pixels[9, 13, 0]) == 255, "it moved"
    assert int(pixels[9, 9, 3]) == 0, "and left a hole"
    doc.undo()
    assert doc.history.head <= head + 1


def test_a_new_document_from_the_selection_crops_to_it(monkeypatch):
    """Through ``open_pixels`` -- the door a sheet import, a sprite draft and a
    rendered sheet already use -- so this pins the *crop* and lets the adoption
    be the one that is already tested."""
    from warlock.studio import inker_mode

    ctx, state, tab = _session()
    opened: list = []
    monkeypatch.setattr(
        inker_mode, "open_pixels", lambda ctx, pixels, title="": opened.append(pixels)
    )
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (4, 4, 12, 12)))
    assert inker_ops.run(ctx, inker_ops.get("new_from_selection"))
    assert opened and opened[0].shape[:2] == (8, 8)


def test_making_a_document_of_nothing_says_so_rather_than_making_one():
    from warlock.studio import inker_mode

    ctx, state, tab = _session()
    assert inker_mode.new_from_selection(ctx, tab) is False
    assert state.tip is not None and "Select something" in state.tip.text


# --- the flatten matte -------------------------------------------------------


def test_toggle_matte_flips_the_document_matte_in_one_step():
    ctx, _state, tab = _session()
    assert tab.doc.matte is None

    inker_ops.run(ctx, inker_ops.get("toggle_matte"))

    assert tab.doc.matte == (255, 255, 255, 255)
    assert tab.doc.undo() is True
    assert tab.doc.matte is None


def test_toggle_matte_ticks_when_the_matte_is_on():
    _ctx, state, tab = _session()
    op = inker_ops.get("toggle_matte")
    assert op.checked is not None
    assert op.checked(state, tab) is False
    tab.doc.toggle_matte()
    assert op.checked(state, tab) is True


def test_toggle_matte_greys_with_a_reason_on_a_background_document():
    """``flatten`` ignores the matte once there is a real background layer."""
    _ctx, state, tab = _session()
    assert tab.doc.to_background() is True
    op = inker_ops.get("toggle_matte")
    assert op.enabled(state, tab) is False
    assert op.reason


def test_a_refused_paste_does_not_switch_the_tool():
    """``stamp_text``'s rule -- switch to Move on success only. An empty
    clipboard used to leave the user holding Move with nothing pasted, and
    ``run`` discards the ``False`` so nothing was said either."""
    from types import SimpleNamespace

    from warlock.studio import inker, inker_ops, inker_state

    state = inker_state.InkerState()
    state.set_tool("brush")
    tab = SimpleNamespace(doc=inker.Document.blank(8, 8), busy=False)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    op = next(o for o in inker_ops.OPS if o.name == "paste")
    assert tab.doc.paste() is False, "nothing on the clipboard to paste"
    op.run(ctx, tab)
    assert state.tool == "brush", "a refused paste leaves the tool alone"


def test_selecting_used_colours_is_one_undo_step():
    """It used to call ``select_colour_range`` once per slot, and that method
    pushes a ``SelectionEdit`` of its own -- so a sixty-colour palette cost
    sixty-one Ctrl+Z to put back and walked the composite sixty times. One
    gesture is one step, the rule the timeline's row ops already follow."""
    from warlock.studio import inker

    doc = inker.Document.blank(8, 8)
    palette = [(index * 8, 0, 0, 255) for index in range(12)]
    doc.set_palette(palette)
    for row in range(8):
        doc.stack[0].pixels[row, :, :] = palette[row % 12]
    doc.invalidate_all()

    used = doc.used_slots()
    assert len(used) == 8, "the fixture draws in eight of the twelve"
    depth = len(doc.history._done)
    assert doc.select_slots(used)
    assert len(doc.history._done) - depth == 1
    assert int((doc.mask.mask > 0).sum()) == 64, "every drawn pixel"


def test_selecting_slots_with_nothing_to_select_refuses():
    from warlock.studio import inker

    doc = inker.Document.blank(4, 4)
    assert doc.select_slots([]) is False
    doc.set_palette([(1, 2, 3, 255)])
    assert doc.select_slots([99]) is False, "a slot off the end of the table"
