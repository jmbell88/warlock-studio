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
    app = SimpleNamespace(inker=state, toasts=[], toast_log=[])
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
    "op", [op for op in inker_ops.OPS if op.enabled is not inker_ops._always],
    ids=lambda op: op.name,
)
def test_an_op_that_can_be_refused_carries_the_sentence(op):
    assert op.reason, f"{op.name} can be greyed out and says nothing about why"


def test_a_refused_op_says_why_rather_than_doing_nothing():
    ctx, state, tab = _session()
    assert inker_ops.run(ctx, inker_ops.get("undo")) is False
    assert state.tip is not None
    assert state.tip.text == inker_ops.get("undo").reason


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
