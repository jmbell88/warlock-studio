"""The animation verbs as ops, so the keyboard and the remapper can reach them.

New frame, duplicate, delete, first, last and the onion toggle lived only on
the timeline's own menus -- so an animator's commonest six actions were six
trips to a context menu, and none of them appeared in the shortcut sheet or in
the remapper, because bindings here are *data* and these were not data.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from warlock.studio import inker, inker_ops, inker_state
from warlock.studio import state as state_mod


def _session(frames=1):
    doc = inker.Document.blank(8, 8)
    if frames > 1:
        doc.ensure_animation()
        for _ in range(frames - 1):
            doc.add_frame()
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState()
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    # ``_toggle`` writes the preference down, so the session needs the settings
    # object every op that persists anything reaches for.
    settings = SimpleNamespace(get=lambda key: {}, set=lambda key, value: None)
    ctx = SimpleNamespace(state=app, toast=app.toast, settings=settings)
    return ctx, state, tab


def _op(name):
    return next(op for op in inker_ops.OPS if op.name == name)


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("new_frame", "Alt+N"),
        ("duplicate_frame", "Alt+D"),
        ("first_frame", "Home"),
        ("last_frame", "End"),
        ("toggle_onion", "F3"),
        ("resize", "Ctrl+Alt+C"),
        ("filter_hue_saturation", "Ctrl+U"),
        ("filter_invert", "Ctrl+I"),
    ],
)
def test_the_new_bindings_are_registered_data(name, key):
    assert _op(name).key == key


def test_delete_frame_is_a_verb_without_a_key():
    """Aseprite has none either: a key that silently drops a frame is the one
    shortcut worth reaching for a menu."""
    assert _op("delete_frame").key == ""


def test_new_frame_appends_a_frame():
    ctx, _, tab = _session(2)
    _op("new_frame").run(ctx, tab)
    assert len(tab.doc.anim.frames) == 3


def test_duplicate_frame_copies_the_cels_rather_than_linking_them():
    ctx, _, tab = _session()
    tab.doc.stack.active.pixels[0, 0] = (1, 2, 3, 255)
    _op("duplicate_frame").run(ctx, tab)
    anim = tab.doc.anim
    assert len(anim.frames) == 2
    track = anim.tracks[0].uid
    assert not anim.is_linked(track, anim.frames[1].uid)


def test_delete_frame_removes_the_current_one():
    ctx, _, tab = _session(3)
    _op("delete_frame").run(ctx, tab)
    assert len(tab.doc.anim.frames) == 2


def test_first_and_last_move_the_playhead_to_the_ends():
    ctx, _, tab = _session(4)
    _op("first_frame").run(ctx, tab)
    assert tab.doc.anim.current == 0
    _op("last_frame").run(ctx, tab)
    assert tab.doc.anim.current == 3


def test_the_onion_toggle_flips_the_session_setting():
    ctx, state, tab = _session(2)
    before = state.onion
    _op("toggle_onion").run(ctx, tab)
    assert state.onion is not before


def test_the_frame_verbs_are_refused_on_a_still_drawing_with_a_reason():
    _, state, tab = _session()
    for name in ("delete_frame", "first_frame", "last_frame"):
        op = _op(name)
        assert op.enabled(state, tab) is False
        assert op.reason


def test_new_frame_is_offered_on_a_still_drawing():
    """It is what animates one -- ``add_frame`` folds the ``AnimateEdit`` in."""
    _, state, tab = _session()
    assert _op("new_frame").enabled(state, tab) is True


def test_the_filter_ops_open_the_popup_on_the_named_filter():
    ctx, state, tab = _session()
    _op("filter_hue_saturation").run(ctx, tab)
    assert state.filter_name == "hue / saturation"
    assert state.pending_dialog == "inker-filter"
    _op("filter_invert").run(ctx, tab)
    assert state.filter_name == "invert"
