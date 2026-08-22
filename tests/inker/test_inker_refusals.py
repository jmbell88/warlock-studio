"""A refusal the user can meet is a refusal the user is told about.

Since W2.6 the *door a gesture arrives at* answers with a status-bar tip rather
than a toast, and the distinction is written down in ``inker_state.Tip``: a tip
answers a gesture, a toast reports a job. So these tests read ``state.tip``,
which is the same claim about the same doors -- one sentence, wherever the
user's hand came from -- said under the cursor instead of over the window.

``INVARIANTS.md`` states the split these tests guard: the engine's answer is a
plain ``False`` because it is asked sixty times a second, and it is the *panes*
that turn one press into one sentence. What had gone wrong is that only some of
the panes' doors held up their end -- a locked layer met with the mouse raised a
toast and the same locked layer met with the arrow keys did nothing at all, so
which answer you got depended on which hand you used.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from warlock.studio import inker, inker_mode, inker_state
from warlock.studio import state as state_mod
from warlock.studio.panes import inker_canvas

SIZE = (32, 32)


def _session(tool: str = "brush"):
    """A one-tab session and a ctx whose toast machinery is the real one.

    The hidden-layer warning coalesces through ``AppState.toast_once``, so the
    double binds the real methods over a bare namespace rather than collecting
    with a lambda -- an append-only lambda would test the double, not the
    dedup. ``app.toasts`` holds the live ``Toast`` objects in the order they
    were raised.
    """
    doc = inker.Document.blank(*SIZE)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState(tool=tool)
    state.add(tab)
    app = SimpleNamespace(inker=state, toasts=[], toast_log=[])
    app.toast = MethodType(state_mod.AppState.toast, app)
    app.toast_once = MethodType(state_mod.AppState.toast_once, app)
    ctx = SimpleNamespace(state=app, toast=app.toast)
    return ctx, state, tab, app


def _sent(app) -> list[tuple[str, str]]:
    return [(t.text, t.level) for t in app.toasts]


def _tip(state) -> str:
    """What the status bar is saying, or ``""``."""
    return "" if state.tip is None else state.tip.text


def _press(monkeypatch, ctx, key):
    import pygame

    return inker_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))


# --- one sentence, both doors ------------------------------------------------


def test_the_locked_sentence_is_written_once() -> None:
    """Both doors read it from ``inker_mode``.

    It was spelled out at the canvas press and nowhere else, which is how the
    keyboard's copies of the same refusal came to be silent: there was nothing
    to reuse, so there was nothing to notice was missing.
    """
    assert "locked" in inker_mode.LOCKED_LAYER
    text = Path(inker_canvas.__file__).read_text(encoding="utf-8")
    assert "inker_mode.LOCKED_LAYER" in text
    assert '"That layer is locked' not in text


def test_a_press_on_a_locked_layer_still_says_so() -> None:
    ctx, state, tab, app = _session()
    tab.doc.stack.active.locked = True
    assert inker_canvas._locked_out(ctx, state, tab) is True
    assert _tip(state) == inker_mode.LOCKED_LAYER
    # And the way out is offered as an *op name*, so the button cannot be one
    # the menus and the keyboard do not have.
    assert state.tip.remedy == "layer_properties"


def test_a_nudge_onto_a_locked_layer_says_the_same_thing(monkeypatch) -> None:
    """It came back ``False`` and the answer was thrown away."""
    import pygame

    ctx, state, tab, app = _session(tool="move")
    tab.doc.stack.active.locked = True
    _press(monkeypatch, ctx, pygame.K_RIGHT)
    assert _tip(state) == inker_mode.LOCKED_LAYER


def test_delete_on_a_locked_layer_says_the_same_thing(monkeypatch) -> None:
    import pygame

    ctx, state, tab, app = _session()
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (2, 2, 8, 8)))
    tab.doc.stack.active.locked = True
    _press(monkeypatch, ctx, pygame.K_DELETE)
    assert _tip(state) == inker_mode.LOCKED_LAYER


def test_a_nudge_that_works_says_nothing(monkeypatch) -> None:
    """The toast is the refusal's, not the gesture's."""
    import pygame

    ctx, state, tab, app = _session(tool="move")
    _press(monkeypatch, ctx, pygame.K_RIGHT)
    assert _sent(app) == []


def test_a_nudge_with_the_brush_in_hand_says_nothing(monkeypatch) -> None:
    """``nudge`` is gated on the move tool, and that refusal is not the lock's.

    The arrows are the only keys a document pane has left to give away, so
    they are deliberately inert on every other tool -- and a toast every time
    somebody pressed Right with a brush selected would be the loudest thing in
    the app.
    """
    import pygame

    ctx, state, tab, app = _session(tool="brush")
    _press(monkeypatch, ctx, pygame.K_RIGHT)
    assert _sent(app) == []


# --- hidden is not refused, but it is not silent either ----------------------


def test_painting_on_a_hidden_layer_says_why_nothing_appeared() -> None:
    """There is no visibility check at any door, so the stroke lands and the
    composite does not show it -- and the only thing on screen that could
    explain that is an eye icon in a different pane."""
    ctx, state, tab, app = _session()
    tab.doc.stack.active.visible = False
    assert inker_canvas._locked_out(ctx, state, tab) is False, "not refused"
    assert "hidden" in _tip(state)
    assert state.tip.remedy == "show_layer"


def test_the_hidden_warning_does_not_stack_per_press() -> None:
    """A multi-click shape tool presses once per vertex and the supported
    paint-under-a-hidden-layer workflow presses once per stroke, without limit,
    and ``AppState.toast`` appends unconditionally -- five clicks were five
    identical toasts on screen at once.

    A tip cannot stack **by construction**: there is one slot, and the newest
    sentence replaces whatever was in it. That is the coalescing the toast door
    had to be taught, and it is why a gesture's answer belongs here."""
    ctx, state, tab, app = _session(tool="polyline")
    tab.doc.stack.active.visible = False
    for _ in range(5):
        assert inker_canvas._locked_out(ctx, state, tab) is False
    assert _sent(app) == []
    assert "hidden" in _tip(state)


def test_an_expired_hidden_warning_is_earned_again() -> None:
    """Not a once-per-session gag: a tip expires, and the next press explains
    itself afresh with a fresh clock."""
    ctx, state, tab, app = _session()
    tab.doc.stack.active.visible = False
    inker_canvas._locked_out(ctx, state, tab)
    first = state.tip
    first.at -= inker_state.TIP_SECONDS + 1.0
    assert not first.alive()
    inker_canvas._locked_out(ctx, state, tab)
    assert state.tip is not first and state.tip.alive()


def test_a_layer_inside_a_hidden_group_is_not_silent() -> None:
    """The composite hides through the group fold, not through the layer's own
    eye -- deliberately, so the eye checkbox keeps showing what the layer is
    set to. The old check read only ``stack.active.visible``, so a visible
    layer inside a hidden group painted invisibly with no warning at all."""
    ctx, state, tab, app = _session()
    doc = tab.doc
    node = doc.group_layers([0])
    assert node is not None
    doc.set_group_props(node.uid, visible=False)
    assert doc.stack.active.visible, "the layer's own eye stays on"
    assert inker_canvas._locked_out(ctx, state, tab) is False, "not refused"
    assert "hidden group" in _tip(state)


def test_a_visible_layer_says_nothing() -> None:
    ctx, state, tab, app = _session()
    assert inker_canvas._locked_out(ctx, state, tab) is False
    assert _sent(app) == [] and _tip(state) == ""


def test_a_visible_group_member_says_nothing() -> None:
    ctx, state, tab, app = _session()
    node = tab.doc.group_layers([0])
    assert node is not None
    assert inker_canvas._locked_out(ctx, state, tab) is False
    assert _sent(app) == [] and _tip(state) == ""


@pytest.mark.parametrize("tool", ["eyedropper", "select", "lasso", "wand"])
def test_a_read_only_tool_is_told_nothing_about_either(tool: str) -> None:
    """Deliberately the same exemptions the lock uses: a pick or a marquee
    writes to no layer, so neither state is any of its business."""
    ctx, state, tab, app = _session(tool=tool)
    tab.doc.stack.active.locked = True
    tab.doc.stack.active.visible = False
    assert inker_canvas._locked_out(ctx, state, tab) is False
    assert _sent(app) == [] and _tip(state) == ""
