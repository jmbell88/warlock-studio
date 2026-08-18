"""A refusal the user can meet is a refusal the user is told about.

``INVARIANTS.md`` states the split these tests guard: the engine's answer is a
plain ``False`` because it is asked sixty times a second, and it is the *panes*
that turn one press into one sentence. What had gone wrong is that only some of
the panes' doors held up their end -- a locked layer met with the mouse raised a
toast and the same locked layer met with the arrow keys did nothing at all, so
which answer you got depended on which hand you used.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock.studio import inker, inker_mode, inker_state
from warlock.studio.panes import inker_canvas

SIZE = (32, 32)


def _session(tool: str = "brush"):
    """A one-tab session and a ctx that collects toasts."""
    doc = inker.Document.blank(*SIZE)
    tab = inker_state.InkerDoc(doc=doc, uid="t1", title="Untitled")
    state = inker_state.InkerState(tool=tool)
    state.add(tab)
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        toast=lambda text, kind="info": toasts.append((text, kind)),
    )
    return ctx, state, tab, toasts


def _press(monkeypatch, ctx, key):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    return inker_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=key))


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
    ctx, state, tab, toasts = _session()
    tab.doc.stack.active.locked = True
    assert inker_canvas._locked_out(ctx, state, tab) is True
    assert toasts == [(inker_mode.LOCKED_LAYER, "warn")]


def test_a_nudge_onto_a_locked_layer_says_the_same_thing(monkeypatch) -> None:
    """It came back ``False`` and the answer was thrown away."""
    import pygame

    ctx, state, tab, toasts = _session(tool="move")
    tab.doc.stack.active.locked = True
    _press(monkeypatch, ctx, pygame.K_RIGHT)
    assert toasts == [(inker_mode.LOCKED_LAYER, "warn")]


def test_delete_on_a_locked_layer_says_the_same_thing(monkeypatch) -> None:
    import pygame

    ctx, state, tab, toasts = _session()
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (2, 2, 8, 8)))
    tab.doc.stack.active.locked = True
    _press(monkeypatch, ctx, pygame.K_DELETE)
    assert toasts == [(inker_mode.LOCKED_LAYER, "warn")]


def test_a_nudge_that_works_says_nothing(monkeypatch) -> None:
    """The toast is the refusal's, not the gesture's."""
    import pygame

    ctx, state, tab, toasts = _session(tool="move")
    _press(monkeypatch, ctx, pygame.K_RIGHT)
    assert toasts == []


def test_a_nudge_with_the_brush_in_hand_says_nothing(monkeypatch) -> None:
    """``nudge`` is gated on the move tool, and that refusal is not the lock's.

    The arrows are the only keys a document pane has left to give away, so
    they are deliberately inert on every other tool -- and a toast every time
    somebody pressed Right with a brush selected would be the loudest thing in
    the app.
    """
    import pygame

    ctx, state, tab, toasts = _session(tool="brush")
    _press(monkeypatch, ctx, pygame.K_RIGHT)
    assert toasts == []


# --- hidden is not refused, but it is not silent either ----------------------


def test_painting_on_a_hidden_layer_says_why_nothing_appeared() -> None:
    """There is no visibility check at any door, so the stroke lands and the
    composite does not show it -- and the only thing on screen that could
    explain that is an eye icon in a different pane."""
    ctx, state, tab, toasts = _session()
    tab.doc.stack.active.visible = False
    assert inker_canvas._locked_out(ctx, state, tab) is False, "not refused"
    assert len(toasts) == 1
    assert "hidden" in toasts[0][0]
    assert toasts[0][1] == "warn"


def test_a_visible_layer_says_nothing() -> None:
    ctx, state, tab, toasts = _session()
    assert inker_canvas._locked_out(ctx, state, tab) is False
    assert toasts == []


@pytest.mark.parametrize("tool", ["eyedropper", "select", "lasso", "wand"])
def test_a_read_only_tool_is_told_nothing_about_either(tool: str) -> None:
    """Deliberately the same exemptions the lock uses: a pick or a marquee
    writes to no layer, so neither state is any of its business."""
    ctx, state, tab, toasts = _session(tool=tool)
    tab.doc.stack.active.locked = True
    tab.doc.stack.active.visible = False
    assert inker_canvas._locked_out(ctx, state, tab) is False
    assert toasts == []
