"""Where a named control ended up on screen, for the frame that just drew it.

``layout.FRAME_PANES`` already answers this for *panes*, and the layout editor
draws its whole overlay from it. A tutorial step points at a **control** -- the
Generate button, the prompt box, one item in the rail -- and a pane's rect is
several hundred pixels too coarse to ring one of those.

So this is the same bargain one level down. A pane calls :func:`mark` straight
after submitting the control, which records imgui's own "the item I just
submitted" rectangle -- exactly what ``widgets.field_error`` and ``focus.item``
already read to draw their rings. The tour overlay then draws *outside* every
pane from these rects, so nothing reaches into a pane's draw list that is not
that pane's own, and ``widgets.section_blocks``' channel split stays out of it
by construction.

Positions are read rather than computed, and that is load-bearing rather than
tidy: the rail alone recomputes every item's box each frame across a
three-rung compression ladder, so a highlight that did its own arithmetic would
drift the moment a window got short enough to compress it.

Cleared once a frame beside ``layout.FRAME_PANES``. A stale rect is worse than a
missing one -- the control is gone and the ring would sit over whatever took its
place, which reads as the tour pointing confidently at the wrong thing.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

#: ``key -> (x, y, w, h)`` in screen px, for this frame only.
FRAME_ANCHORS: dict[str, tuple[float, float, float, float]] = {}


def begin_frame() -> None:
    """Forget last frame's control rects. Called once, before anything draws."""

    FRAME_ANCHORS.clear()


def mark(key: str) -> None:
    """Record the rect of the item that has just been submitted.

    The context check comes *before* the imgui calls rather than as a ``try``
    around them: ``get_item_rect_min`` with no context is an access violation,
    not an exception, because imgui's null check is an assert compiled out of
    the release build. ``get_current_context`` is the one entry point that is
    safe to ask first.
    """
    if imgui.get_current_context() is None:
        return
    low = imgui.get_item_rect_min()
    high = imgui.get_item_rect_max()
    FRAME_ANCHORS[key] = (low.x, low.y, high.x - low.x, high.y - low.y)


def mark_window(key: str) -> None:
    """Record the current window's rect, for pointing at a whole panel.

    Some steps are about a *panel* -- "the toolbox", "the layers list" -- and
    ringing one control inside it would be pointing at the wrong scale. This is
    still a read rather than a computation, so a panel the user has resized or
    that the layout has squeezed is ringed where it actually is.
    """
    if imgui.get_current_context() is None:
        return
    pos = imgui.get_window_pos()
    size = imgui.get_window_size()
    FRAME_ANCHORS[key] = (pos.x, pos.y, size.x, size.y)


def rect(key: str) -> tuple[float, float, float, float] | None:
    """This frame's rect for ``key``, or ``None`` if it did not draw."""

    return FRAME_ANCHORS.get(key)


def marked() -> frozenset[str]:
    """Every key recorded this frame. For tests and for the tour's own guard."""

    return frozenset(FRAME_ANCHORS)


def item(key: str, drawn: Any = None) -> Any:
    """``mark`` as a pass-through, for wrapping a widget call in one line.

    ``if anchors.item("create/generate", widgets.primary_button("Generate")):``
    keeps the mark adjacent to the thing it is about, which is the only way it
    stays correct through a refactor that moves the control.
    """
    mark(key)
    return drawn
