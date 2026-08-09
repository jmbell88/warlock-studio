"""A keyboard focus ring, scoped to the panes that declare one (UX.md Phase 3).

**Not imgui's nav**, which is recorded broken here: ``dialogs`` turned it off
because with it on a focused button drew as focused and did nothing. What this
is instead is the Home-tiles pattern generalised -- a pane states an order over
its own controls, one of them is the cursor, Tab and Shift-Tab move it, and
:func:`widgets.ring` is the one focus visual the whole app uses.

The order is **the order the controls are drawn in**, recorded as the frame
builds them rather than written out beside the pane: a hand-kept list over a
hand-written column of widgets is two orderings, and they drift the first time
a control is inserted -- which is exactly the argument ``landing.TILES`` and
``palette.py`` already make about the two other indices of the same thing.

Two rules keep this from fighting imgui rather than riding on it.

**Focus is *asked for* only on the frame the cursor moved.** ``imgui``'s
``set_keyboard_focus_here`` is a command, not a state: issuing it every frame
would re-focus the control on every one of them, which resets a text cursor
mid-word and makes a slider undraggable. So the request is one-shot and the
ring is what persists.

**And the ring is drawn from the pane's own order rather than from imgui's
idea of focus**, because half these controls are hand-drawn (the segmented
output switch, the count radios) and imgui has no idea about them at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from imgui_bundle import imgui

from . import theme, widgets


def pump(state: Any, pane: str) -> None:
    """Answer Tab / Shift-Tab for ``pane``. Called *before* :func:`begin`.

    Before, because :func:`begin` clears the order and :func:`move` needs the
    one the last frame recorded -- which is also the only order that describes
    what is actually on screen right now.

    Read from imgui rather than from the pygame event loop, and that is not a
    style choice: ``App._shortcut`` is skipped entirely while a text field has
    the keyboard (``io.want_text_input``), and tabbing *out of the prompt box*
    is the single most important thing this has to do.

    Suppressed while any popup is up, for the reason ``App._modal_open`` exists:
    a modal owns the keyboard, and a Tab that also moved the ring behind it
    would leave the cursor somewhere the user never saw it go.
    """
    if imgui.is_popup_open("", imgui.PopupFlags_.any_popup.value):
        return
    if not imgui.is_key_pressed(imgui.Key.tab):
        return
    back = imgui.is_key_down(imgui.Key.left_shift) or imgui.is_key_down(imgui.Key.right_shift)
    move(state, pane, -1 if back else 1)


def begin(state: Any, pane: str) -> None:
    """Start recording ``pane``'s tab order for this frame.

    Called once, before the first :func:`item`. The list is rebuilt every frame
    because what is on screen changes -- a fold closes, a slider appears with
    the LoRA that justifies it -- and a cursor over controls that are not there
    is a Tab key that appears to do nothing.
    """
    state.focus_order[pane] = []


@contextmanager
def item(state: Any, pane: str, key: str) -> Iterator[bool]:
    """Wrap one control. -> whether it is the one the cursor is on.

    ``key`` is a stable name, never an index: the index of a control is a
    function of what else happens to be visible, so a remembered one would
    silently become a different control when a section is collapsed.
    """
    state.focus_order.setdefault(pane, []).append(key)
    focused = state.focus_key.get(pane) == key
    if focused and state.focus_moved:
        # Consumed here rather than at the end of the frame: exactly one
        # control is the target of a move, and leaving the flag set would have
        # every later control in the order ask for focus too -- the last one
        # drawn winning, which is not the one the user tabbed to.
        state.focus_moved = False
        imgui.set_keyboard_focus_here()
    try:
        yield focused
    finally:
        if focused:
            widgets.ring(
                imgui.get_item_rect_min(),
                imgui.get_item_rect_max(),
                theme.ACCENT,
                0.8,
                thick=1.5,
            )


def move(state: Any, pane: str, delta: int) -> bool:
    """Tab (or Shift-Tab). -> whether there was anywhere to go.

    Wraps, as Home's tiles do and unlike the library's arrow keys: a form is a
    fixed ring of controls, where a two-hundred-row list is not.

    A cursor whose control is no longer on screen restarts at the top rather
    than being clamped: it was somewhere that no longer exists, and the nearest
    surviving position is a guess, where the first control is an answer.
    """
    order = state.focus_order.get(pane) or []
    if not order:
        return False
    current = state.focus_key.get(pane)
    index = order.index(current) + delta if current in order else 0
    state.focus_key[pane] = order[index % len(order)]
    state.focus_moved = True
    return True


def activate(state: Any, pane: str) -> str | None:
    """Which control Enter would act on, or ``None``.

    A getter rather than a dispatcher: what "activate" *means* is the pane's
    business -- a button is pressed, a combo is opened, a text field is already
    typing into -- and a table of callbacks here would be a second place that
    knows what a pane's controls are.
    """
    return state.focus_key.get(pane)
