"""What controls a frame actually drew, for a driver that presses them.

``tests/test_studio_smoke.py`` asserts that every pane *builds*. It asserts
nothing about whether a control is wired to anything, whether it is reachable,
or whether pressing it does what its label says --- and
``scripts/screenshot_modes.py`` photographs a mode *at rest*. So the failure
nobody catches is a control that draws correctly and does nothing: clipped past
its content region, disabled with no reason, wired to a handler that was
renamed, or reaching one that raises into the frame and gets swallowed. Every
one of those passes the smoke suite and looks right in a screenshot.

This is the half that makes such a control *addressable*: a per-frame census of
everything :func:`controls._finish_item` saw, with the rect a driver can click.
It is the same bargain :mod:`.anchors` makes one level down --- positions are
**read** rather than computed, because the rail alone recomputes every item's
box each frame across a three-rung compression ladder, and a click target that
did its own arithmetic would land on empty space the moment a window got short
enough to compress it.

Derived rather than hand-kept. ``_finish_item`` is the one chokepoint every
button, field, choice, row and menu item in the studio passes through, so an
inventory taken there cannot fall behind the UI the way a written-out list of
"every control" would --- which is the same reason ``screenshot_modes.py``
derives its mode list from ``modes.KEYS``. The controls that call imgui
directly (all of them in ``widgets.py``) are invisible here; a pinning test
counts them so the blind spot cannot grow quietly, and the driver reports the
number rather than pretending to full coverage.

**Env-gated and read-only.** With ``WARLOCK_UI_PROBE`` unset this costs one
module attribute lookup per control and does nothing else --- the same posture
:mod:`.anchors` has, and the reason the flag is sampled once at import rather
than per call. Nothing here writes to app state.

Cleared once a frame beside ``layout.FRAME_PANES`` and ``anchors.FRAME_ANCHORS``.
A stale census is worse than a missing one, for the reason ``anchors``' own
docstring gives: the control is gone, and a driver would click whatever took
its place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from imgui_bundle import imgui

#: Sampled once, at import. See the module docstring: the point of the gate is
#: that an ungated run pays nothing, and re-reading the environment per control
#: would be most of the cost the gate exists to avoid.
ENABLED = os.environ.get("WARLOCK_UI_PROBE") == "1"


@dataclass(frozen=True)
class Control:
    """One control, as the frame that drew it saw it."""

    #: The label as submitted, imgui id suffix and all.
    label: str
    #: Which ``controls`` function drew it: ``button``, ``checkbox``, ``combo``…
    kind: str
    #: ``(x, y, w, h)`` in screen px, read from imgui.
    rect: tuple[float, float, float, float]
    enabled: bool = True
    reason: str = ""
    selected: bool = False
    #: The hover text, when there is one. Load-bearing rather than decorative:
    #: a tool button's label is a private-use icon codepoint, so the tooltip is
    #: the only human-readable name a report can put next to its picture.
    tooltip: str = ""
    #: imgui's own answer to "was this clipped away". A control that is not
    #: visible is not clickable, and that distinction is the whole point of a
    #: pass that exists to find controls pushed past their content region.
    visible: bool = True
    #: The imgui child window the control was submitted into -- which is the
    #: pane, since ``layout.pane`` names its child by the slot id. Read at
    #: submission time, because that is the only moment it is knowable.
    #:
    #: Not a hit test against ``layout.FRAME_PANES``: that dict is filled by
    #: ``layout.column`` alone, and a workspace that opens a pane directly (the
    #: Inker timeline does) records nothing in it -- so the whole of Inker came
    #: out attributed to no pane at all. imgui's own window is the answer
    #: whatever drew it.
    window: str = ""
    #: The ``layout.FRAME_PANES`` slot whose rect contains this one's centre,
    #: when there is one. Filled in by :func:`census`, never by :func:`record`
    #: -- ``layout.column`` records a pane's rect when the pane *closes*, so at
    #: submission time a control's own pane is not in the dict yet.
    pane: str = ""

    @property
    def text(self) -> str:
        """The part a user reads -- imgui's ``##id`` suffix removed."""

        return self.label.split("##", 1)[0].strip()

    @property
    def name(self) -> str:
        """The best human-readable name there is for this control.

        An icon button's label is a single private-use codepoint from the
        bundled lucide atlas, which is unreadable in a report and unusable in a
        filename -- so the tooltip stands in, and the raw label is the last
        resort.
        """
        readable = self.text
        if readable and readable.isprintable() and readable.isascii():
            return readable
        if self.tooltip:
            return self.tooltip.splitlines()[0].strip()
        # The id half of the label, which for an icon button is the only thing
        # left that says what it is: ``##inker/tool/fill`` -> ``fill``.
        if "##" in self.label:
            return self.label.split("##", 1)[1].rsplit("/", 1)[-1]
        return readable or self.label

    @property
    def where(self) -> str:
        """The best available name for where this control is."""

        return self.pane or self.window or ""

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.rect
        return (x + w * 0.5, y + h * 0.5)


#: Every control drawn this frame, in submission order. This frame only.
FRAME_CONTROLS: list[Control] = []


def begin_frame() -> None:
    """Forget last frame's census. Called once, before anything draws."""

    FRAME_CONTROLS.clear()


def _pane_at(x: float, y: float) -> str:
    """Which pane slot contains ``(x, y)``, or ``""``.

    Imported here rather than at module scope: ``layout`` is a heavier module
    than this one wants to depend on at import time, and the lookup only
    happens on a probe run.
    """
    from . import layout

    for slot, (px, py, pw, ph) in layout.FRAME_PANES.items():
        if px <= x <= px + pw and py <= y <= py + ph:
            return slot
    return ""


def record(
    *,
    label: str,
    kind: str,
    enabled: bool = True,
    reason: str = "",
    selected: bool = False,
    tooltip: str = "",
) -> None:
    """Record the item that has just been submitted. No-op unless enabled.

    The context check comes *before* the imgui reads rather than as a ``try``
    around them, for the reason ``anchors.mark`` gives: ``get_item_rect_min``
    with no context is an access violation, not an exception, because imgui's
    null check is an assert compiled out of the release build.
    """
    if not ENABLED or not kind:
        return
    if imgui.get_current_context() is None:
        return
    low = imgui.get_item_rect_min()
    high = imgui.get_item_rect_max()
    try:
        visible = bool(imgui.is_item_visible())
    except (AttributeError, RuntimeError):
        visible = True
    window = ""
    try:
        # ``ImGuiWindow.name`` is the full id path -- ``##host/.../inker-tools``
        # -- so the last segment is the pane's own id, minus imgui's hash.
        raw = imgui.internal.get_current_window().name
        window = str(raw).rsplit("/", 1)[-1].rsplit("_", 1)[0]
    except (AttributeError, RuntimeError, TypeError):
        window = ""
    FRAME_CONTROLS.append(
        Control(
            label=str(label),
            kind=str(kind),
            rect=(low.x, low.y, high.x - low.x, high.y - low.y),
            enabled=bool(enabled),
            reason=str(reason),
            selected=bool(selected),
            tooltip=str(tooltip),
            visible=visible,
            window=window,
        )
    )


def census() -> list[Control]:
    """This frame's controls, with panes resolved. A snapshot to keep.

    Between frames is the only moment this answer is available: ``layout``
    records a pane's rect as the pane *closes*, so the pane holding a control
    is not in ``FRAME_PANES`` while that control is being submitted. Called
    after a frame and before the next one clears the record, every pane in it
    is complete.
    """
    return [
        replace(one, pane=_pane_at(*one.centre)) for one in FRAME_CONTROLS
    ]
