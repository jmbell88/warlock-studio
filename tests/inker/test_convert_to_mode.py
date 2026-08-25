"""The Convert popup also enters indexed *mode*, with a dither.

The menu row that opens it is called "Colour mode...", and what it did was
snap the pixels onto a palette while leaving the document in RGB. Meanwhile the
mode buttons entered indexed mode with ``"nearest"`` hard-coded and no way to
ask for anything else -- so the one conversion in the app that changes mode was
the one conversion with no dither.

One popup answers both now. ``convert_mode`` says which question it is asking.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from warlock.studio import inker
from warlock.studio.inker_state import InkerDoc, InkerState
from warlock.studio.panes import inker_bridge


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=InkerState())
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, level: str = "info", *_: Any) -> None:
        self.toasts.append((text, level))


def _ramp_tab(ctx: _Ctx) -> InkerDoc:
    doc = inker.Document.blank(16, 4)
    ramp = np.linspace(0, 255, 16).astype("uint8")
    doc.stack.active.pixels[:, :, :3] = ramp[None, :, None]
    doc.stack.active.pixels[:, :, 3] = 255
    doc.invalidate_all()
    tab = InkerDoc(doc=doc, title="ramp")
    ctx.state.inker.add(tab)
    return tab


def _session(ctx: _Ctx, tab: InkerDoc, *, mode: str, method: str) -> None:
    state = ctx.state.inker
    assert tab.doc.begin_convert()
    state.convert_uid = tab.uid
    state.convert_mode = mode
    state.convert_method = method
    state.convert_max = 4
    state.convert_table = tab.doc.built_palette(4)


def test_applying_a_mode_session_enters_indexed_mode():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="indexed", method="nearest")
    assert inker_bridge.apply_convert(ctx, tab)
    assert tab.doc.is_indexed


def test_a_mode_session_uses_the_matrix_that_was_chosen():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="indexed", method="floyd-steinberg")
    assert inker_bridge.apply_convert(ctx, tab)
    plane = tab.doc.composite[..., 0]
    assert not np.array_equal(plane[0], plane[1])


def test_a_plain_session_still_snaps_and_leaves_the_mode_alone():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="", method="nearest")
    assert inker_bridge.apply_convert(ctx, tab)
    assert not tab.doc.is_indexed
    assert len(np.unique(tab.doc.composite[..., 0])) <= 4


def test_applying_closes_the_session_either_way():
    ctx = _Ctx()
    tab = _ramp_tab(ctx)
    _session(ctx, tab, mode="indexed", method="nearest")
    inker_bridge.apply_convert(ctx, tab)
    assert ctx.state.inker.convert_uid == ""
    assert ctx.state.inker.convert_mode == ""
