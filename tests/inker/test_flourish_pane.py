"""The inspector and the insert popup draw, and only when they apply.

A real imgui context with no GL (``_ui_context``), the census on, so what is
asserted is what a user would find: the inspector's controls exist when an
effect layer is active and none of them exist otherwise, and no control here
bypasses ``controls`` (the census counts every one).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import inker, inker_flourish, inker_state, probe
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import presets
from warlock.studio.panes import inker_flourish as pane


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState(), manual=None)
        self.submitted: list[str] = []
        self.toasts: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return False

    def progress(self, key):
        return None

    def submit(self, key, fn, *a, **k):
        self.submitted.append(key)
        return True


def _scene(with_effect: bool = True):
    ctx = _Ctx()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    if with_effect:
        rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=2)
        tab.doc.insert_flourish(B.bake(rec))
    else:
        tab.doc.add_frame()
    return ctx, tab


def _frame(ui, fn):
    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        fn()
    finally:
        ui.end()
        ui.end_frame()
    return probe.census()


def test_the_inspector_draws_controls_for_an_effect_layer(ui):
    ctx, tab = _scene()
    seen = _frame(ui, lambda: pane.draw_inspector(ctx, tab))
    labels = {c.label for c in seen}
    assert any("Regenerate" in label for label in labels)
    assert any("Detach" in label for label in labels)
    # The primitive's own parameters, drawn from its table.
    assert any("radius" in label or "count" in label for label in labels)


def test_the_inspector_draws_nothing_on_an_ordinary_animation(ui):
    ctx, tab = _scene(with_effect=False)
    seen = _frame(ui, lambda: pane.draw_inspector(ctx, tab))
    assert seen == []


def test_the_inspector_draws_nothing_when_the_active_layer_is_outside_the_effect(ui):
    ctx, tab = _scene()
    tab.doc.set_active_layer(0)
    seen = _frame(ui, lambda: pane.draw_inspector(ctx, tab))
    assert seen == []


def test_the_inspector_submits_a_due_render_on_its_tick(ui):
    ctx, tab = _scene()
    state = ctx.state.inker
    group = next(iter(tab.doc.flourish))
    rec = tab.doc.flourish_state(group).recipe
    inker_flourish.set_pending(state, group, dataclasses.replace(rec, seed=3), now=0.0)
    _frame(ui, lambda: pane.draw_inspector(ctx, tab))
    assert ctx.submitted == [inker_flourish.render_key(tab, group)]


def test_the_popup_draws_its_choices_when_open(ui):
    ctx, tab = _scene(with_effect=False)

    def draw():
        pane.open_popup(ctx, tab)
        pane.popup(ctx, tab)

    seen = _frame(ui, draw)
    labels = {c.label for c in seen}
    assert any("Insert" in label for label in labels)
    assert any("Cancel" in label for label in labels)
