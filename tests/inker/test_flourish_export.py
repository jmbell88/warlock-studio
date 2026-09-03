"""Exporting an effect: one sheet per phase through the per-tag export, and
the engine snippet that describes one of those files."""

from __future__ import annotations

import ast
import dataclasses
from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import inker, inker_flourish, inker_mode, inker_ops, inker_state, probe
from warlock.studio.inker import sheetout
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import engines, presets
from warlock.studio.panes import inker_flourish as pane


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState(), manual=None)
        self.toasts: list = []
        self.legs: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return False

    def progress(self, key):
        return None

    def submit(self, key, fn, *a, **k):
        return True


def _scene(with_effect: bool = True):
    ctx = _Ctx()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32), title="spell.ora")
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    if with_effect:
        rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=2)
        tab.doc.insert_flourish(B.bake(rec))
    return ctx, tab


def test_export_is_greyed_without_an_effect_and_offered_with_one():
    ctx, tab = _scene(with_effect=False)
    op = inker_ops.get("flourish_export")
    assert not op.enabled(ctx.state.inker, tab)
    assert inker_ops.reason_for(op, ctx.state.inker, tab) == inker_flourish.NO_EFFECT
    ctx, tab = _scene()
    assert op.enabled(ctx.state.inker, tab)
    assert inker_ops.get("flourish_snippet").enabled(ctx.state.inker, tab)


def test_export_runs_the_per_tag_export_once(monkeypatch):
    ctx, tab = _scene()
    calls: list = []
    monkeypatch.setattr(inker_mode, "export_per_tag", lambda c, t, kind: calls.append((t, kind)))
    assert inker_ops.run(ctx, inker_ops.get("flourish_export"))
    assert calls == [(tab, "sheet")]


def test_the_snippet_describes_the_file_the_per_tag_export_writes():
    ctx, tab = _scene()
    anim = tab.doc.anim
    info = inker_flourish.snippet_info(tab, "sparks")
    tag = next(t for t in anim.tags if t.name == "sparks")
    first, last = sheetout.tag_span(anim, tag)
    stem = sheetout.filename_for(sheetout.DEFAULT_TAG_TEMPLATE, title="spell", tag="sparks")
    assert info["image"] == f"{stem}.png"
    assert info["frames"] == last - first + 1
    assert info["frame_width"] == 32 and info["frame_height"] == 32
    assert info["origin"] == [16, 16]
    assert info["loop"] is False
    assert info["fps"] == round(1000 / anim.frames[first].duration_ms)
    assert inker_flourish.snippet_info(tab, "no-such-tag") is None
    assert inker_flourish.snippet_text(tab, "no-such-tag", "godot") == ""


def test_the_snippet_text_is_the_engine_module_output():
    ctx, tab = _scene()
    for engine in engines.ENGINES:
        text = inker_flourish.snippet_text(tab, "hit", engine)
        assert text == engines.snippet(engine, inker_flourish.snippet_info(tab, "hit"))
    ast.parse(inker_flourish.snippet_text(tab, "hit", "pygame-ce"))


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def test_the_snippet_popup_draws_its_controls(ui):
    ctx, tab = _scene()

    def draw():
        pane.open_snippet_popup(ctx, tab)
        pane.snippet_popup(ctx, tab)

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        draw()
    finally:
        ui.end()
        ui.end_frame()
    labels = {c.label for c in probe.census()}
    assert any("Copy" in label for label in labels)
    assert any("Close" in label for label in labels)
    assert ctx.state.inker.flourish_snippet_tag == "hit"
