"""The sheet-correction strip, censused rather than screenshotted.

``tests/inker/test_context_controls.py``'s rule: a control drawn and wired to
nothing is this codebase's most common historical defect, so the strip is
built through real imgui (no GL) and read back through the probe -- every
control carries a kind, every greyed one carries a reason, and on an ordinary
animation the strip is absent rather than greyed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from warlock.studio import inker_ops, inker_sheet, inker_state, probe, widgets
from warlock.studio.inker.sheetin import document_from_sheet
from warlock.studio.panes import inker_sheet as strip

CELL = 16
DIRECTIONS = ("front", "left", "back", "right")


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _sheet_doc():
    count = len(DIRECTIONS) * 2
    atlas = np.zeros((CELL, count * CELL, 4), dtype=np.uint8)
    cells, tags = [], []
    for d, direction in enumerate(DIRECTIONS):
        tags.append({"name": f"walk_{direction}", "start": d * 2, "end": d * 2 + 1, "loop": True})
        for f in range(2):
            index = d * 2 + f
            atlas[4:12, index * CELL + 6 : index * CELL + 10] = (40, 60, 200, 255)
            cells.append({"x": index * CELL, "y": 0, "w": CELL, "h": CELL})
    return document_from_sheet(atlas, cells, {"tags": tags, "frames": []})


def _scene(monkeypatch, doc):
    state = inker_state.InkerState()
    tab = inker_state.InkerDoc(doc=doc, uid="t")
    state.docs.append(tab)
    state.active_uid = "t"
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        toast=lambda text, level="info", *a, **k: toasts.append((text, level)),
        viewer=None,
    )
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    return ctx, state, tab, toasts


def _frame(imgui, ctx, tab):
    io = imgui.get_io()
    io.add_mouse_pos_event(-100.0, -100.0)
    io.add_mouse_button_event(0, False)
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((1400.0, 700.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    strip.draw_strip(ctx, tab)
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def test_every_strip_op_is_registered_in_the_sheet_menu():
    names = {op.name for op in inker_ops.menu("Sheet")}
    assert set(strip.STRIP_OPS) <= names


def test_the_strip_is_absent_on_an_ordinary_animation(ui, monkeypatch):
    from warlock.studio import inker

    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    ctx, _state, tab, _toasts = _scene(monkeypatch, doc)
    monkeypatch.setattr(probe, "ENABLED", True)
    assert _frame(ui, ctx, tab) == []
    assert tab.sheet_mark is None


def test_every_control_has_a_kind_and_every_greyed_one_a_reason(ui, monkeypatch):
    ctx, _state, tab, _toasts = _scene(monkeypatch, _sheet_doc())
    monkeypatch.setattr(probe, "ENABLED", True)
    controls = _frame(ui, ctx, tab)
    assert controls, "the strip drew nothing on a sheet document"
    for control in controls:
        assert control.kind, control.label
        if not control.enabled:
            assert control.reason, f"{control.label} is greyed with no reason"
    labels = " ".join(c.label for c in controls)
    for name in strip.STRIP_OPS:
        assert f"sheet-{name}" in labels, name


def test_on_front_the_mirror_buttons_say_why_and_propagate_waits_for_a_change(ui, monkeypatch):
    ctx, state, tab, _toasts = _scene(monkeypatch, _sheet_doc())
    monkeypatch.setattr(probe, "ENABLED", True)
    by = {c.label: c for c in _frame(ui, ctx, tab)}
    mirror = by["Apply to mirror##sheet-sheet_mirror"]
    assert not mirror.enabled
    assert "no mirror direction" in mirror.reason
    propagate = by["Propagate patch##sheet-sheet_propagate"]
    assert not propagate.enabled
    assert propagate.reason == inker_sheet.NO_MARK
    # The mark was taken on the first draw, so a change now is measurable.
    assert tab.sheet_mark is not None
    tab.doc.anim.cels[(tab.doc.anim.tracks[0].uid, tab.doc.anim.frames[0].uid)].pixels[
        14, 2
    ] = (255, 0, 0, 255)
    by = {c.label: c for c in _frame(ui, ctx, tab)}
    assert by["Propagate patch##sheet-sheet_propagate"].enabled


def test_pressing_propagate_through_the_registry_sends_the_change(monkeypatch):
    ctx, state, tab, toasts = _scene(monkeypatch, _sheet_doc())
    inker_sheet.sync_mark(tab)
    anim = tab.doc.anim
    cel = anim.cels[(anim.tracks[0].uid, anim.frames[0].uid)]
    cel.pixels[14, 2] = (255, 0, 0, 255)
    assert inker_ops.run(ctx, inker_ops.get("sheet_propagate"))
    for frame in (2, 4, 6):
        other = anim.cels[(anim.tracks[0].uid, anim.frames[frame].uid)]
        assert tuple(other.pixels[14, 2]) == (255, 0, 0, 255)
    assert toasts and toasts[-1][1] == "success"
    # The mark moved with the write, so a second press has nothing to send.
    assert not inker_ops.run(ctx, inker_ops.get("sheet_propagate"))


def test_the_mirror_report_is_cached_on_the_revision(monkeypatch):
    ctx, _state, tab, _toasts = _scene(monkeypatch, _sheet_doc())
    tab.doc.anim.current = 2  # walk_left frame 0
    first = inker_sheet.mirror_report(tab)
    assert first is not None
    assert inker_sheet.mirror_report(tab) is first
    tab.face_fraction = 0.5
    assert inker_sheet.mirror_report(tab) is not first
