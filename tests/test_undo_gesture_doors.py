"""One gesture, one undo step -- at every slider and drag door that writes history.

The 2026-09-02 review's first theme: a slider reports a change on every frame
the pointer moves, and a door that pushes a step per report turns one second of
dragging into sixty steps. With ``UNDO_MAX_DEPTH = 64`` that evicts every
earlier edit in the document, so dragging Tempo once cost the user their whole
session's history. ``controls.fold_undo`` is the fix -- ``UndoStack.mark`` on
activation, ``collapse_since`` on release -- and this file is what pins it to
each door it was applied to.

Three layers, cheapest first: the helper alone against a bare stack; the panes
drawn in a real imgui frame with the item state scripted, so the door's own
code runs and the document's own history is counted; and, for the doors that a
headless frame cannot open -- a context-menu popup, or a panel drawn only in
ordinary tab flow -- a source check that the fold sits between the field and
the write. The 2026-09-04 audit added six rows to that last layer: Clay's
material sliders, Plotter's layer and object opacity and tint, and Sirens's
channel Pan had no ``fold_undo`` call in their files at all.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from imgui_bundle import imgui
from test_sirens_mode import FakeCtx
from test_sirens_panes_smoke import _loaded, _no_device  # noqa: F401
from test_sirens_panes_smoke import frames as frames  # noqa: F401, PLC0414

from warlock.studio import controls, sirens_mode, undo, widgets
from warlock.studio.panes import (
    clay_props,
    inker_colors,
    inker_picker,
    inker_timeline,
    packwright_settings,
    plotter_layers,
    plotter_tileset_editor,
    sirens_effects,
    sirens_instruments,
    sirens_orders,
    sirens_patterns,
    sirens_transport,
)


@dataclass
class _Step(undo.Edit):
    def undo(self, doc: Any) -> None:
        pass

    def redo(self, doc: Any) -> None:
        pass


class _Item:
    """Scripted item state: which frame the drag begins and which it ends.

    imgui has one active item, so the state is reported only while the
    scripted field is the last one drawn (``label``, set by ``_scripted_slider``;
    every field when ``None``).
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, begin: int, end: int) -> None:
        self.frame = 0
        self.begin, self.end = begin, end
        self.label: str | None = None
        self.last: str | None = None
        monkeypatch.setattr(imgui, "is_item_activated", lambda: self._at(self.begin))
        monkeypatch.setattr(imgui, "is_item_deactivated", lambda: self._at(self.end))
        # A test that fails mid-gesture must not leak it into the next.
        monkeypatch.setattr(controls, "_gesture", None)

    def _at(self, frame: int) -> bool:
        return (self.label is None or self.last == self.label) and self.frame == frame

    def dragging(self) -> bool:
        return self.begin <= self.frame < self.end


# --- 1. the helper -------------------------------------------------------------


def test_a_drag_of_seventy_frames_is_one_step_and_evicts_nothing(monkeypatch):
    """Longer than the depth cap on purpose: the eviction is deferred while the
    gesture is open, so the step before the drag survives it."""
    stack = undo.UndoStack()
    earlier = _Step()
    stack.push(earlier)
    item = _Item(monkeypatch, begin=1, end=71)
    for item.frame in range(1, 72):
        controls.fold_undo(stack)
        if item.dragging():
            stack.push(_Step())
    assert len(stack) == 2
    assert stack._done[0] is earlier
    assert isinstance(stack.top, undo.CompoundEdit)
    assert controls._gesture is None


def test_a_drag_that_moved_once_stays_a_plain_step(monkeypatch):
    stack = undo.UndoStack()
    item = _Item(monkeypatch, begin=1, end=2)
    for item.frame in (1, 2):
        controls.fold_undo(stack)
        if item.dragging():
            stack.push(_Step())
    assert len(stack) == 1
    assert not isinstance(stack.top, undo.CompoundEdit)


def test_a_field_without_history_folds_nothing(monkeypatch):
    item = _Item(monkeypatch, begin=1, end=2)
    for item.frame in (1, 2):
        controls.fold_undo(None)
    assert controls._gesture is None


def test_an_orphaned_gesture_is_closed_by_the_next_activation(monkeypatch):
    """A pane that closes mid-drag never reports the deactivation. The next
    drag anywhere closes the stray, so the deferred eviction cannot stay
    switched off for the rest of the session."""
    first, second = undo.UndoStack(), undo.UndoStack()
    item = _Item(monkeypatch, begin=1, end=99)
    item.frame = 1
    controls.fold_undo(first)
    first.push(_Step())
    first.push(_Step())
    assert first._open_gestures == 1
    controls.fold_undo(second)  # frame 1 again: a fresh activation
    assert first._open_gestures == 0, "the stray was closed"
    assert len(first) == 1, "and folded"
    assert controls._gesture[0] is second


# --- 2. the Sirens doors, drawn -----------------------------------------------


def _scripted_field(
    monkeypatch: pytest.MonkeyPatch, item: _Item, attr: str, label: str, values: list
):
    """The real field, with its answer overridden for one label while the
    scripted drag is on -- there is no pointer in a headless frame.

    ``attr`` is the name on ``controls`` (``"slider_int"``, ``"slider_float"``,
    ``"drag_int"``, ``"color_edit4"``, ``"input_float"``...): every one of
    them is called ``(label, value, ...)`` and answers ``(changed, value)``,
    so one wrapper covers them all.
    """
    real = getattr(controls, attr)
    item.label = label

    def scripted(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        item.last = args[0]
        if args[0] == label and item.dragging():
            return True, values[item.frame - item.begin]
        return result

    monkeypatch.setattr(controls, attr, scripted)


def _scripted_slider(monkeypatch: pytest.MonkeyPatch, item: _Item, label: str, values: list):
    """The real slider, with its answer overridden for one label while the
    scripted drag is on -- there is no pointer in a headless frame."""
    _scripted_field(monkeypatch, item, "slider_int", label, values)


def _drag(draw_frame: Any, draw: Any, item: _Item) -> None:
    """Frames 1 .. end: the drag, then the release frame."""
    for frame in range(1, item.end + 1):
        item.frame = frame
        draw_frame(draw)


@pytest.mark.parametrize(
    "label,attr,values",
    [("Tempo", "tempo", [121, 124, 130, 133, 140]), ("Speed", "speed", [7, 8, 9, 10, 11])],
)
def test_transport_tempo_and_speed_drags_are_one_step_each(
    monkeypatch, frames, label, attr, values
):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    before = len(tab.doc.history)
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, label, values)
    _drag(frames, lambda: sirens_transport.draw(ctx), item)
    assert getattr(tab.doc, attr) == values[-1]
    assert len(tab.doc.history) == before + 1
    assert tab.doc.history.undo(tab.doc)
    assert getattr(tab.doc, attr) != values[-1], "one Ctrl+Z takes the whole drag back"


def test_order_list_rows_drag_is_one_step(monkeypatch, frames):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    pattern = tab.doc.patterns[0]
    sirens_mode.set_caret(ctx, pattern=pattern.uid)
    before = len(tab.doc.history)
    values = [60, 56, 50, 44, 40]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, "Rows", values)
    _drag(frames, lambda: sirens_orders.draw(ctx), item)
    assert pattern.rows == values[-1]
    assert len(tab.doc.history) == before + 1


@pytest.mark.parametrize(
    "label,attr,values",
    [("Tempo", "tempo", [121, 124, 130, 133, 140]), ("Speed", "speed", [7, 8, 9, 10, 11])],
)
def test_effect_tempo_and_speed_drags_are_one_step_each(monkeypatch, frames, label, attr, values):
    ctx = FakeCtx()
    tab = _loaded(ctx)
    state = sirens_mode.ensure(ctx)
    effect = tab.doc.oneshot(state.oneshot)
    before = len(tab.doc.history)
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, label, values)
    _drag(frames, lambda: sirens_effects.draw(ctx), item)
    assert getattr(tab.doc.oneshot(effect.uid), attr) == values[-1]
    assert len(tab.doc.history) == before + 1


# --- 3. the Inker picker, drawn -----------------------------------------------


class _PickerDoc:
    def __init__(self) -> None:
        self.palette = [(10, 20, 30, 255), (0, 0, 0, 255)]
        self.is_indexed = True
        self.history = undo.UndoStack()

    def recolour_slot(self, index: int, colour: Any) -> bool:
        self.palette[index] = tuple(colour)
        self.history.push(_Step())
        return True


class _PickerState:
    fg = (10, 20, 30, 255)
    bg = (255, 255, 255, 255)
    fg_slot = 0
    picker_target = "fg"
    palette_usage = None

    def set_fg(self, colour: Any, slot: Any = None) -> None:
        self.fg = tuple(int(c) for c in tuple(colour)[:4])
        self.fg_slot = slot


class _PickerTab:
    def __init__(self) -> None:
        self.doc = _PickerDoc()


def test_a_picker_channel_drag_over_a_palette_slot_is_one_step(monkeypatch, frames):
    tab = _PickerTab()
    state = _PickerState()
    values = [40, 80, 120, 160, 200]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, "##Red", values)
    _drag(frames, lambda: inker_picker._rgb(None, state, tab, 0, tab.doc.palette[0]), item)
    assert tab.doc.palette[0][0] == values[-1]
    assert len(tab.doc.history) == 1


def test_a_picker_drag_over_a_free_colour_opens_no_gesture(monkeypatch, frames):
    """Session state, not history: nothing to fold, and nothing left open."""
    tab = _PickerTab()
    state = _PickerState()
    state.fg_slot = None
    values = [40, 80, 120]
    item = _Item(monkeypatch, begin=1, end=1 + len(values))
    _scripted_slider(monkeypatch, item, "##Red", values)
    _drag(frames, lambda: inker_picker._rgb(None, state, tab, None, state.fg), item)
    assert state.fg[0] == values[-1]
    assert len(tab.doc.history) == 0
    assert tab.doc.history._open_gestures == 0


# --- 4. the popup doors, by source --------------------------------------------


@pytest.mark.parametrize(
    "func,field,write",
    [
        (inker_timeline._group_menu, 'slider_float(\n        "Opacity##group"', "set_group_props("),
        (inker_timeline._cell_menu, '"Opacity##cel"', "set_cel_opacity("),
        (inker_timeline._cell_menu, '"Z##cel"', "set_cel_z("),
        (inker_colors._slots, 'color_edit4("Slot"', "recolour_slot("),
        # The four the 2026-09-04 audit found still unfolded: none of these
        # files called ``fold_undo`` at all, and each setter pushes per report.
        (clay_props._material, '"base colour##bm"', "doc.set_material("),
        (clay_props._material, '"metallic##bm"', "doc.set_material("),
        (plotter_layers._layer_table, '"##layer-opacity"', "doc.set_layer_props("),
        (plotter_layers._layer_table, '"##layer-tint"', "doc.set_layer_props("),
        (plotter_layers._object_fields, '"##obj-opacity"', "doc.set_object("),
        (sirens_patterns._channel_popup, 'f"Pan##{tag}"', "update_channel("),
        # The 2026-09-05 audit's M06 row: the Terrain tab's colour swatch and
        # its probability field, neither folded, each pushing through
        # ``set_wang_colour`` -> ``_write_wangsets`` per report.
        (plotter_tileset_editor._wang_colours, '"##swatch"', "set_wang_colour("),
        (plotter_tileset_editor._wang_colours, '"Probability"', "set_wang_colour("),
    ],
    ids=[
        "group-opacity",
        "cel-opacity",
        "cel-z",
        "palette-slot",
        "clay-base-colour",
        "clay-metallic",
        "layer-opacity",
        "layer-tint",
        "object-opacity",
        "channel-pan",
        "wang-colour",
        "wang-probability",
    ],
)
def test_the_popup_doors_fold_between_the_field_and_the_write(func, field, write):
    """These draw only inside ``begin_popup_context_item``, which a headless
    frame cannot open. The invariant is positional: draw, fold, act."""
    source = inspect.getsource(func)
    after_field = source.split(field, 1)[1]
    # ``fold_undo``'s argument differs by pane (``doc.history``,
    # ``tab.doc.history``), so the call itself is what is looked for.
    fold = after_field.index("controls.fold_undo(")
    assert fold < after_field.index(write), f"{write} runs before the fold"


# --- 5. the name fields -------------------------------------------------------


def test_a_typed_name_is_reported_once_when_the_field_is_left(monkeypatch):
    typed = iter(["l", "le", "lea", "lead", "lead"])
    settled = iter([False, False, False, False, True])
    monkeypatch.setattr(widgets.imgui, "input_text", lambda label, value: (True, next(typed)))
    monkeypatch.setattr(widgets.imgui, "is_item_deactivated_after_edit", lambda: next(settled))
    monkeypatch.setattr(widgets, "note_ime_rect", lambda: None)
    seen = [widgets.input_text("Name", "old", commit=True) for _ in range(5)]
    assert seen == ["old", "old", "old", "old", "lead"]


@pytest.mark.parametrize("pane", [sirens_effects, sirens_instruments])
def test_the_sirens_name_fields_commit_on_release(pane):
    source = inspect.getsource(pane)
    field = source.split('widgets.input_text(\n        "Name"', 1)[1].split(")", 1)[0]
    assert "commit=True" in field


# --- 6. the 2026-09-05 audit's own doors ---------------------------------------


def test_an_interrupted_opacity_drag_still_leaves_one_undo_step(monkeypatch, frames):
    """M05: ``inker_menu.header_controls`` used to set ``layer.opacity``
    directly for the live preview and stash the pre-drag value in a bare
    module-level dict keyed by layer uid, pushing the real undo step only on
    ``is_item_deactivated_after_edit``. A control that vanishes first --
    a mode switch, a panel collapse, the document closing -- never renders
    that release frame: the step was never pushed, the dict entry stuck
    around forever, and a later gesture that reused the same uid started
    from that stale value instead of the layer's real one.

    Routed through ``fold_undo`` and a push on every changed frame instead,
    the interruption is survived by the mechanism every other door in this
    file already relies on: whatever gesture is open closes the moment any
    other item activates, wherever that happens to be.
    """
    import numpy as np

    from warlock.studio import inker
    from warlock.studio.panes import inker_menu

    doc = inker.Document.from_pixels(np.full((4, 4, 4), 255, dtype=np.uint8))
    before = len(doc.history)
    # Opacity is drawn as a percentage (``labeled_slider_float`` infers
    # ``percent`` for a 0..1 range), so the raw ``controls.slider_float``
    # call this scripts sits in 0..100 space; ``header_controls`` divides
    # back down before it ever reaches the layer.
    values_pct = [80, 60, 40]
    item = _Item(monkeypatch, begin=1, end=1 + len(values_pct))
    _scripted_field(monkeypatch, item, "slider_float", "##Opacity", values_pct)
    # The drag runs, but its own release frame (``item.end``) never comes --
    # the panel is torn down mid-gesture.
    for frame in range(1, item.end):
        item.frame = frame
        frames(lambda: inker_menu.header_controls(None, doc))
    assert doc.stack.active.opacity == pytest.approx(values_pct[-1] / 100.0)
    # Each changed frame already pushed its own step (unlike the old direct
    # assignment, which pushed nothing until release) -- the gesture is still
    # open, so nothing has folded yet, but the edit already exists.
    assert len(doc.history) == before + len(values_pct), "every frame's push landed"
    # Nothing above will ever report the deactivation. What closes a stray
    # gesture is the next activation anywhere -- ``test_an_orphaned_gesture_
    # is_closed_by_the_next_activation`` pins the mechanism itself; this
    # reuses it exactly the way a mode switch followed by any other drag
    # would, on a completely unrelated stack standing in for "the user went
    # on and did something else".
    elsewhere = undo.UndoStack()
    item.frame = item.begin
    controls.fold_undo(elsewhere)
    assert len(doc.history) == before + 1, "the interrupted drag folded to one step"
    assert doc.history.undo(doc)
    assert doc.stack.active.opacity == 1.0, "one Ctrl+Z takes the whole drag back"


def test_the_plotter_layer_list_opacity_row_drag_is_one_step(monkeypatch, frames):
    """M06: ``_opacity_row`` -- the Opacity slider drawn over the layer list
    itself, distinct from ``_layer_table``'s own copy in Properties -- called
    ``doc.set_layer_props`` on every changed frame with no fold at all."""
    from test_plotter_mode import FakeCtx as PlotterFakeCtx
    from test_plotter_mode import _tab as _plotter_tab

    from warlock.studio.panes import plotter_layers

    ctx = PlotterFakeCtx()
    tab = _plotter_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    before = len(tab.doc.history)
    # Percent-scaled, same as the inker opacity door above.
    values_pct = [80, 60, 40, 20]
    item = _Item(monkeypatch, begin=1, end=1 + len(values_pct))
    _scripted_field(monkeypatch, item, "slider_float", "##Opacity", values_pct)
    _drag(frames, lambda: plotter_layers._opacity_row(tab.doc, layer, True), item)
    assert layer.opacity == pytest.approx(values_pct[-1] / 100.0)
    assert len(tab.doc.history) == before + 1
    assert tab.doc.history.undo(tab.doc)


@pytest.mark.parametrize(
    "field,write",
    [
        ('"Columns"', "packwright_mode.set_settings("),
        ('"Padding"', "packwright_mode.set_settings("),
        ('"Extrude"', "packwright_mode.set_settings("),
    ],
    ids=["columns", "padding", "extrude"],
)
def test_packwright_settings_drags_fold_between_the_field_and_the_write(field, write):
    """M06: Columns (a drag_int) and Padding/Extrude (sliders) each called
    ``packwright_mode.set_settings`` on every changed frame, which reaches
    the unconditional history push in ``PackDoc.set_settings``
    (``document.py:410``) once per report -- the audit's reminder that
    Columns is not exempt just because it drags rather than slides.

    By source rather than a drawn frame: all three fields live in the same
    ``draw`` and each folds independently, so a scripted single-item drag
    cannot tell one field's activation from another's the way the real imgui
    item stack does -- ``fold_undo``'s own positional contract (draw, fold,
    act) is what a headless frame cannot exercise here without lying about
    which field the pointer is on, and is exactly what a source check proves
    instead. This is the same reason ``clay_props``'s three material fields
    are pinned this way rather than driven live.
    """
    source = inspect.getsource(packwright_settings.draw)
    after_field = source.split(field, 1)[1]
    fold = after_field.index("controls.fold_undo(")
    assert fold < after_field.index(write), f"{write} runs before the fold"
