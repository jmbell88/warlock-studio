"""The Flourish ops and the render loop, driven through a fake context.

The context runs every submitted task synchronously and hands its result back
through ``inker_mode.on_task_done`` exactly as the app would, so what is under
test is the real route: op -> ``inker_mode`` verb -> task -> ``land`` ->
``Document``. Nothing here draws.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from warlock.studio import inker, inker_flourish, inker_mode, inker_ops, inker_state
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import presets
from warlock.studio.tasks import Done


class _Ctx:
    """Runs tasks inline and records toasts; ``on_task_done`` is called for each."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.toasts: list[tuple[str, str]] = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.pending: list[Done] = []
        self.auto_land = True
        self._busy: set[str] = set()

    def toast(self, text: str, level: str = "info", **_: object) -> None:
        self.toasts.append((text, level))

    def busy(self, key: str) -> bool:
        return key in self._busy

    def progress(self, key: str):
        return None

    def submit(self, key: str, fn, *args, **kwargs) -> bool:
        if key in self._busy:
            return False
        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 -- the runner reports, never raises
            done = Done(key=key, error=exc)
        if self.auto_land:
            inker_mode.on_task_done(self, done)
        else:
            self._busy.add(key)
            self.pending.append(done)
        return True

    def land_all(self) -> None:
        while self.pending:
            done = self.pending.pop(0)
            self._busy.discard(done.key)
            inker_mode.on_task_done(self, done)


def _open(ctx: _Ctx, size=(32, 32)) -> inker_state.InkerDoc:
    tab = inker_state.InkerDoc(doc=inker.Document.blank(*size))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    return tab


def _small(name: str = "smoke_puff"):
    return dataclasses.replace(presets.load(name), width=32, height=32, supersample=2)


@pytest.fixture
def ctx():
    return _Ctx()


def test_the_eleven_ops_live_in_the_flourish_menu():
    names = {op.name for op in inker_ops.OPS if op.menu == "Flourish"}
    assert names == {
        "flourish_prompt",
        "flourish_restyle",
        "flourish_insert",
        "flourish_regenerate",
        "flourish_regenerate_all",
        "flourish_keep_edits",
        "flourish_detach",
        "flourish_export",
        "flourish_snippet",
        "flourish_texture_selection",
        "flourish_texture_generate",
    }
    assert "Flourish" in inker_ops.MENUS


def test_every_flourish_op_answers_with_nothing_open():
    state = inker_state.InkerState()
    for op in inker_ops.OPS:
        if op.menu != "Flourish":
            continue
        assert op.enabled(state, None) is False
        assert inker_ops.reason_for(op, state, None)


def test_insert_is_greyed_without_a_document_and_offered_with_one(ctx):
    op = inker_ops.get("flourish_insert")
    assert not op.enabled(ctx.state.inker, None)
    tab = _open(ctx)
    assert op.enabled(ctx.state.inker, tab)
    tab.saving = True
    assert not op.enabled(ctx.state.inker, tab)
    assert inker_ops.reason_for(op, ctx.state.inker, tab) == inker_flourish.BUSY


def test_the_insert_op_opens_the_popup_rather_than_inserting(ctx):
    _open(ctx)
    assert inker_ops.run(ctx, inker_ops.get("flourish_insert"))
    assert ctx.state.inker.pending_dialog == inker_flourish.FLOURISH_POPUP


def test_inserting_through_the_verb_lands_a_group_and_scales_to_the_canvas(ctx):
    tab = _open(ctx, (40, 40))
    assert inker_mode.flourish_insert(ctx, tab, preset="smoke_puff", mode="painterly")
    doc = tab.doc
    assert len(doc.flourish) == 1
    (group, state), = doc.flourish.items()
    assert state.recipe.width == 40 and state.recipe.height == 40
    assert doc.groups[group].name == state.recipe.name
    # The fake runs the task inline, so the landing toast precedes the
    # "rendering..." one; the app sees them the other way round.
    assert any(level == "success" for _, level in ctx.toasts)
    # The inspector shows the group's first layer.
    assert ctx.state.inker.flourish_layer[group] == state.recipe.layers[0].uid


def test_an_unknown_preset_is_refused_with_a_toast(ctx):
    tab = _open(ctx)
    assert not inker_mode.flourish_insert(ctx, tab, preset="no_such_effect")
    assert ctx.toasts[-1][1] == "error"


def test_regenerate_is_greyed_until_the_active_layer_is_in_an_effect(ctx):
    tab = _open(ctx)
    op = inker_ops.get("flourish_regenerate")
    assert not op.enabled(ctx.state.inker, tab)
    assert inker_ops.reason_for(op, ctx.state.inker, tab) == inker_flourish.NO_EFFECT
    tab.doc.insert_flourish(B.bake(_small()))
    assert op.enabled(ctx.state.inker, tab)
    tab.doc.set_active_layer(0)
    assert not op.enabled(ctx.state.inker, tab)


def test_a_pending_edit_rests_then_renders_then_lands_as_one_step(ctx):
    tab = _open(ctx)
    state = ctx.state.inker
    rec = _small()
    group = tab.doc.insert_flourish(B.bake(rec))
    head = tab.doc.history.head
    edited = dataclasses.replace(rec, seed=99)
    inker_flourish.set_pending(state, group, edited, now=10.0)
    assert inker_flourish.current_recipe(state, tab, group) == edited
    # Too soon: nothing goes out.
    assert inker_flourish.tick(ctx, state, tab, now=10.1) == 0
    assert tab.doc.history.head == head
    # Rested: one render, landed synchronously, one step.
    assert inker_flourish.tick(ctx, state, tab, now=10.1 + inker_flourish.DEBOUNCE_SECONDS) == 1
    assert tab.doc.history.head == head + 1
    assert tab.doc.flourish_state(group).recipe == edited
    assert group not in state.flourish_pending
    assert ctx.toasts[-1][1] == "success"
    tab.doc.history.undo(tab.doc)
    assert tab.doc.flourish_state(group).recipe == rec


def test_an_edit_made_during_a_render_is_rendered_next(ctx):
    tab = _open(ctx)
    state = ctx.state.inker
    rec = _small()
    group = tab.doc.insert_flourish(B.bake(rec))
    ctx.auto_land = False
    t0 = inker_flourish.clock()
    first = dataclasses.replace(rec, seed=5)
    inker_flourish.set_pending(state, group, first, now=t0)
    assert inker_flourish.tick(ctx, state, tab, now=t0 + 1.0) == 1
    assert inker_flourish.in_flight(ctx, tab, group)
    second = dataclasses.replace(rec, seed=6)
    inker_flourish.set_pending(state, group, second, now=t0 + 1.0)
    # In flight: the tick waits rather than stacking a second render.
    assert inker_flourish.tick(ctx, state, tab, now=t0 + 2.0) == 0
    ctx.auto_land = True
    ctx.land_all()
    assert tab.doc.flourish_state(group).recipe == first
    # The newer edit is re-armed (at the real clock) and goes out on the next tick.
    assert state.flourish_pending[group] == second
    assert inker_flourish.tick(ctx, state, tab, now=inker_flourish.clock() + 1.0) == 1
    assert tab.doc.flourish_state(group).recipe == second


def test_the_regenerate_op_keeps_paint_and_the_force_op_replaces_it(ctx):
    tab = _open(ctx)
    rec = _small()
    group = tab.doc.insert_flourish(B.bake(rec))
    track_uid = next(iter(tab.doc.flourish_state(group).tracks.values()))
    frame = tab.doc.anim.frames[2]
    tab.doc.anim.cels[(track_uid, frame.uid)].pixels[1, 1] = (7, 7, 7, 255)
    inker_flourish.set_pending(ctx.state.inker, group, dataclasses.replace(rec, seed=3), now=0.0)
    assert inker_ops.run(ctx, inker_ops.get("flourish_regenerate"))
    assert tab.doc.flourish_conflicts(group) == [2]
    assert ctx.toasts[-1][1] == "warn"
    keep = inker_ops.get("flourish_keep_edits")
    assert keep.enabled(ctx.state.inker, tab)
    assert inker_ops.run(ctx, keep)
    assert tab.doc.flourish_conflicts(group) == []
    assert not keep.enabled(ctx.state.inker, tab)
    assert inker_ops.reason_for(keep, ctx.state.inker, tab) == inker_flourish.NO_CONFLICTS
    tab.doc.anim.cels[(track_uid, frame.uid)].pixels[1, 1] = (8, 8, 8, 255)
    inker_flourish.set_pending(ctx.state.inker, group, dataclasses.replace(rec, seed=4), now=0.0)
    assert inker_ops.run(ctx, inker_ops.get("flourish_regenerate_all"))
    assert tab.doc.flourish_conflicts(group) == []
    assert tuple(tab.doc.anim.cels[(track_uid, frame.uid)].pixels[1, 1]) != (8, 8, 8, 255)


def test_detach_through_the_op_drops_pending_state_too(ctx):
    tab = _open(ctx)
    rec = _small()
    group = tab.doc.insert_flourish(B.bake(rec))
    inker_flourish.set_pending(ctx.state.inker, group, dataclasses.replace(rec, seed=3), now=0.0)
    assert inker_ops.run(ctx, inker_ops.get("flourish_detach"))
    assert tab.doc.flourish_state(group) is None
    assert group not in ctx.state.inker.flourish_pending
    assert group not in ctx.state.inker.flourish_due
    assert not inker_ops.get("flourish_detach").enabled(ctx.state.inker, tab)


def test_a_render_for_a_closed_tab_or_detached_group_is_dropped(ctx):
    tab = _open(ctx)
    rec = _small()
    group = tab.doc.insert_flourish(B.bake(rec))
    ctx.auto_land = False
    assert inker_flourish.submit_render(ctx, tab, group, dataclasses.replace(rec, seed=2))
    head = tab.doc.history.head
    tab.doc.detach_flourish(group)
    ctx.auto_land = True
    ctx.land_all()
    assert tab.doc.history.head == head + 1  # the detach only
    assert ctx.toasts[-1][1] == "info"
    # And a tab that has closed: nothing lands, nothing is said.
    ctx.auto_land = False
    group2 = tab.doc.insert_flourish(B.bake(rec))
    assert inker_flourish.submit_render(ctx, tab, group2, rec)
    ctx.state.inker.docs.clear()
    before = len(ctx.toasts)
    ctx.auto_land = True
    ctx.land_all()
    assert len(ctx.toasts) == before


def test_a_failed_render_is_a_warning_not_a_crash(ctx):
    tab = _open(ctx)
    done = Done(key=inker_flourish.render_key(tab, 1), error=RuntimeError("boom"))
    assert not inker_flourish.land(ctx, ctx.state.inker, done, now=0.0)
    assert ctx.toasts[-1][1] == "warn"
