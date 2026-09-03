"""The two ends of the re-render loop, as the app reaches them.

Troupe submits the re-render; Inker finds the result and merges it. Neither
half is any use without the other, and neither had a caller until this landed.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import inker_mode, inker_ops, inker_sheet, troupe_mode
from warlock.studio.inker.sheetin import document_from_sheet


class _Ctx:
    """The narrow slice of the app context these helpers touch.

    ``test_troupe_mode``'s hand-rolled context, for its reason: none of this
    needs a GL context, and opening one to check a conflict flag would be
    untrue about what is being tested."""

    def __init__(self, svc):
        self.svc = svc
        self.cache = svc.store
        self.state = SimpleNamespace(
            troupe=None, preview={}, mode="troupe", inker=None,
            field_errors={}, clear_field_errors=lambda: None,
        )
        self.viewer = None
        self.toasts: list[tuple[str, str]] = []

    def job_dir(self, job_id):
        return self.svc.job_dir(job_id)

    def toast(self, text, level="info", *a, **k):
        self.toasts.append((text, level))

    def busy(self, key):
        return False


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)

# -- the Troupe end ------------------------------------------------------------


def test_the_runs_offered_come_from_the_sheet_rather_than_the_shipped_table(ctx, svc):
    """A sheet built with four directions has four runs per animation, and
    offering it eight would be offering runs it does not contain."""
    from warlock.studio import troupe_mode as mode

    state = mode.ensure(ctx)
    state.job_id, state.sheet_id = "", ""
    layout = mode.preview_layout(ctx)
    runs = mode.sheet_runs(ctx)

    assert len(runs) == len(layout.get("runs") or ())
    assert all({"animation", "direction"} == set(run) for run in runs)


def test_a_re_render_needs_a_selected_sheet_and_some_runs(ctx):
    state = troupe_mode.ensure(ctx)
    state.job_id, state.sheet_id = "", ""
    assert troupe_mode.rerender_runs(ctx, [{"animation": "walk", "direction": "front"}]) is False

    state.job_id, state.sheet_id = "abc", "def"
    assert troupe_mode.rerender_runs(ctx, []) is False


def test_the_pane_offers_the_control_and_names_what_it_costs():
    import inspect

    from warlock.studio.panes import troupe_sheets

    source = inspect.getsource(troupe_sheets)
    assert "rerender_runs" in source, "the pane must reach the controller"
    assert "cost_note" in source
    # A greyed control with no reason is the thing the census exists to catch.
    assert "reason=" in source


# -- finding the re-render -----------------------------------------------------


def _sheet_on_disk(svc, job_id, sheet_id, created, *, tagged=True):
    sheets = svc.job_dir(job_id) / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    (sheets / f"{sheet_id}.png").write_bytes(b"fake")
    record = {
        "id": sheet_id,
        "image": f"{sheet_id}.png",
        "created": created,
        "cells": [],
    }
    if tagged:
        record["animation"] = {"tags": [{"name": "walk_front", "start": 0, "end": 1}]}
    (sheets / f"{sheet_id}.json").write_text(json.dumps(record), "utf-8")


def test_the_newest_sheet_after_the_documents_own_is_the_one_to_merge(svc):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    now = time.time()
    _sheet_on_disk(svc, job_id, "aaaaaaaaaaaa", now)
    _sheet_on_disk(svc, job_id, "bbbbbbbbbbbb", now + 10)
    _sheet_on_disk(svc, job_id, "cccccccccccc", now + 20)

    assert inker_mode.newest_sheet_after(svc, job_id, "aaaaaaaaaaaa") == "cccccccccccc"


def test_an_older_sheet_is_never_offered_as_a_re_render(svc):
    """Merging one would run the comparison backwards -- the "incoming" render
    would be the older picture, and everything since would read as a conflict."""
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    now = time.time()
    _sheet_on_disk(svc, job_id, "aaaaaaaaaaaa", now)
    _sheet_on_disk(svc, job_id, "bbbbbbbbbbbb", now + 10)

    assert inker_mode.newest_sheet_after(svc, job_id, "bbbbbbbbbbbb") == ""


def test_a_plain_sheet_is_not_a_re_render_of_a_character(svc):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    now = time.time()
    _sheet_on_disk(svc, job_id, "aaaaaaaaaaaa", now)
    _sheet_on_disk(svc, job_id, "bbbbbbbbbbbb", now + 10, tagged=False)

    assert inker_mode.newest_sheet_after(svc, job_id, "aaaaaaaaaaaa") == ""


# -- the Inker end -------------------------------------------------------------

CELL = 8


def _doc(values=(10, 20, 30, 40)):
    atlas = np.zeros((CELL, len(values) * CELL, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    for i, value in enumerate(values):
        atlas[:, i * CELL : (i + 1) * CELL, 0] = value
    cells = [{"x": i * CELL, "y": 0, "w": CELL, "h": CELL} for i in range(len(values))]
    anim = {
        "tags": [{"name": "walk_front", "start": 0, "end": len(values) - 1, "loop": True}],
        "frames": [],
    }
    return document_from_sheet(atlas, cells, anim, source={"job": "J", "sheet": "S"})


class _Tab:
    def __init__(self, doc):
        self.doc = doc


def test_a_document_from_a_rendered_sheet_can_be_merged_into():
    tab = _Tab(_doc())
    assert inker_sheet.has_base(tab) is True


def test_a_document_with_no_recorded_render_says_why_it_cannot_merge():
    """Greyed with a reason, never silently disabled."""
    tab = _Tab(_doc())
    tab.doc.sheet_base = None

    class _State:
        pass

    reason = inker_sheet.merge_reason(_State(), tab)
    assert reason
    assert "not opened from a rendered sheet" in reason


def test_conflicts_come_back_as_frame_indices():
    doc = _doc()
    tab = _Tab(doc)
    doc.sheet_base.conflicts.add(doc.anim.frames[2].uid)
    assert inker_sheet.conflicts(tab) == [2]


def test_walking_the_conflicts_wraps():
    """A reader who starts halfway down should not have to scroll back up."""
    doc = _doc()
    tab = _Tab(doc)
    doc.sheet_base.conflicts.update({doc.anim.frames[1].uid, doc.anim.frames[3].uid})

    assert inker_sheet.next_conflict(tab, 0) == 1
    assert inker_sheet.next_conflict(tab, 1) == 3
    assert inker_sheet.next_conflict(tab, 3) == 1, "wraps to the first"


def test_no_conflicts_is_none_rather_than_a_wrong_cell():
    assert inker_sheet.next_conflict(_Tab(_doc()), 0) is None


def test_keeping_the_edit_clears_the_flag_and_is_undoable(ctx):
    doc = _doc()
    tab = _Tab(doc)
    flagged = doc.anim.frames[2].uid
    doc.sheet_base.conflicts.add(flagged)
    doc.history.clear()

    assert inker_sheet.resolve_keep(ctx, tab, [2]) is True
    assert doc.sheet_base.conflicts == set()

    doc.undo()
    assert doc.sheet_base.conflicts == {flagged}


def test_keeping_an_unflagged_cell_does_nothing(ctx):
    doc = _doc()
    assert inker_sheet.resolve_keep(ctx, _Tab(doc), [0]) is False


# -- the ops -------------------------------------------------------------------


def test_the_merge_ops_are_registered_under_the_sheet_menu():
    registry = inker_ops.registry() if callable(getattr(inker_ops, "registry", None)) else None
    names = {"sheet_merge", "sheet_conflict_next", "sheet_keep_edit"}
    source = __import__("inspect").getsource(inker_ops)
    for name in names:
        assert f'"{name}"' in source, name
    assert registry is None or names <= set(registry)


def test_every_merge_op_carries_a_reason_for_being_greyed():
    """The census rule: a greyed control with no reason is one the user cannot
    act on and cannot find out why."""
    import inspect

    source = inspect.getsource(inker_ops)
    block = source[source.index('"sheet_merge"') : source.index('"sheet_propagate"')]
    assert block.count("reason=") == 3
    assert block.count("enabled=") == 3
