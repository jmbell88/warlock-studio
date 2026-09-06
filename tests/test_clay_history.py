"""One press, one Ctrl+Z -- and a history panel that says what each step was.

Three claims, and each was false before 2026-09-01:

* An op is **one** undo step however many objects it touched.
  ``clay_ops.run_mesh_op`` pushes a ``set_mesh`` per object, so extruding faces
  across three objects cost three presses to undo a gesture made once.
* That step is **named after the op**. A fold reads as "compound" and an
  unfolded ``MeshEdit`` reads as "mesh", neither of which tells a reader what
  they are about to reverse.
* A jump through the stack **drops the element selection it invalidates**, the
  way a plain undo already did. ``UndoStack.step_to`` walks the stack's own
  undo and redo, which do not run ``ClayDoc._forget_elements``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import clay_ops, undo
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import primitives as bp
from warlock.studio.clay_state import ClayState


class _Ctx:
    """A ctx that carries Clay state, which is what ``run`` records against."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(clay=ClayState())
        self.errors: list[str] = []

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.errors.append(message)


def _doc(count: int = 1) -> tuple[bd.ClayDoc, list[int]]:
    doc = bd.ClayDoc()
    uids = [
        doc.add_object(
            bd.Obj(uid=bd.new_uid(), name=f"Box{index}", mesh=bp.box())
        ).uid
        for index in range(count)
    ]
    doc.select(uids)
    return doc, uids


def _faces(doc: bd.ClayDoc, uids: list[int]) -> None:
    doc.element_mode = "face"
    for uid in uids:
        doc.set_element_sel(uid, el.ElementSel(faces=[0]))


# --- one press, one step -----------------------------------------------------


def test_an_op_across_three_objects_is_a_single_undo_step():
    doc, uids = _doc(3)
    _faces(doc, uids)
    depth = len(doc.history)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))

    assert len(doc.history) == depth + 1, "three set_mesh pushes, one gesture"


def test_one_ctrl_z_puts_all_three_objects_back():
    """The failure the fold exists to prevent is worse than an extra press: two
    presses in, the document showed one object extruded and two not -- a state
    the user never made."""
    doc, uids = _doc(3)
    _faces(doc, uids)
    before = [len(doc.by_uid(uid).mesh.positions) for uid in uids]

    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    assert [len(doc.by_uid(uid).mesh.positions) for uid in uids] != before

    doc.undo()
    assert [len(doc.by_uid(uid).mesh.positions) for uid in uids] == before


def test_a_single_object_op_is_not_wrapped_in_a_compound():
    """``collapse_since`` refuses a run of one, and that is deliberate: a lone
    ``CompoundEdit`` around one edit would read as "compound" where the edit
    reads as what it did."""
    doc, uids = _doc(1)
    _faces(doc, uids)
    depth = len(doc.history)

    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))

    assert len(doc.history) == depth + 1
    assert not isinstance(doc.history.top, undo.CompoundEdit)


def test_a_single_object_op_is_named_too():
    """The one the first cut of ``_one_step`` got wrong. Reading the head at
    the top of that function reads it *after* the op has run, so the guard only
    fired when the collapse itself had pushed -- which is to say a multi-object
    op got its name and a single-object one silently kept "mesh"."""
    doc, uids = _doc(1)
    _faces(doc, uids)

    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))

    assert doc.history.history()[-1][0] == "Extrude"


def _drag(doc: bd.ClayDoc, uids: list[int], *, tool: str = "move") -> None:
    """Move every selected object, then commit the drag the way a release does.

    A stub rather than a ``ClayView``: ``_commit_drag`` is a ``DragOps`` method
    and reads exactly these four attributes, so the real GL view (and its
    context) is not part of the claim under test.
    """
    from warlock.studio._view_drag import DragOps

    start = {
        uid: tuple(np.array(v, copy=True) for v in doc.by_uid(uid).trs())
        for uid in uids
    }
    for uid in uids:
        doc.by_uid(uid).translation = np.array([1.0, 0.0, 0.0])
    view = SimpleNamespace(
        _drag_start=start,
        _drag_uids=list(uids),
        _key_kind="",
        state=SimpleNamespace(tool=tool),
    )
    DragOps._commit_drag(view, doc)


def test_a_gizmo_drag_across_three_objects_is_a_single_undo_step():
    """The reversal of "one history step per object". A later item drops a
    whole multi-object figure into Clay in one click, so scaling a placed
    humanoid cost sixteen Ctrl+Z -- and the states in between showed some limbs
    moved and some not, which is the failure the fold exists to prevent."""
    doc, uids = _doc(3)
    depth = len(doc.history)

    _drag(doc, uids)

    assert len(doc.history) == depth + 1, "three set_transform pushes, one drag"


def test_one_ctrl_z_puts_all_three_dragged_objects_back():
    doc, uids = _doc(3)
    before = [np.array(doc.by_uid(uid).translation, copy=True) for uid in uids]

    _drag(doc, uids)
    assert [list(doc.by_uid(uid).translation) for uid in uids] != [
        list(v) for v in before
    ]

    doc.undo()

    for uid, was in zip(uids, before, strict=True):
        assert list(doc.by_uid(uid).translation) == list(was)


def test_a_drag_step_is_labelled_with_the_tool_that_made_it():
    doc, uids = _doc(3)

    _drag(doc, uids, tool="scale")

    assert doc.history.history()[-1][0] == "Scale"


def test_a_drag_that_moved_nothing_pushes_no_step_and_relabels_none():
    """``collapse_since`` on an empty run must not leave a compound behind, and
    the label must not land on whatever step happens to be underneath."""
    doc, uids = _doc(3)
    before = doc.history.history()

    from warlock.studio._view_drag import DragOps

    view = SimpleNamespace(
        _drag_start={
            uid: tuple(np.array(v, copy=True) for v in doc.by_uid(uid).trs())
            for uid in uids
        },
        _drag_uids=list(uids),
        _key_kind="",
        state=SimpleNamespace(tool="move"),
    )
    DragOps._commit_drag(view, doc)

    assert doc.history.history() == before


# --- the step says what it was ----------------------------------------------


def test_the_step_is_labelled_with_the_op_that_made_it():
    doc, uids = _doc(2)
    _faces(doc, uids)

    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))

    label, is_done = doc.history.history()[-1]
    assert is_done and label == "Extrude"


def test_a_parameterised_op_drops_the_ellipsis_from_its_label():
    """"Inset Faces..." is an invitation to a dialog; a history row is a record
    of something that already happened."""
    doc, uids = _doc(1)
    _faces(doc, uids)

    op = clay_ops.get("inset")
    assert op.label.endswith("...")
    assert clay_ops.run(_Ctx(), doc, op)

    assert doc.history.history()[-1][0] == "Inset Faces"


def test_an_op_that_changes_no_document_labels_nothing():
    """Select All pushes no step. Labelling the *previous* one with this op's
    name would be a lie in the one place a user goes to read what happened."""
    doc, uids = _doc(1)
    _faces(doc, uids)
    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    before = doc.history.history()

    clay_ops.run(_Ctx(), doc, clay_ops.get("select-all"))

    assert doc.history.history() == before


def test_an_edit_with_no_label_still_reads_as_its_class_name():
    """The fallback is what keeps the twenty existing edits from each needing a
    string, and it must survive the override being added."""
    doc, _uids = _doc(1)

    assert doc.history.history()[-1][0] == "object add"


# --- what was run last -------------------------------------------------------


def test_running_an_op_records_what_it_ran_against():
    doc, uids = _doc(2)
    _faces(doc, uids)
    depth = len(doc.history)
    ctx = _Ctx()

    clay_ops.run(ctx, doc, clay_ops.get("inset"), thickness=0.25)

    last = ctx.state.clay.last_op
    assert last is not None
    assert last.name == "inset"
    # The clamped, defaulted values rather than the ones passed: what the card
    # must re-run with is what actually ran, and ``run`` fills and clamps.
    assert last.params["thickness"] == pytest.approx(0.25)
    assert "depth" in last.params, "an unpassed parameter is recorded at its default"
    assert last.depth_before == depth
    assert last.head_after == doc.history.head, "the guard the card re-runs behind"


def test_the_recorded_selection_is_the_one_the_op_ran_against():
    """Extrude leaves the *new* faces selected, so re-running from the current
    selection would extrude the extrusion. The card has to put back what the op
    started from."""
    doc, uids = _doc(1)
    _faces(doc, uids)
    started_from = doc.element_sel_of(uids[0])
    ctx = _Ctx()

    clay_ops.run(ctx, doc, clay_ops.get("extrude"))

    recorded = ctx.state.clay.last_op.element_sel[uids[0]]
    assert recorded.same_as(started_from)
    assert ctx.state.clay.last_op.element_mode == "face"


def test_a_refused_op_records_nothing():
    """A card offering to adjust an op that never ran would re-run it on the
    press of a slider -- which is a refusal turned into an action."""
    doc, uids = _doc(1)
    doc.element_mode = "face"
    doc.set_element_sel(uids[0], el.ElementSel())
    ctx = _Ctx()

    assert not clay_ops.run(ctx, doc, clay_ops.get("extrude"))
    assert ctx.state.clay.last_op is None


def test_a_ctx_with_no_clay_state_records_nothing_and_does_not_raise():
    """``run`` is the choke point every surface funnels through, headless calls
    included. It reaches for the state rather than requiring one."""
    doc, uids = _doc(1)
    _faces(doc, uids)

    bare = SimpleNamespace(state=None, toast=lambda *a: None)
    assert clay_ops.run(bare, doc, clay_ops.get("extrude"))


# --- jumping through the stack -----------------------------------------------


def test_step_history_walks_to_a_position_and_back():
    doc, uids = _doc(1)
    _faces(doc, uids)
    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    top = len(doc.history)

    assert doc.step_history(0)
    assert len(doc.history) == 0
    assert doc.step_history(top)
    assert len(doc.history) == top


def test_step_history_clamps_rather_than_raising():
    doc, _uids = _doc(1)
    depth = len(doc.history)

    assert not doc.step_history(depth), "already there"
    doc.step_history(99)
    assert len(doc.history) == depth
    doc.step_history(-5)
    assert len(doc.history) == 0


def test_a_jump_drops_an_element_selection_it_invalidated():
    """``UndoStack.step_to`` would not: it walks the stack's own undo and redo,
    which do not run ``_forget_elements``. Jumping past the step that made a
    mesh would leave the selection naming faces of a mesh that is gone."""
    doc, uids = _doc(1)
    _faces(doc, uids)
    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    assert doc.element_sel_of(uids[0]) is not None

    doc.step_history(0)

    assert not doc.element_sel


def test_the_mode_door_forgets_the_last_op_when_it_moves():
    """A jump can undo the very op the adjust card is offering to re-run."""
    from warlock.studio import clay_mode

    doc, uids = _doc(1)
    _faces(doc, uids)
    ctx = _Ctx()
    tab = SimpleNamespace(doc=doc)
    clay_ops.run(ctx, doc, clay_ops.get("extrude"))
    assert ctx.state.clay.last_op is not None

    assert clay_mode.step_history(ctx, tab, 0)
    assert ctx.state.clay.last_op is None


def test_a_jump_that_moves_nothing_leaves_the_last_op_alone():
    from warlock.studio import clay_mode

    doc, uids = _doc(1)
    _faces(doc, uids)
    ctx = _Ctx()
    tab = SimpleNamespace(doc=doc)
    clay_ops.run(ctx, doc, clay_ops.get("extrude"))

    assert not clay_mode.step_history(ctx, tab, len(doc.history))
    assert ctx.state.clay.last_op is not None
