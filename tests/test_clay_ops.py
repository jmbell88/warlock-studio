"""The Clay ops registry: one list, three surfaces, no imgui.

The registry exists because there used to be three lists -- the tools pane, the
key handler and (now) the context menu -- and the interesting one was always the
one nobody updated. So the assertions here are mostly about *agreement*: that a
key fires the op the menu shows for it, that enablement is one predicate, and
that nothing here reaches for a GUI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.studio import clay_ops
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_topo
from warlock.studio.clay import primitives as bp


class _Toasts:
    """Only what ``Ctx`` really offers. The double used to carry an ``error``
    method under a ``ctx.toasts`` attribute the app has never had, which is
    exactly how every refusal passed here and vanished in the real app."""

    def __init__(self) -> None:
        self.errors: list[str] = []


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()

    def toast(self, message: str, level: str = "info") -> None:
        if level == "error":
            self.toasts.errors.append(message)


def _doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bp.box(),
            generator="box",
            params={"size": (1.0, 1.0, 1.0)},
        )
    )
    return doc, obj.uid


def _faces(doc: bd.ClayDoc, uid: int, *faces: int) -> None:
    doc.set_element_mode("face")
    doc.set_element_sel(uid, el.ElementSel(faces=list(faces)))


# --- the registry itself ----------------------------------------------------


def test_every_op_has_a_unique_name_and_at_least_one_mode() -> None:
    names = [op.name for op in clay_ops.OPS]
    assert len(names) == len(set(names))
    for op in clay_ops.OPS:
        assert op.modes
        assert set(op.modes) <= set(clay_ops.ALL_MODES)


def test_registering_a_duplicate_name_is_refused() -> None:
    existing = clay_ops.OPS[0]
    with pytest.raises(ValueError, match="already registered"):
        clay_ops.register(
            clay_ops.Op(name=existing.name, label="x", modes=("object",), run=lambda *_: None)
        )


def test_the_menu_is_filtered_by_mode_and_keeps_registration_order() -> None:
    face = [op.name for op in clay_ops.menu("face")]
    assert "extrude" in face
    assert "bevel" not in face, "bevel is an edge op"
    assert face == [op.name for op in clay_ops.OPS if "face" in op.modes]

    edges = [op.name for op in clay_ops.menu("edge")]
    assert "bevel" in edges and "loop-cut" in edges
    # Extrude is now one row across all three modes, dispatched on the mode the
    # way Dissolve is: it means the same thing everywhere and only the
    # implementation differs, so three rows would be exposing that.
    assert "extrude" in edges
    assert "bridge" in edges
    assert "bridge" not in face, "bridge joins two boundary loops, which is an edge selection"


def test_a_key_resolves_to_the_op_the_menu_shows_for_it() -> None:
    op = clay_ops.by_key("face", "E")
    assert op is not None and op.name == "extrude"
    # The same op, and therefore the same key, in every element mode.
    assert clay_ops.by_key("edge", "E") is op
    assert clay_ops.by_key("vertex", "E") is op
    assert clay_ops.by_key("face", "nope") is None


def test_the_registry_imports_no_gui() -> None:
    """``clay_ops`` is testable precisely because it draws nothing.

    The layering rule for the whole package: nothing under ``clay/`` and nothing
    in the registry knows imgui exists; only panes and ``main`` draw.
    """
    import ast
    import importlib
    from pathlib import Path

    source = importlib.import_module("warlock.studio.clay_ops").__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "imgui_bundle" not in imported
    assert "imgui" not in imported


# --- enablement -------------------------------------------------------------


def test_an_element_op_is_disabled_with_nothing_selected() -> None:
    doc, uid = _doc()
    doc.set_element_mode("face")
    extrude = clay_ops.get("extrude")
    assert not extrude.enabled(doc)
    _faces(doc, uid, 0)
    assert extrude.enabled(doc)


def test_an_op_that_is_disabled_does_not_run() -> None:
    doc, _ = _doc()
    doc.set_element_mode("face")
    ctx = _Ctx()
    assert clay_ops.run(ctx, doc, clay_ops.get("extrude")) is False
    assert len(doc.history) == 1, "the add, and nothing else"


def test_object_ops_need_an_object_selection() -> None:
    doc, uid = _doc()
    assert not clay_ops.get("duplicate").enabled(doc)
    doc.select([uid])
    assert clay_ops.get("duplicate").enabled(doc)


# --- running ----------------------------------------------------------------


def test_running_extrude_edits_the_mesh_and_selects_the_caps() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("extrude")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 10
    assert doc.element_sel_of(uid).faces.tolist() == [0]


def test_a_topology_op_freezes_the_generator_exactly_once() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert doc.by_uid(uid).generator == "box"
    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    assert doc.by_uid(uid).generator is None
    assert doc.by_uid(uid).params == {}

    depth = len(doc.history)
    _faces(doc, uid, 1)
    clay_ops.run(_Ctx(), doc, clay_ops.get("extrude"))
    assert len(doc.history) == depth + 1, "already frozen: one mesh step, no props step"


@pytest.mark.parametrize("key", ["bake", "mirror-x", "mirror-y", "mirror-z"])
def test_every_op_that_changes_geometry_freezes_it(key: str) -> None:
    """The regression: the freeze lived in ``run_mesh_op`` and Smooth, so Bake
    Transform and Mirror went straight to ``set_mesh`` and left the object
    still claiming to be "box, size 1". The properties panel keeps offering
    that size field, and touching it rebuilds a pristine box -- so the bake or
    the mirror vanished with no warning.
    """
    doc, uid = _doc()
    doc.select([uid])
    doc.set_transform(uid, translation=[1.0, 2.0, 3.0])

    clay_ops.run(_Ctx(), doc, clay_ops.get(key))

    assert doc.by_uid(uid).generator is None
    assert doc.by_uid(uid).params == {}


def test_deleting_elements_freezes_the_generator() -> None:
    """The same hole reached from ``clay_mode._delete`` rather than an op: it
    calls ``doc.set_mesh`` directly, and nothing there used to freeze."""
    doc, uid = _doc()
    _faces(doc, uid, 0)
    before = bm.face_count(doc.by_uid(uid).mesh)

    mesh, sel = ops_topo.delete_faces(doc.by_uid(uid).mesh, doc.element_sel_of(uid))
    doc.set_mesh(uid, mesh, select=sel)

    assert bm.face_count(doc.by_uid(uid).mesh) < before
    assert doc.by_uid(uid).generator is None


def test_one_undo_takes_back_the_edit_and_its_freeze_together() -> None:
    """The freeze was a second history step, so a single Ctrl+Z restored the
    generator claim over the still-edited mesh -- the exact state the freeze
    exists to prevent -- and only a second press undid the edit it belongs to.
    Touching the size field in between rebuilt a pristine box over the edit."""
    doc, uid = _doc()
    _faces(doc, uid, 0)
    mesh, sel = ops_topo.delete_faces(doc.by_uid(uid).mesh, doc.element_sel_of(uid))
    original = doc.by_uid(uid).mesh
    doc.set_mesh(uid, mesh, select=sel)

    assert doc.undo() is True

    obj = doc.by_uid(uid)
    assert obj.mesh is original
    assert obj.generator == "box"
    assert obj.params == {"size": (1.0, 1.0, 1.0)}


def test_a_rebuild_from_the_generator_is_the_one_thing_that_keeps_it() -> None:
    """Otherwise editing a box's size would freeze it on the first keystroke
    and the field would disappear under the user's hands."""
    doc, uid = _doc()

    doc.set_mesh(uid, bp.box(size=(2.0, 2.0, 2.0)), keep_generator=True)

    assert doc.by_uid(uid).generator == "box"


def test_a_refusal_becomes_a_toast_and_records_no_edit() -> None:
    doc, uid = _doc()
    doc.set_element_mode("edge")
    a = doc.by_uid(uid).mesh
    doc.set_element_sel(uid, el.ElementSel(edges=[[int(a.loops[0]), int(a.loops[1])]]))
    ctx = _Ctx()
    depth = len(doc.history)

    assert clay_ops.run(ctx, doc, clay_ops.get("fill-hole")) is False
    assert ctx.toasts.errors and "boundary" in ctx.toasts.errors[0]
    assert len(doc.history) == depth


def test_a_refusal_on_one_object_does_not_abandon_the_others() -> None:
    doc, first = _doc()
    second = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box())).uid
    doc.set_element_mode("face")
    doc.set_element_sel(first, el.ElementSel(faces=[0]))
    doc.set_element_sel(second, el.ElementSel(faces=[0]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("extrude")) is True
    assert bm.face_count(doc.by_uid(first).mesh) == 10
    assert bm.face_count(doc.by_uid(second).mesh) == 10


def test_parameters_fall_back_to_their_declared_defaults() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    inset = clay_ops.get("inset")
    assert clay_ops.defaults_for(inset) == {"thickness": 0.1, "depth": 0.0}
    assert clay_ops.run(_Ctx(), doc, inset) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 10


def test_a_parameter_passed_in_overrides_the_default() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    clay_ops.run(_Ctx(), doc, clay_ops.get("inset"), thickness=99.0)
    inner = doc.by_uid(uid).mesh
    corners = inner.positions[inner.loops[inner.starts[0] : inner.starts[1]]]
    assert np.allclose(corners, corners[0]), "collapsed onto the centroid"


def test_a_parameter_out_of_range_is_clamped_to_what_it_declared() -> None:
    """``run`` is the choke point, so the declared range holds on every surface.

    The popup clamps its own live fields, but the key path, the tools pane and
    a remembered value from a previous session all arrive here instead -- and a
    subdivision at 99 levels is not something each op should have to refuse for
    itself. An integer parameter also comes out an ``int``, so an op can index
    with it.
    """
    seen: dict[str, Any] = {}

    def record(ctx: Any, doc: Any, **params: Any) -> None:
        del ctx, doc
        seen.update(params)

    # Unregistered: the registry is global, and a probe op that joined it would
    # be there for every later test in the process.
    probe = clay_ops.Op(
        name="probe-clamp",
        label="Probe",
        modes=("object",),
        run=record,
        params=(
            clay_ops.Param("t", "position", 0.5, 0.05, low=0.0, high=1.0),
            clay_ops.Param("levels", "levels", 1.0, 1.0, low=1.0, high=4.0, integer=True),
        ),
    )

    assert clay_ops.run(_Ctx(), _doc()[0], probe, t=5.0, levels=99.0) is True
    assert seen == {"t": 1.0, "levels": 4}
    assert isinstance(seen["levels"], int)


def test_a_parameter_below_its_floor_is_clamped_up() -> None:
    seen: dict[str, Any] = {}

    def record(ctx: Any, doc: Any, **params: Any) -> None:
        del ctx, doc
        seen.update(params)

    probe = clay_ops.Op(
        name="probe-floor",
        label="Probe",
        modes=("object",),
        run=record,
        params=(clay_ops.Param("t", "position", 0.5, 0.05, low=0.25, high=1.0),),
    )

    clay_ops.run(_Ctx(), _doc()[0], probe, t=-3.0)
    assert seen == {"t": 0.25}


def test_dissolve_dispatches_on_the_mode() -> None:
    doc, uid = _doc()
    doc.set_element_mode("vertex")
    doc.set_element_sel(uid, el.ElementSel(verts=[0]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("dissolve")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 4


def test_merge_faces_is_dissolve_under_the_name_a_user_looks_for() -> None:
    """Two adjacent faces of a box become one, leaving five."""
    doc, uid = _doc()
    _faces(doc, uid, 0, 2)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("merge_faces")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 5


def test_merge_faces_offers_itself_only_in_face_mode() -> None:
    """Vertex and edge selections have their own dissolve; a Merge Faces row
    over an edge selection would be a third name for the same dispatch."""
    assert "merge_faces" in [op.name for op in clay_ops.menu("face")]
    assert "merge_faces" not in [op.name for op in clay_ops.menu("edge")]
    assert "merge_faces" not in [op.name for op in clay_ops.menu("vertex")]
    assert "merge_faces" not in [op.name for op in clay_ops.menu("object")]


def test_merge_faces_needs_a_face_selection() -> None:
    doc, uid = _doc()
    doc.set_element_mode("face")
    merge = clay_ops.get("merge_faces")
    assert not merge.enabled(doc)
    _faces(doc, uid, 0, 2)
    assert merge.enabled(doc)


def test_merge_faces_and_dissolve_agree_in_face_mode() -> None:
    """The whole licence for the second name: it must stay the same operation.

    Run on two documents that started identical, the two ops have to produce
    the same face count -- if they ever diverge, one of them has become a
    second implementation and the manual's claim that they are one thing is a
    lie the user finds out about by undoing the wrong result.
    """
    counts = []
    for name in ("merge_faces", "dissolve"):
        doc, uid = _doc()
        _faces(doc, uid, 0, 2)
        assert clay_ops.run(_Ctx(), doc, clay_ops.get(name)) is True
        counts.append(bm.face_count(doc.by_uid(uid).mesh))
    assert counts[0] == counts[1]


def test_merge_faces_refuses_a_selection_it_cannot_represent() -> None:
    """Opposite faces of a box touch nowhere, so there is no single n-gon to
    make. The refusal is a toast and the mesh is left alone."""
    doc, uid = _doc()
    before = bm.face_count(doc.by_uid(uid).mesh)
    _faces(doc, uid, 0, 1)
    ctx = _Ctx()
    clay_ops.run(ctx, doc, clay_ops.get("merge_faces"))
    assert bm.face_count(doc.by_uid(uid).mesh) == before


def test_smooth_ignores_the_element_selection_and_takes_whole_objects() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("smooth"), levels=1) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 24


def test_delete_in_face_mode_removes_faces_rather_than_the_object() -> None:
    doc, uid = _doc()
    _faces(doc, uid, 0)
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("delete")) is True
    assert len(doc.objects) == 1, "the object survives"
    assert bm.face_count(doc.by_uid(uid).mesh) == 5


def test_delete_in_object_mode_removes_the_object() -> None:
    doc, uid = _doc()
    doc.select([uid])
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("delete")) is True
    assert doc.objects == []


def test_mirror_bakes_into_the_mesh_and_leaves_the_scale_positive() -> None:
    doc, uid = _doc()
    doc.select([uid])
    clay_ops.run(_Ctx(), doc, clay_ops.get("mirror-x"))
    assert np.allclose(doc.by_uid(uid).scale, [1.0, 1.0, 1.0])
    # Two steps: the transform, and the mesh change with its freeze compounded
    # into it. The freeze used to be a third, so one Ctrl+Z put the generator
    # claim back over the mirrored mesh.
    assert len(doc.history) == 2
    assert doc.by_uid(uid).generator is None


# --- merge ------------------------------------------------------------------


def _two_boxes(apart: float = 5.0) -> tuple[bd.ClayDoc, int, int]:
    doc, first = _doc()
    second = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="Box.001", mesh=bp.box(), translation=[apart, 0.0, 0.0])
    )
    doc.select([first, second.uid])
    return doc, first, second.uid


def test_merge_is_disabled_below_two_objects() -> None:
    """Merging one object is the identity, and an enabled button that does
    nothing is worse than a greyed one."""
    doc, uid = _doc()
    op = clay_ops.get("join")
    assert not op.enabled(doc)
    doc.select([uid])
    assert not op.enabled(doc)
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=bp.box()))
    doc.select([o.uid for o in doc.objects])
    assert op.enabled(doc)


def test_merge_keeps_the_topmost_object_and_absorbs_the_rest() -> None:
    doc, first, second = _two_boxes()
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("join"))
    assert [o.uid for o in doc.objects] == [first]
    assert doc.by_uid(first).name == "Box"
    assert doc.selection == {first}
    assert second not in {o.uid for o in doc.objects}


def test_merge_takes_the_geometry_of_everything_it_absorbed() -> None:
    doc, first, _ = _two_boxes()
    faces = bm.face_count(doc.by_uid(first).mesh)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"))
    merged = doc.by_uid(first).mesh
    bm.validate(merged)
    assert bm.face_count(merged) == faces * 2
    lo, hi = bm.bounds(merged)
    assert np.allclose(lo, [-0.5, -0.5, -0.5]) and np.allclose(hi, [5.5, 0.5, 0.5])


def test_merge_freezes_the_generator_and_undoes_in_one_press() -> None:
    doc, first, _ = _two_boxes()
    depth = len(doc.history)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"))
    assert doc.by_uid(first).generator is None
    assert len(doc.history) == depth + 1

    assert doc.undo()
    assert [o.name for o in doc.objects] == ["Box", "Box.001"]
    assert doc.by_uid(first).generator == "box"


def test_merge_welds_at_the_distance_it_is_given() -> None:
    """The parameter is what turns two shells into one surface; at zero it is
    a group, which is a legitimate thing to ask for."""
    doc, first, _ = _two_boxes(apart=0.0)
    verts = len(doc.by_uid(first).mesh.positions)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"), weld=0.0)
    assert len(doc.by_uid(first).mesh.positions) == 2 * verts

    doc, first, _ = _two_boxes(apart=0.0)
    clay_ops.run(_Ctx(), doc, clay_ops.get("join"), weld=1e-4)
    assert len(doc.by_uid(first).mesh.positions) == verts


def test_merge_refuses_a_zero_scaled_target_as_a_toast() -> None:
    doc, first, _ = _two_boxes()
    doc.set_transform(first, scale=[0.0, 1.0, 1.0])
    depth = len(doc.history)
    ctx = _Ctx()
    assert not clay_ops.run(ctx, doc, clay_ops.get("join"))
    assert ctx.toasts.errors
    assert len(doc.objects) == 2
    assert len(doc.history) == depth


# --- parameter formatting ---------------------------------------------------


def test_a_sub_millimetre_parameter_is_drawn_with_enough_decimals():
    """imgui's default "%.3f" printed both weld distances as 0.000 -- a field
    whose value cannot be read, whose step arrows appear to do nothing, and
    which hands back a different number than the one it was showing."""
    for name in ("weld", "join"):
        param = next(p for p in clay_ops.get(name).params if "distance" in p.label)
        assert param.default < 1e-3
        shown = clay_ops.format_for(param) % param.default
        assert float(shown) == pytest.approx(param.default), shown


def test_every_parameter_can_be_read_back_from_what_it_is_drawn_as():
    """The property, for the whole registry rather than for the one that was
    wrong: what the field shows must round-trip to what the op will be given."""
    for op in clay_ops.OPS:
        for param in op.params:
            if param.integer:
                continue
            shown = clay_ops.format_for(param) % param.default
            assert float(shown) == pytest.approx(param.default), f"{op.name}.{param.name}={shown}"


def test_an_ordinary_parameter_still_reads_at_three_decimals():
    """Widened only downwards: a parameter that was legible before must not
    grow a tail of zeros because of the parameter that was not."""
    assert clay_ops.format_for(clay_ops.Param("w", "width (m)", 0.05, 0.01)) == "%.3f"
    assert clay_ops.format_for(clay_ops.Param("t", "position", 0.5, 0.05)) == "%.3f"


def _one_hidden() -> tuple[bd.ClayDoc, int, int]:
    """Two selected objects, the second hidden after the fact."""
    doc, first, second = _two_boxes()
    doc.set_props(second, visible=False)
    return doc, first, second


def test_merge_is_disabled_when_only_one_selected_object_is_visible():
    """Greyed rather than refused: the op would skip the hidden one, so a row
    enabled by an object the merge ignores is the enabled-button-that-does-
    nothing problem again."""
    doc, _first, _second = _one_hidden()
    assert not clay_ops.get("join").enabled(doc)


def test_merge_never_absorbs_a_hidden_object():
    """``_select_all`` no longer hands one over; this is the other half, an
    object hidden after it was selected. Forced past ``enabled`` because that is
    exactly what a stale predicate would do."""
    doc, first, second = _one_hidden()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="C", mesh=bp.box(), translation=[0.0, 5.0, 0.0]))
    doc.select([o.uid for o in doc.objects])
    faces = bm.face_count(doc.by_uid(first).mesh)

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("join"))

    assert second in {o.uid for o in doc.objects}, "the hidden object survives untouched"
    assert bm.face_count(doc.by_uid(first).mesh) == faces * 2, "A and C, not A, B and C"


# --- Union's binding ----------------------------------------------------------
#
# Union Objects did the job the manual describes from the day it landed and
# almost nobody found it: it sat in the context menu with no key while the weld
# beside it held Ctrl+J, so the discoverable half of the pair was the half that
# leaves the interior walls in. The binding is the fix, and these pin it.


def test_union_is_bound_beside_merge_rather_than_buried() -> None:
    """Ctrl+Shift+J to Ctrl+J: the same key, shifted, for the same question
    answered the other way. A user who knows one is one keystroke from the
    other, which is the whole of what was missing."""
    assert clay_ops.get("join").key == "Ctrl+J"
    assert clay_ops.get("union").key == "Ctrl+Shift+J"


def test_no_two_ops_in_one_mode_claim_the_same_key() -> None:
    """The registry's own promise, asserted rather than assumed -- adding a
    binding is exactly the change that can break it, and ``by_key`` returns the
    first match, so a collision fails silently in favour of registration order."""
    for mode in clay_ops.ALL_MODES:
        keys = [op.key for op in clay_ops.menu(mode) if op.key]
        duplicates = {key for key in keys if keys.count(key) > 1}
        assert not duplicates, f"{mode}: {sorted(duplicates)}"


def test_union_and_merge_are_enabled_together() -> None:
    """One predicate, so the pair is never half-offered: a selection that can
    be welded can be unioned, and the choice between them is the user's."""
    doc, _first, _second = _two_boxes()
    assert clay_ops.get("join").enabled(doc)
    assert clay_ops.get("union").enabled(doc)


def test_merge_points_at_union_from_inside_its_own_dialog() -> None:
    """The pair is only a choice if you know both halves exist.

    The dialog is where the pointer belongs rather than the menu: it is the one
    moment the user has committed to "make these one object" and can still pick
    which meaning of that they wanted, and it is the moment the weld's cost --
    the walls it is about to bury -- is still undone.
    """
    hint = clay_ops.get("join").hint
    assert "Union" in hint
    assert clay_ops.get("union").key in hint


def test_every_op_hint_names_a_binding_that_exists() -> None:
    """A hint citing a key is a second place the binding is written down, and
    the failure mode of those is that they go stale in silence."""
    bindings = {op.key for op in clay_ops.OPS if op.key}
    for op in clay_ops.OPS:
        for token in op.hint.replace(",", " ").replace("(", " ").replace(")", " ").split():
            if token.startswith("Ctrl+") or token.startswith("Shift+"):
                assert token in bindings, f"{op.name}: {token} is not bound to anything"
