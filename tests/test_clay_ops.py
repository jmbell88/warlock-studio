"""The Clay ops registry: one list, three surfaces, no imgui.

The registry exists because there used to be three lists -- the tools pane, the
key handler and (now) the context menu -- and the interesting one was always the
one nobody updated. So the assertions here are mostly about *agreement*: that a
key fires the op the menu shows for it, that enablement is one predicate, and
that nothing here reaches for a GUI.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import clay_ops
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as bp


class _Toasts:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)


class _Ctx:
    def __init__(self) -> None:
        self.toasts = _Toasts()


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
    assert "extrude" not in edges


def test_a_key_resolves_to_the_op_the_menu_shows_for_it() -> None:
    op = clay_ops.by_key("face", "E")
    assert op is not None and op.name == "extrude"
    assert clay_ops.by_key("edge", "E") is None
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


def test_dissolve_dispatches_on_the_mode() -> None:
    doc, uid = _doc()
    doc.set_element_mode("vertex")
    doc.set_element_sel(uid, el.ElementSel(verts=[0]))
    assert clay_ops.run(_Ctx(), doc, clay_ops.get("dissolve")) is True
    assert bm.face_count(doc.by_uid(uid).mesh) == 4


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
    assert len(doc.history) == 2
