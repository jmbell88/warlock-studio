"""Union Objects: the one op in Clay that removes geometry it did not select.

The properties asserted here are the ones that separate a union from the weld
beside it. A weld of two overlapping cubes keeps both sets of interior walls, so
its edges are used by four faces and its volume is the sum of the parts. A union
keeps neither: every edge is used by exactly two faces, and the volume is less
than the sum, by exactly the overlap. Those two facts are the whole difference
and are why both ops exist.

The stated costs are pinned too -- no UVs, triangles only -- because they are
what the manual promises and what makes *Merge Objects* worth keeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import clay_ops
from warlock.studio.clay import document as bd
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_boolean
from warlock.studio.clay import primitives as bp
from warlock.studio.clay.elements import OpError

from .topo_asserts import assert_closed

pytest.importorskip("manifold3d")


def _obj(name: str, *, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _volume(mesh: bm.Mesh) -> float:
    """The enclosed volume, by the divergence theorem over triangles."""
    tris, _face = bm.triangulate(mesh)
    p = mesh.positions[tris].astype("f8")
    return float(np.abs(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6.0)


# --- the geometry -----------------------------------------------------------


def test_two_overlapping_cubes_come_out_one_closed_solid():
    a = _obj("A")
    b = _obj("B", translation=[0.5, 0.5, 0.5])

    merged = ops_boolean.union([a, b])

    assert_closed(merged)


def test_the_overlap_is_removed_rather_than_kept():
    """The whole difference from Merge Objects: a weld would report 2.0."""
    a = _obj("A")
    b = _obj("B", translation=[0.5, 0.5, 0.5])

    volume = _volume(ops_boolean.union([a, b]))

    assert volume < 2.0
    assert volume == pytest.approx(1.875, abs=1e-3)


def test_disjoint_objects_still_union_into_two_shells():
    """Not an error and not special-cased: the union of two solids that do not
    touch is a two-shell solid, and refusing it would make the op behave
    differently depending on where something had been dragged."""
    a = _obj("A")
    b = _obj("B", translation=[5.0, 0.0, 0.0])

    merged = ops_boolean.union([a, b])

    assert_closed(merged)
    assert _volume(merged) == pytest.approx(2.0, abs=1e-3)


def test_the_result_lands_in_the_targets_local_frame():
    """``ops.join``'s convention: the target keeps its transform, so nothing
    about it moves and everything else is carried into its frame."""
    a = _obj("A", translation=[10.0, 0.0, 0.0])
    b = _obj("B", translation=[10.5, 0.5, 0.5])

    lo, hi = bm.bounds(ops_boolean.union([a, b]))

    # Local coordinates, so the target's own -0.5..0.5 box is still at the
    # origin rather than out at x = 10.
    assert lo[0] == pytest.approx(-0.5, abs=1e-4)
    assert hi[0] == pytest.approx(1.0, abs=1e-4)


# --- the stated costs -------------------------------------------------------


def test_the_union_carries_no_uvs():
    """A boolean recuts every face it touches, so no output corner corresponds
    to any input corner. ``None`` is the honest answer."""
    a = _obj("A", mesh=bp.box())
    b = _obj("B", translation=[0.5, 0.5, 0.5])

    assert ops_boolean.union([a, b]).uv is None


def test_the_union_comes_back_as_triangles():
    merged = ops_boolean.union([_obj("A"), _obj("B", translation=[0.5, 0.5, 0.5])])
    counts = np.diff(merged.starts)
    assert set(counts.tolist()) == {3}


def test_the_material_slot_is_the_targets():
    """There is no per-face answer to carry across, and the target is already
    what a merge keeps the name and the transform of."""
    mesh = bp.box()
    coloured = bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=np.full(bm.face_count(mesh), 3, dtype="i4"),
        smooth=mesh.smooth,
    )
    a = _obj("A", mesh=coloured)
    b = _obj("B", translation=[0.5, 0.5, 0.5])

    assert set(ops_boolean.union([a, b]).material.tolist()) == {3}


# --- refusals ---------------------------------------------------------------


def test_one_object_is_refused():
    with pytest.raises(OpError, match="at least two"):
        ops_boolean.union([_obj("A")])


def test_an_open_surface_is_refused_by_name():
    """A union is only defined over solids: an open surface has no inside, so
    there is nothing to say whether a point is in it."""
    box = bp.box()
    open_box = bm.Mesh(
        positions=box.positions,
        loops=box.loops[: box.starts[5]],
        starts=box.starts[:6],
        material=box.material[:5],
        smooth=box.smooth[:5],
    )
    with pytest.raises(OpError, match="closed solid"):
        ops_boolean.union([_obj("A", mesh=open_box), _obj("B", translation=[0.5, 0.5, 0.5])])


def test_a_refusal_reaches_the_user_as_a_toast_and_changes_nothing():
    """Through the registry, which is where every other refusal is caught."""

    class _Ctx:
        def __init__(self) -> None:
            self.errors: list[str] = []

        def toast(self, message: str, level: str = "info") -> None:
            if level == "error":
                self.errors.append(message)

    box = bp.box()
    open_box = bm.Mesh(
        positions=box.positions,
        loops=box.loops[: box.starts[5]],
        starts=box.starts[:6],
        material=box.material[:5],
        smooth=box.smooth[:5],
    )
    doc = bd.ClayDoc()
    first = doc.add_object(_obj("A", mesh=open_box))
    second = doc.add_object(_obj("B", translation=[0.5, 0.5, 0.5]))
    doc.select([first.uid, second.uid])
    ctx = _Ctx()

    assert clay_ops.run(ctx, doc, clay_ops.get("union")) is False
    assert ctx.errors and "closed solid" in ctx.errors[0]
    assert len(doc.objects) == 2, "nothing was absorbed"


# --- through the registry ---------------------------------------------------


def _two_boxes() -> tuple[bd.ClayDoc, int, int]:
    doc = bd.ClayDoc()
    first = doc.add_object(_obj("A", generator="box", params={"size": (1.0, 1.0, 1.0)}))
    second = doc.add_object(_obj("B", translation=[0.5, 0.5, 0.5]))
    doc.select([first.uid, second.uid])
    return doc, first.uid, second.uid


class _Ctx:
    def toast(self, message: str, level: str = "info") -> None:
        raise AssertionError(f"unexpected toast: {message}")


def test_the_op_keeps_the_topmost_object_and_absorbs_the_rest():
    doc, first, _second = _two_boxes()

    assert clay_ops.run(_Ctx(), doc, clay_ops.get("union")) is True

    assert [obj.uid for obj in doc.objects] == [first]
    assert doc.selection == {first}


def test_the_op_is_disabled_below_two_visible_objects():
    doc, first, second = _two_boxes()
    union = clay_ops.get("union")
    assert union.enabled(doc)

    doc.set_visibility({second: False})
    assert not union.enabled(doc)


def test_the_op_freezes_the_generator_and_undoes_in_one_press():
    """The whole edit is one ``CompoundEdit``: a Ctrl+Z that put the absorbed
    object back while the target still carried the union would show a state
    that never happened."""
    doc, first, second = _two_boxes()
    before = bm.face_count(doc.by_uid(first).mesh)
    clay_ops.run(_Ctx(), doc, clay_ops.get("union"))
    assert doc.by_uid(first).generator is None

    doc.undo()

    assert [obj.uid for obj in doc.objects] == [first, second]
    assert bm.face_count(doc.by_uid(first).mesh) == before
    assert doc.by_uid(first).generator == "box"


# --- difference and intersection (Clay W5 groundwork) -------------------------


def test_a_difference_keeps_only_the_first_object():
    """The one boolean whose **order matters**: "the block minus the hole" and
    "the hole minus the block" are different shapes, and only one of them is
    ever what was meant."""
    first = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    second = bd.Obj(
        uid=bd.new_uid(), name="b", mesh=bp.box(), translation=np.array([0.5, 0.0, 0.0])
    )

    mesh = ops_boolean.difference([first, second])
    xs = np.asarray(mesh.positions, dtype="f8")[:, 0]

    # The half of the first cube the second does not overlap.
    assert xs.min() == pytest.approx(-0.5, abs=1e-6)
    assert xs.max() == pytest.approx(0.0, abs=1e-6)


def test_a_difference_is_not_symmetric():
    first = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    second = bd.Obj(
        uid=bd.new_uid(), name="b", mesh=bp.box(), translation=np.array([0.5, 0.0, 0.0])
    )

    one = np.asarray(ops_boolean.difference([first, second]).positions)[:, 0]
    other = np.asarray(ops_boolean.difference([second, first]).positions)[:, 0]

    assert one.max() != pytest.approx(other.max())


def test_an_intersection_keeps_only_the_overlap():
    first = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    second = bd.Obj(
        uid=bd.new_uid(), name="b", mesh=bp.box(), translation=np.array([0.5, 0.0, 0.0])
    )

    xs = np.asarray(ops_boolean.intersection([first, second]).positions)[:, 0]

    assert xs.min() == pytest.approx(0.0, abs=1e-6)
    assert xs.max() == pytest.approx(0.5, abs=1e-6)


def test_an_intersection_of_disjoint_solids_is_refused_by_name():
    """An empty result is *correct* and is also an object with nothing in it,
    which on screen is indistinguishable from the operation having failed. So
    it is named rather than applied -- and the sentence says which of the three
    operations produced it, which the union-only wording could not."""
    first = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    far = bd.Obj(
        uid=bd.new_uid(), name="b", mesh=bp.box(), translation=np.array([9.0, 0.0, 0.0])
    )

    with pytest.raises(OpError, match="intersection came out empty"):
        ops_boolean.intersection([first, far])


def test_union_still_means_what_it_did():
    """The generalisation is backward compatible: the old entry point is the
    new one with a default."""
    first = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    second = bd.Obj(
        uid=bd.new_uid(), name="b", mesh=bp.box(), translation=np.array([0.5, 0.0, 0.0])
    )

    direct = ops_boolean.union([first, second])
    by_kind = ops_boolean.boolean([first, second], "union")

    assert len(direct.positions) == len(by_kind.positions)


def test_an_unknown_kind_is_refused_by_name():
    from warlock.studio.clay.elements import OpError

    first = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    second = bd.Obj(uid=bd.new_uid(), name="b", mesh=bp.box())

    with pytest.raises(OpError, match="not one of"):
        ops_boolean.boolean([first, second], "nonsense")


@pytest.mark.parametrize("kind", list(ops_boolean.KINDS))
def test_every_kind_refuses_a_single_object_with_its_own_verb(kind):
    from warlock.studio.clay.elements import OpError

    only = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())

    with pytest.raises(OpError, match="at least two"):
        ops_boolean.boolean([only], kind)


# --- the triangle budget -----------------------------------------------------


def test_the_kernel_never_runs_past_the_triangle_budget(monkeypatch):
    """Two boxes at 12 triangles each is nowhere near the real ceiling, so
    lowering it is what stands in for the import-sized inputs it exists to
    catch -- and the point is that the refusal fires before ``manifold3d``
    ever sees the meshes, not after it runs out of memory computing them."""
    from warlock.studio.clay.elements import OpError

    first = bd.Obj(uid=bd.new_uid(), name="a", mesh=bp.box())
    second = bd.Obj(
        uid=bd.new_uid(), name="b", mesh=bp.box(), translation=np.array([0.5, 0.0, 0.0])
    )
    monkeypatch.setattr(ops_boolean, "MAX_BOOLEAN_TRIANGLES", 20)

    with pytest.raises(OpError, match="union"):
        ops_boolean.union([first, second])


def test_the_budget_is_the_sum_of_every_input_not_the_largest(monkeypatch):
    """A budget read off one mesh would pass three boxes that together exceed
    it, since no single input does."""
    from warlock.studio.clay import mesh as bm
    from warlock.studio.clay.elements import OpError

    box = bp.box()
    one_box_triangles = len(box.loops) - 2 * bm.face_count(box)
    objs = [
        bd.Obj(uid=bd.new_uid(), name=f"o{i}", mesh=bp.box(), translation=np.array([i, 0.0, 0.0]))
        for i in range(3)
    ]
    monkeypatch.setattr(ops_boolean, "MAX_BOOLEAN_TRIANGLES", one_box_triangles * 2)

    with pytest.raises(OpError, match="union"):
        ops_boolean.union(objs)
