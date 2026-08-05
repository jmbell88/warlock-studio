"""The object-level operations: mirror, snap, bake, duplicate.

The load-bearing one is mirror. A mirrored object could be expressed as a
negative node scale in a single line, and the reason it is not -- glTF readers
disagree about whether a negative scale flips winding -- is invisible until an
asset reaches an engine that decided the other way. So the test here does not
check that a mesh was mirrored; it checks that the mirrored mesh is still
consistently and *outwardly* wound, which is the property the negative-scale
version cannot promise.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from warlock.studio.build import document as bd
from warlock.studio.build import mesh as bm
from warlock.studio.build import ops
from warlock.studio.build import primitives as bp
from warlock.studio.viewer import math3d as m3


def _obj(name: str = "A", mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _face_normal(mesh: bm.Mesh, i: int) -> np.ndarray:
    """Newell normal for one face, by the textbook formula rather than by
    importing the one the renderer uses -- see ``test_primitives`` for why."""
    p = mesh.positions[bm.face(mesh, i)].astype("f8")
    q = np.roll(p, -1, axis=0)
    return np.stack(
        [
            ((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
            ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
            ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum(),
        ]
    )


def _volume_sum(mesh: bm.Mesh) -> float:
    """Six times the enclosed volume; positive iff the shell faces outward."""
    lo, hi = bm.bounds(mesh)
    centre = (lo + hi) * 0.5
    return sum(
        float(
            (mesh.positions[bm.face(mesh, i)].astype("f8").mean(axis=0) - centre)
            @ _face_normal(mesh, i)
        )
        for i in range(bm.face_count(mesh))
    )


def _max_directed_edge_use(mesh: bm.Mesh) -> int:
    """1 on a consistently oriented shell; 2 where two faces traverse an edge
    the same way, which is what a single flipped face produces."""
    counts: Counter[tuple[int, int]] = Counter()
    for i in range(bm.face_count(mesh)):
        loop = [int(v) for v in bm.face(mesh, i)]
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            counts[(a, b)] += 1
    return max(counts.values()) if counts else 0


def _world_positions(obj: bd.Obj) -> np.ndarray:
    matrix = m3.compose(obj.translation, obj.rotation, obj.scale)
    homo = np.hstack([obj.mesh.positions.astype("f8"), np.ones((len(obj.mesh.positions), 1))])
    return (matrix @ homo.T).T[:, :3]


CLOSED = ["box", "cylinder", "cone", "uv_sphere", "torus"]


# --- mirror ------------------------------------------------------------------


@pytest.mark.parametrize("name", CLOSED)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_mirrored_object_is_still_wound_outward(name: str, axis: int) -> None:
    defaults, build = bp.GENERATORS[name]
    obj = _obj(name, build(**defaults))
    out = ops.mirror(obj, axis)

    assert _max_directed_edge_use(out.mesh) == 1
    assert _volume_sum(out.mesh) > 0.0


def test_mirror_bakes_into_the_mesh_rather_than_negating_the_scale() -> None:
    obj = _obj(translation=(1.0, 2.0, 3.0), scale=(2.0, 2.0, 2.0))
    out = ops.mirror(obj, 0)

    assert np.allclose(out.scale, [2.0, 2.0, 2.0])
    assert (out.scale > 0.0).all()
    assert np.allclose(out.translation, obj.translation)
    assert out.mesh is not obj.mesh
    assert np.allclose(out.mesh.positions[:, 0], -obj.mesh.positions[:, 0])
    # The other two axes are untouched, so this really is a reflection and not
    # a rotation that happens to land on one.
    assert np.allclose(out.mesh.positions[:, 1:], obj.mesh.positions[:, 1:])


def test_mirroring_twice_returns_the_original_geometry() -> None:
    obj = _obj("A", bp.cone())
    twice = ops.mirror(ops.mirror(obj, 2), 2)
    assert np.allclose(twice.mesh.positions, obj.mesh.positions)
    assert np.array_equal(twice.mesh.loops, obj.mesh.loops)


def test_mirror_keeps_the_uid_because_it_edits_one_object() -> None:
    obj = _obj()
    assert ops.mirror(obj, 1).uid == obj.uid


def test_mirror_rejects_an_axis_that_is_not_one_of_three() -> None:
    with pytest.raises(ValueError):
        ops.mirror(_obj(), 3)


# --- snapping ----------------------------------------------------------------


def test_snapping_at_step_zero_is_the_identity() -> None:
    """Snapping *off* must not quantise everything to zero."""
    assert ops.snap_value(0.3456, 0.0) == pytest.approx(0.3456)
    assert np.allclose(ops.snap_translation((0.3, -1.7, 9.9), 0.0), [0.3, -1.7, 9.9])
    q = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(37.0))
    assert np.allclose(ops.snap_rotation(q, 0.0), q)


def test_snapping_is_symmetric_around_zero() -> None:
    assert ops.snap_value(0.34, 0.25) == pytest.approx(0.25)
    assert ops.snap_value(-0.34, 0.25) == pytest.approx(-0.25)


def test_a_value_exactly_between_two_steps_rounds_away_from_zero() -> None:
    """Half-away-from-zero, not Python's half-to-even: the grid must look the
    same on both sides of the origin, and ``round`` breaks that tie by parity."""
    assert ops.snap_value(0.125, 0.25) == pytest.approx(0.25)
    assert ops.snap_value(-0.125, 0.25) == pytest.approx(-0.25)


def test_snap_translation_snaps_each_component() -> None:
    out = ops.snap_translation((0.34, -0.9, 1.51), 0.5)
    assert np.allclose(out, [0.5, -1.0, 1.5])


def test_snap_rotation_quantises_the_angle_and_keeps_the_axis() -> None:
    axis = m3.vec3(0.0, 0.0, 1.0)
    q = ops.snap_rotation(m3.quat_from_axis_angle(axis, math.radians(37.0)), 15.0)
    expected = m3.quat_from_axis_angle(axis, math.radians(30.0))
    assert np.allclose(q, expected)


def test_snap_rotation_leaves_an_identity_rotation_alone() -> None:
    """The axis of a zero rotation is undefined, so there is nothing to keep."""
    assert np.allclose(ops.snap_rotation(m3.quat_identity(), 15.0), [0.0, 0.0, 0.0, 1.0])


def test_a_snapped_rotation_is_still_a_unit_quaternion() -> None:
    q = m3.quat_from_axis_angle(m3.vec3(1.0, 2.0, 3.0), math.radians(100.0))
    assert np.linalg.norm(ops.snap_rotation(q, 15.0)) == pytest.approx(1.0)


# --- baking ------------------------------------------------------------------


def test_bake_transform_folds_the_trs_in_and_resets_it() -> None:
    obj = _obj(
        translation=(1.0, 2.0, 3.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), math.radians(30.0)),
        scale=(2.0, 1.0, 0.5),
    )
    before = _world_positions(obj)
    out = ops.bake_transform(obj)

    assert np.allclose(out.translation, [0.0, 0.0, 0.0])
    assert np.allclose(out.rotation, [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(out.scale, [1.0, 1.0, 1.0])
    assert np.allclose(_world_positions(out), before, atol=1e-5)


def test_baking_leaves_the_world_bounds_where_they_were() -> None:
    obj = _obj(
        translation=(0.0, 5.0, 0.0),
        rotation=m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), math.radians(45.0)),
        scale=(3.0, 3.0, 3.0),
    )
    before = _world_positions(obj)
    lo, hi = bm.bounds(ops.bake_transform(obj).mesh)
    assert np.allclose(lo, before.min(axis=0), atol=1e-5)
    assert np.allclose(hi, before.max(axis=0), atol=1e-5)


def test_baking_a_negative_scale_keeps_the_winding_outward() -> None:
    """A negative scale is exactly what mirror refuses to leave on the node, so
    the one function that can be handed one has to deal with it too."""
    obj = _obj("A", bp.uv_sphere(), scale=(-1.0, 1.0, 1.0))
    out = ops.bake_transform(obj)
    assert _max_directed_edge_use(out.mesh) == 1
    assert _volume_sum(out.mesh) > 0.0


def test_baking_an_untransformed_object_changes_nothing() -> None:
    obj = _obj()
    out = ops.bake_transform(obj)
    assert np.allclose(out.mesh.positions, obj.mesh.positions)


# --- duplicate ---------------------------------------------------------------


def test_duplicate_gets_a_new_uid_the_same_geometry_and_a_new_name() -> None:
    obj = _obj("Box", translation=(1.0, 0.0, 0.0))
    copy = ops.duplicate(obj, bd.new_uid())

    assert copy.uid != obj.uid
    assert copy.name != obj.name
    assert np.allclose(copy.mesh.positions, obj.mesh.positions)
    assert np.allclose(copy.translation, obj.translation)


def test_a_duplicate_shares_the_mesh_because_a_mesh_is_immutable() -> None:
    obj = _obj("Box")
    assert ops.duplicate(obj, bd.new_uid()).mesh is obj.mesh


def test_duplicate_avoids_the_names_it_is_told_are_taken() -> None:
    obj = _obj("Box")
    copy = ops.duplicate(obj, bd.new_uid(), taken=["Box", "Box.001", "Box.002"])
    assert copy.name == "Box.003"


def test_duplicating_a_duplicate_counts_up_rather_than_nesting() -> None:
    obj = _obj("Box.007")
    assert ops.duplicate(obj, bd.new_uid()).name == "Box.008"


def test_a_duplicates_transform_arrays_are_its_own() -> None:
    obj = _obj("Box", translation=(1.0, 0.0, 0.0))
    copy = ops.duplicate(obj, bd.new_uid())
    copy.translation[0] = 9.0
    assert obj.translation[0] == 1.0


def test_a_duplicates_params_are_its_own() -> None:
    """Sharing the dict would make editing the copy's radius edit the
    original's, which is the one thing a duplicate must never do."""
    obj = _obj("Box", generator="cylinder", params={"radius": 0.5})
    copy = ops.duplicate(obj, bd.new_uid())
    copy.params["radius"] = 1.0
    assert obj.params == {"radius": 0.5}
    assert copy.generator == "cylinder"
