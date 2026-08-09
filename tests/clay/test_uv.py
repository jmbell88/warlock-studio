"""Texture coordinates: the projection, and what every generator produces.

``Mesh.uv`` has been a field since the CSR layout was written; this is the
first thing that fills it in. The rules that matter are all about *corners*
rather than vertices -- a wrap seam is one position with two coordinates, which
is the only reason the field is per corner at all.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as prim
from warlock.studio.clay import uv as uv_mod


def _bare(mesh: bm.Mesh) -> bm.Mesh:
    return replace(mesh, uv=None)


def _faces(mesh: bm.Mesh):
    starts = mesh.starts
    return [mesh.uv[starts[i] : starts[i + 1]] for i in range(bm.face_count(mesh))]


# --- every generator ---------------------------------------------------------


def test_every_generator_produces_one_coordinate_per_corner():
    for name, (defaults, build) in prim.GENERATORS.items():
        mesh = build(**defaults)
        assert mesh.uv is not None, name
        assert mesh.uv.shape == (len(mesh.loops), 2), name
        assert mesh.uv.dtype == np.float32, name


def test_every_generator_stays_inside_the_unit_square():
    """Outside it a texture tiles, which is a decision, not a default."""
    for name, (defaults, build) in prim.GENERATORS.items():
        mesh = build(**defaults)
        assert mesh.uv.min() >= -1e-5, name
        assert mesh.uv.max() <= 1.0 + 1e-5, name


def test_no_generator_produces_a_degenerate_island():
    """A face whose corners all land on one point bakes to a single texel."""
    for name, (defaults, build) in prim.GENERATORS.items():
        for index, island in enumerate(_faces(build(**defaults))):
            span = island.max(axis=0) - island.min(axis=0)
            assert float(span.max()) > 1e-4, (name, index)


def test_a_wrap_seam_closes_at_one_rather_than_folding_back_to_zero():
    """The whole reason the field is per corner: the last quad around a
    cylinder runs to u = 1, sharing positions with the first quad at u = 0."""
    mesh = prim.cylinder(segments=8)
    sides = _faces(mesh)[:8]
    assert float(sides[0][:, 0].min()) == pytest.approx(0.0)
    assert float(sides[-1][:, 0].max()) == pytest.approx(1.0)


def test_a_cylinders_caps_do_not_overlap_its_sides():
    """A canonical unwrap whose islands overlap is one that cannot be baked
    to, and baking is what these coordinates exist for."""
    mesh = prim.cylinder(segments=8)
    islands = _faces(mesh)
    sides, caps = islands[:8], islands[8:]
    assert min(float(i[:, 1].min()) for i in sides) >= 0.5 - 1e-5
    assert max(float(i[:, 1].max()) for i in caps) <= 0.5 - 1e-5


def test_a_cylinders_two_caps_do_not_overlap_each_other():
    caps = _faces(prim.cylinder(segments=8))[8:]
    assert float(caps[0][:, 0].max()) < float(caps[1][:, 0].min())


def test_a_cones_apex_gets_a_different_coordinate_per_side():
    """It is one vertex and many corners, which is exactly the case a
    per-corner field exists for -- a shared apex uv smears the whole top of
    the texture into a point."""
    mesh = prim.cone(segments=6)
    apex_us = [float(island[1, 0]) for island in _faces(mesh)[:6]]
    assert len(set(apex_us)) == 6


def test_a_sphere_runs_pole_to_pole_across_the_whole_square():
    mesh = prim.uv_sphere(segments=8, rings=4)
    assert float(mesh.uv[:, 1].min()) == pytest.approx(0.0)
    assert float(mesh.uv[:, 1].max()) == pytest.approx(1.0)
    assert float(mesh.uv[:, 0].max()) == pytest.approx(1.0)


def test_a_torus_closes_both_of_its_seams():
    mesh = prim.torus(segments=6, sides=4)
    assert float(mesh.uv[:, 0].max()) == pytest.approx(1.0)
    assert float(mesh.uv[:, 1].max()) == pytest.approx(1.0)


def test_an_icospheres_seam_faces_stay_contiguous_rather_than_wrapping():
    """A per-vertex longitude hands a straddling triangle ``0.98, 0.02`` and
    wraps the whole texture backwards across it. The island is unwrapped to one
    branch and then *translated* into the square -- never scaled, so the mapping
    stays exact and the only thing paid is continuity, at the seam."""
    mesh = prim.icosphere(subdivisions=2)
    for index, island in enumerate(_faces(mesh)):
        assert float(island[:, 0].max() - island[:, 0].min()) < 0.5, index


def test_an_icospheres_axial_corner_borrows_its_faces_own_longitude():
    """A corner on the Y axis has no longitude at all; without the fallback it
    reads as zero and drags its island across half the texture."""
    mesh = prim.icosphere(subdivisions=1)
    assert np.all(np.isfinite(mesh.uv))


def test_a_capsules_v_follows_arc_length_rather_than_height():
    """Height squashes each hemisphere into a band as tall as its own bulge,
    which on a stubby capsule is most of the texture in a tenth of the square."""
    mesh = prim.capsule(radius=0.5, height=0.1, segments=8, rings=4)
    equator = float(mesh.uv[:, 1].max())
    assert equator == pytest.approx(1.0)
    # The cylinder is a tenth of the profile's length, so it must take about a
    # tenth of v -- the two rows at y = +-0.05 sit either side of the middle.
    rows = sorted({round(float(v), 4) for v in mesh.uv[:, 1]})
    middle = [v for v in rows if 0.4 < v < 0.6]
    assert len(middle) == 2
    assert middle[1] - middle[0] == pytest.approx(0.1 / (0.1 + np.pi * 0.5), abs=1e-3)


def test_a_grid_wraps_one_texture_over_the_whole_sheet():
    """The coordinates describe the surface, not the faces it happens to be cut
    into, so a texture fits it once however many divisions it has."""
    mesh = prim.grid(divisions=4)
    assert float(mesh.uv.min()) == pytest.approx(0.0)
    assert float(mesh.uv.max()) == pytest.approx(1.0)


def test_a_generators_uvs_survive_a_parameter_change():
    """The panel rebuilds the mesh from the generator, so a resized box has to
    come back with coordinates rather than without."""
    assert prim.box(size=(3.0, 0.2, 1.0)).uv is not None


# --- the projection -----------------------------------------------------------


def test_a_box_unwrap_gives_every_face_the_whole_square():
    islands = _faces(uv_mod.box_unwrap(_bare(prim.box())))
    for island in islands:
        assert float(island[:, 0].min()) == pytest.approx(0.0)
        assert float(island[:, 0].max()) == pytest.approx(1.0)
        assert float(island[:, 1].min()) == pytest.approx(0.0)
        assert float(island[:, 1].max()) == pytest.approx(1.0)


def test_a_box_is_its_own_unwrap():
    """The generator calls the op rather than keeping a second table of the
    same six squares, so the two cannot drift."""
    assert np.array_equal(prim.box().uv, uv_mod.box_unwrap(_bare(prim.box())).uv)


def test_texel_density_is_shared_rather_than_per_island():
    """Normalised over the mesh's own bounding box, so a long thin object is
    not stretched into a square. Scaling each island to fill the square
    independently would put different sized checkers on every panel.
    """
    mesh = uv_mod.box_unwrap(_bare(prim.box(size=(4.0, 1.0, 1.0))))
    islands = _faces(mesh)
    # The +Y face spans the long axis in u and the short one in v.
    top = islands[1]
    assert float(top[:, 0].max() - top[:, 0].min()) == pytest.approx(1.0)
    assert float(top[:, 1].max() - top[:, 1].min()) == pytest.approx(0.25)


def test_opposite_faces_are_mirrored_rather_than_projected_alike():
    """They share the square, and without the flip one of the pair reads
    backwards -- lettering on the far side of a box comes out reversed.

    Asserted geometrically rather than by comparing the two arrays: a mirrored
    face is also wound the other way, so the flip makes the two corner *lists*
    come out identical, which is the evidence rather than a counterexample.
    What says it worked is that the corner at a given world x carries a
    different u on each of the two faces.
    """
    mesh = uv_mod.box_unwrap(_bare(prim.box()))
    positions = mesh.positions[mesh.loops]

    def u_at(face: int, x: float) -> float:
        s, e = mesh.starts[face], mesh.starts[face + 1]
        which = np.flatnonzero(np.isclose(positions[s:e, 0], x))
        return float(mesh.uv[s:e][which[0], 0])

    front, back = 2, 3  # +Z and -Z
    assert u_at(front, -0.5) == pytest.approx(0.0)
    assert u_at(back, -0.5) == pytest.approx(1.0)


def test_a_planar_unwrap_flattens_everything_down_one_axis():
    mesh = uv_mod.planar_unwrap(_bare(prim.box()), axis=1)
    # Every corner's u comes from X and its v from Z, so the two -Y/+Y faces
    # and the four sides all project into the same square.
    assert mesh.uv.shape == (len(mesh.loops), 2)
    assert float(mesh.uv.max()) == pytest.approx(1.0)


def test_unwrapping_changes_no_geometry_at_all():
    """UVs are not geometry: the positions, the topology, the materials and the
    shading flags all have to come back untouched, which is what lets the op
    keep the object's generator."""
    before = _bare(prim.cylinder())
    after = uv_mod.box_unwrap(before)
    assert np.array_equal(after.positions, before.positions)
    assert np.array_equal(after.loops, before.loops)
    assert np.array_equal(after.starts, before.starts)
    assert np.array_equal(after.material, before.material)
    assert np.array_equal(after.smooth, before.smooth)


def test_an_empty_mesh_unwraps_to_itself_rather_than_raising():
    empty = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    assert uv_mod.box_unwrap(empty) is empty


def test_a_flat_mesh_does_not_divide_by_its_zero_extent():
    """A plane at y = 0 has no extent on one axis; one is the scale that leaves
    the coordinates as they are."""
    mesh = uv_mod.box_unwrap(_bare(prim.plane()))
    assert np.all(np.isfinite(mesh.uv))


def test_the_dominant_axis_is_the_one_the_normal_is_closest_to():
    normals = np.array([[0.9, 0.1, 0.0], [0.0, -1.0, 0.0], [0.2, 0.3, -0.8]])
    assert uv_mod.dominant_axis(normals).tolist() == [0, 1, 2]


# --- the op -------------------------------------------------------------------


def test_the_unwrap_op_keeps_the_generator():
    """A box that has been unwrapped is still describable as "box, size 1":
    nothing about its geometry changed, so editing the size must still
    rebuild it rather than being refused as a frozen mesh."""
    from warlock.studio import clay_ops
    from warlock.studio.clay import document as bd

    doc = bd.ClayDoc()
    obj = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="B", mesh=_bare(prim.box()), generator="box",
               params={"size": (1.0, 1.0, 1.0)})
    )
    doc.select([obj.uid])
    assert clay_ops.run(None, doc, clay_ops.get("unwrap")) is not False
    assert doc.by_uid(obj.uid).mesh.uv is not None
    assert doc.by_uid(obj.uid).generator == "box"


def test_the_unwrap_op_is_one_undo_step_per_object():
    from warlock.studio import clay_ops
    from warlock.studio.clay import document as bd

    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="B", mesh=_bare(prim.box())))
    doc.select([obj.uid])
    depth = len(doc.history)
    clay_ops.run(None, doc, clay_ops.get("unwrap"))
    assert len(doc.history) == depth + 1
    doc.undo()
    assert doc.by_uid(obj.uid).mesh.uv is None
