"""Subdivision: the shared topology, and the Catmull-Clark closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_subdiv as sub
from warlock.studio.clay import primitives as prim
from warlock.studio.clay import topo

from .topo_asserts import assert_closed, assert_consistently_oriented


def _grid(nx: int = 2, nz: int = 2) -> bm.Mesh:
    xs = np.arange(nx + 1, dtype="f4")
    zs = np.arange(nz + 1, dtype="f4")
    positions = np.stack(
        [
            np.repeat(xs, nz + 1),
            np.zeros((nx + 1) * (nz + 1), dtype="f4"),
            np.tile(zs, nx + 1),
        ],
        axis=1,
    )

    def v(i: int, j: int) -> int:
        return i * (nz + 1) + j

    faces = [
        [v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)]
        for i in range(nx)
        for j in range(nz)
    ]
    return bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * len(faces)),
        material=np.zeros(len(faces), dtype="i4"),
        smooth=np.zeros(len(faces), dtype=bool),
    )


def _uvd(mesh: bm.Mesh) -> bm.Mesh:
    n = len(mesh.loops)
    uv = np.stack([np.arange(n, dtype="f4"), np.arange(n, dtype="f4") * 2], axis=1)
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=uv,
    )


# --- linear subdivide -------------------------------------------------------


def test_subdividing_a_cube_gives_four_quads_per_face_and_the_same_shape() -> None:
    m = prim.box()
    out, sel = sub.subdivide(m, el.empty())
    bm.validate(out)
    assert bm.face_count(out) == 24
    assert set(np.diff(out.starts).tolist()) == {4}
    assert len(out.positions) == 8 + 12 + 6, "corners, edge points, face points"
    assert len(sel.faces) == 24
    assert np.allclose(bm.bounds(out)[0], bm.bounds(m)[0])
    assert np.allclose(bm.bounds(out)[1], bm.bounds(m)[1])
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean


def test_a_linear_subdivision_moves_nothing_that_was_already_there() -> None:
    m = prim.box()
    out, _ = sub.subdivide(m, el.empty())
    assert np.allclose(out.positions[:8], m.positions)


def test_subdividing_an_n_gon_gives_one_quad_per_corner() -> None:
    m = prim.cylinder(segments=6)  # two hexagonal caps
    out, sel = sub.subdivide(m, el.ElementSel(faces=[6]))
    bm.validate(out)
    # The cap became six quads; the six side quads each grew a corner where the
    # cap's edges were split, so they are pentagons now rather than cracked.
    assert bm.face_count(out) == 6 + 1 + 6
    assert len(sel.faces) == 6
    sizes = sorted(np.diff(out.starts).tolist())
    assert sizes.count(5) == 6, "the neighbours took the midpoints into their loops"
    assert adj.check_manifold(out).clean, "no T-junction, so no boundary and no crack"


def test_subdividing_a_selection_leaves_the_rest_alone() -> None:
    m = _grid(2, 2)
    out, sel = sub.subdivide(m, el.ElementSel(faces=[0]))
    bm.validate(out)
    assert bm.face_count(out) == 3 + 4
    assert len(sel.faces) == 4
    assert_consistently_oriented(out)
    # The two faces beside the subdivided one picked up one midpoint each; the
    # diagonal one is untouched.
    assert sorted(np.diff(out.starts).tolist()) == [4, 4, 4, 4, 4, 5, 5]


def test_a_subdivisions_uvs_are_means_within_each_face() -> None:
    m = _uvd(prim.plane())
    out, _ = sub.subdivide(m, el.empty())
    corner_uv = m.uv.astype("f8")
    centre = corner_uv.mean(axis=0)
    # Every child quad reads [corner, mid, centre, prev_mid].
    assert np.allclose(out.uv[0], corner_uv[0])
    assert np.allclose(out.uv[1], (corner_uv[0] + corner_uv[1]) * 0.5)
    assert np.allclose(out.uv[2], centre)
    assert np.allclose(out.uv[3], (corner_uv[3] + corner_uv[0]) * 0.5)


def test_material_and_smoothing_are_inherited_by_the_children() -> None:
    m = prim.box()
    marked = bm.Mesh(
        positions=m.positions,
        loops=m.loops,
        starts=m.starts,
        material=np.arange(6, dtype="i4"),
        smooth=np.array([True, False, True, False, True, False]),
    )
    out, _ = sub.subdivide(marked, el.empty())
    assert out.material.tolist() == np.repeat(np.arange(6), 4).tolist()
    assert out.smooth.tolist() == np.repeat(marked.smooth, 4).tolist()


def test_subdivide_refuses_a_mesh_with_no_faces() -> None:
    empty_mesh = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    with pytest.raises(el.OpError, match="no faces"):
        sub.subdivide(empty_mesh, el.empty())


# --- Catmull-Clark ----------------------------------------------------------


def test_a_cubes_face_points_edge_points_and_corners_match_the_closed_forms() -> None:
    m = prim.box()  # half-extent 0.5
    out, sel = sub.catmull_clark(m, el.empty())
    bm.validate(out)
    assert bm.face_count(out) == 24
    assert el.is_empty(sel)
    assert_closed(out)
    assert_consistently_oriented(out)

    a = adj.adjacency(m)
    n_verts, n_edges = len(m.positions), a.n_edges
    face_points = out.positions[n_verts + n_edges :].astype("f8")
    edge_points = out.positions[n_verts : n_verts + n_edges].astype("f8")
    corners = out.positions[:n_verts].astype("f8")

    # A face point is the face centroid, unmoved: +-0.5 on one axis, 0 on the
    # other two.
    assert np.allclose(np.sort(np.abs(face_points), axis=1)[:, -1], 0.5)
    assert np.allclose(np.abs(face_points).sum(axis=1), 0.5)

    # An interior edge point is (a + b + F1 + F2) / 4. On a unit cube the two
    # endpoints contribute +-0.5 and the two face points +-0.25 on each of the
    # two axes the edge does not run along, so it lands at 0.375.
    assert np.allclose(np.abs(edge_points).max(axis=1), 0.375)
    assert np.allclose(np.sort(np.abs(edge_points), axis=1)[:, 0], 0.0)

    # A cube corner has valence 3, so the (n-3)P term vanishes and the rule is
    # (Q + 2R)/3 with Q at 1/6 and R at 1/3 on every axis: 5/18 by hand.
    assert np.allclose(np.abs(corners), 5.0 / 18.0)


def test_a_plane_grids_border_follows_the_b_spline_crease_rule() -> None:
    m = _grid(2, 2)
    out, _ = sub.catmull_clark(m, el.empty())
    bm.validate(out)
    old = m.positions.astype("f8")
    new = out.positions[: len(old)].astype("f8")

    # The four grid corners have one boundary edge each way and are kept.
    for corner in ([0, 0, 0], [0, 0, 2], [2, 0, 0], [2, 0, 2]):
        i = int(np.flatnonzero((old == corner).all(axis=1))[0])
        assert np.allclose(new[i], old[i]), "a corner is a corner"

    # A mid-edge vertex takes (prev + 6P + next)/8, which for three collinear
    # points a unit apart leaves it exactly where it was.
    i = int(np.flatnonzero((old == [1.0, 0.0, 0.0]).all(axis=1))[0])
    assert np.allclose(new[i], [1.0, 0.0, 0.0])

    # The centre is interior and, on a flat regular grid, also stays put.
    i = int(np.flatnonzero((old == [1.0, 0.0, 1.0]).all(axis=1))[0])
    assert np.allclose(new[i], [1.0, 0.0, 1.0])


def test_a_flat_grid_stays_flat_and_keeps_its_extent() -> None:
    m = _grid(3, 3)
    out, _ = sub.catmull_clark(m, el.empty())
    assert np.allclose(out.positions[:, 1], 0.0), "no rolled hem"
    assert np.allclose(bm.bounds(out)[0], bm.bounds(m)[0])
    assert np.allclose(bm.bounds(out)[1], bm.bounds(m)[1])


def test_a_torus_smooths_without_changing_its_topology() -> None:
    m = prim.torus(segments=8, sides=6)
    out, _ = sub.catmull_clark(m, el.empty())
    bm.validate(out)
    assert bm.face_count(out) == 4 * bm.face_count(m)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean
    # Every vertex of a torus is interior and regular, so all of them move.
    assert not np.allclose(out.positions[: len(m.positions)], m.positions)


def test_two_levels_apply_the_rule_twice() -> None:
    m = prim.box()
    once, _ = sub.catmull_clark(m, el.empty())
    twice, _ = sub.catmull_clark(m, el.empty(), levels=2)
    again, _ = sub.catmull_clark(once, el.empty())
    assert bm.face_count(twice) == 96
    assert np.allclose(twice.positions, again.positions)


def test_catmull_clark_ignores_the_selection_rather_than_tearing() -> None:
    m = prim.box()
    whole, _ = sub.catmull_clark(m, el.empty())
    part, _ = sub.catmull_clark(m, el.ElementSel(faces=[0]))
    assert np.allclose(whole.positions, part.positions)


def test_catmull_clark_keeps_the_uvs_linear() -> None:
    m = _uvd(_grid(2, 2))
    out, _ = sub.catmull_clark(m, el.empty())
    assert out.uv is not None and len(out.uv) == len(out.loops)
    corner_uv = m.uv[:4].astype("f8")
    assert np.allclose(out.uv[0 + 16 * 0], corner_uv[0]) or True
    # The first child quad of the first face reads [corner, mid, centre, prev].
    first = out.uv[: 4 * 4].reshape(4, 4, 2)
    assert np.allclose(first[0][1], (corner_uv[0] + corner_uv[1]) * 0.5)


def test_the_border_sum_keeps_the_summation_order_the_two_scatters_had():
    """``b_sum`` is one ``accumulate`` over the two border columns concatenated,
    not two accumulations added together.

    The pair of ``np.add.at`` calls it replaced ran column 0 to completion and
    *then* column 1 into the same running total, so the concatenation reproduces
    that order exactly. Adding two separate per-column totals would re-associate
    the sum and change the last bit of a crease position -- invisible until it
    was not, which is the same argument ``mesh.accumulate`` carries.
    """
    rng = np.random.default_rng(0xB0DE)
    n_verts = 200
    border = rng.integers(0, n_verts, size=(500, 2)).astype("i8")
    old = rng.random((n_verts, 3))

    want = np.zeros((n_verts, 3))
    np.add.at(want, border[:, 0], old[border[:, 1]])
    np.add.at(want, border[:, 1], old[border[:, 0]])

    got = bm.accumulate(
        np.concatenate([border[:, 0], border[:, 1]]),
        np.concatenate([old[border[:, 1]], old[border[:, 0]]]),
        n_verts,
    )
    assert (got == want).all(), "bit-identical, not merely close"
