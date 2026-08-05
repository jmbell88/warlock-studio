"""The CSR mesh: its validity rules, its fans, its normals and its copies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from warlock.studio.clay import mesh as bm
from warlock.studio.viewer import math3d as m3

# The unit box, centred on the origin, one metre on a side. Every loop is
# wound counter-clockwise seen from outside, so a correct Newell normal points
# away from the centre -- which is the whole of the outward-normals test.
BOX_POSITIONS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5],
        [-0.5, +0.5, +0.5],
    ],
    dtype="f4",
)
BOX_FACES = [
    [0, 3, 2, 1],  # -Z
    [4, 5, 6, 7],  # +Z
    [0, 1, 5, 4],  # -Y
    [3, 7, 6, 2],  # +Y
    [0, 4, 7, 3],  # -X
    [1, 2, 6, 5],  # +X
]


def _from_faces(positions: np.ndarray, faces: list[list[int]], **kw: object) -> bm.Mesh:
    loops = np.array([i for f in faces for i in f], dtype="i4")
    starts = np.concatenate([[0], np.cumsum([len(f) for f in faces])]).astype("i4")
    n = len(faces)
    return bm.Mesh(
        positions=positions,
        loops=loops,
        starts=starts,
        material=kw.get("material", np.zeros(n, dtype="i4")),
        smooth=kw.get("smooth", np.zeros(n, dtype=bool)),
    )


def box(**kw: object) -> bm.Mesh:
    return _from_faces(BOX_POSITIONS, BOX_FACES, **kw)


def quad() -> bm.Mesh:
    return _from_faces(
        np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            dtype="f4",
        ),
        [[0, 1, 2, 3]],
    )


def triangle_then_pentagon() -> bm.Mesh:
    angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    pent = np.stack([np.cos(angles), np.sin(angles), np.full(5, 2.0)], axis=1)
    positions = np.concatenate(
        [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"), pent.astype("f4")]
    )
    return _from_faces(positions, [[0, 1, 2], [3, 4, 5, 6, 7]])


# --- validate ---------------------------------------------------------------


def test_a_well_formed_csr_mesh_validates() -> None:
    bm.validate(box())


def test_an_empty_mesh_is_a_valid_csr_mesh() -> None:
    bm.validate(
        bm.Mesh(
            positions=np.zeros((0, 3), dtype="f4"),
            loops=np.zeros(0, dtype="i4"),
            starts=np.zeros(1, dtype="i4"),
            material=np.zeros(0, dtype="i4"),
            smooth=np.zeros(0, dtype=bool),
        )
    )


def test_validate_rejects_positions_that_are_not_three_wide() -> None:
    with pytest.raises(ValueError, match=r"positions must be \(V, 3\)"):
        bm.validate(
            bm.Mesh(
                positions=np.zeros((4, 5), dtype="f4"),
                loops=np.zeros(0, dtype="i4"),
                starts=np.zeros(1, dtype="i4"),
                material=np.zeros(0, dtype="i4"),
                smooth=np.zeros(0, dtype=bool),
            )
        )


def test_validate_rejects_empty_positions_that_are_not_three_wide() -> None:
    # An empty mesh is valid, but only as (0, 3) -- a (0, 5) array is as
    # malformed as any other and used to slip through the empty case.
    with pytest.raises(ValueError, match=r"positions must be \(V, 3\)"):
        bm.validate(
            bm.Mesh(
                positions=np.zeros((0, 5), dtype="f4"),
                loops=np.zeros(0, dtype="i4"),
                starts=np.zeros(1, dtype="i4"),
                material=np.zeros(0, dtype="i4"),
                smooth=np.zeros(0, dtype=bool),
            )
        )


def test_validate_rejects_starts_that_are_not_monotonic() -> None:
    m = box()
    starts = np.array(m.starts)
    starts[3], starts[4] = starts[4], starts[3]
    with pytest.raises(ValueError, match="monotonic"):
        bm.validate(bm.Mesh(m.positions, m.loops, starts, m.material, m.smooth))


def test_validate_rejects_starts_that_do_not_begin_at_zero() -> None:
    m = box()
    starts = np.array(m.starts)
    starts[0] = 1
    with pytest.raises(ValueError, match="start at 0"):
        bm.validate(bm.Mesh(m.positions, m.loops, starts, m.material, m.smooth))


def test_validate_rejects_starts_that_do_not_end_at_the_loop_count() -> None:
    m = box()
    starts = np.array(m.starts)
    starts[-1] = len(m.loops) - 1
    with pytest.raises(ValueError, match="end at len"):
        bm.validate(bm.Mesh(m.positions, m.loops, starts, m.material, m.smooth))


def test_validate_rejects_a_face_with_fewer_than_three_corners() -> None:
    m = _from_faces(BOX_POSITIONS, [[0, 1]])
    with pytest.raises(ValueError, match="fewer than 3 corners"):
        bm.validate(m)


def test_validate_rejects_a_loop_index_out_of_range_for_positions() -> None:
    m = box()
    loops = np.array(m.loops)
    loops[5] = len(m.positions)
    with pytest.raises(ValueError, match="out of range"):
        bm.validate(bm.Mesh(m.positions, loops, m.starts, m.material, m.smooth))


def test_validate_rejects_a_material_array_of_the_wrong_length() -> None:
    m = box()
    with pytest.raises(ValueError, match="material"):
        bm.validate(
            bm.Mesh(m.positions, m.loops, m.starts, np.zeros(3, "i4"), m.smooth)
        )


def test_validate_rejects_a_smooth_array_of_the_wrong_length() -> None:
    m = box()
    with pytest.raises(ValueError, match="smooth"):
        bm.validate(
            bm.Mesh(m.positions, m.loops, m.starts, m.material, np.zeros(3, bool))
        )


# --- accessors --------------------------------------------------------------


def test_face_count_is_one_less_than_the_length_of_starts() -> None:
    m = box()
    assert bm.face_count(m) == len(m.starts) - 1 == 6


def test_face_is_the_slice_of_loops_between_consecutive_starts() -> None:
    m = box()
    for i in range(bm.face_count(m)):
        expected = m.loops[m.starts[i] : m.starts[i + 1]]
        assert np.array_equal(bm.face(m, i), expected)
        assert list(bm.face(m, i)) == BOX_FACES[i]


def test_edges_are_unique_undirected_pairs() -> None:
    m = box()
    e = bm.edges(m)
    assert e.shape == (12, 2)  # a box has twelve edges, each shared by two faces
    assert (e[:, 0] < e[:, 1]).all()  # normalised low-to-high
    assert len(np.unique(e, axis=0)) == 12


# --- triangulate ------------------------------------------------------------


def test_triangulate_fans_a_quad_into_two_triangles_of_the_same_face() -> None:
    tris, tri_face = bm.triangulate(quad())
    assert tris.shape == (2, 3)
    assert np.array_equal(tris, [[0, 1, 2], [0, 2, 3]])
    assert np.array_equal(tri_face, [0, 0])


def test_triangulate_a_triangle_then_a_pentagon_yields_one_plus_three() -> None:
    tris, tri_face = bm.triangulate(triangle_then_pentagon())
    assert len(tris) == 1 + 3
    assert np.array_equal(tri_face, [0, 1, 1, 1])
    assert np.array_equal(tris[0], [0, 1, 2])
    assert np.array_equal(tris[1:], [[3, 4, 5], [3, 5, 6], [3, 6, 7]])


def test_triangulate_an_empty_mesh_yields_empty_arrays() -> None:
    m = bm.Mesh(
        np.zeros((0, 3), "f4"),
        np.zeros(0, "i4"),
        np.zeros(1, "i4"),
        np.zeros(0, "i4"),
        np.zeros(0, bool),
    )
    tris, tri_face = bm.triangulate(m)
    assert tris.shape == (0, 3)
    assert tri_face.shape == (0,)


# --- render_arrays ----------------------------------------------------------


def test_render_arrays_splits_a_vertex_per_corner_on_a_flat_cube() -> None:
    positions, normals, indices = bm.render_arrays(box())
    assert len(positions) == 24
    assert len(normals) == 24
    assert len(indices) == 6 * 2 * 3


def test_render_arrays_shares_vertices_on_a_smooth_cube() -> None:
    positions, normals, indices = bm.render_arrays(box(smooth=np.ones(6, bool)))
    assert len(positions) == 8
    assert len(normals) == 8
    assert len(indices) == 6 * 2 * 3


def test_render_arrays_normals_on_a_unit_box_point_outward() -> None:
    for smooth in (np.zeros(6, bool), np.ones(6, bool)):
        positions, normals, _ = bm.render_arrays(box(smooth=smooth))
        centre = np.zeros(3)
        assert float(np.sum((positions - centre) * normals)) > 0.0


def test_render_arrays_normals_are_unit_length() -> None:
    _, normals, _ = bm.render_arrays(box())
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


def test_render_arrays_indices_stay_inside_the_emitted_vertices() -> None:
    mixed = box(smooth=np.array([True] * 3 + [False] * 3))
    positions, _, indices = bm.render_arrays(mixed)
    assert indices.max() < len(positions)
    # Three smooth faces touch every one of the eight corners; the three flat
    # ones contribute four corners each.
    assert len(positions) == 8 + 12


# --- transformed ------------------------------------------------------------


def test_transformed_returns_a_new_mesh_and_leaves_the_input_untouched() -> None:
    m = box()
    before = np.array(m.positions)
    out = bm.transformed(m, m3.translation(m3.vec3(1.0, 2.0, 3.0)))
    assert out is not m
    assert np.array_equal(m.positions, before)
    assert np.allclose(out.positions, before + np.array([1.0, 2.0, 3.0]), atol=1e-5)


def test_transformed_applies_the_full_four_by_four() -> None:
    m = quad()
    matrix = m3.translation(m3.vec3(0.0, 0.0, 5.0)) @ m3.scaling(m3.vec3(2.0, 3.0, 1.0))
    out = bm.transformed(m, matrix)
    assert np.allclose(
        out.positions,
        np.array([[0, 0, 5], [2, 0, 5], [2, 3, 5], [0, 3, 5]], dtype="f4"),
        atol=1e-5,
    )


def test_transformed_by_a_mirror_reverses_every_face_loop() -> None:
    m = box()
    out = bm.transformed(m, m3.scaling(m3.vec3(-1.0, 1.0, 1.0)))
    for i in range(bm.face_count(m)):
        assert list(bm.face(out, i)) == list(bm.face(m, i))[::-1]


def test_transformed_by_a_mirror_keeps_normals_pointing_outward() -> None:
    out = bm.transformed(box(), m3.scaling(m3.vec3(-1.0, 1.0, 1.0)))
    positions, normals, _ = bm.render_arrays(out)
    assert float(np.sum(positions * normals)) > 0.0


def test_transformed_without_a_mirror_keeps_every_face_loop_as_it_was() -> None:
    m = box()
    spin = m3.quat_to_mat4(m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.7))
    out = bm.transformed(m, spin @ m3.scaling(2.0))
    assert np.array_equal(out.loops, m.loops)


# --- bounds -----------------------------------------------------------------


def test_bounds_on_a_known_box_are_exact() -> None:
    lo, hi = bm.bounds(box())
    assert np.array_equal(lo, [-0.5, -0.5, -0.5])
    assert np.array_equal(hi, [0.5, 0.5, 0.5])


def test_bounds_on_an_empty_mesh_are_two_zeroes() -> None:
    lo, hi = bm.bounds(
        bm.Mesh(
            np.zeros((0, 3), "f4"),
            np.zeros(0, "i4"),
            np.zeros(1, "i4"),
            np.zeros(0, "i4"),
            np.zeros(0, bool),
        )
    )
    assert np.array_equal(lo, np.zeros(3))
    assert np.array_equal(hi, np.zeros(3))


# --- immutability -----------------------------------------------------------


def test_a_mesh_copies_the_arrays_it_is_built_from() -> None:
    positions = np.array(BOX_POSITIONS)
    m = _from_faces(positions, BOX_FACES)
    positions[0] = [9.0, 9.0, 9.0]
    assert np.array_equal(m.positions[0], BOX_POSITIONS[0])


def test_a_meshs_arrays_are_read_only() -> None:
    m = box()
    for arr in (m.positions, m.loops, m.starts, m.material, m.smooth):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 0


def test_a_face_slice_cannot_be_written_through() -> None:
    m = box()
    with pytest.raises(ValueError):
        bm.face(m, 0)[0] = 7


def test_a_mesh_field_cannot_be_reassigned() -> None:
    m = box()
    with pytest.raises(FrozenInstanceError):
        m.positions = np.zeros((1, 3), "f4")  # type: ignore[misc]


def test_a_meshs_arrays_own_their_bytes() -> None:
    # A view of a big shared buffer reports its own tiny nbytes while keeping
    # the base alive, and nbytes is what undo eviction is driven by. Built
    # from a slice of a 10k-vertex scratch buffer, the mesh must still own
    # every byte it claims -- base is None on all five fields, never merely a
    # smaller base.
    big = np.zeros((10_000, 3), dtype="f4")
    m = _from_faces(big[:8], BOX_FACES)
    for arr in (m.positions, m.loops, m.starts, m.material, m.smooth):
        assert arr.base is None
