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
        uv=kw.get("uv"),
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
    positions, normals, uvs, indices = bm.render_arrays(box())
    assert len(positions) == 24
    assert len(normals) == 24
    assert len(indices) == 6 * 2 * 3
    assert uvs is None


def test_render_arrays_shares_vertices_on_a_smooth_cube() -> None:
    positions, normals, uvs, indices = bm.render_arrays(box(smooth=np.ones(6, bool)))
    assert len(positions) == 8
    assert len(normals) == 8
    assert len(indices) == 6 * 2 * 3
    assert uvs is None


def test_render_arrays_normals_on_a_unit_box_point_outward() -> None:
    for smooth in (np.zeros(6, bool), np.ones(6, bool)):
        positions, normals, _, _ = bm.render_arrays(box(smooth=smooth))
        centre = np.zeros(3)
        assert float(np.sum((positions - centre) * normals)) > 0.0


def test_render_arrays_normals_are_unit_length() -> None:
    _, normals, _, _ = bm.render_arrays(box())
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


def test_render_arrays_indices_stay_inside_the_emitted_vertices() -> None:
    mixed = box(smooth=np.array([True] * 3 + [False] * 3))
    positions, _, _, indices = bm.render_arrays(mixed)
    assert indices.max() < len(positions)
    # Three smooth faces touch every one of the eight corners; the three flat
    # ones contribute four corners each.
    assert len(positions) == 8 + 12


# --- uv ---------------------------------------------------------------------


def _uv_box() -> bm.Mesh:
    """The unit box with a distinct uv per corner, so nothing can share one."""
    m = box()
    uv = np.arange(len(m.loops) * 2, dtype="f4").reshape(-1, 2) / 100.0
    return bm.Mesh(
        positions=m.positions,
        loops=m.loops,
        starts=m.starts,
        material=m.material,
        smooth=m.smooth,
        uv=uv,
    )


def test_a_mesh_without_uvs_carries_none_rather_than_an_empty_array() -> None:
    assert box().uv is None


def test_uvs_are_copied_and_made_read_only_like_every_other_array() -> None:
    given = np.zeros((24, 2), dtype="f4")
    m = _from_faces(BOX_POSITIONS, BOX_FACES, uv=given)
    assert m.uv is not given
    with pytest.raises(ValueError):
        m.uv[0, 0] = 1.0


def test_validate_accepts_one_uv_per_face_corner() -> None:
    bm.validate(_uv_box())


def test_validate_rejects_uvs_that_are_not_one_per_corner() -> None:
    m = box()
    with pytest.raises(ValueError, match="uv must be one"):
        bm.validate(
            bm.Mesh(
                positions=m.positions,
                loops=m.loops,
                starts=m.starts,
                material=m.material,
                smooth=m.smooth,
                uv=np.zeros((len(m.positions), 2), dtype="f4"),
            )
        )


def test_a_textured_mesh_renders_one_vertex_per_corner() -> None:
    m = _uv_box()
    positions, normals, uvs, indices = bm.render_arrays(m)
    assert len(positions) == len(m.loops) == 24
    assert len(normals) == 24
    assert uvs is not None
    assert np.array_equal(uvs, m.uv)
    assert indices.max() < len(positions)


def test_a_textured_mesh_gives_up_sharing_but_not_smooth_normals() -> None:
    """A smooth textured cube emits 24 vertices carrying the 8 shared normals."""
    smooth = np.ones(6, bool)
    shared = bm.render_arrays(box(smooth=smooth))
    m = _uv_box()
    textured = bm.Mesh(
        positions=m.positions,
        loops=m.loops,
        starts=m.starts,
        material=m.material,
        smooth=smooth,
        uv=m.uv,
    )
    positions, normals, _, _ = bm.render_arrays(textured)
    assert len(positions) == 24 and len(shared[0]) == 8
    # Every corner's normal is the one its vertex got on the shared path.
    for corner, vertex in enumerate(textured.loops):
        assert np.allclose(normals[corner], shared[1][vertex], atol=1e-6)


def test_a_textured_empty_mesh_renders_an_empty_uv_array_not_none() -> None:
    empty = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
        uv=np.zeros((0, 2), dtype="f4"),
    )
    _, _, uvs, _ = bm.render_arrays(empty)
    assert uvs is not None and uvs.shape == (0, 2)


def test_mesh_bytes_skips_an_absent_uv_and_counts_a_present_one() -> None:
    from warlock.studio.clay.edits import mesh_bytes

    plain, textured = box(), _uv_box()
    assert mesh_bytes(textured) == mesh_bytes(plain) + textured.uv.nbytes


def test_a_mirror_reverses_the_uvs_with_the_loops_they_belong_to() -> None:
    m = _uv_box()
    out = bm.transformed(m, m3.scaling(m3.vec3(-1.0, 1.0, 1.0)))
    assert out.uv is not None
    for i in range(bm.face_count(m)):
        lo, hi = int(m.starts[i]), int(m.starts[i + 1])
        assert np.array_equal(out.uv[lo:hi], m.uv[lo:hi][::-1])


def test_a_transform_without_a_mirror_carries_the_uvs_through_unchanged() -> None:
    m = _uv_box()
    out = bm.transformed(m, m3.translation(m3.vec3(1.0, 0.0, 0.0)))
    assert np.array_equal(out.uv, m.uv)


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
    positions, normals, _, _ = bm.render_arrays(out)
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


# --- the smooth-normal scatter ----------------------------------------------


def test_bincount_accumulation_is_bit_identical_to_add_at():
    """``_accumulate`` replaced ``np.add.at`` for speed and nothing else.

    ``np.add.at`` is the unbuffered ufunc path and is an order of magnitude
    slower; ``np.bincount`` sums the same values in the same *input* order, so
    the float result is identical rather than merely close. Asserting that is
    the whole point -- the output feeds ``_normalize`` and then the GPU, where
    a changed last bit has nothing to announce it.
    """
    rng = np.random.default_rng(90210)
    n_verts = 137
    # Heavy repetition, so a vertex accumulates many contributions and the
    # summation order actually has something to disagree about.
    loops = rng.integers(0, n_verts, size=4096, dtype="i8")
    values = (rng.random((4096, 3)) - 0.5) * 1e3

    reference = np.zeros((n_verts, 3), dtype="f8")
    np.add.at(reference, loops, values)

    got = bm._accumulate(loops, values, n_verts)
    assert got.dtype == reference.dtype
    assert np.array_equal(got, reference)


def test_accumulation_of_nothing_is_the_zeroed_table():
    """A mesh with no smooth face still needs a full-length accumulator."""
    empty = bm._accumulate(
        np.zeros(0, dtype="i8"), np.zeros((0, 3), dtype="f8"), 9
    )
    assert empty.shape == (9, 3)
    assert not empty.any()
    # Float, not the integer table bincount hands back for an empty input:
    # ``_normalize`` divides into this and refuses an int output array.
    assert empty.dtype == np.dtype("f8")


# --- the render layout ------------------------------------------------------
#
# ``render_arrays`` is a composition of ``render_layout`` (what a moved vertex
# cannot change) and ``render_from_layout`` (what it can). The split exists so
# an element drag can hold the first across a whole drag; these assert that it
# is a *split* and not a rewrite -- byte identity, not closeness, because the
# outputs go to the GPU where a changed last bit has nothing to announce it.


def _layout_cases() -> list[tuple[str, bm.Mesh]]:
    smooth_box = _from_faces(BOX_POSITIONS, BOX_FACES, smooth=np.ones(6, dtype="?"))
    mixed = _from_faces(
        BOX_POSITIONS, BOX_FACES, smooth=np.array([1, 0, 1, 0, 1, 0], dtype="?")
    )
    textured = _from_faces(
        BOX_POSITIONS,
        BOX_FACES,
        uv=np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), (6, 1)),
    )
    # An L, whose reflex corner sends it down the ear-clipping path.
    concave = bm.Mesh(
        positions=np.array(
            [[0, 0, 0], [2, 0, 0], [2, 0, 1], [1, 0, 1], [1, 0, 2], [0, 0, 2]],
            dtype="f4",
        ),
        loops=np.arange(6, dtype="i4"),
        starts=np.array([0, 6], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype="?"),
    )
    empty = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype="?"),
    )
    return [
        ("flat box", _from_faces(BOX_POSITIONS, BOX_FACES)),
        ("smooth box", smooth_box),
        ("mixed box", mixed),
        ("textured box", textured),
        ("concave face", concave),
        ("empty", empty),
    ]


@pytest.mark.parametrize(
    "name,mesh", _layout_cases(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_the_layout_split_reproduces_render_arrays_byte_for_byte(name, mesh) -> None:
    want = bm.render_arrays(mesh)
    got = bm.render_from_layout(bm.render_layout(mesh), mesh.positions)
    for a, b in zip(want, got, strict=True):
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert a.dtype == b.dtype, name
        assert np.array_equal(a, b), name


@pytest.mark.parametrize(
    "name,mesh", _layout_cases(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_a_reused_layout_answers_for_moved_vertices(name, mesh) -> None:
    """What a drag actually does: build the layout once, move the vertices,
    and ask again. The positions and normals must be what a full rebuild on the
    moved mesh would have produced -- the *indices* are deliberately the
    layout's, which is what the GPU is still drawing during a drag."""
    from dataclasses import replace

    rng = np.random.default_rng(4242)
    moved = np.array(mesh.positions, dtype="f4")
    if len(moved):
        moved[0] += rng.random(3).astype("f4") * 0.1

    layout = bm.render_layout(mesh)
    got = bm.render_from_layout(layout, moved)
    want = bm.render_arrays(replace(mesh, positions=moved))
    for index, (a, b) in enumerate(zip(want, got, strict=True)):
        if index == 3:  # indices: the layout's, by design
            continue
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert np.array_equal(a, b), f"{name}: field {index}"


def test_a_layout_is_not_disturbed_by_being_used() -> None:
    """A drag reuses one layout for every frame, so a frame that wrote into it
    would make the second frame answer about the first one's positions."""
    mesh = _from_faces(BOX_POSITIONS, BOX_FACES, smooth=np.ones(6, dtype="?"))
    layout = bm.render_layout(mesh)
    first = bm.render_from_layout(layout, mesh.positions)

    moved = np.array(mesh.positions, dtype="f4")
    moved[0] += 1.0
    bm.render_from_layout(layout, moved)

    again = bm.render_from_layout(layout, mesh.positions)
    for a, b in zip(first, again, strict=True):
        if a is None:
            assert b is None
            continue
        assert np.array_equal(a, b)
