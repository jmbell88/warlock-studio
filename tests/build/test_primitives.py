"""The six primitive generators: valid CSR, closed shells and outward winding.

Almost everything here is parametrised over ``primitives.GENERATORS`` rather
than written out per shape, because the registry is the thing a properties
panel will be generated from and a seventh primitive added later should be
covered by tests that already exist rather than by tests somebody remembers to
write. The per-shape tests below the parametrised block are the ones that
genuinely cannot be generic: an exact vertex count, an exact span, the value a
segment count is clamped to.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from warlock.studio.build import mesh as bm
from warlock.studio.build import primitives as bp

# ``plane`` is the one generator that is deliberately not a closed shell: it is
# a single face, so it has a boundary, no volume and no meaningful "outward".
# Naming it here rather than scattering ``if name == "plane"`` through the
# parametrised tests keeps the exception assertable as an exception -- if a
# later change accidentally closes it, or accidentally opens something else,
# exactly one of the two tests below fails.
OPEN = {"plane"}
CLOSED = sorted(set(bp.GENERATORS) - OPEN)


def _default(name: str) -> bm.Mesh:
    defaults, builder = bp.GENERATORS[name]
    return builder(**defaults)


def _face_normal(mesh: bm.Mesh, i: int) -> np.ndarray:
    """Newell normal for one face, computed here rather than imported.

    ``mesh._face_normals`` would do it, but a generator's winding checked with
    the same private helper the renderer uses would pass a test that only says
    the two agree. This is four lines of the textbook formula and is
    independent of it.
    """
    p = mesh.positions[bm.face(mesh, i)].astype("f8")
    q = np.roll(p, -1, axis=0)
    return np.stack(
        [
            ((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])).sum(),
            ((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])).sum(),
            ((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])).sum(),
        ]
    )


def _centroid(mesh: bm.Mesh, i: int) -> np.ndarray:
    return mesh.positions[bm.face(mesh, i)].astype("f8").mean(axis=0)


def _edge_use_counts(mesh: bm.Mesh) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for i in range(bm.face_count(mesh)):
        loop = [int(v) for v in bm.face(mesh, i)]
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            counts[(min(a, b), max(a, b))] += 1
    return counts


# --- parametrised over the registry -----------------------------------------


@pytest.mark.parametrize("name", sorted(bp.GENERATORS))
def test_every_generator_called_with_its_own_defaults_validates(name: str) -> None:
    bm.validate(_default(name))


@pytest.mark.parametrize("name", sorted(bp.GENERATORS))
def test_every_generator_is_centred_on_the_origin(name: str) -> None:
    lo, hi = bm.bounds(_default(name))
    assert np.allclose(lo + hi, 0.0, atol=1e-6)


@pytest.mark.parametrize("name", sorted(bp.GENERATORS))
def test_every_generators_faces_are_convex_enough_to_fan(name: str) -> None:
    """Every corner turns the same way, which is what ``triangulate`` assumes.

    ``triangulate`` fans from the first corner, and a fan across a reflex
    corner puts a triangle outside the polygon. Convexity is therefore a
    property of the generators, not a hope about them.
    """
    mesh = _default(name)
    for i in range(bm.face_count(mesh)):
        p = mesh.positions[bm.face(mesh, i)].astype("f8")
        normal = _face_normal(mesh, i)
        edge = np.roll(p, -1, axis=0) - p
        turn = np.cross(edge, np.roll(edge, -1, axis=0)) @ normal
        assert (turn > -1e-9).all(), f"{name} face {i} has a reflex corner"


@pytest.mark.parametrize("name", CLOSED)
def test_every_closed_generator_is_wound_outward(name: str) -> None:
    """The divergence-theorem check: it is six times the enclosed volume.

    This is the test that catches a flipped cap, which back-face culling is off
    for in the viewport and which turns an exported GLB inside out in an engine
    with nothing in the file to say why.
    """
    mesh = _default(name)
    lo, hi = bm.bounds(mesh)
    centre = (lo + hi) * 0.5
    total = sum(
        float((_centroid(mesh, i) - centre) @ _face_normal(mesh, i))
        for i in range(bm.face_count(mesh))
    )
    assert total > 0.0


@pytest.mark.parametrize("name", CLOSED)
def test_every_closed_generator_uses_each_edge_exactly_twice(name: str) -> None:
    mesh = _default(name)
    counts = _edge_use_counts(mesh)
    assert set(counts.values()) == {2}
    assert len(counts) == len(bm.edges(mesh))


@pytest.mark.parametrize("name", sorted(bp.GENERATORS))
def test_every_generator_leaves_every_vertex_used(name: str) -> None:
    mesh = _default(name)
    assert set(int(v) for v in mesh.loops) == set(range(len(mesh.positions)))


@pytest.mark.parametrize("name", sorted(bp.GENERATORS))
def test_every_registry_default_is_a_complete_call(name: str) -> None:
    """What the properties panel does: splat the defaults and expect a mesh.

    A default that is missing for a parameter without one is a ``TypeError``
    the panel would raise the first time a user picked that primitive.
    """
    defaults, builder = bp.GENERATORS[name]
    assert isinstance(builder(**defaults), bm.Mesh)
    assert builder.__name__ == name


def test_the_plane_is_the_one_generator_with_a_boundary() -> None:
    counts = _edge_use_counts(bp.plane())
    assert set(counts.values()) == {1}


def test_the_planes_single_face_points_up() -> None:
    """No centroid test can say which way a flat sheet faces, so say it here.

    +Y because the whole editor is Y-up glTF space and a floor a user places is
    a floor, not a ceiling.
    """
    mesh = bp.plane()
    normal = _face_normal(mesh, 0)
    assert np.allclose(normal / np.linalg.norm(normal), [0.0, 1.0, 0.0])


# --- exact counts ------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh", "verts", "faces"),
    [
        (bp.box(), 8, 6),
        (bp.plane(), 4, 1),
        # Two caps as n-gons rather than fans: 16 side quads plus 2 faces, and
        # no cap-centre vertex, so 2 * 16 positions and not 34.
        (bp.cylinder(segments=16), 32, 18),
        # One apex, one n-gon base, 16 side triangles.
        (bp.cone(segments=16), 17, 17),
        # Two poles plus 7 intermediate rings of 16; 2 * 16 pole triangles
        # plus 6 * 16 quads.
        (bp.uv_sphere(segments=16, rings=8), 114, 128),
        (bp.torus(segments=24, sides=12), 288, 288),
    ],
    ids=["box", "plane", "cylinder", "cone", "uv_sphere", "torus"],
)
def test_the_exact_vertex_and_face_counts(mesh: bm.Mesh, verts: int, faces: int) -> None:
    assert (len(mesh.positions), bm.face_count(mesh)) == (verts, faces)


def test_a_uv_spheres_poles_are_triangles_and_its_bands_are_quads() -> None:
    mesh = bp.uv_sphere(segments=16, rings=8)
    sizes = Counter(int(n) for n in np.diff(mesh.starts))
    assert sizes == {4: 96, 3: 32}


def test_a_cylinders_caps_are_single_n_gons() -> None:
    mesh = bp.cylinder(segments=16)
    sizes = Counter(int(n) for n in np.diff(mesh.starts))
    assert sizes == {4: 16, 16: 2}


# --- sizes -------------------------------------------------------------------


def test_a_box_spans_exactly_the_size_it_was_asked_for() -> None:
    lo, hi = bm.bounds(bp.box(size=(2.0, 1.0, 3.0)))
    assert np.allclose(lo, [-1.0, -0.5, -1.5])
    assert np.allclose(hi, [+1.0, +0.5, +1.5])


def test_a_plane_spans_exactly_the_size_it_was_asked_for() -> None:
    lo, hi = bm.bounds(bp.plane(size=(2.0, 3.0)))
    assert np.allclose(lo, [-1.0, 0.0, -1.5])
    assert np.allclose(hi, [+1.0, 0.0, +1.5])


def test_a_cylinder_spans_its_radius_and_height() -> None:
    lo, hi = bm.bounds(bp.cylinder(radius=2.0, height=3.0, segments=16))
    assert np.allclose(lo, [-2.0, -1.5, -2.0], atol=1e-6)
    assert np.allclose(hi, [+2.0, +1.5, +2.0], atol=1e-6)


def test_a_cone_spans_its_radius_and_height() -> None:
    lo, hi = bm.bounds(bp.cone(radius=2.0, height=3.0, segments=16))
    assert np.allclose(lo, [-2.0, -1.5, -2.0], atol=1e-6)
    assert np.allclose(hi, [+2.0, +1.5, +2.0], atol=1e-6)


def test_a_uv_sphere_spans_its_diameter() -> None:
    lo, hi = bm.bounds(bp.uv_sphere(radius=2.0, segments=16, rings=8))
    assert np.allclose(lo, -2.0, atol=1e-6)
    assert np.allclose(hi, +2.0, atol=1e-6)


def test_a_torus_spans_its_outer_diameter_and_its_tube() -> None:
    lo, hi = bm.bounds(bp.torus(radius=2.0, tube=0.5, segments=24, sides=12))
    assert np.allclose(lo, [-2.5, -0.5, -2.5], atol=1e-6)
    assert np.allclose(hi, [+2.5, +0.5, +2.5], atol=1e-6)


# --- clamping ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["cylinder", "cone", "uv_sphere", "torus"])
def test_a_degenerate_segment_count_is_clamped_rather_than_obeyed(name: str) -> None:
    """A slider dragged to zero must not produce a mesh ``validate`` rejects.

    Clamping at the low end rather than raising is the right shape for a
    control the user is *dragging*: an exception mid-drag would have to be
    caught by the panel, and there is no sensible thing for it to show.
    """
    defaults, builder = bp.GENERATORS[name]
    degenerate = builder(**{**defaults, "segments": 1})
    clamped = builder(**{**defaults, "segments": bp.MIN_SEGMENTS})
    bm.validate(degenerate)
    assert (len(degenerate.positions), bm.face_count(degenerate)) == (
        len(clamped.positions),
        bm.face_count(clamped),
    )


def test_a_cylinder_clamped_to_three_segments_is_a_triangular_prism() -> None:
    mesh = bp.cylinder(segments=1)
    assert (len(mesh.positions), bm.face_count(mesh)) == (6, 5)
    bm.validate(mesh)


def test_a_uv_spheres_rings_are_clamped_too() -> None:
    mesh = bp.uv_sphere(segments=8, rings=0)
    bm.validate(mesh)
    assert (len(mesh.positions), bm.face_count(mesh)) == (10, 16)


def test_a_toruss_sides_are_clamped_too() -> None:
    mesh = bp.torus(segments=8, sides=1)
    bm.validate(mesh)
    assert (len(mesh.positions), bm.face_count(mesh)) == (24, 24)


# --- the registry itself -----------------------------------------------------


def test_the_registry_names_every_generator_the_module_exports() -> None:
    assert set(bp.GENERATORS) == {
        "box",
        "plane",
        "cylinder",
        "cone",
        "uv_sphere",
        "torus",
    }


def test_every_generated_mesh_starts_flat_shaded_on_material_zero() -> None:
    for name in bp.GENERATORS:
        mesh = _default(name)
        assert not mesh.smooth.any()
        assert not mesh.material.any()
