"""The six primitive generators: valid CSR, closed shells and outward winding.

Almost everything here is parametrised over ``primitives.GENERATORS`` rather
than written out per shape, because the registry is the thing a properties
panel will be generated from and a seventh primitive added later should be
covered by tests that already exist rather than by tests somebody remembers to
write. Each parametrised case runs at the defaults *and* at the clamped low
end, where the topology genuinely differs. The per-shape tests below the
parametrised block are the ones that cannot be generic: an exact vertex count,
an exact span, the value a segment count is clamped to.

**Winding takes two assertions.** The volume sum -- ``(centroid - centre) . n``
over the faces -- says the shell is oriented *outward*, and that is all it
says: reversing one face moves it by only twice that face's contribution, and
on a cone's base it lands on ``7e-18``, which is on the passing side of
``> 0``. Directed-edge uniqueness says the orientation is *consistent*, which
is the half that catches a flipped face, and
``test_a_single_flipped_face_is_caught`` demonstrates it firing on every face
of every closed primitive rather than leaving its strength to be assumed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as bp

from .topo_asserts import directed_edge_counts, edge_use_counts

# ``plane`` is the one generator that is deliberately not a closed shell: it is
# a single face, so it has a boundary, no volume and no meaningful "outward".
# Naming it here rather than scattering ``if name == "plane"`` through the
# parametrised tests keeps the exception assertable as an exception -- if a
# later change accidentally closes it, or accidentally opens something else,
# exactly one of the two tests below fails.
OPEN = {"plane"}
CLOSED = sorted(set(bp.GENERATORS) - OPEN)

# Every generic test runs at the defaults *and* at the clamped low end, because
# a clamp is where a primitive's topology changes shape: a three-segment
# cylinder's cap is a triangle, a two-band sphere is two fans of triangles with
# no quad between them, and those are exactly the configurations a hand
# derivation of the winding is least likely to have been done for.
COUNT_MINIMA = {
    "segments": bp.MIN_SEGMENTS,
    "sides": bp.MIN_SEGMENTS,
    "rings": bp.MIN_RINGS,
}


def _variants() -> list[pytest.param]:  # type: ignore[valid-type]
    out = []
    for name in sorted(bp.GENERATORS):
        defaults, _ = bp.GENERATORS[name]
        out.append(pytest.param(name, dict(defaults), id=f"{name}-defaults"))
        low = {k: COUNT_MINIMA.get(k, v) for k, v in defaults.items()}
        if low != defaults:
            out.append(pytest.param(name, low, id=f"{name}-minimum"))
    return out


VARIANTS = _variants()
CLOSED_VARIANTS = [p for p in VARIANTS if p.values[0] not in OPEN]


def _build(name: str, params: dict) -> bm.Mesh:
    _, builder = bp.GENERATORS[name]
    return builder(**params)


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


# Both live in ``topo_asserts`` now, so the op tests assert the same two things
# these do -- and are re-exported here under their original names because the
# argument this module makes reads as one piece.
_directed_edge_counts = directed_edge_counts
"""How many faces traverse each *ordered* corner pair."""

_edge_use_counts = edge_use_counts
"""How many faces use each *undirected* edge -- orientation-blind by design.

This one answers "is the shell closed" and cannot answer anything about
winding: it sorts each pair, so a reversed face uses exactly the same
undirected edges it used before. That is why :func:`_directed_edge_counts`
exists beside it.
"""


# --- parametrised over the registry -----------------------------------------


@pytest.mark.parametrize(("name", "params"), VARIANTS)
def test_every_generator_produces_a_valid_csr(name: str, params: dict) -> None:
    bm.validate(_build(name, params))


@pytest.mark.parametrize("name", sorted(bp.GENERATORS))
def test_every_generator_is_centred_on_the_origin(name: str) -> None:
    """At the defaults only -- the *bounds* of a coarse ring are not centred.

    A three-segment cylinder has one vertex at +r and two at -r/2, so its
    measured box is off-centre while its geometry is not. Every default segment
    count here is divisible by four, where the two coincide.
    """
    lo, hi = bm.bounds(_default(name))
    assert np.allclose(lo + hi, 0.0, atol=1e-6)


@pytest.mark.parametrize(("name", "params"), VARIANTS)
def test_every_generators_faces_are_convex_enough_to_fan(name: str, params: dict) -> None:
    """Every corner turns the same way, which is what ``triangulate`` assumes.

    ``triangulate`` fans from the first corner, and a fan across a reflex
    corner puts a triangle outside the polygon. Convexity is therefore a
    property of the generators, not a hope about them.

    The turn is divided through by the two edge lengths and the normal's
    magnitude, so what is compared against the epsilon is the *sine* of the
    corner's turn angle rather than a quantity that scales as edge-length
    squared times twice the face area. Without that, the same absolute
    tolerance is savage at metre scale and meaningless at millimetre scale --
    a reflex corner on a small primitive would pass.
    """
    mesh = _build(name, params)
    for i in range(bm.face_count(mesh)):
        p = mesh.positions[bm.face(mesh, i)].astype("f8")
        normal = _face_normal(mesh, i)
        normal = normal / np.linalg.norm(normal)
        edge = np.roll(p, -1, axis=0) - p
        length = np.linalg.norm(edge, axis=1)
        turn = np.cross(edge, np.roll(edge, -1, axis=0)) @ normal
        turn = turn / (length * np.roll(length, -1))
        assert (turn > -1e-9).all(), f"{name} face {i} has a reflex corner"


@pytest.mark.parametrize(("name", "params"), VARIANTS)
def test_every_generator_is_consistently_oriented(name: str, params: dict) -> None:
    """No ordered corner pair is traversed twice -- the flipped-face check.

    Two faces sharing an edge traverse it in *opposite* directions when they
    agree about which side is out, so across a consistently oriented mesh every
    ordered pair ``(a, b)`` occurs at most once. Reverse one face's loop and
    each of its edges collides with its neighbour's, and this assertion fires.

    It is here because the volume sum below cannot do it. Reversing a single
    face changes that sum by only twice the face's own contribution, which is
    small beside the whole enclosed volume: measured on these generators,
    flipping the cylinder's top cap leaves it at +3.06 out of +4.59, and
    flipping the cone's base leaves it at 7e-18 -- still on the passing side of
    ``> 0``, by float noise. The undirected edge count cannot do it either; it
    sorts each pair and is blind to orientation by construction. Two
    assertions, two different claims: this one says *consistent*, the volume
    sum says *outward*, and only together do they say what the module promises.
    """
    counts = _directed_edge_counts(_build(name, params))
    assert max(counts.values()) == 1


@pytest.mark.parametrize(("name", "params"), CLOSED_VARIANTS)
def test_every_closed_generator_is_wound_outward(name: str, params: dict) -> None:
    """The divergence-theorem check: it is six times the enclosed volume.

    Positive means the consistent orientation the test above establishes is the
    *outward* one rather than the inward one -- an inside-out shell is equally
    consistent and encloses a negative volume.
    """
    mesh = _build(name, params)
    lo, hi = bm.bounds(mesh)
    centre = (lo + hi) * 0.5
    total = sum(
        float((_centroid(mesh, i) - centre) @ _face_normal(mesh, i))
        for i in range(bm.face_count(mesh))
    )
    assert total > 0.0


@pytest.mark.parametrize(("name", "params"), CLOSED_VARIANTS)
def test_every_closed_generator_uses_each_edge_exactly_twice(name: str, params: dict) -> None:
    mesh = _build(name, params)
    counts = _edge_use_counts(mesh)
    assert set(counts.values()) == {2}
    assert len(counts) == len(bm.edges(mesh))


def _with_face_reversed(mesh: bm.Mesh, i: int) -> bm.Mesh:
    loops = [list(bm.face(mesh, f)) for f in range(bm.face_count(mesh))]
    loops[i] = loops[i][::-1]
    return bm.Mesh(
        positions=mesh.positions,
        loops=np.array([v for f in loops for v in f], dtype="i4"),
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
    )


@pytest.mark.parametrize("name", CLOSED)
def test_a_single_flipped_face_is_caught(name: str) -> None:
    """The orientation assertion earns its place: flip one face, it fires.

    A test whose failure mode has never been observed is a test nobody knows
    the strength of. Each of these meshes with exactly one face reversed still
    passes ``validate``, still uses every undirected edge twice, and (for the
    cone's base) still passes the volume sum -- and this is what catches it.
    """
    mesh = _default(name)
    for i in range(bm.face_count(mesh)):
        broken = _with_face_reversed(mesh, i)
        bm.validate(broken)  # the defect is geometric; the CSR is untouched
        assert max(_directed_edge_counts(broken).values()) == 2


def test_the_volume_sum_alone_would_miss_a_flipped_cone_base() -> None:
    """Why the orientation assertion exists, stated as an executable fact.

    If this ever starts failing because the sum went properly negative, the
    volume check has become sufficient on its own and this test's comment --
    not the assertion above it -- is what needs revisiting.
    """
    mesh = bp.cone()
    base = bm.face_count(mesh) - 1
    broken = _with_face_reversed(mesh, base)
    lo, hi = bm.bounds(broken)
    centre = (lo + hi) * 0.5
    total = sum(
        float((_centroid(broken, i) - centre) @ _face_normal(broken, i))
        for i in range(bm.face_count(broken))
    )
    assert abs(total) < 1e-12
    assert set(_edge_use_counts(broken).values()) == {2}


@pytest.mark.parametrize(("name", "params"), VARIANTS)
def test_every_generator_leaves_every_vertex_used(name: str, params: dict) -> None:
    mesh = _build(name, params)
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


# --- negative extents --------------------------------------------------------


@pytest.mark.parametrize(
    ("negative", "positive"),
    [
        (lambda: bp.box(size=(-2.0, 1.0, 3.0)), lambda: bp.box(size=(2.0, 1.0, 3.0))),
        (lambda: bp.plane(size=(2.0, -3.0)), lambda: bp.plane(size=(2.0, 3.0))),
        (
            lambda: bp.cylinder(radius=0.5, height=-1.0),
            lambda: bp.cylinder(radius=0.5, height=1.0),
        ),
        (
            lambda: bp.cone(radius=-0.5, height=1.0),
            lambda: bp.cone(radius=0.5, height=1.0),
        ),
        (lambda: bp.uv_sphere(radius=-0.5), lambda: bp.uv_sphere(radius=0.5)),
        (lambda: bp.torus(radius=0.35, tube=-0.15), lambda: bp.torus()),
    ],
    ids=["box", "plane", "cylinder", "cone", "uv_sphere", "torus"],
)
def test_a_negative_extent_is_taken_as_its_magnitude(negative: object, positive: object) -> None:
    """A minus sign in a numeric field must not turn a primitive inside out.

    Without the ``abs``, a negative height puts a cylinder's bottom ring above
    its top and every side quad and both caps then face inward -- and
    ``validate`` accepts it, because the CSR is impeccable and only the
    geometry is wrong. Mirroring is ``mesh.transformed``'s job, which reverses
    the loops to keep the winding honest.
    """
    mesh = negative()  # type: ignore[operator]
    assert np.allclose(mesh.positions, positive().positions)  # type: ignore[operator]
    assert max(_directed_edge_counts(mesh).values()) == 1


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
