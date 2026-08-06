"""Scale: the paths that must stay vectorised, measured on a 100k-face mesh.

The one thing that turns this editor from usable into unusable is a Python loop
over faces creeping into a hot path -- and it is not visible in any of the other
tests, because they all run on a cube. So this file builds a synthetic soup at
roughly the scale of a real ``model.glb`` and puts a wall-clock bound on the
four operations that run per interaction: building the adjacency, triangulating,
subdividing, and sweeping a marquee.

The bounds are deliberately loose -- several times what the vectorised versions
actually take on a slow machine -- because the failure they are guarding
against is not "10% slower", it is "a Python loop over 100,000 faces", which is
two orders of magnitude, not two per cent. A generous bound that never flakes
and still catches that is worth more than a tight one that fails on a busy CI
box.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_subdiv as sub
from warlock.studio.clay import pick as bp
from warlock.studio.clay import topo
from warlock.studio.viewer import math3d as m3

# A grid this many quads on a side: 200 * 200 = 40,000 quads, 80,000 triangles,
# 40,401 vertices. Big enough that a per-face Python loop is unmistakable and
# small enough that the vectorised path is a few tens of milliseconds.
SIDE = 200

BUDGET = 3.0  # seconds; see the module docstring on why this is generous


@pytest.fixture(scope="module")
def soup() -> bm.Mesh:
    """A ``SIDE`` x ``SIDE`` grid of quads -- an ordinary manifold at scale."""
    xs = np.arange(SIDE + 1, dtype="f4")
    positions = np.stack(
        [
            np.repeat(xs, SIDE + 1),
            np.zeros((SIDE + 1) ** 2, dtype="f4"),
            np.tile(xs, SIDE + 1),
        ],
        axis=1,
    )
    i, j = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    i, j = i.reshape(-1), j.reshape(-1)

    def v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * (SIDE + 1) + b

    loops = np.stack(
        [v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)], axis=1
    ).reshape(-1)
    return bm.Mesh(
        positions=positions,
        loops=loops.astype("i4"),
        starts=topo.starts_from_counts(np.full(SIDE * SIDE, 4)),
        material=np.zeros(SIDE * SIDE, dtype="i4"),
        smooth=np.zeros(SIDE * SIDE, dtype=bool),
    )


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_the_fixture_really_is_at_scale(soup: bm.Mesh) -> None:
    assert bm.face_count(soup) == SIDE * SIDE
    assert len(soup.positions) == (SIDE + 1) ** 2


def test_building_the_adjacency_is_one_sort_not_a_loop(soup: bm.Mesh) -> None:
    elapsed = _timed(lambda: adj._build(soup))
    assert elapsed < BUDGET, f"adjacency took {elapsed:.2f}s"


def test_triangulating_stays_on_the_fan_fast_path(soup: bm.Mesh) -> None:
    """A convex mesh must never reach the per-face ear search.

    The concavity screen is one vectorised pass; if it started reporting an
    ordinary grid as suspect, every face would go through a Python ear clip and
    this would take minutes rather than milliseconds.
    """
    from warlock.studio.clay import earclip as ec

    suspect = ec.concave_faces(
        soup.positions, soup.loops, soup.starts, bm._face_normals(soup)
    )
    assert not suspect.any(), "an ordinary grid must not trip the ear search"
    elapsed = _timed(lambda: bm.triangulate(soup))
    assert elapsed < BUDGET, f"triangulate took {elapsed:.2f}s"


def test_subdividing_the_whole_mesh_is_vectorised(soup: bm.Mesh) -> None:
    def run() -> None:
        out, _ = sub.subdivide(soup, el.empty())
        assert bm.face_count(out) == 4 * SIDE * SIDE

    assert _timed(run) < BUDGET * 2, "subdivide is the heaviest of the four"


def test_a_marquee_over_every_vertex_is_two_comparisons(soup: bm.Mesh) -> None:
    eye = m3.vec3(SIDE * 0.5, SIDE * 1.5, SIDE * 0.5)
    view = m3.look_at(eye, m3.vec3(SIDE * 0.5, 0.0, SIDE * 0.5), m3.vec3(0, 0, -1))
    view_proj = m3.perspective(60.0, 4 / 3, 0.1, 10_000.0) @ view
    screen = bp.project(soup.positions, m3.identity(), view_proj, eye, 1600, 1200)
    rect = (0.0, 0.0, 1600.0, 1200.0)

    def run() -> None:
        assert len(bp.marquee_verts(screen, rect)) > 0
        bp.marquee_edges(screen, adj.adjacency(soup).edge_verts, rect)
        bp.marquee_faces(screen, soup.loops, soup.starts, rect)

    assert _timed(run) < BUDGET, "a marquee runs while the mouse is moving"


def test_element_helpers_do_not_walk_face_by_face(soup: bm.Mesh) -> None:
    """``corner_spans`` and ``region_boundary_corners`` run per interaction."""
    faces = np.arange(0, SIDE * SIDE, 2)
    assert _timed(lambda: topo.corner_spans(soup.starts, faces)) < BUDGET
    assert _timed(lambda: topo.region_boundary_corners(soup, faces)) < BUDGET
    # ``affected_verts`` on a face selection is the one deliberate exception --
    # it walks the selected faces in Python -- so it is exercised on a slice
    # rather than on the whole mesh, which is what a real selection is.
    sel = el.ElementSel(faces=faces[:2000])
    assert _timed(lambda: el.affected_verts(soup, sel)) < BUDGET
