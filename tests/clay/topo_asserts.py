"""Topology assertions shared by every op test.

Lifted out of ``test_primitives`` and **re-exported there rather than moved**:
those tests read as a self-contained argument about what "consistently
oriented" means, and rewriting them into imports would cost that. The functions
are the same objects, so a change here changes both.

The pair is the point. :func:`directed_edge_counts` answers *is the winding
consistent* -- neighbouring faces traverse their shared edge in opposite
directions, so no ordered pair occurs twice -- and :func:`edge_use_counts`
answers *is the shell closed*, sorting each pair and therefore blind to winding
by construction. An op that reverses a face passes the second and fails the
first, which is exactly the defect that renders perfectly and exports inside
out.
"""

from __future__ import annotations

from collections import Counter

from warlock.studio.clay import mesh as bm


def directed_edge_counts(mesh: bm.Mesh) -> Counter[tuple[int, int]]:
    """How many faces traverse each *ordered* corner pair."""
    counts: Counter[tuple[int, int]] = Counter()
    for i in range(bm.face_count(mesh)):
        loop = [int(v) for v in bm.face(mesh, i)]
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            counts[(a, b)] += 1
    return counts


def edge_use_counts(mesh: bm.Mesh) -> Counter[tuple[int, int]]:
    """How many faces use each *undirected* edge -- orientation-blind by design.

    This one answers "is the shell closed" and cannot answer anything about
    winding: it sorts each pair, so a reversed face uses exactly the same
    undirected edges it used before. That is why
    :func:`directed_edge_counts` exists beside it.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for (a, b), n in directed_edge_counts(mesh).items():
        counts[(min(a, b), max(a, b))] += n
    return counts


def assert_consistently_oriented(mesh: bm.Mesh) -> None:
    """No ordered corner pair is traversed twice."""
    counts = directed_edge_counts(mesh)
    if counts and max(counts.values()) != 1:
        worst = [pair for pair, n in counts.items() if n > 1]
        raise AssertionError(f"directed edges traversed more than once: {worst[:8]}")


def assert_closed(mesh: bm.Mesh) -> None:
    """Every undirected edge is used by exactly two faces."""
    counts = edge_use_counts(mesh)
    if counts and set(counts.values()) != {2}:
        bad = sorted({n for n in counts.values()} - {2})
        raise AssertionError(f"edge use counts other than 2: {bad}")


def assert_wound_outward(mesh: bm.Mesh) -> None:
    """Six times the enclosed volume is positive -- the shell faces outward."""
    import numpy as np

    total = 0.0
    normals = bm._face_normals(mesh)
    centre = mesh.positions.astype("f8").mean(axis=0)
    for i in range(bm.face_count(mesh)):
        centroid = mesh.positions[bm.face(mesh, i)].astype("f8").mean(axis=0)
        total += float(np.dot(centroid - centre, normals[i]))
    if total <= 0.0:
        raise AssertionError(f"the shell is wound inward: volume sum {total}")
