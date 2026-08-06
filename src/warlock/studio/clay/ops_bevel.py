"""Loop cut and bevel: the two ops that walk the surface before they touch it.

Both start from a seed edge and follow the mesh outward -- the loop cut along a
quad strip, the bevel around each beveled vertex's fan -- so both are the ops
that :mod:`.adjacency`'s conservative ``twin`` was built for. A ``twin`` of −1
on a boundary, a non-manifold edge or a flipped pair stops the walk, which is
exactly what these want: continuing across one would produce a ring that
doubles back or a fan that never closes, and neither failure is visible until
the geometry is exported.

**Loop cut is walk-oriented, and that is the subtle part.** A ring of edges
around a cylinder has no global "left"; each edge only knows its two endpoints,
in whatever order the vertex numbering happens to give. Cutting at ``t = 0.25``
per edge with no shared orientation puts the new vertex a quarter of the way
along from a *different* end on roughly half the ring, and the cut zigzags. So
the walk records, per cut edge, which endpoint ``t`` is measured from -- carried
across each quad from its entry edge to the far side of the same two corners --
and every later step converts to that convention rather than assuming one.
"""

from __future__ import annotations

import numpy as np

from . import topo
from .adjacency import Adjacency, adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh, _face_normals, face_count

__all__ = ["bevel_edges", "loop_cut"]


def _arity(mesh: Mesh) -> np.ndarray:
    return np.diff(mesh.starts.astype("i8"))


def _walk(
    mesh: Mesh, a: Adjacency, start: int, flip: bool
) -> tuple[list[int], list[tuple[int, int]], bool]:
    """Follow the quad strip from entry corner *start*.

    Returns ``(faces, oriented_edges, closed)``. Each ``oriented_edges`` entry
    is ``(edge_id, from_vertex)`` -- the endpoint the cut parameter is measured
    from, which is what keeps the ring straight (see the module docstring).
    ``flip`` reverses that convention for the backward half of the walk, whose
    entry corner points the other way along the seed edge by construction.
    """
    arity = _arity(mesh)
    faces: list[int] = []
    edges: list[tuple[int, int]] = []
    seen: set[int] = set()
    corner = int(start)
    closed = False

    while corner >= 0:
        face = int(a.corner_face[corner])
        if face in seen:
            closed = True
            break
        if arity[face] != 4:
            break
        seen.add(face)
        faces.append(face)

        head = int(mesh.loops[corner])
        tail = int(mesh.loops[a.next_corner[corner]])
        edges.append((int(a.corner_edge[corner]), tail if flip else head))

        exit_corner = int(a.next_corner[a.next_corner[corner]])
        exit_head = int(mesh.loops[exit_corner])
        exit_tail = int(mesh.loops[a.next_corner[exit_corner]])
        # The exit edge is recorded here rather than left to the next face's
        # entry: the walk can stop *after* crossing into a triangle, and the
        # quad we just left would then be split against an edge nothing cut.
        edges.append((int(a.corner_edge[exit_corner]), exit_head if flip else exit_tail))
        if a.twin[exit_corner] < 0:
            break
        corner = int(a.twin[exit_corner])

    return faces, edges, closed


def loop_cut(mesh: Mesh, sel: ElementSel, *, t: float = 0.5) -> tuple[Mesh, ElementSel]:
    """Ring a quad strip with a new edge loop, cut at *t* along each rung.

    The strip is walked in both directions from the selected edge and stops at
    the first thing that is not a quad crossed by exactly two faces: a triangle
    or an n-gon, a boundary, or a non-manifold edge. Stopping is the whole
    behaviour -- a partial loop across the quads that *are* there is what a user
    modelling a hard-surface asset wants, and it is far better than a ring that
    wanders off through a cap.

    **Crossed quads split in two; the faces at the open ends are spliced.** The
    end face is not part of the strip and keeps its shape, but the new vertex
    lands on one of its edges, so it takes that vertex into its loop and becomes
    an n-gon -- no T-junction, exactly as in :mod:`.ops_subdiv`.

    UV **interpolated** per face: each new corner's uv is a lerp between the two
    corner uvs of the edge it sits on, within the face that owns them, so a cut
    across a seam leaves the seam where it was.

    Selection out: the new ring, as edges.
    """
    if len(sel.edges) != 1:
        raise OpError("Select exactly one edge to cut a loop through.")
    a = adjacency(mesh)
    seed = int(a.edge_ids(sel.edges)[0])
    if seed < 0:
        raise OpError("That edge is not part of this mesh.")
    if a.edge_uses[seed] != 2:
        raise OpError(
            "A loop cut runs between two faces, and that edge has "
            f"{int(a.edge_uses[seed])}."
        )
    corners = np.flatnonzero(a.corner_edge == seed)
    if (_arity(mesh)[a.corner_face[corners]] != 4).all():
        raise OpError("A loop cut runs through quads, and neither face there is one.")

    start = int(corners[0])
    forward_faces, forward_edges, closed = _walk(mesh, a, start, flip=False)
    if not forward_faces:
        start = int(corners[1])
        forward_faces, forward_edges, closed = _walk(mesh, a, start, flip=False)
    back_faces: list[int] = []
    back_edges: list[tuple[int, int]] = []
    if not closed and a.twin[start] >= 0:
        # The other side of the seed edge is a face the forward walk never
        # visits, so the backward half keeps its first face; only the seed edge
        # itself is a duplicate, and ``setdefault`` below drops it.
        back_faces, back_edges, _ = _walk(mesh, a, int(a.twin[start]), flip=True)

    strip = np.array(list(dict.fromkeys(forward_faces + back_faces)), dtype="i8")
    ordered: dict[int, int] = {}
    for edge, origin in forward_edges + back_edges:
        ordered.setdefault(edge, origin)
    cut_edges = np.array(sorted(ordered), dtype="i8")

    # New vertex per cut edge, at ``t`` from the endpoint the walk chose.
    n_verts = len(mesh.positions)
    new_index = np.full(a.n_edges, -1, dtype="i8")
    new_index[cut_edges] = n_verts + np.arange(len(cut_edges), dtype="i8")
    origin = np.zeros(a.n_edges, dtype="i8")
    for edge, from_vertex in ordered.items():
        origin[edge] = from_vertex

    other = np.where(
        a.edge_verts[cut_edges, 0] == origin[cut_edges],
        a.edge_verts[cut_edges, 1],
        a.edge_verts[cut_edges, 0],
    )
    lo = mesh.positions[origin[cut_edges]].astype("f8")
    hi = mesh.positions[other].astype("f8")
    positions = np.concatenate([mesh.positions.astype("f8"), lo + (hi - lo) * float(t)])

    def corner_t(corner: np.ndarray) -> np.ndarray:
        """``t`` as measured from *this* corner's own head vertex."""
        edge = a.corner_edge[corner]
        same = mesh.loops[corner] == origin[edge]
        return np.where(same, float(t), 1.0 - float(t))[:, None]

    # Split each crossed quad into two.
    entry = _entry_corners(mesh, a, strip, new_index)
    nxt = a.next_corner[entry].astype("i8")
    nxt2 = a.next_corner[nxt].astype("i8")
    nxt3 = a.next_corner[nxt2].astype("i8")
    v0, v1 = mesh.loops[entry].astype("i8"), mesh.loops[nxt].astype("i8")
    v2, v3 = mesh.loops[nxt2].astype("i8"), mesh.loops[nxt3].astype("i8")
    m_in = new_index[a.corner_edge[entry]]
    m_out = new_index[a.corner_edge[nxt2]]
    halves = np.stack(
        [
            np.stack([v0, m_in, m_out, v3], axis=1),
            np.stack([m_in, v1, v2, m_out], axis=1),
        ],
        axis=1,
    ).reshape(-1)

    child_uv = None
    if mesh.uv is not None:
        uv = mesh.uv.astype("f8")
        uv_in = uv[entry] + (uv[nxt] - uv[entry]) * corner_t(entry)
        uv_out = uv[nxt2] + (uv[nxt3] - uv[nxt2]) * corner_t(nxt2)
        child_uv = np.stack(
            [
                np.stack([uv[entry], uv_in, uv_out, uv[nxt3]], axis=1),
                np.stack([uv_in, uv[nxt], uv[nxt2], uv_out], axis=1),
            ],
            axis=1,
        ).reshape(-1, 2)

    # Splice each new vertex into the face at an open end of the strip.
    in_strip = np.zeros(face_count(mesh), dtype=bool)
    in_strip[strip] = True
    outside = np.flatnonzero(~in_strip[a.corner_face] & (new_index[a.corner_edge] >= 0))
    value_uv = None
    if mesh.uv is not None and len(outside):
        uv = mesh.uv.astype("f8")
        value_uv = uv[outside] + (uv[a.next_corner[outside]] - uv[outside]) * corner_t(
            outside
        )
    loops2, starts2, uv2 = topo.splice_corners(
        mesh.loops,
        mesh.starts,
        mesh.uv,
        after=outside,
        values=new_index[a.corner_edge[outside]],
        counts=np.ones(len(outside), dtype="i8"),
        value_uv=value_uv,
    )
    base = topo.take_faces(
        topo.rebuild(positions, loops2, starts2, mesh.material, mesh.smooth, uv=uv2),
        np.flatnonzero(~in_strip),
    )

    n_halves = 2 * len(entry)
    out = topo.rebuild(
        positions,
        np.concatenate([base.loops.astype("i8"), halves]),
        np.concatenate(
            [
                base.starts.astype("i8"),
                int(base.starts[-1]) + 4 * np.arange(1, n_halves + 1, dtype="i8"),
            ]
        ),
        np.concatenate([base.material, np.repeat(mesh.material[strip], 2)]),
        np.concatenate([base.smooth, np.repeat(mesh.smooth[strip], 2)]),
        uv=None if child_uv is None else np.concatenate([base.uv, child_uv]),
    )
    ring = np.stack([m_in, m_out], axis=1)
    return out, ElementSel(edges=ring)


def _entry_corners(
    mesh: Mesh, a: Adjacency, strip: np.ndarray, new_index: np.ndarray
) -> np.ndarray:
    """One corner per strip face: the one whose edge the cut enters through.

    A quad crossed by the loop has exactly two cut edges, opposite each other,
    and either could be called the entry -- so the lower corner index is taken,
    deterministically, and the split reads from there.
    """
    out = []
    for face in strip.tolist():
        span = np.arange(mesh.starts[face], mesh.starts[face + 1], dtype="i8")
        cut = span[new_index[a.corner_edge[span]] >= 0]
        out.append(int(cut[0]))
    return np.array(out, dtype="i8")


# --- bevel ------------------------------------------------------------------

# Below this, two edges at a corner count as collinear and the in-face bisector
# is undefined -- the miter distance formula divides by sin(theta/2), which is 1
# there, so only the *direction* needs the fallback.
_BISECTOR_EPS = 1e-9


def bevel_edges(
    mesh: Mesh, sel: ElementSel, *, width: float = 0.05
) -> tuple[Mesh, ElementSel]:
    """Replace each selected edge with a flat quad, rebuilding the corners.

    **Slide-vertex fan reconstruction.** Every vertex touched by a beveled edge
    is replaced, in each face around it, by one or two new vertices, and which
    ones is a three-row table over the corner's own two edges:

    ===================  ======================================================
    corner's two edges   what replaces the corner
    ===================  ======================================================
    neither beveled      two **slide** vertices, one along each edge
    exactly one          one **slide** vertex, along the *unbeveled* edge
    both beveled         one **miter** vertex, along the in-face bisector
    ===================  ======================================================

    A slide vertex is keyed on ``(vertex, edge)`` and therefore **shared by both
    faces flanking that edge**. That sharing is the whole crack-freeness
    argument: two faces that push their corners back along the same edge push
    them back to the same point, by construction rather than by both computing
    the same number and hoping the floats agree.

    A miter vertex is per corner, at ``width / sin(theta/2)`` along the bisector
    -- the distance at which two bevel quads meeting at that corner have their
    edges exactly ``width`` from the original ones. It is clamped to half the
    shorter incident edge so a wide bevel on a small feature collapses rather
    than turning inside out.

    Then a **bevel quad per edge**, wound from the traversal of the face that
    reads the edge forwards, and a **vertex polygon** wherever three or more
    distinct new vertices land on one original vertex -- which is exactly the
    textbook result: bevel one edge of a cube and the opposite face becomes a
    pentagon with no polygon at the corner (only two new vertices there); bevel
    all three edges at a corner and you get a miter triangle; bevel all twelve
    and the cube becomes 6 + 12 + 8 = 26 faces.

    **Refusals**: a boundary or non-manifold edge (there is no second face to
    wind the quad against), a boundary vertex carrying two or more beveled edges
    (its fan does not close, so the polygon has no ring), and a fan whose ring
    forks -- all named, all with a reason.

    **Stated limits**, none of them bugs: a single segment (no rounded profile);
    the width clamped per edge rather than solved globally, so a bevel wider
    than a feature flattens it instead of self-intersecting; the miter along the
    in-face bisector rather than the true intersection of the two offset planes,
    which drifts on a strongly non-planar corner; and no self-intersection guard
    at all.

    UV: slides **interpolated** along their edge within each face, miters keep
    their corner's uv, and the new quads and polygons copy the uv of the source
    corner each of their vertices came from -- flat in UV, and said out loud
    rather than dressed up as a projection.
    """
    if len(sel.edges) == 0:
        raise OpError("Select an edge to bevel.")
    a = adjacency(mesh)
    ids = a.edge_ids(sel.edges)
    if (ids < 0).any():
        raise OpError("That edge is not part of this mesh.")
    bad = a.edge_uses[ids] != 2
    if bad.any():
        pair = sel.edges[bad][0]
        uses = int(a.edge_uses[ids[bad][0]])
        why = "is on a boundary" if uses == 1 else f"has {uses} faces on it"
        raise OpError(
            f"Edge {int(pair[0])}-{int(pair[1])} {why}, so there is no second "
            "face to wind a bevel against."
        )

    if (a.twin[np.isin(a.corner_edge, ids)] < 0).any():
        raise OpError(
            "One of those edges has its two faces wound against each other, so "
            "a bevel there would inherit the flip. Fix the normals first."
        )

    beveled = set(ids.tolist())
    positions = mesh.positions.astype("f8")
    loops = mesh.loops.astype("i8")
    corner_uv = None if mesh.uv is None else mesh.uv.astype("f8")
    _check_bevel_vertices(mesh, a, beveled)

    minted: list[np.ndarray] = []
    n_verts = len(positions)

    def mint(point: np.ndarray) -> int:
        minted.append(point)
        return n_verts + len(minted) - 1

    slides: dict[tuple[int, int], tuple[int, float]] = {}

    def slide(vertex: int, edge: int) -> tuple[int, float]:
        key = (vertex, edge)
        if key not in slides:
            ends = a.edge_verts[edge]
            far = int(ends[1]) if int(ends[0]) == vertex else int(ends[0])
            delta = positions[far] - positions[vertex]
            length = float(np.linalg.norm(delta))
            s = 0.5 if length == 0.0 else min(abs(float(width)) / length, 0.5)
            slides[key] = (mint(positions[vertex] + delta * s), s)
        return slides[key]

    miters: dict[int, int] = {}

    def miter(corner: int) -> int:
        if corner not in miters:
            here = int(loops[corner])
            back = positions[int(loops[a.prev_corner[corner]])] - positions[here]
            fore = positions[int(loops[a.next_corner[corner]])] - positions[here]
            n_back, n_fore = float(np.linalg.norm(back)), float(np.linalg.norm(fore))
            d1 = back / n_back if n_back else back
            d2 = fore / n_fore if n_fore else fore
            bisector = d1 + d2
            if float(np.linalg.norm(bisector)) < _BISECTOR_EPS:
                # A straight-through corner: the bisector is degenerate, so take
                # the in-face perpendicular instead of dividing by zero.
                normal = _face_normals(mesh)[int(a.corner_face[corner])]
                bisector = np.cross(normal.astype("f8"), d2)
            bisector = bisector / max(float(np.linalg.norm(bisector)), _BISECTOR_EPS)
            half = np.arccos(np.clip(float(d1 @ d2), -1.0, 1.0)) * 0.5
            reach = abs(float(width)) / max(float(np.sin(half)), 1e-6)
            limit = 0.5 * min(n_back or reach, n_fore or reach)
            miters[corner] = mint(positions[here] + bisector * min(reach, limit))
        return miters[corner]

    touched = set(a.edge_verts[ids].reshape(-1).tolist())

    def replace(corner: int) -> list[tuple[int, np.ndarray | None]]:
        """The new corners replacing this one, each with its uv."""
        here = int(loops[corner])
        out_edge = int(a.corner_edge[corner])
        in_edge = int(a.corner_edge[a.prev_corner[corner]])
        out_bev, in_bev = out_edge in beveled, in_edge in beveled
        if out_bev and in_bev:
            return [(miter(corner), None if corner_uv is None else corner_uv[corner])]

        def one(edge: int, toward: int) -> tuple[int, np.ndarray | None]:
            index, s = slide(here, edge)
            if corner_uv is None:
                return index, None
            base = corner_uv[corner]
            return index, base + (corner_uv[toward] - base) * s

        back = int(a.prev_corner[corner])
        fore = int(a.next_corner[corner])
        if out_bev:
            return [one(in_edge, back)]
        if in_bev:
            return [one(out_edge, fore)]
        return [one(in_edge, back), one(out_edge, fore)]

    # --- rewrite every face -------------------------------------------------
    new_loops: list[int] = []
    new_uv: list[np.ndarray] = []
    counts: list[int] = []
    starts = mesh.starts.astype("i8")
    for face in range(face_count(mesh)):
        size = 0
        for corner in range(int(starts[face]), int(starts[face + 1])):
            if int(loops[corner]) not in touched:
                new_loops.append(int(loops[corner]))
                if corner_uv is not None:
                    new_uv.append(corner_uv[corner])
                size += 1
                continue
            for index, uv_row in replace(corner):
                new_loops.append(index)
                if corner_uv is not None:
                    new_uv.append(uv_row)
                size += 1
        counts.append(size)

    # --- one quad per beveled edge, and the darts the fans chain on ---------
    darts: dict[int, list[tuple[int, int, int]]] = {}

    def dart(vertex: int, tail: int, head: int, source: int) -> None:
        darts.setdefault(vertex, []).append((tail, head, source))

    for corner in range(len(loops)):
        edge = int(a.corner_edge[corner])
        if edge not in beveled or int(loops[corner]) > int(loops[a.next_corner[corner]]):
            continue
        c1, c2 = corner, int(a.twin[corner])
        u, v = int(loops[c1]), int(loops[c2])
        a_u = replace(c1)[0]
        a_v = replace(int(a.next_corner[c1]))[0]
        b_v = replace(c2)[0]
        b_u = replace(int(a.next_corner[c2]))[0]
        for index, uv_row in (a_v, a_u, b_u, b_v):
            new_loops.append(index)
            if corner_uv is not None:
                new_uv.append(uv_row)
        counts.append(4)
        dart(v, a_v[0], b_v[0], int(a.next_corner[c1]))
        dart(u, b_u[0], a_u[0], int(a.next_corner[c2]))

    for vertex in sorted(touched):
        for corner in a.vertex_corners(vertex).tolist():
            pair = replace(int(corner))
            if len(pair) == 2:
                dart(vertex, pair[1][0], pair[0][0], int(corner))
        ring = _chain_darts(vertex, darts.get(vertex, []))
        if ring is None:
            continue
        for index, source in ring:
            new_loops.append(index)
            if corner_uv is not None:
                new_uv.append(corner_uv[source])
        counts.append(len(ring))

    n_faces = face_count(mesh)
    n_new = len(counts) - n_faces
    out, _ = topo.compact_vertices(
        topo.rebuild(
            np.concatenate([positions, np.array(minted).reshape(-1, 3)]),
            np.array(new_loops, dtype="i8"),
            topo.starts_from_counts(counts),
            np.concatenate([mesh.material, np.zeros(n_new, dtype="i4")]),
            np.concatenate([mesh.smooth, np.zeros(n_new, dtype=bool)]),
            uv=None if corner_uv is None else np.array(new_uv).reshape(-1, 2),
        )
    )
    return out, ElementSel(faces=np.arange(n_faces, n_faces + n_new))


def _check_bevel_vertices(mesh: Mesh, a: Adjacency, beveled: set[int]) -> None:
    """Refuse a boundary vertex carrying two or more beveled edges.

    One beveled edge ending on an open border is fine -- the two slide vertices
    it makes are all that vertex needs. Two of them want a polygon between them,
    and the fan they would have to close around does not close.
    """
    for vertex in sorted(set(a.edge_verts[sorted(beveled)].reshape(-1).tolist())):
        corners = a.vertex_corners(vertex)
        incident = set(a.corner_edge[corners].tolist()) | set(
            a.corner_edge[a.prev_corner[corners]].tolist()
        )
        if len(incident & beveled) < 2:
            continue
        open_edge = [e for e in sorted(incident) if a.edge_uses[e] != 2]
        if open_edge:
            pair = a.edge_verts[open_edge[0]]
            raise OpError(
                f"Vertex {vertex} is on the border (at edge {int(pair[0])}-"
                f"{int(pair[1])}) and carries more than one beveled edge, so the "
                "corner there cannot be closed. Bevel one of them, or fill the "
                "hole first."
            )


def _chain_darts(
    vertex: int, darts: list[tuple[int, int, int]]
) -> list[tuple[int, int]] | None:
    """Chain a vertex's darts into a ring, or ``None`` when there is no polygon.

    Fewer than three distinct new vertices means the corner closed itself --
    two slide vertices and a bevel quad between them need nothing more -- and
    that is the common case, not an error.
    """
    if len({tail for tail, _, _ in darts}) < 3:
        return None
    successor: dict[int, tuple[int, int]] = {}
    for tail, head, source in darts:
        if tail in successor:
            raise OpError(
                f"The corner at vertex {vertex} folds back on itself, so it "
                "cannot be closed with one face. Bevel fewer edges there."
            )
        successor[tail] = (head, source)

    start = darts[0][0]
    ring: list[tuple[int, int]] = []
    cursor = start
    while True:
        step = successor.get(cursor)
        if step is None:
            raise OpError(
                f"The faces around vertex {vertex} do not close into a ring, so "
                "the bevel there has no corner to fill. Fix the surface first."
            )
        ring.append((cursor, step[1]))
        cursor = step[0]
        if cursor == start:
            break
        if len(ring) > len(successor):  # pragma: no cover - the fork check covers it
            raise OpError(f"The corner at vertex {vertex} could not be closed.")
    if len(ring) != len(successor):
        raise OpError(
            f"The faces around vertex {vertex} form more than one ring, so the "
            "bevel there cannot be capped with a single face."
        )
    return ring
