"""Dissolve: removing an element by merging what it separated.

Delete and dissolve are different operations and the difference is the whole
module. Deleting an edge would have to delete the faces on both sides of it,
leaving a hole; dissolving it merges those two faces into one and leaves the
surface intact. That is what a modeller means by "get rid of this edge", and it
is why every op here is one shape: **find the faces the selection joins, work
out the outline of each connected group, and replace the group with a single
n-gon wound along that outline.**

Three ops, one core. Edge-dissolve groups faces joined by a selected edge;
face-dissolve groups the selected faces themselves; vertex-dissolve groups every
face around a selected vertex. After that they are identical, which is why the
group-to-n-gon step lives in :func:`merge_groups` and not three times over.

**This is where refusing matters.** The core walks a group's border
head-to-tail, and a walk has no defined next step when the border forks or
splits, so rather than guessing it names the element and stops:

* An **annulus** -- a group whose border is two separate rings, which happens
  when the selection encircles a face it does not include. A face with a hole in
  it is not representable in CSR at all, and the alternatives (leave the island,
  bridge it with a slit) are both worse than saying so.
* A **bowtie ring** -- a border that visits one vertex twice. The n-gon would
  self-intersect.
* A **boundary edge**, for edge-dissolve: there is only one face there, so
  there is nothing to merge it with.
* A **non-manifold edge**, anywhere: three faces meet, so "the other side" is
  not a single face.
* A **boundary or non-manifold vertex**, for vertex-dissolve: the fan around it
  does not close, so it has no ring.

**Results are routinely concave**, and that is the first real consumer of
:mod:`.earclip` -- dissolving the edge between two triangles of an L gives a
polygon a fan would triangulate outside itself.

**A stated limitation:** the two ends of a dissolved edge stay in the merged
n-gon as two-valence collinear corners. They are harmless (the Newell normal is
stable across them, and :mod:`.earclip` tolerates them), they keep the vertex
count honest about what the user removed, and removing them would be a separate
"dissolve vertices" pass the user has a control for.

UV **preserved**: a border corner keeps its own uv and an interior corner is
dropped along with the geometry it described.
"""

from __future__ import annotations

import numpy as np

from . import topo
from .adjacency import adjacency
from .elements import ElementSel, OpError
from .mesh import Mesh, face_count

__all__ = ["dissolve_edges", "dissolve_faces", "dissolve_verts", "merge_groups"]


class _Union:
    """Union-find over face indices, kept tiny and local on purpose."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


#: The largest outline a dissolve will produce. ``ops_subdiv`` refuses past
#: ``MAX_SUBDIVIDED_FACES`` and states why; this is the same argument for the
#: other unbounded growth an edit can ask for. The n-gon a dissolve makes is
#: handed to ``earclip``, whose ear search is worst-case quadratic in the
#: corner count, in Python, one triangle removed per scan of the remainder --
#: and ``clay_ops.run_mesh_op`` calls it synchronously from the key handler,
#: which is the frame thread. Past this the app stops responding rather than
#: becoming slower, which is the outcome every ceiling in this package exists
#: to prevent.
#:
#: Twenty thousand keeps that search well under a second. No hand-made
#: selection approaches it: a dissolve of a whole subdivided face loop on a
#: dense mesh is a few thousand corners.
MAX_DISSOLVED_RING = 20_000


def _refuse_ring(rings: list[np.ndarray]) -> None:
    """Refuse before the merge when the outline is too big to triangulate."""
    worst = max((len(r) for r in rings), default=0)
    if worst > MAX_DISSOLVED_RING:
        raise OpError(
            f"That merge would make a face with {worst:,} corners, past the "
            f"{MAX_DISSOLVED_RING:,} Clay can triangulate without stalling. "
            "Dissolve a smaller region."
        )


def _ring_corners(mesh: Mesh, group: np.ndarray) -> np.ndarray:
    """The group's outline as an ordered array of corner indices.

    Ordered by chaining the border's directed edges head to tail, which is what
    makes the resulting n-gon wound consistently with everything still around
    it: each border corner reads ``a -> b`` because its own face does, and the
    merged face inherits exactly those traversals.
    """
    border = topo.region_boundary_corners(mesh, group)
    if len(border) == 0:
        raise OpError(
            "Those faces make up a closed surface on their own, so there is "
            "nothing left to merge them into."
        )
    a = adjacency(mesh)
    heads = mesh.loops[border].astype("i8")
    succ: dict[int, int] = {}
    for corner, head in zip(border.tolist(), heads.tolist(), strict=True):
        if head in succ:
            raise OpError(
                f"Vertex {head} appears twice on the outline of that selection, "
                "so it cannot be merged into one face. Dissolve a smaller "
                "region."
            )
        succ[head] = corner

    start = int(border[0])
    ring = [start]
    cursor = int(mesh.loops[a.next_corner[start]])
    while cursor != int(mesh.loops[start]):
        nxt = succ.get(cursor)
        if nxt is None:  # pragma: no cover - a fork is caught above
            raise OpError(
                f"The outline of that selection breaks at vertex {cursor}, so it "
                "cannot be merged into one face. Dissolve a smaller region."
            )
        ring.append(nxt)
        cursor = int(mesh.loops[a.next_corner[nxt]])

    if len(ring) != len(border):
        raise OpError(
            "That selection surrounds a face it does not include, so merging it "
            "would need a face with a hole in it. Include the middle, or "
            "dissolve a smaller region."
        )
    return np.array(ring, dtype="i8")


def merge_groups(mesh: Mesh, groups: list[np.ndarray]) -> tuple[Mesh, ElementSel]:
    """Replace each group of faces with one n-gon along its outline.

    Groups of a single face are dropped rather than rebuilt: there is nothing to
    merge, and rebuilding would move the face to the end of the list for no
    reason.

    Refused past :data:`MAX_DISSOLVED_RING`, for the reason ``ops_subdiv``
    refuses past ``MAX_SUBDIVIDED_FACES``.
    """
    real = [np.asarray(g, dtype="i8") for g in groups if len(g) > 1]
    if not real:
        raise OpError(
            "Nothing there to dissolve: a dissolve merges neighbours, so it "
            "needs at least two of them touching."
        )

    rings = [_ring_corners(mesh, g) for g in real]
    _refuse_ring(rings)
    consumed = np.concatenate(real)
    keep = np.ones(face_count(mesh), dtype=bool)
    keep[consumed] = False
    kept = np.flatnonzero(keep)

    counts = np.array([len(r) for r in rings], dtype="i8")
    corners = np.concatenate(rings)
    base = topo.take_faces(mesh, kept)

    out, _ = topo.compact_vertices(
        topo.rebuild(
            mesh.positions,
            np.concatenate([base.loops.astype("i8"), mesh.loops[corners].astype("i8")]),
            np.concatenate([base.starts.astype("i8"), int(base.starts[-1]) + np.cumsum(counts)]),
            np.concatenate([base.material, [mesh.material[g[0]] for g in real]]),
            np.concatenate([base.smooth, [mesh.smooth[g[0]] for g in real]]),
            uv=None if mesh.uv is None else np.concatenate([base.uv, mesh.uv[corners]]),
        )
    )
    n_kept = len(kept)
    return out, ElementSel(faces=np.arange(n_kept, n_kept + len(rings)))


def _check_edges(mesh: Mesh, edges: np.ndarray) -> np.ndarray:
    a = adjacency(mesh)
    ids = a.edge_ids(edges)
    if (ids < 0).any():
        raise OpError("That edge is not part of this mesh.")
    uses = a.edge_uses[ids]
    if (uses == 1).any():
        bad = edges[uses == 1][0]
        raise OpError(
            f"Edge {int(bad[0])}-{int(bad[1])} is on a boundary, so there is only "
            "one face there and nothing to merge it with."
        )
    if (uses >= 3).any():
        bad = edges[uses >= 3][0]
        raise OpError(
            f"Edge {int(bad[0])}-{int(bad[1])} has {int(uses[uses >= 3][0])} faces on "
            "it, so there is no single face on the other side. Fix the "
            "non-manifold edge first."
        )
    return ids


def dissolve_edges(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Merge the pair of faces across each selected edge."""
    if len(sel.edges) == 0:
        raise OpError("Select an edge to dissolve.")
    ids = _check_edges(mesh, sel.edges)

    a = adjacency(mesh)
    union = _Union(face_count(mesh))
    # The corner list is sorted by edge **once** and each selected edge's pair
    # of faces is found by bisection. It used to be ``corner_face[corner_edge
    # == e]`` inside the loop -- a full scan of every corner in the mesh per
    # selected edge -- so dissolving a loop of 400 edges on a 200k-corner
    # sculpt was 80 million comparisons for an answer one sort already holds.
    order = np.argsort(a.corner_edge, kind="stable")
    by_edge = a.corner_edge[order]
    faces_by_edge = a.corner_face[order]
    lo = np.searchsorted(by_edge, ids, side="left")
    hi = np.searchsorted(by_edge, ids, side="right")
    for start, stop in zip(lo.tolist(), hi.tolist(), strict=True):
        # ``_check_edges`` has already refused anything but a manifold pair, so
        # the slice is exactly two.
        union.union(int(faces_by_edge[start]), int(faces_by_edge[stop - 1]))
    return merge_groups(mesh, [np.array(g) for g in union.groups()])


def dissolve_faces(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Merge each connected block of selected faces into one face."""
    if len(sel.faces) == 0:
        raise OpError("Select the faces to dissolve into one.")
    a = adjacency(mesh)
    chosen = np.zeros(face_count(mesh), dtype=bool)
    chosen[sel.faces] = True

    union = _Union(face_count(mesh))
    interior = np.flatnonzero(chosen[a.corner_face] & (a.twin >= 0))
    for corner in interior.tolist():
        other = int(a.corner_face[a.twin[corner]])
        if chosen[other]:
            union.union(int(a.corner_face[corner]), other)

    groups = [np.array(g) for g in union.groups() if chosen[g[0]]]
    return merge_groups(mesh, groups)


def dissolve_verts(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Merge the fan of faces around each selected vertex into one face."""
    if len(sel.verts) == 0:
        raise OpError("Select a vertex to dissolve.")
    a = adjacency(mesh)
    union = _Union(face_count(mesh))
    for v in sel.verts.astype("i8").tolist():
        if v >= len(mesh.positions):
            raise OpError(f"Vertex {v} is not part of this mesh.")
        corners = a.vertex_corners(v)
        if len(corners) == 0:
            raise OpError(f"Vertex {v} belongs to no face.")
        uses = a.edge_uses[a.corner_edge[corners]]
        incoming = a.edge_uses[a.corner_edge[a.prev_corner[corners]]]
        touch = np.concatenate([uses, incoming])
        if (touch == 1).any():
            raise OpError(
                f"Vertex {v} is on a boundary, so the faces around it do not "
                "close into a ring. Fill the hole first, or delete the vertex."
            )
        if (touch >= 3).any():
            raise OpError(
                f"Vertex {v} sits on a non-manifold edge, so the faces around it "
                "have no single order. Fix that edge first."
            )
        faces = a.corner_face[corners].astype("i8")
        for f in faces[1:].tolist():
            union.union(int(faces[0]), f)

    touched = np.zeros(face_count(mesh), dtype=bool)
    touched[a.corner_face[np.concatenate([a.vertex_corners(int(v)) for v in sel.verts])]] = True
    groups = [np.array(g) for g in union.groups() if touched[g[0]]]
    return merge_groups(mesh, groups)
