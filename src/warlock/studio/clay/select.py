"""Selection verbs: loops, rings, linked, grow, shrink, boundary, mirror.

Every one of these is a thing a modeller does dozens of times an hour and none
of them existed. Clay could select an element and add another with Shift, and
that was the whole vocabulary -- so selecting the ring of edges round a cylinder
meant clicking each of them, and selecting one of two objects welded into one
mesh was not possible at all.

**Pure, and over the adjacency.** Nothing here touches a document, a selection
object or a view: each takes a mesh and a set of indices and returns a set of
indices, which is what lets "does a loop stop at a pole" be a plain assertion.
``ops_bevel`` already walks a quad strip for its own purposes and this shares
that walk rather than writing a second one -- an edge ring and a bevel's strip
are the same traversal, and two of them would disagree the first time either
was fixed.

Two conventions travel with the package. **An edge is a vertex pair**, not an
id: ids renumber globally whenever the topology changes and pairs do not, which
is why ``ElementSel`` stores pairs and why everything here returns them.
And **a refusal is empty, never an exception**: these are reached by a keystroke
during a selection, and a key that raises is a key that takes the window down.
"""

from __future__ import annotations

import numpy as np

from .adjacency import Adjacency, adjacency
from .mesh import Mesh


def _arity(mesh: Mesh) -> np.ndarray:
    """How many corners each face has."""
    starts = np.asarray(mesh.starts, dtype="i8")
    return (starts[1:] - starts[:-1]).astype("i8")


def _pairs(edge_ids: np.ndarray, a: Adjacency) -> np.ndarray:
    """Edge ids back to the vertex pairs a selection stores."""
    ids = np.asarray(edge_ids, dtype="i8").reshape(-1)
    if not len(ids):
        return np.zeros((0, 2), dtype="i4")
    ids = np.unique(ids[(ids >= 0) & (ids < a.n_edges)])
    return a.edge_verts[ids].astype("i4")


# --- loops and rings ----------------------------------------------------------


def quad_strip(mesh: Mesh, a: Adjacency, corner: int) -> tuple[list[int], list[int]]:
    """The strip of quads reached from ``corner``. -> ``(faces, edge ids)``.

    The traversal an edge ring and a face loop are both made of, and the one
    ``ops_bevel`` already does for its own cuts: step across a quad to the
    opposite edge, through its twin, and on until a triangle, a boundary or the
    starting face stops it.

    Written here rather than imported from ``ops_bevel`` because that one also
    records the endpoint each cut is measured from -- a bevel needs to know
    which way along an edge it is going and a selection does not -- and a
    function that returns what half its callers throw away is a function two
    callers are reading differently.
    """
    arity = _arity(mesh)
    faces: list[int] = []
    edges: list[int] = []
    seen: set[int] = set()
    at = int(corner)
    while at >= 0:
        face = int(a.corner_face[at])
        if face in seen or arity[face] != 4:
            break
        seen.add(face)
        faces.append(face)
        edges.append(int(a.corner_edge[at]))
        exit_corner = int(a.next_corner[a.next_corner[at]])
        edges.append(int(a.corner_edge[exit_corner]))
        at = int(a.twin[exit_corner])
    return faces, edges


def _corners_of_edge(a: Adjacency, edge: int) -> list[int]:
    """Every corner that leaves along ``edge``. One or two on a sane mesh."""
    return [int(c) for c in np.flatnonzero(a.corner_edge == int(edge))]


def edge_ring(mesh: Mesh, edge: tuple[int, int]) -> np.ndarray:
    """Every edge parallel to ``edge`` across the quad strip. -> vertex pairs.

    The ring, not the loop: a cylinder's ring is the band of edges running
    *round* it, each one the far side of a quad from the last. Both directions
    from the seed, so a seed in the middle of a strip reaches both ends.
    """
    a = adjacency(mesh)
    ids = a.edge_ids(np.asarray([edge], dtype="i4"))
    if not len(ids) or ids[0] < 0:
        return np.zeros((0, 2), dtype="i4")
    found: set[int] = {int(ids[0])}
    for corner in _corners_of_edge(a, int(ids[0])):
        _faces, walked = quad_strip(mesh, a, corner)
        found.update(walked)
    return _pairs(np.fromiter(found, dtype="i8", count=len(found)), a)


def edge_loop(mesh: Mesh, edge: tuple[int, int]) -> np.ndarray:
    """Every edge continuing ``edge`` end to end. -> vertex pairs.

    The loop, not the ring: it runs *along* the seed rather than across it, and
    it is what Alt+click gives in every modelling package.

    The rule at each vertex is the one that makes a loop stop where a modeller
    expects it to. Continue through a vertex of **exactly four edges** to the
    edge opposite the one arrived on; stop at anything else. A pole -- the tip
    of a cone, the centre of a fan -- has some other number, and a loop that
    ran through one would wander off round the mesh.
    """
    a = adjacency(mesh)
    ids = a.edge_ids(np.asarray([edge], dtype="i4"))
    if not len(ids) or ids[0] < 0:
        return np.zeros((0, 2), dtype="i4")
    seed = int(ids[0])
    found: set[int] = {seed}
    for end in (0, 1):
        current = seed
        vertex = int(a.edge_verts[seed][end])
        while True:
            nxt = _opposite_edge(a, vertex, current)
            if nxt is None or nxt in found:
                break
            found.add(nxt)
            pair = a.edge_verts[nxt]
            vertex = int(pair[1]) if int(pair[0]) == vertex else int(pair[0])
            current = nxt
    return _pairs(np.fromiter(found, dtype="i8", count=len(found)), a)


def _opposite_edge(a: Adjacency, vertex: int, edge: int) -> int | None:
    """The edge across ``vertex`` from ``edge``, or None at anything but a
    four-edge vertex. See :func:`edge_loop` for why four."""
    corners = a.vertex_corners(int(vertex))
    around: set[int] = set()
    for corner in corners:
        around.add(int(a.corner_edge[corner]))
        around.add(int(a.corner_edge[a.prev_corner[corner]]))
    if len(around) != 4 or int(edge) not in around:
        return None
    # The two edges of the face the seed is in are its neighbours; the fourth
    # is the one opposite. Two of the four share a face with the seed at this
    # vertex, and the remaining one is the answer.
    neighbours: set[int] = set()
    for corner in corners:
        pair = {int(a.corner_edge[corner]), int(a.corner_edge[a.prev_corner[corner]])}
        if int(edge) in pair:
            neighbours |= pair
    rest = around - neighbours
    return int(next(iter(rest))) if len(rest) == 1 else None


def face_loop(mesh: Mesh, face: int) -> np.ndarray:
    """The strip of faces running through ``face``. -> face indices.

    Both directions, which for a quad means both of its two strips: a face sits
    on two loops at right angles and picking one arbitrarily would make the
    verb's result depend on corner order rather than on anything the user can
    see. Both is the honest answer and is what Blender's face loop gives from a
    face rather than from an edge.
    """
    a = adjacency(mesh)
    face = int(face)
    if not (0 <= face < len(mesh.starts) - 1) or _arity(mesh)[face] != 4:
        return np.zeros(0, dtype="i4")
    found: set[int] = {face}
    start = int(mesh.starts[face])
    for offset in (0, 1):
        for corner in (start + offset, int(a.next_corner[a.next_corner[start + offset]])):
            faces, _edges = quad_strip(mesh, a, corner)
            found.update(faces)
    return np.array(sorted(found), dtype="i4")


# --- everything connected -----------------------------------------------------


def linked(mesh: Mesh, verts: np.ndarray) -> np.ndarray:
    """Every vertex reachable from ``verts`` along edges. -> vertex indices.

    The verb that makes two objects welded into one mesh separable again: L
    over one of them takes the whole shell. Label propagation over the edge
    list rather than a queue -- it is a handful of numpy passes over an array
    that is already built, where a per-vertex walk in Python is not something to
    run on a keystroke over a 200k-vertex import.
    """
    seeds = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    count = len(mesh.positions)
    if not len(seeds) or count == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    inside = np.zeros(count, dtype=bool)
    inside[seeds[(seeds >= 0) & (seeds < count)]] = True
    if a.n_edges == 0:
        return np.flatnonzero(inside).astype("i4")
    lo = a.edge_verts[:, 0].astype("i8")
    hi = a.edge_verts[:, 1].astype("i8")
    while True:
        grown = inside.copy()
        grown[lo[inside[hi]]] = True
        grown[hi[inside[lo]]] = True
        if bool(np.array_equal(grown, inside)):
            break
        inside = grown
    return np.flatnonzero(inside).astype("i4")


# --- more, less, and the edge of it -------------------------------------------


def grow(mesh: Mesh, verts: np.ndarray) -> np.ndarray:
    """One ring outward. -> vertex indices."""
    seeds = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    count = len(mesh.positions)
    if not len(seeds) or count == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    inside = np.zeros(count, dtype=bool)
    inside[seeds[(seeds >= 0) & (seeds < count)]] = True
    if a.n_edges:
        lo = a.edge_verts[:, 0].astype("i8")
        hi = a.edge_verts[:, 1].astype("i8")
        inside[lo[inside[hi]]] = True
        inside[hi[inside[lo]]] = True
    return np.flatnonzero(inside).astype("i4")


def shrink(mesh: Mesh, verts: np.ndarray) -> np.ndarray:
    """One ring inward: every selected vertex whose neighbours are all selected.

    The exact inverse of :func:`grow` only on an infinite lattice, and this is
    the definition that matters rather than the symmetry: shrinking peels the
    *boundary* off a selection, which is what a user reaching for it wants --
    "the middle of what I have", not "whatever grow would undo".
    """
    seeds = np.unique(np.asarray(verts, dtype="i8").reshape(-1))
    count = len(mesh.positions)
    if not len(seeds) or count == 0:
        return np.zeros(0, dtype="i4")
    a = adjacency(mesh)
    inside = np.zeros(count, dtype=bool)
    inside[seeds[(seeds >= 0) & (seeds < count)]] = True
    if a.n_edges == 0:
        return np.flatnonzero(inside).astype("i4")
    lo = a.edge_verts[:, 0].astype("i8")
    hi = a.edge_verts[:, 1].astype("i8")
    # A vertex is peeled when it has a neighbour outside the selection **or**
    # it sits on the mesh's own border.
    #
    # The second half is not a refinement, it is the case the verb is mostly
    # used in: Select Less over a fully selected grid has no unselected
    # neighbour anywhere, so without it the answer is "everything" and the key
    # appears to do nothing. A closed solid has no border and so is unchanged,
    # which is also right -- there is nothing to peel off a cube.
    edge_of = np.zeros(count, dtype=bool)
    edge_of[lo[~inside[hi]]] = True
    edge_of[hi[~inside[lo]]] = True
    border = a.edge_uses == 1
    if bool(border.any()):
        edge_of[lo[border]] = True
        edge_of[hi[border]] = True
    return np.flatnonzero(inside & ~edge_of).astype("i4")


def boundary(mesh: Mesh) -> np.ndarray:
    """Every edge with exactly one face on it. -> vertex pairs.

    The mesh's open border, which is what a hole is and what a Fill Hole is
    about to act on -- so selecting it is how you look at what you are about to
    close.
    """
    a = adjacency(mesh)
    if a.n_edges == 0:
        return np.zeros((0, 2), dtype="i4")
    return _pairs(np.flatnonzero(a.edge_uses == 1), a)


def by_material(mesh: Mesh, slot: int) -> np.ndarray:
    """Every face using palette slot ``slot``. -> face indices.

    The one selection verb that is about the *document* rather than the
    topology, and it earns its place for the reason a material slot exists at
    all: "show me everything painted with this" is how a slot gets reassigned,
    and there was no way to ask.
    """
    material = np.asarray(mesh.material, dtype="i8")
    return np.flatnonzero(material == int(slot)).astype("i4")


# --- symmetry -----------------------------------------------------------------


def mirror_pairs(mesh: Mesh, axis: int = 0, eps: float = 1e-4) -> dict[int, int]:
    """``{vertex: its mirror}`` across the plane ``axis == 0``.

    What X-mirror editing needs and what it can only be as good as: a mesh that
    is not actually symmetric has no pairs to find, and this reports the ones it
    can rather than pretending. A vertex *on* the plane maps to itself, which is
    the case that has to be handled rather than excluded -- those are the ones a
    mirrored drag must slide along the plane instead of moving off it.

    ``eps`` is a distance in the mesh's own units. Bucketed on the rounded
    coordinate rather than compared pairwise, because pairwise is O(V^2) and a
    50k-vertex import would take minutes.
    """
    positions = np.asarray(mesh.positions, dtype="f8")
    if not len(positions):
        return {}
    axis = int(axis)
    quantum = max(float(eps), 1e-9)
    mirrored = positions.copy()
    mirrored[:, axis] *= -1.0
    keys = np.round(positions / quantum).astype("i8")
    wanted = np.round(mirrored / quantum).astype("i8")
    lookup: dict[tuple[int, int, int], int] = {}
    for index, row in enumerate(keys):
        lookup.setdefault((int(row[0]), int(row[1]), int(row[2])), index)
    out: dict[int, int] = {}
    for index, row in enumerate(wanted):
        twin = lookup.get((int(row[0]), int(row[1]), int(row[2])))
        if twin is not None:
            out[index] = int(twin)
    return out
