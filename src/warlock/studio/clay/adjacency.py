"""Derived connectivity over the CSR mesh: a half-edge table without the storage.

CSR stays canonical. Nothing here is stored on a :class:`~.mesh.Mesh` and
nothing here is a winged-edge structure -- every table is *derived* from
``loops``/``starts`` and cached against the mesh that produced it, so a mesh
remains the small pair of arrays it has always been and an op that returns a
new mesh never has to maintain a second representation in step with the first.

**The half-edge here is a corner.** Entry *c* of ``loops`` is both "the *c*-th
face corner" and "the directed edge leaving that corner towards the next corner
of its face", and those are the same thing: a face's corners in order *are* its
directed edges in order. So the whole table is indexed by corner, which is
already the index everything else in Clay uses -- ``uv`` is per corner,
:func:`~.mesh.render_arrays` splits per corner, and an op that rewrites a face
rewrites a span of corners. There is no separate half-edge id to keep in step.

**One sort builds all of it.** The undirected edge list, the per-corner edge id,
the use counts, the twin table and the flipped-pair report all fall out of a
single ``np.unique`` over the ``(L, 2)`` array of sorted vertex pairs, which is
O(L log L) once rather than a Python loop over faces. That matters at the scale
this module exists for: an imported ``model.glb`` is hundreds of thousands of
corners and every element-mode interaction touches this table.

**A cached value must not hold its key.** :class:`Adjacency` stores arrays and
nothing else -- no ``Mesh`` reference, not even an index back into one --
because the cache is a :class:`weakref.WeakKeyDictionary` and a value that
referenced its own key would keep the key alive forever, which is a leak dressed
up as a cache. The dictionary is keyed by the mesh *object*, never by ``id()``:
an ``id()``-keyed dict is a use-after-free waiting to happen, since CPython
reuses addresses and the next mesh allocated at that address would silently
inherit the dead one's adjacency.

``twin`` is deliberately conservative. It is −1 not only on a boundary but also
on a non-manifold edge (three or more faces: there is no single opposite) *and*
on a consistently-flipped pair (two faces traversing their shared edge in the
same direction, which means one of them is wound inside out). Every walking op
-- loop cut, dissolve, bevel -- steps through ``twin``, and a walk that crossed
a flipped edge would silently produce geometry with a reversed patch in it. A
−1 stops the walk, which is the outcome those ops want; the pairs themselves are
reported in ``flipped_pairs`` so :func:`check_manifold` can name them.

:func:`check_manifold` is a **report, not a gate**. Nothing here refuses a mesh:
importing a real-world GLB routinely yields flipped triangles, doubled faces and
stray vertices, and an editor that refused to open one would be useless exactly
when it is most needed. Ops read the ``Adjacency`` flags directly and decide for
themselves -- the local ones proceed best-effort, the walking ones refuse with a
message naming the element -- while the report exists for the UI and the tests,
which want the whole picture rather than one op's opinion of it.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np

from .mesh import Mesh, triangulate

__all__ = [
    "Adjacency",
    "ManifoldReport",
    "adjacency",
    "boundary_loops",
    "cached_triangulation",
    "check_manifold",
]


def _freeze(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.setflags(write=False)


@dataclass(frozen=True, eq=False)
class Adjacency:
    """Derived connectivity for one mesh. Every array is read-only.

    Corner-indexed tables are ``(L,)``; edge-indexed ones are ``(E,)`` over
    :attr:`edge_verts`, which is lexsorted-unique with the low vertex first --
    the same canonical form :func:`~.mesh.edges` returns, so an edge id here is
    an index into that array too.
    """

    next_corner: np.ndarray  # (L,) i4  the corner after this one, wrapping per face
    prev_corner: np.ndarray  # (L,) i4  the corner before it
    corner_face: np.ndarray  # (L,) i4  which face owns the corner
    edge_verts: np.ndarray  # (E, 2) i4 low-vertex-first, lexsorted unique
    corner_edge: np.ndarray  # (L,) i4  the undirected edge the corner leaves along
    edge_uses: np.ndarray  # (E,) i4  1 boundary, 2 interior, >=3 non-manifold
    twin: np.ndarray  # (L,) i4  the opposing corner, or -1
    vc_starts: np.ndarray  # (V+1,) i4 CSR offsets into vc_corners
    vc_corners: np.ndarray  # (L,) i4  corners grouped by their vertex
    flipped_pairs: np.ndarray  # (K, 2) i4 corner pairs sharing an edge, same direction

    @property
    def n_edges(self) -> int:
        return len(self.edge_verts)

    def vertex_corners(self, v: int) -> np.ndarray:
        """The corners at vertex *v* -- a read-only view into ``vc_corners``."""
        return self.vc_corners[self.vc_starts[v] : self.vc_starts[v + 1]]

    def edge_ids(self, pairs: np.ndarray) -> np.ndarray:
        """Map ``(n, 2)`` vertex pairs to edge ids, −1 where there is no edge.

        The counterpart of storing a selection as vertex *pairs* rather than as
        edge ids: ids renumber globally whenever the topology changes, pairs do
        not, so a selection is stored as pairs and mapped through here whenever
        it needs to index an edge-shaped table.
        """
        rows = np.asarray(pairs, dtype="i4").reshape(-1, 2)
        if len(rows) == 0:
            return np.zeros(0, dtype="i4")
        lo = np.minimum(rows[:, 0], rows[:, 1]).astype("i8")
        hi = np.maximum(rows[:, 0], rows[:, 1]).astype("i8")
        if self.n_edges == 0:
            return np.full(len(rows), -1, dtype="i4")
        span = int(self.edge_verts.max()) + 1
        keys = self.edge_verts[:, 0].astype("i8") * span + self.edge_verts[:, 1]
        want = lo * span + hi
        pos = np.searchsorted(keys, want)
        pos = np.clip(pos, 0, len(keys) - 1)
        hit = keys[pos] == want
        return np.where(hit, pos, -1).astype("i4")


def _build(mesh: Mesh) -> Adjacency:
    loops = mesh.loops.astype("i8")
    starts = mesh.starts.astype("i8")
    n_loops = len(loops)
    n_verts = len(mesh.positions)
    n_faces = len(starts) - 1

    # Corner -> next/prev within its face. The last corner wraps to the first,
    # which is what makes one flat pass over ``loops`` emit every polygon edge
    # exactly once.
    nxt = np.arange(1, n_loops + 1, dtype="i8")
    if n_faces:
        nxt[starts[1:] - 1] = starts[:-1]
    prv = np.empty(n_loops, dtype="i8")
    if n_loops:
        prv[nxt] = np.arange(n_loops, dtype="i8")

    counts = np.diff(starts)
    corner_face = np.repeat(np.arange(n_faces, dtype="i8"), counts)

    # The one sort: undirected edge keys, deduplicated.
    a = loops
    b = loops[nxt] if n_loops else loops
    pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
    if n_loops:
        # Packed into one integer per edge so ``unique`` sorts on a *scalar*.
        # ``np.unique(..., axis=0)`` goes through a structured-void view and a
        # lexsort, which measures 777 ms against 130 ms here on a 200k-vertex
        # mesh -- and this is the dominant cost of entering element mode.
        #
        # ``lo * stride + hi`` is order-preserving in exactly the same sense
        # the lexsort is, but only while both columns are non-negative and
        # below ``stride``; both are vertex indices, so that is asserted rather
        # than assumed. The square must also fit i8, which the assert covers.
        stride = int(pairs.max()) + 1
        assert pairs.min() >= 0 and stride <= 3_000_000_000
        packed = pairs[:, 0] * stride + pairs[:, 1]
        keys, corner_edge = np.unique(packed, return_inverse=True)
        edge_verts = np.stack([keys // stride, keys % stride], axis=1).astype("i8")
        corner_edge = corner_edge.reshape(-1)
    else:
        edge_verts = np.zeros((0, 2), dtype="i8")
        corner_edge = np.zeros(0, dtype="i8")
    edge_uses = np.bincount(corner_edge, minlength=len(edge_verts)).astype("i8")

    # Twins. Group corners by edge, then pair up the two-use edges; a pair whose
    # two corners traverse the shared edge in the *same* direction is flipped
    # rather than opposed, so it gets no twin and is reported instead.
    forward = a < b
    order = np.argsort(corner_edge, kind="stable")
    group = np.concatenate([[0], np.cumsum(edge_uses)])
    two = np.flatnonzero(edge_uses == 2)
    c1 = order[group[two]] if len(two) else np.zeros(0, dtype="i8")
    c2 = order[group[two] + 1] if len(two) else np.zeros(0, dtype="i8")
    opposed = forward[c1] != forward[c2] if len(two) else np.zeros(0, dtype=bool)
    twin = np.full(n_loops, -1, dtype="i8")
    twin[c1[opposed]] = c2[opposed]
    twin[c2[opposed]] = c1[opposed]
    flipped = np.stack([c1[~opposed], c2[~opposed]], axis=1)

    # Vertex -> corners, as CSR. Stable so a vertex's corners come back in
    # corner order, which makes a fan walk deterministic.
    vc_corners = np.argsort(loops, kind="stable")
    vc_starts = np.concatenate([[0], np.cumsum(np.bincount(loops, minlength=n_verts))]).astype("i8")

    out = Adjacency(
        next_corner=nxt.astype("i4"),
        prev_corner=prv.astype("i4"),
        corner_face=corner_face.astype("i4"),
        edge_verts=edge_verts.astype("i4").reshape(-1, 2),
        corner_edge=corner_edge.astype("i4"),
        edge_uses=edge_uses.astype("i4"),
        twin=twin.astype("i4"),
        vc_starts=vc_starts.astype("i4"),
        vc_corners=vc_corners.astype("i4"),
        flipped_pairs=flipped.astype("i4").reshape(-1, 2),
    )
    _freeze(
        out.next_corner,
        out.prev_corner,
        out.corner_face,
        out.edge_verts,
        out.corner_edge,
        out.edge_uses,
        out.twin,
        out.vc_starts,
        out.vc_corners,
        out.flipped_pairs,
    )
    return out


_CACHE: weakref.WeakKeyDictionary[Mesh, Adjacency] = weakref.WeakKeyDictionary()


def adjacency(mesh: Mesh) -> Adjacency:
    """The connectivity tables for *mesh*, built once and cached against it.

    The same mesh object hands back the same :class:`Adjacency` object, which is
    what lets a caller key a GPU buffer or a memo on ``id(adj)``; a mesh that
    becomes unreachable takes its entry with it.
    """
    got = _CACHE.get(mesh)
    if got is None:
        got = _build(mesh)
        _CACHE[mesh] = got
    return got


_F8: weakref.WeakKeyDictionary[Mesh, np.ndarray] = weakref.WeakKeyDictionary()


def cached_positions_f8(mesh: Mesh) -> np.ndarray:
    """``mesh.positions`` as f8, memoised against the mesh, read-only.

    The ray tests work in f8 and the mesh stores f4, so every pick converted
    the whole vertex array first -- once per object, on every mouse move, for a
    mesh that is frozen and cannot have changed. Cheap next to the ray cast
    itself, but it is a full copy of an array the cast then only reads.
    """
    got = _F8.get(mesh)
    if got is None:
        got = mesh.positions.astype("f8")
        _freeze(got)
        _F8[mesh] = got
    return got


_TRIS: weakref.WeakKeyDictionary[Mesh, tuple[np.ndarray, np.ndarray]] = weakref.WeakKeyDictionary()


def cached_triangulation(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """:func:`~.mesh.triangulate`, memoised against the mesh, arrays read-only.

    Picking, the selection overlays and the marquee all want the same triangle
    list within one frame, and re-fanning a 200k-corner mesh three times per
    frame is the difference between an interactive viewport and a slideshow.
    """
    got = _TRIS.get(mesh)
    if got is None:
        tris, tri_face = triangulate(mesh)
        _freeze(tris, tri_face)
        got = (tris, tri_face)
        _TRIS[mesh] = got
    return got


# --- boundary rings ---------------------------------------------------------


def boundary_loops(mesh: Mesh) -> tuple[list[np.ndarray], np.ndarray]:
    """``(rings, pinched)`` -- the open borders, and the vertices that fork.

    A ring is an ordered ``(n,)`` i4 array of vertices **wound in the hole
    direction**: a face built by using the returned order as its corner loop
    traverses each shared edge opposite to the face already on it, which is
    exactly the consistency rule the whole mesh is held to. That is why the walk
    follows the *reverse* of each boundary corner's own direction -- the corner
    reads ``a -> b`` because its face does, so the hole reads ``b -> a``. Fill
    hole therefore needs no winding decision of its own, and cannot get it
    backwards.

    ``pinched`` names the vertices where two boundary edges leave -- an
    hourglass touching at a point, or a hole that pinches shut against itself.
    There the ring is genuinely ambiguous: the walk picks a successor
    deterministically (the lowest corner) so the result is stable, but a caller
    that cares about correctness rather than display must refuse, which is what
    fill-hole does.

    Only ``edge_uses == 1`` counts as boundary. A non-manifold edge is not a
    border even though it is not an ordinary interior edge either; three faces
    meeting there means there is no hole to fill.
    """
    a = adjacency(mesh)
    if len(mesh.loops) == 0:
        return [], np.zeros(0, dtype="i4")
    corners = np.flatnonzero(a.edge_uses[a.corner_edge] == 1)
    if len(corners) == 0:
        return [], np.zeros(0, dtype="i4")

    src = mesh.loops[a.next_corner[corners]].astype("i8")
    dst = mesh.loops[corners].astype("i8")
    pinched = np.flatnonzero(np.bincount(src, minlength=len(mesh.positions)) > 1).astype("i4")

    pool: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(src.tolist()):
        pool[s].append(i)
    seen = np.zeros(len(corners), dtype=bool)

    rings: list[np.ndarray] = []
    for start in range(len(corners)):
        if seen[start]:
            continue
        ring: list[int] = []
        i = start
        while not seen[i]:
            seen[i] = True
            ring.append(int(src[i]))
            nxt = [j for j in pool[int(dst[i])] if not seen[j]]
            if not nxt:
                break
            i = nxt[0]
        rings.append(np.array(ring, dtype="i4"))
    return rings, pinched


# --- the report -------------------------------------------------------------


@dataclass(frozen=True)
class ManifoldReport:
    """What is wrong with a mesh, said plainly. Empty everywhere means clean.

    Edges are ``(n, 2)`` vertex pairs rather than edge ids so the report can be
    read without the :class:`Adjacency` it came from -- ids renumber on any
    topology change, pairs do not.
    """

    boundary_edges: np.ndarray  # (n, 2) i4 -- one face, an open border
    nonmanifold_edges: np.ndarray  # (n, 2) i4 -- three or more faces
    flipped_edges: np.ndarray  # (n, 2) i4 -- two faces, same traversal
    repeated_corner_faces: np.ndarray  # (n,) i4 -- a vertex twice in one loop
    duplicate_faces: np.ndarray  # (n,) i4 -- faces over the same vertex set
    unused_verts: np.ndarray  # (n,) i4 -- referenced by no face

    @property
    def clean(self) -> bool:
        return not any(len(getattr(self, f.name)) for f in fields(self))


def check_manifold(mesh: Mesh) -> ManifoldReport:
    """Measure *mesh* against every defect the CSR layout cannot prevent.

    A boundary is listed as a finding because the callers that ask are asking
    "is this closed", but an open sheet is a legitimate mesh -- ``clean`` is a
    strict reading, not a verdict on usability.

    **No pane calls this per frame, and that is the whole constraint on where
    it is surfaced.** It builds a full adjacency, which is O(corners) and not
    frame-thread work on a real model -- so the properties panel runs it from a
    button and holds the result against the ``Mesh`` it measured, which is sound
    because a ``Mesh`` is immutable and every op replaces it. Drawing it
    unconditionally would re-measure a 200k-corner mesh sixty times a second to
    show a line that had not changed. The other callers need the answer *about a
    mesh in hand*: the topology ops' own tests, and ``serialize.read_wblk``'s
    validation, which validates rather than trusts because ``edges`` and
    ``face_normals`` go quietly wrong on a short face instead of raising.
    ``diagnose.rows_for`` is what turns the six arrays into something a panel
    can draw and a click can select.
    """
    a = adjacency(mesh)
    loops = mesh.loops.astype("i8")
    n_faces = len(mesh.starts) - 1

    flipped = (
        a.edge_verts[a.corner_edge[a.flipped_pairs[:, 0]]]
        if len(a.flipped_pairs)
        else np.zeros((0, 2), dtype="i4")
    )

    counts = np.diff(mesh.starts.astype("i8"))
    repeated: list[int] = []
    duplicate: list[int] = []
    if n_faces:
        # Corners sorted within their own face, so a repeated vertex lands
        # beside itself and two faces over the same vertex set become identical
        # rows -- both without a per-face Python loop.
        face_of = np.repeat(np.arange(n_faces, dtype="i8"), counts)
        order = np.lexsort((loops, face_of))
        sorted_loops = loops[order]
        same_face = face_of[1:] == face_of[:-1]
        dup_corner = same_face & (sorted_loops[1:] == sorted_loops[:-1])
        repeated = np.unique(face_of[1:][dup_corner]).tolist()

        # Group by arity: within one arity the sorted loops are a rectangle, so
        # "two faces over the same vertex set" is one np.unique over rows.
        for arity in np.unique(counts):
            which = np.flatnonzero(counts == arity)
            rows = sorted_loops[_row_spans(counts, which, int(arity))]
            _, inv, cnt = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
            duplicate.extend(which[cnt[inv.reshape(-1)] > 1].tolist())

    used = np.zeros(len(mesh.positions), dtype=bool)
    used[loops] = True

    return ManifoldReport(
        boundary_edges=a.edge_verts[a.edge_uses == 1],
        nonmanifold_edges=a.edge_verts[a.edge_uses >= 3],
        flipped_edges=np.asarray(flipped, dtype="i4").reshape(-1, 2),
        repeated_corner_faces=np.array(sorted(repeated), dtype="i4"),
        duplicate_faces=np.array(sorted(duplicate), dtype="i4"),
        unused_verts=np.flatnonzero(~used).astype("i4"),
    )


def _row_spans(counts: np.ndarray, which: np.ndarray, arity: int) -> np.ndarray:
    """Indices into the face-sorted corner array for the faces in *which*.

    ``sorted_loops`` is laid out face by face in face order, so a face's span
    begins at the cumulative corner count before it -- the same offsets as
    ``starts``, which is what this recomputes for the subset.
    """
    offs = np.concatenate([[0], np.cumsum(counts)])[:-1]
    return offs[which][:, None] + np.arange(arity, dtype="i8")[None, :]
