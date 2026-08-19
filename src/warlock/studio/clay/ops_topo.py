"""Element ops that rewrite a mesh's faces: delete, flip, extrude, inset.

**Every op here has the one signature**
``op(mesh, sel, **params) -> tuple[Mesh, ElementSel]``: the new mesh, and the
selection the UI should show next. Returning the selection is not a
convenience -- it is the difference between an editor and a batch tool. Extrude
hands back the *caps* so the very next thing the user does is drag them with the
gizmo; delete hands back nothing, because what was selected is gone and leaving
stale indices selected would light up whichever faces inherited those numbers.
An op refuses by raising :class:`~.elements.OpError`, which the caller shows as
a toast and records no edit for.

**Non-manifold policy: local and additive ops are best-effort.** Everything in
this module either removes faces or grows new ones outward from the ones it was
given, and neither needs to walk across the mesh to do it -- so a stray extra
face hinged on a selected edge changes what the result looks like without
making the op meaningless. The walking ops (dissolve, fill hole, bevel) are the
ones that refuse, because a walk across a non-manifold edge has no defined next
step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import topo
from .adjacency import adjacency, boundary_loops
from .elements import ElementSel, OpError, empty
from .mesh import Mesh, face_count, face_normals, reversed_corner_perm

__all__ = [
    "bridge_edges",
    "collapse",
    "delete_faces",
    "extrude_edges",
    "extrude_faces",
    "extrude_verts",
    "fill_hole",
    "flip_normals",
    "inset_faces",
    "merge_vertices",
    "weld",
]


def _require_faces(sel: ElementSel, what: str) -> np.ndarray:
    if len(sel.faces) == 0:
        raise OpError(f"Select at least one face to {what}.")
    return sel.faces.astype("i8")


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0.0)


def delete_faces(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Remove the selected faces, then drop the vertices nothing uses.

    UV **preserved**: every surviving corner keeps the uv it had, because the
    corners travel whole and only the face list is shorter.

    Compaction is not optional. Leaving orphaned vertices behind would show them
    as loose points in vertex mode, put them in the bounding box the camera
    frames against, and export them into a GLB that every validator complains
    about -- all for geometry the user just asked to be rid of.
    """
    if len(sel.faces) == 0:
        raise OpError("Select at least one face to delete.")
    n = face_count(mesh)
    keep = np.ones(n, dtype=bool)
    keep[sel.faces] = False
    if not keep.any():
        # Deleting every face is legal and leaves an empty mesh, not an error:
        # the object is still there and the user can undo.
        return topo.rebuild(
            np.zeros((0, 3), dtype="f4"),
            np.zeros(0, dtype="i4"),
            np.zeros(1, dtype="i4"),
            np.zeros(0, dtype="i4"),
            np.zeros(0, dtype=bool),
            uv=None if mesh.uv is None else np.zeros((0, 2), dtype="f4"),
        ), empty()
    kept, _ = topo.compact_vertices(topo.take_faces(mesh, np.flatnonzero(keep)))
    return kept, empty()


def flip_normals(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Reverse the winding of the selected faces, or of all of them.

    UV **preserved**: the uv belongs to the corner, and the corners are the same
    corners in the other order, so the same permutation moves both. Reversing
    ``loops`` without reversing ``uv`` would rotate the texture around every
    flipped face by one corner, which is the sort of defect that only shows up
    on an exported asset.

    Flipping a *subset* deliberately leaves the mesh inconsistently oriented,
    and that is the point of the tool: it is how a user fixes an imported mesh
    with a handful of inverted triangles, and refusing on the grounds that the
    result is non-manifold would refuse exactly the repair being attempted.
    """
    n = face_count(mesh)
    faces = sel.faces if len(sel.faces) else np.arange(n, dtype="i4")
    if n == 0:
        raise OpError("This object has no faces to flip.")

    perm = np.arange(len(mesh.loops), dtype="i8")
    rev = reversed_corner_perm(mesh.starts)
    corners = topo.corner_spans(mesh.starts, faces)
    perm[corners] = rev[corners]

    out = topo.rebuild(
        mesh.positions,
        mesh.loops[perm],
        mesh.starts,
        mesh.material,
        mesh.smooth,
        uv=None if mesh.uv is None else mesh.uv[perm],
    )
    # The same faces stay selected: the user is looking at them and may want to
    # flip back. Edge and vertex selections are unaffected by a rewind.
    return out, sel


# --- extrude ----------------------------------------------------------------


def extrude_faces(mesh: Mesh, sel: ElementSel, *, offset: float = 0.0) -> tuple[Mesh, ElementSel]:
    """Pull the selected faces off the surface, walling in the gap they leave.

    **Region-aware.** A connected block of selected faces extrudes as one lid
    with walls only around its outside, not as a set of separate boxes with
    interior walls between them -- which is what per-face extrusion of adjacent
    faces produces, and it is never what the user meant by selecting a patch.
    The border comes from :func:`~.topo.region_boundary_corners`, so the rule is
    exactly "an edge no other selected face shares".

    **The caps keep their face indices**, along with their arity, material,
    smooth flag and uv. That is what makes the returned selection -- the caps
    again -- point at the same faces the user had selected a moment ago, so the
    highlight does not jump, and it is what lets the whole interaction be
    "extrude, then drag": the op itself moves nothing.

    **Zero offset by default**, which is the point. An extrude that guessed a
    distance would be undone and redone at a different distance every single
    time; an extrude at zero followed by a gizmo drag of the returned cap
    selection is one undo step for the topology and one for the movement, both
    of them things the user chose. A non-zero *offset* is still accepted, along
    the region's area-weighted mean normal, for callers that have a number.

    UV: caps **preserved** (their corners are untouched), walls **inherited** --
    each wall quad copies its two source corners' uvs, twice over, so a
    textured extrusion comes out with the side stripes running from the edge it
    grew from rather than with garbage.
    """
    faces = _require_faces(sel, "extrude")
    a = adjacency(mesh)
    n_verts = len(mesh.positions)

    cap_corners = topo.corner_spans(mesh.starts, faces)
    used = np.unique(mesh.loops[cap_corners])
    new_index = np.full(n_verts, -1, dtype="i8")
    new_index[used] = n_verts + np.arange(len(used), dtype="i8")

    direction = _unit(face_normals(mesh)[faces].sum(axis=0))
    positions = np.concatenate(
        [
            mesh.positions.astype("f8"),
            mesh.positions[used].astype("f8") + direction * float(offset),
        ]
    )

    loops = mesh.loops.astype("i8").copy()
    loops[cap_corners] = new_index[mesh.loops[cap_corners]]

    border = topo.region_boundary_corners(mesh, faces)
    nxt = a.next_corner[border].astype("i8")
    v_a, v_b = mesh.loops[border].astype("i8"), mesh.loops[nxt].astype("i8")
    # [a, b, new_b, new_a]: the wall traverses the bottom edge the way the
    # outside neighbour does not, and the top edge the way the cap does not,
    # which is what "consistently oriented" means on both of its shared edges.
    walls = np.stack([v_a, v_b, new_index[v_b], new_index[v_a]], axis=1).reshape(-1)

    uv = mesh.uv
    if uv is not None:
        uv_a, uv_b = uv[border], uv[nxt]
        uv = np.concatenate([uv, np.stack([uv_a, uv_b, uv_b, uv_a], axis=1).reshape(-1, 2)])

    owner = a.corner_face[border].astype("i8")
    out, _ = topo.compact_vertices(
        topo.rebuild(
            positions,
            np.concatenate([loops, walls]),
            np.concatenate(
                [
                    mesh.starts.astype("i8"),
                    int(mesh.starts[-1]) + 4 * np.arange(1, len(border) + 1, dtype="i8"),
                ]
            ),
            np.concatenate([mesh.material, mesh.material[owner]]),
            np.concatenate([mesh.smooth, mesh.smooth[owner]]),
            uv=uv,
        )
    )
    return out, ElementSel(faces=faces)


# --- inset ------------------------------------------------------------------


def inset_faces(
    mesh: Mesh,
    sel: ElementSel,
    *,
    thickness: float = 0.1,
    depth: float = 0.0,
    region: bool = False,
) -> tuple[Mesh, ElementSel]:
    """Shrink each selected face in place, ringing it with the rim it vacated.

    Two modes, because they are genuinely different operations. **Per-face**
    (the default) insets every selected face independently: no ring vertices are
    shared between faces, so two adjacent selected faces each get their own rim
    and a doubled edge between them. That is what "inset each of these" means,
    and it is exact -- the inner corners are a lerp toward the face's own
    centroid in position *and* uv, which reads no other face's corners and is
    therefore seam-safe.

    **Region** insets the outline of the selected block as a whole. It is
    implemented as an extrude at zero followed by a pull of each new cap vertex
    toward the mean centroid of the cap faces meeting at it, and that is a
    documented approximation rather than a true bisector offset: exact on a flat
    regular region, drifting on a strongly non-planar one. The honest
    alternative is a straight-skeleton offset, which is a great deal of
    machinery for a control whose result the user is looking at.

    ``thickness`` is a distance, turned per corner into a fraction of that
    corner's own distance from the centroid and clamped at 1 -- so a thickness
    larger than the face collapses the inner ring onto the centroid rather than
    turning the face inside out. ``depth`` then moves the inner ring along the
    face normal, which is what makes inset-then-push a panel.
    """
    faces = _require_faces(sel, "inset")
    if region:
        return _inset_region(mesh, faces, thickness=thickness, depth=depth)

    starts = mesh.starts.astype("i8")
    corners = topo.corner_spans(mesh.starts, faces)
    counts = starts[faces + 1] - starts[faces]
    offsets, nxt, _ = topo.flat_next(counts)
    total = len(corners)

    pts = mesh.positions[mesh.loops[corners]].astype("f8")
    centroid = np.add.reduceat(pts, offsets[:-1], axis=0) / counts[:, None]
    per_corner = np.repeat(centroid, counts, axis=0)
    t = _shrink_fraction(pts, per_corner, thickness)
    normals = np.repeat(_unit(face_normals(mesh)[faces]), counts, axis=0)
    inner = pts + (per_corner - pts) * t + normals * float(depth)

    n_verts = len(mesh.positions)
    inner_v = n_verts + np.arange(total, dtype="i8")
    loops = mesh.loops.astype("i8").copy()
    loops[corners] = inner_v

    outer_v = mesh.loops[corners].astype("i8")
    rim = np.stack([outer_v, outer_v[nxt], inner_v[nxt], inner_v], axis=1).reshape(-1)

    uv = mesh.uv
    if uv is not None:
        uv = uv.astype("f8").copy()
        outer_uv = uv[corners]
        uv_centroid = np.repeat(
            np.add.reduceat(outer_uv, offsets[:-1], axis=0) / counts[:, None],
            counts,
            axis=0,
        )
        inner_uv = outer_uv + (uv_centroid - outer_uv) * t
        uv[corners] = inner_uv
        uv = np.concatenate(
            [
                uv,
                np.stack([outer_uv, outer_uv[nxt], inner_uv[nxt], inner_uv], axis=1).reshape(-1, 2),
            ]
        )

    owner = np.repeat(faces, counts)
    out = topo.rebuild(
        np.concatenate([mesh.positions.astype("f8"), inner]),
        np.concatenate([loops, rim]),
        np.concatenate([starts, int(starts[-1]) + 4 * np.arange(1, total + 1, dtype="i8")]),
        np.concatenate([mesh.material, mesh.material[owner]]),
        np.concatenate([mesh.smooth, mesh.smooth[owner]]),
        uv=uv,
    )
    return out, ElementSel(faces=faces)


def _shrink_fraction(pts: np.ndarray, target: np.ndarray, thickness: float) -> np.ndarray:
    """How far each point travels toward its target, as an ``(n, 1)`` fraction.

    A distance divided by the point's own distance from the target, clamped at
    one. A point already at its target does not move -- dividing by zero there
    would be a NaN in the positions, and a NaN in a vertex buffer is a mesh that
    disappears with nothing in the log.
    """
    radius = np.linalg.norm(target - pts, axis=1)
    return np.clip(
        np.divide(
            abs(float(thickness)),
            radius,
            out=np.zeros_like(radius),
            where=radius > 0.0,
        ),
        0.0,
        1.0,
    )[:, None]


def _inset_region(
    mesh: Mesh, faces: np.ndarray, *, thickness: float, depth: float
) -> tuple[Mesh, ElementSel]:
    """Extrude at zero, then pull each new cap vert toward its local centroid."""
    out, sel = extrude_faces(mesh, ElementSel(faces=faces), offset=0.0)
    starts = out.starts.astype("i8")
    corners = topo.corner_spans(out.starts, faces)
    counts = starts[faces + 1] - starts[faces]
    offsets, _, _ = topo.flat_next(counts)

    pts = out.positions[out.loops[corners]].astype("f8")
    centroid = np.add.reduceat(pts, offsets[:-1], axis=0) / counts[:, None]
    verts = out.loops[corners].astype("i8")

    n_verts = len(out.positions)
    total = np.zeros((n_verts, 3))
    hits = np.zeros(n_verts)
    np.add.at(total, verts, np.repeat(centroid, counts, axis=0))
    np.add.at(hits, verts, 1.0)
    touched = np.flatnonzero(hits > 0)

    positions = out.positions.astype("f8").copy()
    target = total[touched] / hits[touched][:, None]
    t = _shrink_fraction(positions[touched], target, thickness)
    direction = _unit(face_normals(out)[faces].sum(axis=0))
    positions[touched] += (target - positions[touched]) * t + direction * float(depth)

    return topo.rebuild(positions, out.loops, out.starts, out.material, out.smooth, uv=out.uv), sel


# --- merging vertices -------------------------------------------------------

# Above this many vertices the 27-neighbour search is skipped and the plain
# grid is used instead. The difference is real and worth stating: the
# neighbour search finds every pair within ``eps`` of each other, while the
# grid finds only pairs that quantise into the same cell and therefore misses
# two vertices a nanometre apart across a cell boundary. On a selection the
# user made by hand the exact search costs nothing; on a whole imported mesh even
# the SciPy pair query and its connected-components pass are more than the
# answer is worth, and a weld that takes a visible pause is worse than a weld
# that misses the occasional straddling pair.
#
# The limit predates the SciPy rewrite, which moved the exact arm from minutes
# of Python to seconds of C -- so it is now a conservative bound rather than a
# necessary one, and re-measuring it is a reasonable future exercise.
WELD_SEARCH_LIMIT = 20_000

def merge_vertices(mesh: Mesh, remap: np.ndarray, positions: np.ndarray) -> Mesh:
    """Rewrite a mesh with vertices identified, and clean up what that breaks.

    The shared aftermath of weld and collapse, in one place because getting any
    step of it wrong produces a mesh that passes :func:`~.mesh.validate` and
    then misbehaves somewhere else entirely:

    * **Remap the corners.** Every corner now names its representative.
    * **Drop consecutive duplicates.** Two corners of one face that merged into
      the same vertex leave a zero-length edge in the loop; left in, it makes
      ``edges`` report an edge from a vertex to itself and gives every
      triangulation a degenerate triangle.
    * **Drop faces under three corners.** A quad whose corners merged in pairs
      is a line, not a face, and ``validate`` refuses it outright.
    * **Compact.** The merged-away vertices are referenced by nothing.

    **Doubled faces are left alone and reported**, not deleted. Welding the rim
    of a folded sheet genuinely produces two faces over the same corners, and
    which one is redundant depends on what the user is doing;
    :func:`~.adjacency.check_manifold` names them, and deleting geometry the
    user can still see would be the surprising choice.

    UV **preserved**: a surviving corner keeps its own uv, so a texture seam
    across the weld survives it -- which is the whole reason uvs are per corner
    rather than per vertex.
    """
    remap = np.asarray(remap, dtype="i8").reshape(-1)
    loops = remap[mesh.loops]
    if len(loops) == 0:
        return topo.rebuild(positions, loops, mesh.starts, mesh.material, mesh.smooth, uv=mesh.uv)

    starts = mesh.starts.astype("i8")
    nxt = np.arange(1, len(loops) + 1, dtype="i8")
    nxt[starts[1:] - 1] = starts[:-1]
    keep = loops != loops[nxt]

    face_of = np.repeat(np.arange(len(starts) - 1, dtype="i8"), np.diff(starts))
    counts = np.bincount(face_of[keep], minlength=len(starts) - 1)
    alive = counts >= 3
    keep &= alive[face_of]

    corners = np.flatnonzero(keep)
    out, _ = topo.compact_vertices(
        topo.rebuild(
            positions,
            loops[corners],
            topo.starts_from_counts(counts[alive]),
            mesh.material[alive],
            mesh.smooth[alive],
            uv=None if mesh.uv is None else mesh.uv[corners],
        )
    )
    return out


def _clusters(points: np.ndarray, eps: float) -> np.ndarray:
    """Group *points* within ``eps``; returns a label per point.

    Two strategies, chosen by size -- see :data:`WELD_SEARCH_LIMIT`.
    """
    if len(points) == 0:
        return np.zeros(0, dtype="i8")
    cells = np.floor(points / eps).astype("i8")
    if len(points) > WELD_SEARCH_LIMIT:
        return np.unique(cells, axis=0, return_inverse=True)[1].reshape(-1)

    # A Python dict spatial hash plus a hand-rolled union-find was the whole of
    # this, and every part of it -- the 27-cell neighbourhood walk, the pair
    # test, the path-compressing ``find`` -- ran in Python per vertex. SciPy is
    # already a **main** dependency, so this invents nothing: ``query_pairs``
    # is the same "everything within eps" question asked of a KD-tree in C, and
    # ``connected_components`` is the same union-find in C. The above-limit
    # cell-grouping arm is untouched, because it answers a different (and
    # cheaper, and deliberately approximate) question.
    #
    # The labels themselves are not the same *numbers* the union-find produced
    # and nothing depended on them being: ``weld`` puts each cluster's
    # representative at the centroid precisely so the answer cannot depend on
    # vertex order, and the tests assert the geometry rather than the labelling.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    count = len(points)
    if not len(pairs):
        return np.arange(count, dtype="i8")
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
        shape=(count, count),
    )
    _n, labels = connected_components(graph, directed=False)
    return np.unique(labels, return_inverse=True)[1].reshape(-1).astype("i8")


def weld(mesh: Mesh, sel: ElementSel, *, eps: float = 1e-4) -> tuple[Mesh, ElementSel]:
    """Merge vertices closer together than ``eps`` into one at their centroid.

    With nothing selected this welds the whole mesh, which is the repair a
    freshly imported model usually wants; with a vertex selection it welds only
    those, which is how a user closes one seam without disturbing a deliberate
    split elsewhere.

    The representative sits at the **cluster centroid** rather than at one
    member: picking a member would make the result depend on vertex order, so
    the same weld run on the same geometry saved two different ways would land
    in two different places.
    """
    verts = sel.verts.astype("i8") if len(sel.verts) else np.arange(len(mesh.positions), dtype="i8")
    if len(verts) < 2:
        raise OpError("Select at least two vertices to weld.")
    if eps <= 0.0:
        raise OpError("The weld distance must be greater than zero.")

    labels = _clusters(mesh.positions[verts].astype("f8"), float(eps))
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    sums = np.zeros((n_clusters, 3))
    hits = np.zeros(n_clusters)
    np.add.at(sums, labels, mesh.positions[verts].astype("f8"))
    np.add.at(hits, labels, 1.0)

    positions = mesh.positions.astype("f8").copy()
    remap = np.arange(len(mesh.positions), dtype="i8")
    # The lowest-numbered member of each cluster is its representative index;
    # its *position* is the centroid, so the choice of index has no geometric
    # consequence and only keeps the numbering stable.
    first = np.full(n_clusters, len(mesh.positions), dtype="i8")
    np.minimum.at(first, labels, verts)
    remap[verts] = first[labels]
    positions[first] = sums / hits[:, None]

    out = merge_vertices(mesh, remap, positions)
    return out, empty()


def collapse(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Pull each selected edge or face down to a single vertex at its centre.

    Edges collapse by connected component, not one at a time: a chain of three
    selected edges becomes one vertex, because collapsing them in sequence would
    give an answer that depends on the order they were visited. Faces collapse
    to their own centroid.

    A bare *vertex* selection is refused with a pointer at weld -- merging
    vertices that are not joined by anything selected is what weld does, and
    guessing between the two would silently do the wrong one.
    """
    if len(sel.edges) == 0 and len(sel.faces) == 0:
        if len(sel.verts):
            raise OpError("Collapse works on edges and faces; use Weld for vertices.")
        raise OpError("Select an edge or a face to collapse.")

    n_verts = len(mesh.positions)
    parent = np.arange(n_verts, dtype="i8")

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    groups: list[np.ndarray] = [sel.edges.astype("i8")]
    starts = mesh.starts.astype("i8")
    for f in sel.faces.astype("i8").tolist():
        loop = mesh.loops[starts[f] : starts[f + 1]].astype("i8")
        groups.append(np.stack([loop, np.roll(loop, -1)], axis=1))
    for a_v, b_v in np.concatenate(groups).tolist():
        ra, rb = find(int(a_v)), find(int(b_v))
        if ra != rb:
            parent[ra] = rb

    remap = np.array([find(i) for i in range(n_verts)], dtype="i8")
    positions = mesh.positions.astype("f8").copy()
    sums = np.zeros((n_verts, 3))
    hits = np.zeros(n_verts)
    np.add.at(sums, remap, mesh.positions.astype("f8"))
    np.add.at(hits, remap, 1.0)
    touched = np.flatnonzero(hits > 1)
    positions[touched] = sums[touched] / hits[touched][:, None]

    return merge_vertices(mesh, remap, positions), empty()


# --- fill hole --------------------------------------------------------------


def fill_hole(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Cap the boundary ring each selected edge belongs to, with one n-gon.

    One face rather than a fan, because that is what the CSR storage is for and
    because a fan invents a centre vertex the user did not ask for -- and
    because the ring may be concave, which :mod:`.earclip` now handles at
    render time without the topology having to commit to a triangulation.

    The winding needs no decision here: :func:`~.adjacency.boundary_loops`
    returns the ring already oriented in the hole direction, so using it as the
    corner loop makes the cap traverse every shared edge opposite to the face
    already on it.

    **Refusals**, all of them because the ring is genuinely ambiguous rather
    than merely awkward: a seed edge that is not on a boundary, a pinched
    vertex (two boundary edges leave it, so which way the ring continues is a
    coin toss), and a ring that visits a vertex twice (a figure eight, whose cap
    would self-intersect).

    UV is **copied from an adjacent boundary corner** at each ring vertex -- a
    documented placeholder. There is no correct answer without a projection or
    an unwrap, and copying at least puts the cap inside the same region of the
    texture as the surface around it rather than at the origin.
    """
    if len(sel.edges) == 0:
        raise OpError("Select an edge on the boundary of a hole to fill it.")

    a = adjacency(mesh)
    ids = a.edge_ids(sel.edges)
    if (ids < 0).any():
        raise OpError("That edge is not part of this mesh.")
    uses = a.edge_uses[ids]
    if (uses != 1).any():
        bad = sel.edges[uses != 1][0]
        raise OpError(
            f"Edge {int(bad[0])}-{int(bad[1])} is not on a boundary, so there is "
            "no hole there to fill."
        )

    rings, pinched = boundary_loops(mesh)
    wanted = set(map(int, sel.edges.reshape(-1).tolist()))
    chosen = [r for r in rings if wanted & set(r.tolist())]
    if not chosen:
        raise OpError("No boundary ring runs through the selected edge.")

    pinched_set = set(pinched.tolist())
    new_loops: list[np.ndarray] = []
    for ring in chosen:
        hit = pinched_set & set(ring.tolist())
        if hit:
            raise OpError(
                f"Vertex {min(hit)} has two boundary edges, so the hole through it "
                "has no single ring. Weld or delete the pinch first."
            )
        if len(np.unique(ring)) != len(ring):
            raise OpError(
                "That boundary crosses itself, so it cannot be capped with one "
                "face. Split it first."
            )
        new_loops.append(ring.astype("i8"))

    counts = np.array([len(r) for r in new_loops], dtype="i8")
    corners = np.concatenate(new_loops)
    n_faces = face_count(mesh)

    uv = mesh.uv
    if uv is not None:
        rows = np.stack([uv[a.vertex_corners(int(v))[0]] for v in corners])
        uv = np.concatenate([uv, rows])

    out = topo.rebuild(
        mesh.positions,
        np.concatenate([mesh.loops.astype("i8"), corners]),
        np.concatenate([mesh.starts.astype("i8"), int(mesh.starts[-1]) + np.cumsum(counts)]),
        np.concatenate([mesh.material, np.zeros(len(counts), dtype="i4")]),
        np.concatenate([mesh.smooth, np.zeros(len(counts), dtype=bool)]),
        uv=uv,
    )
    return out, ElementSel(faces=np.arange(n_faces, n_faces + len(counts)))


# --- the boundary ops: extrude an edge, bridge two loops ---------------------
#
# Both of these are about a mesh's *border* rather than its interior, and both
# rest on one fact that is easy to forget and expensive to rediscover: a
# ``Mesh`` stores no wire edges. ``edges`` is derived from face corners, so an
# edge exists exactly while some face uses it -- which is why a vertex extrude
# is not implementable here at all (see :func:`extrude_edges`) and why both ops
# below refuse an interior edge rather than guessing what to do with it.


def _boundary_owner(mesh: Mesh, sel: ElementSel, what: str) -> tuple[Any, np.ndarray, np.ndarray]:
    """``(adjacency, edge ids, owning corners)`` for a boundary-edge selection.

    Shared by both ops because the refusals are the interesting part and two
    copies of them is two chances to word one of them differently. A boundary
    edge has exactly one corner leaving along it, so that corner is the edge's
    whole context: the face that owns it, the direction that face traverses it,
    and the two uvs to inherit.
    """
    if len(sel.edges) == 0:
        raise OpError(f"Select one or more boundary edges to {what}.")
    a = adjacency(mesh)
    ids = a.edge_ids(sel.edges).astype("i8")
    if (ids < 0).any():
        raise OpError("That edge is not part of this mesh.")
    uses = a.edge_uses[ids]
    if (uses != 1).any():
        bad = sel.edges[uses != 1][0]
        raise OpError(
            f"Edge {int(bad[0])}-{int(bad[1])} has faces on both sides, so there is "
            f"no open side to {what}. Select edges on the border of the mesh."
        )
    # One corner per edge, and the assignment is unambiguous precisely because
    # every selected edge is used exactly once.
    owner = np.full(a.n_edges, -1, dtype="i8")
    owner[a.corner_edge.astype("i8")] = np.arange(len(mesh.loops), dtype="i8")
    return a, ids, owner[ids]


def extrude_edges(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Grow one quad outward from each selected boundary edge.

    **It takes no distance at all**, which is the one place it departs from
    :func:`extrude_faces`. That op accepts an ``offset`` along the region's mean
    normal, and here the same quantity is *degenerate on the commonest case*: a
    closed rim's owning faces point radially outward all the way round, so their
    mean is zero and any number handed in moves nothing. A parameter that
    silently does nothing is worse than no parameter, so the new rim lands
    exactly on the old one and the gizmo moves it -- which is what the returned
    selection is for, and what "extrude, then drag" already meant.

    Each new quad traverses its shared edge **opposite** to the face already on
    it -- ``[v_b, v_a, new_a, new_b]`` against that face's ``v_a -> v_b`` --
    which is what keeps the shell consistently oriented. That is the mirror of
    the wall :func:`extrude_faces` builds, and it is the other way round for a
    reason worth stating: there the cap *moves* onto the new vertices, so the
    wall keeps the original direction, while here the face stays exactly where
    it is.

    **A vertex extrude is not offered, and cannot be.** A ``Mesh`` stores no
    wire edges, so an extruded lone vertex would be a position no face uses --
    which ``compact_vertices`` drops on the way out and every exporter would
    drop again. Extrude in vertex mode therefore converts the selection to the
    edges it implies and extrudes those, which is what a user selecting a run of
    border vertices means; a selection implying no boundary edge is refused by
    name rather than silently doing nothing.

    UV **inherited**: each new quad copies its source corners' two uvs, twice
    over, exactly as an extruded wall does.
    """
    a, _ids, corners = _boundary_owner(mesh, sel, "extrude")
    nxt = a.next_corner[corners].astype("i8")
    v_a, v_b = mesh.loops[corners].astype("i8"), mesh.loops[nxt].astype("i8")

    used = np.unique(np.concatenate([v_a, v_b]))
    n_verts = len(mesh.positions)
    new_index = np.full(n_verts, -1, dtype="i8")
    new_index[used] = n_verts + np.arange(len(used), dtype="i8")

    owner_face = a.corner_face[corners].astype("i8")
    positions = np.concatenate([mesh.positions.astype("f8"), mesh.positions[used].astype("f8")])

    quads = np.stack([v_b, v_a, new_index[v_a], new_index[v_b]], axis=1).reshape(-1)
    uv = mesh.uv
    if uv is not None:
        uv_a, uv_b = uv[corners], uv[nxt]
        uv = np.concatenate([uv, np.stack([uv_b, uv_a, uv_a, uv_b], axis=1).reshape(-1, 2)])

    out, old_to_new = topo.compact_vertices(
        topo.rebuild(
            positions,
            np.concatenate([mesh.loops.astype("i8"), quads]),
            np.concatenate(
                [
                    mesh.starts.astype("i8"),
                    int(mesh.starts[-1]) + 4 * np.arange(1, len(corners) + 1, dtype="i8"),
                ]
            ),
            np.concatenate([mesh.material, mesh.material[owner_face]]),
            np.concatenate([mesh.smooth, mesh.smooth[owner_face]]),
            uv=uv,
        )
    )
    grown = np.stack([new_index[v_a], new_index[v_b]], axis=1)
    return out, ElementSel(edges=old_to_new[grown])


def extrude_verts(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Extrude the *border* edges a vertex selection implies.

    See :func:`extrude_edges` for why there is no such thing as extruding a
    vertex here. What a user selecting a run of border vertices means is the
    border between them, so that is what this takes -- and the interior edges
    the same selection also implies are dropped rather than refused, because
    including them is the one reading nobody intends: a patch of selected
    vertices in the middle of a surface implies a web of interior edges and no
    border at all.
    """
    from .elements import convert

    edges = convert(mesh, sel, "edge").edges
    if len(edges):
        a = adjacency(mesh)
        ids = a.edge_ids(edges)
        edges = edges[(ids >= 0) & (a.edge_uses[np.clip(ids, 0, None)] == 1)]
    if not len(edges):
        raise OpError(
            "Those vertices have no border edge between them. A mesh stores no "
            "wire edges, so a vertex on its own has nothing to extrude -- select "
            "two or more connected vertices on the edge of the surface."
        )
    return extrude_edges(mesh, ElementSel(edges=edges))


def _directed_runs(pairs: np.ndarray, what: str) -> list[list[int]]:
    """Ordered vertex runs from directed edges, one per connected component.

    A closed run comes back with its k vertices and no repeat of the first; an
    open one with k + 1. The direction is the caller's -- here always the way
    the owning face traverses the edge -- so the walk is a plain ``u -> v``
    follow rather than an undirected trace needing a turn rule, and a component
    that is not a single run is refused rather than cut into pieces.
    """
    out_of: dict[int, int] = {}
    into: dict[int, int] = {}
    for u, v in np.asarray(pairs, dtype="i8").tolist():
        if u in out_of or v in into:
            raise OpError(
                f"The selected edges fork at vertex {u if u in out_of else v}, so "
                f"there is no single loop to {what}."
            )
        out_of[int(u)], into[int(v)] = int(v), int(u)

    runs: list[list[int]] = []
    seen: set[int] = set()
    for start in out_of:
        if start in seen:
            continue
        # Walk back to a beginning first, so an open run is reported from its
        # end rather than from wherever the iteration happened to start.
        head = start
        while head in into and into[head] not in seen and into[head] != start:
            head = into[head]
            if head == start:
                break
        run = [head]
        seen.add(head)
        while run[-1] in out_of:
            nxt = out_of[run[-1]]
            if nxt == run[0]:
                break  # closed: the first vertex is not repeated
            if nxt in seen:
                raise OpError(
                    f"The selected edges cross themselves, so there is nothing to {what}."
                )
            run.append(nxt)
            seen.add(nxt)
        runs.append(run)
    return runs


def _bridge_offset(a_pos: np.ndarray, b_pos: np.ndarray) -> int:
    """Which rotation of loop B lines up with loop A.

    A closed loop carries no starting vertex -- a ring of sixteen is the same
    ring however it is numbered -- so the correspondence has to be *measured*,
    and the measurement is the plain one: the rotation minimising the summed
    squared distance between paired vertices. Getting it wrong produces no
    refusal at all, just a bridge with a visible twist in it, which is why this
    is arithmetic rather than a convention.
    """
    return int(
        np.argmin(
            [float(((a_pos - np.roll(b_pos, -s, axis=0)) ** 2).sum()) for s in range(len(a_pos))]
        )
    )


def bridge_edges(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Join two selected boundary loops with a strip of quads.

    The op that turns two tubes into one, closes the gap left by deleting a band
    of faces, and skins a pair of profiles -- and it is worth having because
    every alternative is wrong: filling both holes leaves the geometry inside,
    and extruding one loop and welding it onto the other is four steps and a
    distance the user has to get exactly right.

    **Both loops must be boundaries**, which is the one refusal that will
    surprise somebody. Bridging two *interior* rings means deleting the faces
    between them first, and doing that silently would remove geometry nobody
    asked to lose. Everything else refused here is genuinely ambiguous: a
    selection that is not exactly two runs, two runs of different lengths, a
    fork, a self-crossing run, or one closed loop against one open chain, which
    have no correspondence at all.

    **The winding needs no decision and no measurement.** Each loop is ordered
    by the direction its own faces traverse it, and every quad then traverses
    both of its shared edges the opposite way -- ``[a1, a0, b0, b1]`` -- which is
    exactly what "consistently oriented" means, on both sides at once. A
    geometric guess (does this normal point away from the axis?) would be a
    second opinion that could disagree with the faces already there.

    UV **inherited**: each quad takes its four coordinates from the two boundary
    corners it grew between, which puts the strip in the same region of the
    texture as the surfaces it joins.
    """
    a, _ids, corners = _boundary_owner(mesh, sel, "bridge")
    nxt = a.next_corner[corners].astype("i8")
    v_a, v_b = mesh.loops[corners].astype("i8"), mesh.loops[nxt].astype("i8")

    # Corner lookup by the directed pair, so the walk can hand back plain vertex
    # runs and the uvs and owning faces can still be found afterwards.
    corner_of = {(int(u), int(v)): int(c) for u, v, c in zip(v_a, v_b, corners, strict=True)}
    runs = _directed_runs(np.stack([v_a, v_b], axis=1), "bridge")
    if len(runs) != 2:
        raise OpError(
            f"Bridge joins exactly two boundary loops; the selection makes {len(runs)}."
        )
    left, right = runs
    closed_left = (int(left[-1]), int(left[0])) in corner_of
    closed_right = (int(right[-1]), int(right[0])) in corner_of
    if closed_left != closed_right:
        raise OpError(
            "One selected loop is closed and the other is not, so there is no way "
            "to pair them up."
        )
    if len(left) != len(right):
        raise OpError(
            f"The two loops have {len(left)} and {len(right)} vertices; bridge needs "
            "the same number on each side."
        )

    positions = mesh.positions.astype("f8")
    # B reversed: the strip walks A forwards and B backwards, which is what makes
    # quad i and quad i+1 share an edge rather than a single vertex.
    flipped = list(right[::-1])
    if closed_right:
        shift = _bridge_offset(positions[left], positions[flipped])
        flipped = [int(v) for v in np.roll(np.asarray(flipped, dtype="i8"), -shift)]
    span = len(left) if closed_left else len(left) - 1

    quads: list[int] = []
    uv_rows: list[np.ndarray] = []
    owner_face: list[int] = []
    for i in range(span):
        a0, a1 = int(left[i]), int(left[(i + 1) % len(left)])
        b0, b1 = int(flipped[i]), int(flipped[(i + 1) % len(flipped)])
        quads.extend([a1, a0, b0, b1])
        ca = corner_of[(a0, a1)]
        # ``flipped`` runs against its own faces, so its directed edge is the
        # reversed pair -- which is the corner the uvs and the face come from.
        cb = corner_of[(b1, b0)]
        owner_face.append(int(a.corner_face[ca]))
        if mesh.uv is not None:
            na, nb = int(a.next_corner[ca]), int(a.next_corner[cb])
            uv_rows.append(np.stack([mesh.uv[na], mesh.uv[ca], mesh.uv[nb], mesh.uv[cb]]))

    faces = np.asarray(owner_face, dtype="i8")
    uv = mesh.uv
    if uv is not None:
        uv = np.concatenate([uv, np.concatenate(uv_rows)])
    out = topo.rebuild(
        mesh.positions,
        np.concatenate([mesh.loops.astype("i8"), np.asarray(quads, dtype="i8")]),
        np.concatenate(
            [
                mesh.starts.astype("i8"),
                int(mesh.starts[-1]) + 4 * np.arange(1, span + 1, dtype="i8"),
            ]
        ),
        np.concatenate([mesh.material, mesh.material[faces]]),
        np.concatenate([mesh.smooth, mesh.smooth[faces]]),
        uv=uv,
    )
    n_faces = face_count(mesh)
    return out, ElementSel(faces=np.arange(n_faces, n_faces + span))
