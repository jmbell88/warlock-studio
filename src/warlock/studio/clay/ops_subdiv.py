"""Subdivision: one topology, two sets of positions.

Linear subdivide and Catmull-Clark are the *same* refinement -- split every
edge, add a point per face, and replace each n-gon with n quads -- differing
only in where the new and old vertices end up. So the topology is built once, in
:func:`subdivide_topology`, and the two ops are that function plus a choice of
positions. Writing them separately is how the smooth version ends up with a
subtly different face order or a different uv rule than the linear one, and
neither difference is visible until an asset comes back wrong.

**No T-junctions, ever.** Subdividing a *selection* splits edges that unselected
faces also use, and the unselected face on the other side still runs straight
from one original vertex to the next -- the new midpoint sits on its edge
without being one of its corners. That is a T-junction: it renders as a hairline
crack that crawls as the camera moves, it survives export, and it is invisible
in a wireframe. :func:`~.topo.splice_corners` puts the midpoint into the
neighbour's loop, turning its quad into a pentagon, and the crack closes with no
extra faces and no extra pass.

**UV is interpolated per face**, never across one. A midpoint's uv is the mean
of *that face's* two corner uvs and a centre's is the mean of that face's
corners, so a corner on a texture seam -- two corners at one position carrying
different uvs -- subdivides into two midpoints that are still on the seam. A
per-vertex average would have closed the seam and smeared the texture across it.
Catmull-Clark uses the same linear rule rather than smoothing the uvs, which is
stated rather than hidden: smoothing them would move a seam's two sides by
different amounts and open a visible gap in the texture.

**Face order changes**: the untouched faces come first and the children follow.
Nothing depends on face index stability across an op -- the returned selection
is what the UI uses -- and preserving it would mean interleaving children back
into the middle of the array for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import topo
from .adjacency import adjacency
from .elements import ElementSel, OpError, empty
from .mesh import Mesh, face_count

__all__ = ["Subdivision", "catmull_clark", "subdivide", "subdivide_topology"]


@dataclass(frozen=True)
class Subdivision:
    """The refined mesh plus the index maps Catmull-Clark needs to move it.

    ``edge_vertex`` and ``face_vertex`` are −1 where nothing was created, so a
    caller can address "the new vertex on edge *e*" without re-deriving the
    numbering the topology pass chose.
    """

    mesh: Mesh
    edge_vertex: np.ndarray  # (E,) i8 -- new vertex per split edge, else -1
    face_vertex: np.ndarray  # (F,) i8 -- new centre vertex per face, else -1
    children: ElementSel  # the quads that replaced the selection
    n_verts: int  # how many vertices the original mesh had


def subdivide_topology(mesh: Mesh, faces: np.ndarray) -> Subdivision:
    """Split *faces* into quads, with every new vertex placed linearly.

    The quad for corner *k* is ``[v_k, mid_k, centre, mid_(k-1)]`` -- out along
    its own edge, in to the face point, back along the previous edge. That
    traversal is what keeps the winding: the two halves of a split edge are
    traversed in the same direction the original face traversed the whole edge,
    so a neighbour that was consistent stays consistent.
    """
    a = adjacency(mesh)
    starts = mesh.starts.astype("i8")
    n_faces = face_count(mesh)
    chosen = np.zeros(n_faces, dtype=bool)
    chosen[faces] = True
    faces = np.flatnonzero(chosen)
    unselected = np.flatnonzero(~chosen)

    corners = topo.corner_spans(mesh.starts, faces)
    counts = starts[faces + 1] - starts[faces]
    offsets, nxt_local, prev_local = topo.flat_next(counts)

    split = np.unique(a.corner_edge[corners])
    n_verts = len(mesh.positions)
    edge_vertex = np.full(a.n_edges, -1, dtype="i8")
    edge_vertex[split] = n_verts + np.arange(len(split), dtype="i8")
    face_vertex = np.full(n_faces, -1, dtype="i8")
    face_vertex[faces] = n_verts + len(split) + np.arange(len(faces), dtype="i8")

    pts = mesh.positions[mesh.loops[corners]].astype("f8")
    positions = np.concatenate(
        [
            mesh.positions.astype("f8"),
            mesh.positions[a.edge_verts[split]].astype("f8").mean(axis=1),
            np.add.reduceat(pts, offsets[:-1], axis=0) / counts[:, None],
        ]
    )

    v = mesh.loops[corners].astype("i8")
    here = edge_vertex[a.corner_edge[corners]]
    centre = np.repeat(face_vertex[faces], counts)
    quads = np.stack([v, here, centre, here[prev_local]], axis=1).reshape(-1)

    child_uv = None
    value_uv = None
    if mesh.uv is not None:
        corner_uv = mesh.uv[corners].astype("f8")
        mid_uv = (corner_uv + corner_uv[nxt_local]) * 0.5
        centre_uv = np.repeat(
            np.add.reduceat(corner_uv, offsets[:-1], axis=0) / counts[:, None],
            counts,
            axis=0,
        )
        child_uv = np.stack(
            [corner_uv, mid_uv, centre_uv, mid_uv[prev_local]], axis=1
        ).reshape(-1, 2)

    # Splice each new midpoint into the unselected face across that edge.
    outside = np.flatnonzero(~chosen[a.corner_face] & (edge_vertex[a.corner_edge] >= 0))
    if mesh.uv is not None and len(outside):
        value_uv = (
            mesh.uv[outside].astype("f8") + mesh.uv[a.next_corner[outside]].astype("f8")
        ) * 0.5
    loops2, starts2, uv2 = topo.splice_corners(
        mesh.loops,
        mesh.starts,
        mesh.uv,
        after=outside,
        values=edge_vertex[a.corner_edge[outside]],
        counts=np.ones(len(outside), dtype="i8"),
        value_uv=value_uv,
    )
    base = topo.take_faces(
        topo.rebuild(positions, loops2, starts2, mesh.material, mesh.smooth, uv=uv2),
        unselected,
    )

    out = topo.rebuild(
        positions,
        np.concatenate([base.loops.astype("i8"), quads]),
        np.concatenate(
            [
                base.starts.astype("i8"),
                int(base.starts[-1]) + 4 * np.arange(1, len(corners) + 1, dtype="i8"),
            ]
        ),
        np.concatenate([base.material, np.repeat(mesh.material[faces], counts)]),
        np.concatenate([base.smooth, np.repeat(mesh.smooth[faces], counts)]),
        uv=None if child_uv is None else np.concatenate([base.uv, child_uv]),
    )
    return Subdivision(
        mesh=out,
        edge_vertex=edge_vertex,
        face_vertex=face_vertex,
        children=ElementSel(
            faces=np.arange(len(unselected), len(unselected) + len(corners))
        ),
        n_verts=n_verts,
    )


def subdivide(mesh: Mesh, sel: ElementSel) -> tuple[Mesh, ElementSel]:
    """Split the selected faces (or all of them) into quads, shape unchanged.

    Linear: every new vertex is a plain average of the corners it came from, so
    a cube stays a cube and a flat panel stays flat. That is what makes this the
    *modelling* subdivision -- more topology to work with, and no surprise
    change to the silhouette the user placed -- and Catmull-Clark the
    *smoothing* one.
    """
    faces = sel.faces if len(sel.faces) else np.arange(face_count(mesh), dtype="i4")
    if face_count(mesh) == 0:
        raise OpError("This object has no faces to subdivide.")
    result = subdivide_topology(mesh, np.asarray(faces, dtype="i8"))
    return result.mesh, result.children


def catmull_clark(
    mesh: Mesh, sel: ElementSel, *, levels: int = 1
) -> tuple[Mesh, ElementSel]:
    """The Catmull-Clark limit surface, one level at a time.

    **The whole mesh, whatever is selected.** A smoothing subdivision moves the
    original vertices, and moving only some of them tears the surface along the
    edge of the selection -- so the selection is deliberately ignored rather
    than half-honoured.

    The three closed forms, which the tests pin by hand on a cube, a torus and a
    plane grid:

    * **Face point** -- the centroid of the face. (The linear pass already
      places it there.)
    * **Edge point** -- ``(a + b + F1 + F2) / 4`` for an interior edge, and the
      plain midpoint on a boundary or non-manifold edge. That second case is the
      crease rule: without it, an open sheet pulls its own border inward and a
      flat plane grid develops a rolled hem.
    * **Vertex point** -- ``(Q + 2R + (n-3)P) / n`` in the interior, where *Q*
      averages the touching face points, *R* the touching edge midpoints and *n*
      is the valence. A boundary vertex with exactly two boundary edges takes
      the B-spline rule ``(prev + 6P + next) / 8``, which keeps the border a
      smooth curve rather than letting the interior drag it about -- *unless*
      it has only those two edges, which makes it a corner of the border and
      it is kept. Rounding a corner is how a square sheet becomes a lozenge one
      level at a time. Irregular vertices (one boundary edge, or three) are
      kept too, for want of any rule that means anything there.

    **Never refuses.** Every one of those rules has an answer for every vertex,
    and a smoothing pass that stopped halfway through a mesh would be worse than
    one that leaves an odd corner unmoved.

    UV is **linear** (see the module docstring), not CC-smoothed.
    """
    del sel
    if face_count(mesh) == 0:
        raise OpError("This object has no faces to smooth.")
    for _ in range(max(1, int(levels))):
        mesh = _one_level(mesh)
    return mesh, empty()


def _one_level(mesh: Mesh) -> Mesh:
    a = adjacency(mesh)
    n_verts = len(mesh.positions)
    n_edges = a.n_edges
    result = subdivide_topology(mesh, np.arange(face_count(mesh), dtype="i8"))
    positions = result.mesh.positions.astype("f8").copy()

    face_points = positions[n_verts + n_edges :]
    mids = mesh.positions[a.edge_verts].astype("f8").mean(axis=1)

    # Edge points: the average of the two endpoints and the two face points,
    # which is the midpoint again on a boundary because there is only one face.
    fsum = np.zeros((n_edges, 3))
    np.add.at(fsum, a.corner_edge, face_points[a.corner_face])
    interior = a.edge_uses == 2
    edge_points = mids.copy()
    edge_points[interior] = (mids[interior] * 2.0 + fsum[interior]) / 4.0
    # Subdividing every face splits every edge, so the topology pass numbered
    # the edge vertices ``n_verts + e`` in edge order and this is a plain slice.
    assert (result.edge_vertex == n_verts + np.arange(n_edges)).all()
    positions[n_verts : n_verts + n_edges] = edge_points

    # Vertex points.
    old = mesh.positions.astype("f8")
    q_sum = np.zeros((n_verts, 3))
    np.add.at(q_sum, mesh.loops, face_points[a.corner_face])
    q_count = np.bincount(mesh.loops, minlength=n_verts).astype("f8")

    ends = a.edge_verts.reshape(-1)
    r_sum = np.zeros((n_verts, 3))
    np.add.at(r_sum, ends, np.repeat(mids, 2, axis=0))
    valence = np.bincount(ends, minlength=n_verts).astype("f8")

    border = a.edge_verts[a.edge_uses == 1]
    b_count = np.bincount(border.reshape(-1), minlength=n_verts)
    b_sum = np.zeros((n_verts, 3))
    if len(border):
        np.add.at(b_sum, border[:, 0], old[border[:, 1]])
        np.add.at(b_sum, border[:, 1], old[border[:, 0]])

    n = np.maximum(valence, 1.0)[:, None]
    smooth = (
        q_sum / np.maximum(q_count, 1.0)[:, None]
        + 2.0 * (r_sum / n)
        + (np.maximum(valence, 1.0)[:, None] - 3.0) * old
    ) / n
    crease = (b_sum + 6.0 * old) / 8.0

    # A boundary vertex with exactly two edges in total is a *corner* of the
    # border, and stays exactly where it is -- the B-spline rule would round it
    # off, which turns a square sheet into a lozenge one level at a time.
    is_crease = (b_count == 2) & (valence >= 3)
    is_interior = (b_count == 0) & (valence >= 3) & (q_count > 0)
    new_old = old.copy()
    new_old[is_interior] = smooth[is_interior]
    new_old[is_crease] = crease[is_crease]
    positions[:n_verts] = new_old

    return topo.rebuild(
        positions,
        result.mesh.loops,
        result.mesh.starts,
        result.mesh.material,
        result.mesh.smooth,
        uv=result.mesh.uv,
    )
