"""Triangulating an n-gon that a fan cannot handle, and deciding which those are.

:func:`~.mesh.triangulate` fans every face from its first corner, which is
correct for a convex polygon and wrong for a concave one -- a fan across a
reflex corner puts a triangle outside the polygon, and the result is a shaded
mesh with a wedge that is not there and a face pick that hits empty space. Every
primitive generator produces convex faces, so the fan was right until the
dissolve family started merging faces into arbitrary n-gons and GLB import
started accepting whatever an exporter wrote.

**The screen comes first, and it is the whole performance story.** A face is
*suspect* only if some corner turns the wrong way about the face's own Newell
normal, measured as the sine of the turn between its two unit edge vectors --
scale-free, so one tolerance works from a millimetre bracket to a kilometre
terrain. That test is one vectorised pass with a ``reduceat`` min per face, and
on the meshes this editor actually loads it says "no" for every face: imported
triangle and quad soup is convex by construction, so it never leaves the fan
path, and the fan output for a non-suspect mesh is *byte-identical* to what it
was before this module existed. Only the faces that fail the screen pay for the
O(n²) ear search, in Python, one at a time.

**Rendering never raises.** The ear search is tolerant rather than strict: it
accepts collinear and repeated corners, and if it stalls -- a self-intersecting
loop, a zero-area face, a polygon whose corners are all collinear -- it fans the
remainder rather than giving up. A triangulation that is wrong is a visual
defect the user can see and fix; an exception here would take down the frame
loop from inside a draw, which they cannot. The count is exactly ``n - 2``
triangles either way, so callers that index by face never have to special-case
a failure.

This module deliberately takes plain arrays rather than a
:class:`~.mesh.Mesh` -- ``mesh`` imports it, not the other way round -- and
returns triangles as *corner* indices into ``loops``. Corners, not vertices,
because :func:`~.mesh.render_arrays` needs to look up a per-corner uv and a
per-corner split vertex for each triangle it emits, and a vertex index has
thrown that away.
"""

from __future__ import annotations

import numpy as np

__all__ = ["concave_faces", "corner_triangles", "fan_corners"]

# The turn between two unit edge vectors, as a sine. A corner flatter than this
# is treated as straight rather than as reflex: a fan through a collinear corner
# is harmless (the triangle is degenerate but inside), while ear-clipping a face
# because of float noise on a flat corner would cost the fast path on ordinary
# subdivided geometry.
TURN_EPS = 1e-6


def fan_corners(starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(tri_corners, tri_face)`` fanning every face from its first corner.

    ``tri_corners`` is ``(T, 3)`` i8 indexing *loops*; ``tri_face`` is ``(T,)``
    i4. This is the fast path and the fallback both.
    """
    starts = starts.astype("i8")
    counts = np.diff(starts)
    per_face = np.maximum(counts - 2, 0)
    total = int(per_face.sum())
    if total == 0:
        return np.zeros((0, 3), dtype="i8"), np.zeros(0, dtype="i4")
    tri_face = np.repeat(np.arange(len(counts), dtype="i8"), per_face)
    offsets = np.concatenate([[0], np.cumsum(per_face)[:-1]])
    local = np.arange(total, dtype="i8") - offsets[tri_face]
    first = starts[:-1][tri_face]
    corners = np.stack([first, first + local + 1, first + local + 2], axis=1)
    return corners, tri_face.astype("i4")


def concave_faces(
    positions: np.ndarray,
    loops: np.ndarray,
    starts: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """A ``(F,)`` bool mask: faces with at least one reflex corner.

    *normals* is one unnormalised Newell normal per face, exactly what
    :func:`~.mesh._face_normals` returns -- passed in rather than recomputed
    because both callers already have it.
    """
    starts = starts.astype("i8")
    n_faces = len(starts) - 1
    if n_faces == 0 or len(loops) == 0:
        return np.zeros(n_faces, dtype=bool)

    nxt = np.arange(1, len(loops) + 1, dtype="i8")
    nxt[starts[1:] - 1] = starts[:-1]
    prv = np.empty(len(loops), dtype="i8")
    prv[nxt] = np.arange(len(loops), dtype="i8")

    p = positions[loops[prv]].astype("f8")
    c = positions[loops].astype("f8")
    q = positions[loops[nxt]].astype("f8")
    e1 = _unit(c - p)
    e2 = _unit(q - c)
    turn = np.cross(e1, e2)

    counts = np.diff(starts)
    unit_n = _unit(normals.astype("f8"))
    sine = np.einsum("ij,ij->i", turn, unit_n[np.repeat(np.arange(n_faces), counts)])
    worst = np.minimum.reduceat(sine, starts[:-1])
    # A triangle is convex however its corners turn, and screening one costs
    # nothing to skip -- but a zero-area triangle's noisy sine would otherwise
    # send it down the slow path.
    return (worst < -TURN_EPS) & (counts > 3)


def corner_triangles(
    positions: np.ndarray,
    loops: np.ndarray,
    starts: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(tri_corners, tri_face)`` -- fans where a fan is right, ears where not.

    Face order is preserved and each face contributes exactly ``n - 2``
    triangles, so ``tri_face`` is identical to the pure-fan version and a caller
    that groups by face sees no difference.
    """
    corners, tri_face = fan_corners(starts)
    suspect = np.flatnonzero(concave_faces(positions, loops, starts, normals))
    if len(suspect) == 0:
        return corners, tri_face

    starts = starts.astype("i8")
    counts = np.diff(starts)
    per_face = np.maximum(counts - 2, 0)
    offsets = np.concatenate([[0], np.cumsum(per_face)[:-1]])
    unit_n = _unit(normals.astype("f8"))
    for f in suspect.tolist():
        lo, hi = int(starts[f]), int(starts[f + 1])
        pts = _flatten(positions[loops[lo:hi]].astype("f8"), unit_n[f])
        local = _earclip(pts)
        corners[offsets[f] : offsets[f] + per_face[f]] = lo + local
    return corners, tri_face


# --- the ear search ---------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0.0)


def _flatten(pts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Drop the axis the face faces along, keeping the winding.

    The two surviving axes are taken in cyclic order after the dropped one --
    ``(Z, X)`` for a Y-facing polygon -- because that is the pair whose
    counter-clockwise sense is ``+axis``. A face pointing down the *negative*
    axis swaps them, which flips the sense back. Getting this wrong would not
    crash: it would make every ear test reject and every concave face fall back
    to a fan, silently.
    """
    k = int(np.argmax(np.abs(normal))) if normal.any() else 2
    u, v = (k + 1) % 3, (k + 2) % 3
    if normal[k] < 0.0:
        u, v = v, u
    return np.stack([pts[:, u], pts[:, v]], axis=1)


def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _inside(pts: np.ndarray, a: int, b: int, c: int, p: int) -> bool:
    """Is corner *p* strictly inside triangle *abc*? Ties count as outside."""
    d1 = _cross2(pts[a], pts[b], pts[p])
    d2 = _cross2(pts[b], pts[c], pts[p])
    d3 = _cross2(pts[c], pts[a], pts[p])
    return d1 > 0.0 and d2 > 0.0 and d3 > 0.0


def _earclip(pts: np.ndarray) -> np.ndarray:
    """``(n - 2, 3)`` i8 local corner indices for one flattened polygon."""
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), dtype="i8")
    idx = list(range(n))
    if sum(_cross2(pts[0], pts[i], pts[i + 1]) for i in range(1, n - 1)) < 0.0:
        idx.reverse()

    tris: list[tuple[int, int, int]] = []
    p = 0
    stalled = 0
    while len(idx) > 3 and stalled <= len(idx):
        m = len(idx)
        a, b, c = idx[p % m], idx[(p + 1) % m], idx[(p + 2) % m]
        rest = [i for i in idx if i not in (a, b, c)]
        if _cross2(pts[a], pts[b], pts[c]) > 0.0 and not any(
            _inside(pts, a, b, c, i) for i in rest
        ):
            tris.append((a, b, c))
            del idx[(p + 1) % m]
            p = max(0, p - 1)
            stalled = 0
        else:
            p += 1
            stalled += 1

    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    else:
        # Stalled on a loop no ear search can resolve (self-intersecting, or
        # entirely collinear). Fanning the remainder keeps the triangle count
        # at n - 2, which is the contract every caller indexes by.
        tris.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    return np.array(tris, dtype="i8").reshape(-1, 3)
