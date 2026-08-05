"""The immutable CSR n-gon mesh, and the pure functions over it.

Build mode stores a mesh the way a sparse matrix is stored: one flat array of
face corners (``loops``) and an array of offsets into it (``starts``), so face
*i* is the slice ``loops[starts[i]:starts[i + 1]]``. The alternative -- a
fixed-arity ``(F, 3)`` or ``(F, 4)`` index array -- was rejected for two
reasons that both bite later rather than now:

* **An n-gon needs no triangulation to be *stored*.** A cylinder's cap is one
  face with thirty-two corners, and a user who selects it should get one face,
  not thirty fans. Triangles are a *rendering* concern, produced on the way to
  the GPU by :func:`triangulate` and never written back.
* **Phase 2 adds faces by concatenation.** Extrude and inset append corners to
  ``loops`` and offsets to ``starts``; nothing has to decide an arity up front
  or rebuild an index when a quad becomes a pentagon.

The cost is that ``starts`` has to be kept honest -- hence :func:`validate`,
which every op that builds a mesh by hand should be tested against.

**Every array is a copy, and every copy is read-only.** ``Mesh`` is a frozen
dataclass, but ``frozen=True`` only stops the *fields* being reassigned; a
caller holding ``mesh.positions`` could still write through it and silently
change a mesh someone else is holding. So :meth:`Mesh.__post_init__` copies
each array it is given and clears its writeable flag, which propagates to
every view taken of it -- ``face(mesh, i)`` hands back a slice that cannot be
written through. The copy also matters to *undo*: a view of a shared vertex
buffer reports its own tiny ``nbytes`` while keeping the whole base array
alive, and ``nbytes`` is exactly what ``studio.undo``'s eviction budget is
driven by. An edit that owns its arrays is an edit whose size is the truth.

Immutability is load-bearing for a third reason, outside this module
entirely: the viewport's GPU cache keys on ``id(obj.mesh)``. Every op here is
``Mesh -> Mesh``, so a changed mesh is a *different object* and the cache
misses exactly when it should. An op that mutated in place would leave the
cache holding a live key over stale geometry, and the viewport would render
the old shape forever with nothing in the data to say why. It is also what
makes undo a snapshot rather than an inverse operation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

_DTYPES = {
    "positions": "f4",
    "loops": "i4",
    "starts": "i4",
    "material": "i4",
    "smooth": "?",
}


@dataclass(frozen=True, eq=False)
class Mesh:
    """A face-corner mesh in compressed-sparse-row form.

    ``eq=False`` deliberately: numpy arrays have no truthy ``==``, and identity
    comparison is what the GPU cache wants anyway.
    """

    positions: np.ndarray  # (V, 3) f4
    loops: np.ndarray  # (L,)   i4  vertex index per face corner
    starts: np.ndarray  # (F+1,) i4  CSR offsets into loops
    material: np.ndarray  # (F,)   i4  index into the document palette
    smooth: np.ndarray  # (F,)   bool

    def __post_init__(self) -> None:
        for f in fields(self):
            arr = np.array(getattr(self, f.name), dtype=_DTYPES[f.name], copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, f.name, arr)


# --- validity ---------------------------------------------------------------


def validate(mesh: Mesh) -> None:
    """Raise :class:`ValueError` on a CSR that does not describe a mesh.

    Cheap enough to call after any hand-built mesh in a test, and every
    generator in ``primitives.py`` is tested through it. The rules are the
    ones a slice-by-offsets layout cannot enforce by construction.
    """
    if mesh.positions.ndim != 2 or (
        mesh.positions.size and mesh.positions.shape[1] != 3
    ):
        raise ValueError(f"positions must be (V, 3), got {mesh.positions.shape}")
    if mesh.loops.ndim != 1 or mesh.starts.ndim != 1:
        raise ValueError("loops and starts must each be one-dimensional")
    if len(mesh.starts) == 0:
        raise ValueError("starts must start at 0 and end at len(loops); it is empty")
    if mesh.starts[0] != 0:
        raise ValueError(f"starts must start at 0, got {int(mesh.starts[0])}")
    if mesh.starts[-1] != len(mesh.loops):
        raise ValueError(
            f"starts must end at len(loops)={len(mesh.loops)}, "
            f"got {int(mesh.starts[-1])}"
        )
    counts = np.diff(mesh.starts)
    if counts.size and counts.min() < 0:
        raise ValueError("starts must be monotonic; it decreases")
    if counts.size and counts.min() < 3:
        bad = int(np.argmin(counts))
        raise ValueError(f"face {bad} has fewer than 3 corners")
    n_faces = len(mesh.starts) - 1
    if len(mesh.loops) and (
        mesh.loops.min() < 0 or mesh.loops.max() >= len(mesh.positions)
    ):
        raise ValueError(
            f"a loop index is out of range for {len(mesh.positions)} positions"
        )
    if len(mesh.material) != n_faces:
        raise ValueError(
            f"material must be one per face ({n_faces}), got {len(mesh.material)}"
        )
    if len(mesh.smooth) != n_faces:
        raise ValueError(
            f"smooth must be one per face ({n_faces}), got {len(mesh.smooth)}"
        )


# --- accessors --------------------------------------------------------------


def face_count(mesh: Mesh) -> int:
    return len(mesh.starts) - 1


def face(mesh: Mesh, i: int) -> np.ndarray:
    """Face *i*'s corner vertex indices -- a read-only view, not a copy."""
    return mesh.loops[mesh.starts[i] : mesh.starts[i + 1]]


def _next_corner(mesh: Mesh) -> np.ndarray:
    """For every corner, the index *in loops* of the corner after it.

    The last corner of a face wraps to that face's first, which is what makes
    a single flat pass over ``loops`` produce every polygon edge exactly once.
    """
    nxt = np.arange(1, len(mesh.loops) + 1, dtype="i8")
    ends = mesh.starts[1:].astype("i8") - 1
    nxt[ends] = mesh.starts[:-1]
    return nxt


def edges(mesh: Mesh) -> np.ndarray:
    """Unique undirected edges as an ``(E, 2)`` i4 array, low index first.

    Undirected because an edge shared by two faces is drawn once in the
    wireframe and picked once by an edge-select, and the two faces traverse it
    in opposite directions.
    """
    if len(mesh.loops) == 0:
        return np.zeros((0, 2), dtype="i4")
    pairs = np.stack([mesh.loops, mesh.loops[_next_corner(mesh)]], axis=1)
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).astype("i4")


# --- triangulation ----------------------------------------------------------


def _fan_corners(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """``(tri_corners, tri_face)`` where tri_corners index *loops*, not vertices.

    Shared by :func:`triangulate` and :func:`render_arrays`: the first maps
    the corners through ``loops``, the second through its own split-or-share
    corner table, and neither should reimplement the fan.
    """
    counts = np.diff(mesh.starts).astype("i8")
    per_face = np.maximum(counts - 2, 0)
    total = int(per_face.sum())
    if total == 0:
        return np.zeros((0, 3), dtype="i8"), np.zeros(0, dtype="i4")
    tri_face = np.repeat(np.arange(len(counts), dtype="i8"), per_face)
    offsets = np.concatenate([[0], np.cumsum(per_face)[:-1]])
    local = np.arange(total, dtype="i8") - offsets[tri_face]
    first = mesh.starts[:-1].astype("i8")[tri_face]
    corners = np.stack([first, first + local + 1, first + local + 2], axis=1)
    return corners, tri_face.astype("i4")


def triangulate(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """``(tris, tri_face)``: an ``(T, 3)`` i4 index array and its face owners.

    A **fan from each face's first corner**. That is correct for convex faces
    and wrong for concave ones -- a fan across a reflex vertex puts a triangle
    outside the polygon -- and it is chosen knowingly: every primitive
    generator here produces convex faces, and so do Phase 2's extrude and
    inset. If a face-level tool ever produces a concave n-gon, this is the
    function that has to grow an ear-clipping path, not its callers.

    ``tri_face`` is returned rather than left to be derived because both its
    consumers would otherwise derive it differently: face picking maps a hit
    triangle back to the face the user selected, and the renderer groups
    triangles by ``material[tri_face]`` into draw ranges.
    """
    corners, tri_face = _fan_corners(mesh)
    if len(corners) == 0:
        return np.zeros((0, 3), dtype="i4"), tri_face
    return mesh.loops[corners].astype("i4"), tri_face


# --- normals and render arrays ----------------------------------------------


def _face_normals(mesh: Mesh) -> np.ndarray:
    """Unnormalised Newell normals, one per face, magnitude twice the area.

    Newell rather than the cross product of the first three corners: that
    shortcut degenerates to a zero vector whenever the first three corners are
    collinear, which an n-gon cap or a subdivided face reaches routinely, and
    it ignores every corner after the third on a face that is not quite
    planar. Newell sums over the whole loop and is the projected-area formula,
    so it is stable on both.

    Left unnormalised on purpose: the magnitude is proportional to the face's
    area, which makes an area-weighted smooth normal a plain sum.
    """
    if face_count(mesh) == 0:
        return np.zeros((0, 3), dtype="f8")
    a = mesh.positions[mesh.loops].astype("f8")
    b = mesh.positions[mesh.loops[_next_corner(mesh)]].astype("f8")
    contrib = np.stack(
        [
            (a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2]),
            (a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0]),
            (a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1]),
        ],
        axis=1,
    )
    return np.add.reduceat(contrib, mesh.starts[:-1].astype("i8"), axis=0)


def _normalize(v: np.ndarray) -> np.ndarray:
    """Unit vectors, leaving a zero-length row as zero rather than as NaN."""
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0.0)


def render_arrays(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(positions, normals, indices)`` ready for a GPU upload.

    ``positions`` is ``(N, 3)`` f4, ``normals`` ``(N, 3)`` f4 and ``indices``
    a flat ``(T * 3,)`` u4 -- the same shapes ``viewer.gltf.Primitive`` uses,
    so the viewport's existing upload path takes them unchanged.

    **A flat face splits a vertex per corner; a smooth face shares one.** A
    flat cube therefore emits twenty-four vertices and a smooth cube eight.
    There is no averaging anywhere near a flat face: a shading implementation
    that averaged the six normals meeting at a cube corner would round every
    hard edge into a bevel that is not in the geometry, which is the single
    most visible way a low-poly editor can lie about the mesh it is about to
    export. A vertex touched by both a smooth and a flat face gets both
    treatments -- the smooth faces share it, the flat ones each get their own
    copy -- and only the smooth faces contribute to the shared normal.
    """
    n_faces = face_count(mesh)
    if n_faces == 0:
        return (
            np.zeros((0, 3), dtype="f4"),
            np.zeros((0, 3), dtype="f4"),
            np.zeros(0, dtype="u4"),
        )

    raw = _face_normals(mesh)
    unit = _normalize(raw)
    counts = np.diff(mesh.starts).astype("i8")
    face_of_corner = np.repeat(np.arange(n_faces, dtype="i8"), counts)
    smooth_corner = mesh.smooth[face_of_corner]

    # Shared half: only vertices that a smooth face touches get an entry, and
    # each accumulates the area-weighted normals of its smooth faces alone.
    n_verts = len(mesh.positions)
    used = np.zeros(n_verts, dtype=bool)
    smooth_loops = mesh.loops[smooth_corner]
    used[smooth_loops] = True
    n_shared = int(used.sum())
    accum = np.zeros((n_verts, 3), dtype="f8")
    np.add.at(accum, smooth_loops, raw[face_of_corner[smooth_corner]])
    remap = np.zeros(n_verts, dtype="i8")
    remap[used] = np.arange(n_shared, dtype="i8")

    # Split half: one vertex per corner of a flat face, carrying that face's
    # own normal.
    flat_corners = np.flatnonzero(~smooth_corner)
    n_flat = len(flat_corners)

    positions = np.concatenate(
        [mesh.positions[used], mesh.positions[mesh.loops[flat_corners]]]
    ).astype("f4")
    normals = np.concatenate(
        [_normalize(accum[used]), unit[face_of_corner[flat_corners]]]
    ).astype("f4")

    corner_vertex = np.zeros(len(mesh.loops), dtype="i8")
    corner_vertex[smooth_corner] = remap[smooth_loops]
    corner_vertex[flat_corners] = n_shared + np.arange(n_flat, dtype="i8")

    corners, _ = _fan_corners(mesh)
    indices = corner_vertex[corners].reshape(-1).astype("u4")
    return positions, normals, indices


# --- transforms and measurement ---------------------------------------------


def transformed(mesh: Mesh, matrix: np.ndarray) -> Mesh:
    """The mesh through a 4x4, as a new :class:`Mesh`.

    Column-vector convention, ``M @ v``, the same one ``viewer.math3d`` uses.

    **A mirroring matrix reverses every face's loop.** If the linear part's
    determinant is negative the transform swaps handedness, so a loop that was
    counter-clockwise seen from outside becomes clockwise and every normal
    computed from it points inward. Reversing the loops puts the winding back
    where it was -- the mesh really is mirrored, and it really is still
    outward-facing. This is not optional bookkeeping: glTF has no negative
    scale that renders correctly everywhere, back-face culling would turn the
    object inside out in the viewport, and an exporter that shipped it would
    produce the classic "the model is inside out in the engine" bug with
    nothing in the file to explain it.
    """
    m = np.asarray(matrix, dtype="f8")
    if len(mesh.positions):
        ones = np.ones((len(mesh.positions), 1))
        homo = np.hstack([mesh.positions.astype("f8"), ones])
        positions = (m @ homo.T).T[:, :3]
    else:
        positions = np.zeros((0, 3), dtype="f8")

    loops = mesh.loops
    if np.linalg.det(m[:3, :3]) < 0.0:
        loops = np.concatenate(
            [face(mesh, i)[::-1] for i in range(face_count(mesh))]
            or [np.zeros(0, dtype="i4")]
        )

    return Mesh(
        positions=positions,
        loops=loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
    )


def bounds(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The axis-aligned box as ``(lo, hi)``, both ``(3,)`` f8.

    f8 rather than the positions' f4 because the consumers are camera framing
    and gizmo placement, which live in ``math3d``'s f8 world. An empty mesh
    measures as two zeroes rather than raising: an object with no geometry yet
    is a state the outliner can be in, and infinities would poison the
    document-wide box every other object is framed against.
    """
    if len(mesh.positions) == 0:
        return np.zeros(3, dtype="f8"), np.zeros(3, dtype="f8")
    return (
        mesh.positions.min(axis=0).astype("f8"),
        mesh.positions.max(axis=0).astype("f8"),
    )
