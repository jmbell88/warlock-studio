"""The immutable CSR n-gon mesh, and the pure functions over it.

Clay stores a mesh the way a sparse matrix is stored: one flat array of
face corners (``loops``) and an array of offsets into it (``starts``), so face
*i* is the slice ``loops[starts[i]:starts[i + 1]]``. The alternative -- a
fixed-arity ``(F, 3)`` or ``(F, 4)`` index array -- was rejected for two
reasons that both bite later rather than now:

* **An n-gon needs no triangulation to be *stored*.** A cylinder's cap is one
  face with thirty-two corners, and a user who selects it should get one face,
  not thirty fans. Triangles are a *rendering* concern, produced on the way to
  the GPU by :func:`triangulate` and never written back.
* **The element ops add faces by concatenation.** Extrude and inset append
  corners to ``loops`` and offsets to ``starts``; nothing has to decide an
  arity up front or rebuild an index when a quad becomes a pentagon.

The cost is that ``starts`` has to be kept honest -- hence :func:`validate`,
which every op that builds a mesh by hand should be tested against. Nothing
else here re-checks it: :func:`edges`, :func:`_next_corner` and
:func:`face_normals` all *assume* an already-validated mesh, and on an
invalid one they do not raise, they quietly produce wrong answers -- a face
with fewer than three corners makes ``np.add.reduceat`` hand back a bare row
instead of a sum, and makes ``_next_corner`` write its wrap index into the
previous face's span. A primitive generator or an element op that builds a
``Mesh`` by hand should therefore run :func:`validate` in its tests -- and
every one of them funnels through :func:`~.topo.rebuild` for exactly that
reason.

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

from .earclip import corner_triangles, fan_corners

_DTYPES = {
    "positions": "f4",
    "loops": "i4",
    "starts": "i4",
    "material": "i4",
    "smooth": "?",
    "uv": "f4",
}


@dataclass(frozen=True, eq=False)
class Mesh:
    """A face-corner mesh in compressed-sparse-row form.

    ``eq=False`` deliberately: numpy arrays have no truthy ``==``, and identity
    comparison is what the GPU cache wants anyway.

    **UVs are per corner, and optional.** ``uv`` is one row per entry in
    ``loops`` rather than one per vertex, because a texture seam is exactly the
    case where one vertex carries two different texture coordinates -- two
    corners at one position -- and a per-vertex array cannot express it without
    splitting the position too, which would split the *topology* to describe a
    property of the shading. ``None`` means "this mesh has no texture
    coordinates", which is what every primitive generator produces and what
    keeps an authored document byte-identical to what it was before UVs
    existed. Every op states its policy for the field: preserved (the corner
    keeps its uv), interpolated (a new corner's uv is a lerp within the face
    that made it, which is seam-safe because it never reads another face's
    corners) or inherited (a new face copies the corners it was extruded from).
    """

    positions: np.ndarray  # (V, 3) f4
    loops: np.ndarray  # (L,)   i4  vertex index per face corner
    starts: np.ndarray  # (F+1,) i4  CSR offsets into loops
    material: np.ndarray  # (F,)   i4  index into the document palette
    smooth: np.ndarray  # (F,)   bool
    uv: np.ndarray | None = None  # (L, 2) f4, per face corner, or absent

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            arr = np.array(value, dtype=_DTYPES[f.name], copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, f.name, arr)


# --- validity ---------------------------------------------------------------


def validate(mesh: Mesh) -> None:
    """Raise :class:`ValueError` on a CSR that does not describe a mesh.

    Cheap enough to call after any hand-built mesh in a test, and every
    generator in ``primitives.py`` is tested through it. The rules are the
    ones a slice-by-offsets layout cannot enforce by construction.
    """
    if mesh.positions.ndim != 2 or mesh.positions.shape[1] != 3:
        raise ValueError(f"positions must be (V, 3), got {mesh.positions.shape}")
    if mesh.loops.ndim != 1 or mesh.starts.ndim != 1:
        raise ValueError("loops and starts must each be one-dimensional")
    if len(mesh.starts) == 0:
        raise ValueError("starts must start at 0 and end at len(loops); it is empty")
    if mesh.starts[0] != 0:
        raise ValueError(f"starts must start at 0, got {int(mesh.starts[0])}")
    if mesh.starts[-1] != len(mesh.loops):
        raise ValueError(
            f"starts must end at len(loops)={len(mesh.loops)}, got {int(mesh.starts[-1])}"
        )
    counts = np.diff(mesh.starts)
    if counts.size and counts.min() < 0:
        raise ValueError("starts must be monotonic; it decreases")
    if counts.size and counts.min() < 3:
        bad = int(np.argmin(counts))
        raise ValueError(f"face {bad} has fewer than 3 corners")
    n_faces = len(mesh.starts) - 1
    if len(mesh.loops) and (mesh.loops.min() < 0 or mesh.loops.max() >= len(mesh.positions)):
        raise ValueError(f"a loop index is out of range for {len(mesh.positions)} positions")
    if len(mesh.material) != n_faces:
        raise ValueError(f"material must be one per face ({n_faces}), got {len(mesh.material)}")
    if len(mesh.smooth) != n_faces:
        raise ValueError(f"smooth must be one per face ({n_faces}), got {len(mesh.smooth)}")
    if mesh.uv is not None and mesh.uv.shape != (len(mesh.loops), 2):
        raise ValueError(
            f"uv must be one (u, v) per face corner ({len(mesh.loops)}, 2), got {mesh.uv.shape}"
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


def reversed_corner_perm(starts: np.ndarray) -> np.ndarray:
    """A permutation of the corners that reverses every face's loop in place.

    ``loops[perm]`` is the same CSR with every face traversed the other way --
    the operation a mirroring transform needs, and the one a flip-normals op is
    made of. Vectorised because it runs on imported meshes: a face's corners
    span ``[s, e)``, so corner *i* maps to ``s + e - 1 - i``, which is the whole
    formula once ``s`` and ``e`` are broadcast per corner.
    """
    starts = starts.astype("i8")
    n_faces = len(starts) - 1
    if n_faces <= 0:
        return np.zeros(0, dtype="i8")
    counts = np.diff(starts)
    lo = np.repeat(starts[:-1], counts)
    hi = np.repeat(starts[1:], counts)
    return lo + hi - 1 - np.arange(int(starts[-1]), dtype="i8")


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

    The fan alone, with no concavity screen. Nothing in the package calls it:
    it survives as the *reference* the ear-clipping tests measure against, the
    same role the numpy fallbacks play beside the native kernels. Everything
    that actually triangulates goes through :func:`_corner_triangles`.
    """
    return fan_corners(mesh.starts)


def _corner_triangles(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The dispatcher: fans for convex faces, ears for the rest.

    Shared by :func:`triangulate` and :func:`render_arrays` -- the first maps
    the corners through ``loops``, the second through its own split-or-share
    corner table -- so shading, picking and export can never disagree about
    where a concave face's triangles are.
    """
    return corner_triangles(mesh.positions, mesh.loops, mesh.starts, face_normals(mesh))


def triangulate(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """``(tris, tri_face)``: an ``(T, 3)`` i4 index array and its face owners.

    A **fan from each face's first corner where that is correct, ear-clipping
    where it is not**. :mod:`.earclip` screens every face for a reflex corner in
    one vectorised pass; a mesh with none -- every primitive here, and every
    triangle-or-quad soup an exporter writes -- comes back byte-identical to the
    plain fan, and only the faces that fail pay for the search. The count is
    ``n - 2`` triangles per face either way.

    ``tri_face`` is returned rather than left to be derived because both its
    consumers would otherwise derive it differently: face picking maps a hit
    triangle back to the face the user selected, and the renderer groups
    triangles by ``material[tri_face]`` into draw ranges.
    """
    corners, tri_face = _corner_triangles(mesh)
    if len(corners) == 0:
        return np.zeros((0, 3), dtype="i4"), tri_face
    return mesh.loops[corners].astype("i4"), tri_face


# --- normals and render arrays ----------------------------------------------


def face_normals(mesh: Mesh) -> np.ndarray:
    """Unnormalised Newell normals, one per face, magnitude twice the area.

    **Assumes an already-validated mesh** -- as every function in this module
    does, and this one most consequentially: a ``starts`` array that does not
    describe the ``loops`` it indexes produces normals that are numbers rather
    than an exception, and those numbers go on to decide triangulation, shading
    and picking. That precondition is why ``serialize.read_wblk`` calls
    :func:`validate` before it hands a loaded mesh to anything.

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
    return _newell(
        mesh.positions,
        mesh.loops,
        mesh.loops[_next_corner(mesh)],
        mesh.starts[:-1].astype("i8"),
    )


def _newell(
    positions: np.ndarray,
    loops: np.ndarray,
    loops_next: np.ndarray,
    starts_head: np.ndarray,
) -> np.ndarray:
    """:func:`face_normals` over arrays rather than a :class:`Mesh`.

    Split out for :func:`render_from_layout`, which has the corner permutation
    already resolved and must not pay for ``_next_corner`` on every frame of a
    drag. ``loops_next`` is ``loops[_next_corner(mesh)]`` -- the *vertex* after
    each corner, not the corner -- so the two gathers here are the same two
    :func:`face_normals` always did, in the same order.
    """
    a = positions[loops].astype("f8")
    b = positions[loops_next].astype("f8")
    contrib = np.stack(
        [
            (a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2]),
            (a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0]),
            (a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1]),
        ],
        axis=1,
    )
    return np.add.reduceat(contrib, starts_head, axis=0)


# Compat: the tests and INVARIANTS.md name the private form, and it was the
# spelling three other modules in this package imported across the boundary --
# which is what the promotion above is for. The alias keeps both readable.
_face_normals = face_normals


def _normalize(v: np.ndarray) -> np.ndarray:
    """Unit vectors, leaving a zero-length row as zero rather than as NaN."""
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0.0)


def accumulate(loops: np.ndarray, values: np.ndarray, n_verts: int) -> np.ndarray:
    """``(n_verts, 3)`` f8 scatter-add of *values* onto vertex *loops*.

    ``np.bincount`` rather than ``np.add.at``: the latter is numpy's unbuffered
    ufunc path, which is an order of magnitude slower and is the single slowest
    primitive in Clay's rebuild-on-every-mesh-edit path
    (``clay_view.ClayView.sync`` keys its GPU cache on ``id(obj.mesh)``, so
    every extrude, inset and element-drag commit pays this).

    Both accumulate in *input* order, so the float sum order -- and therefore
    the result, bit for bit -- is unchanged. That is asserted rather than
    assumed: the output feeds :func:`_normalize` and then the GPU, where a
    changed last bit would be invisible right up until it was not.
    """
    return np.stack(
        [
            np.bincount(loops, weights=values[:, k], minlength=n_verts)
            for k in range(3)
        ],
        axis=1,
        # ``bincount`` ignores ``weights`` entirely when the input is empty and
        # hands back an *integer* table -- which a mesh with no smooth face at
        # all reaches, and which ``_normalize`` then refuses to divide into.
    ).astype("f8", copy=False)


# Compat: promoted from ``_accumulate`` when ops_topo, ops_subdiv and the
# viewer stopped being the only callers -- the same promotion ``face_normals``
# got above, for the same reason and with the same alias kept.
_accumulate = accumulate


def render_arrays(
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """``(positions, normals, uvs, indices)`` ready for a GPU upload.

    ``positions`` is ``(N, 3)`` f4, ``normals`` ``(N, 3)`` f4, ``uvs``
    ``(N, 2)`` f4 or ``None``, and ``indices`` a flat ``(T * 3,)`` u4 -- the
    same shapes ``viewer.gltf.Primitive`` uses, so the viewport's existing
    upload path takes them unchanged.

    **A flat face splits a vertex per corner; a smooth face shares one.** A
    flat cube therefore emits twenty-four vertices and a smooth cube eight.
    There is no averaging anywhere near a flat face: a shading implementation
    that averaged the six normals meeting at a cube corner would round every
    hard edge into a bevel that is not in the geometry, which is the single
    most visible way a low-poly editor can lie about the mesh it is about to
    export. A vertex touched by both a smooth and a flat face gets both
    treatments -- the smooth faces share it, the flat ones each get their own
    copy -- and only the smooth faces contribute to the shared normal.

    **A textured mesh gives that sharing up entirely**, and emits one vertex
    per corner. A UV is a per-corner property, so two corners of a smooth face
    that meet at one position and carry different texture coordinates cannot be
    one GPU vertex however the normals work out -- and a seam is precisely that
    case. Rather than split only the corners that disagree, which would make
    the emitted vertex count depend on the UV values, a mesh with ``uv`` splits
    all of them: the normals are computed exactly as above (the smooth ones
    still share their *value*, accumulated across the face's smooth
    neighbours), only the buffer is no longer de-duplicated. The extra vertices
    cost upload bandwidth on an imported mesh and nothing at all on an authored
    one, which has no UVs.

    **The split into :func:`render_layout` and :func:`render_from_layout` is
    what makes an element drag affordable**, and it costs this function
    nothing: the layout is built from the same mesh and consumed immediately,
    so every array below is produced by the same operations in the same order
    it always was. See :class:`RenderLayout` for what a drag reuses and why it
    is allowed to.
    """
    return render_from_layout(render_layout(mesh), mesh.positions)


@dataclass(frozen=True)
class RenderLayout:
    """Everything :func:`render_arrays` decides that moving a vertex cannot change.

    A ``Mesh`` is immutable, so the corner permutation, the shared/split
    vertex table and the index buffer are all pure functions of one mesh
    object -- which is what lets a drag build this once and reuse it for every
    frame, recomputing only the two arrays that a moved vertex actually
    changes: the positions and the normals.

    **The triangulation is in here even though it is not purely topological.**
    :func:`~.earclip.corner_triangles` screens faces for a reflex corner, and
    deforming a concave n-gon far enough could in principle change the answer.
    Reusing the pre-drag triangulation is nonetheless exactly right for the
    preview, because the index buffer *on the GPU* is the pre-drag one:
    ``scene.GpuPrimitive.update_vertices`` rewrites the vertex data and leaves
    the IBO alone, so recomputing indices per frame produced an array that was
    thrown away. The drag's commit goes through the ordinary
    :func:`render_arrays` path and re-triangulates.

    ``uv is not None`` discriminates the two shapes, the same rule
    :func:`render_arrays` reads: a mesh with UVs emits one vertex per corner
    and shares nothing, so ``used``/``flat_loops``/``foc_flat`` are unused and
    ``face_of_corner`` is.
    """

    n_faces: int
    n_verts: int
    loops: np.ndarray
    loops_next: np.ndarray
    starts_head: np.ndarray
    smooth_corner: np.ndarray
    smooth_loops: np.ndarray
    foc_smooth: np.ndarray  # face_of_corner[smooth_corner]
    indices: np.ndarray
    used: np.ndarray | None = None  # shared path
    flat_loops: np.ndarray | None = None  # shared path: loops[flat_corners]
    foc_flat: np.ndarray | None = None  # shared path
    face_of_corner: np.ndarray | None = None  # per-corner path
    uv: np.ndarray | None = None  # per-corner path


_EMPTY_LAYOUT_FIELDS = dict(
    n_verts=0,
    loops=np.zeros(0, dtype="i4"),
    loops_next=np.zeros(0, dtype="i8"),
    starts_head=np.zeros(0, dtype="i8"),
    smooth_corner=np.zeros(0, dtype="?"),
    smooth_loops=np.zeros(0, dtype="i4"),
    foc_smooth=np.zeros(0, dtype="i8"),
    indices=np.zeros(0, dtype="u4"),
)


def render_layout(mesh: Mesh) -> RenderLayout:
    """The topology half of :func:`render_arrays`, for reuse across a drag."""
    n_faces = face_count(mesh)
    if n_faces == 0:
        empty_uv = None if mesh.uv is None else np.zeros((0, 2), dtype="f4")
        return RenderLayout(n_faces=0, uv=empty_uv, **_EMPTY_LAYOUT_FIELDS)

    raw = face_normals(mesh)
    counts = np.diff(mesh.starts).astype("i8")
    face_of_corner = np.repeat(np.arange(n_faces, dtype="i8"), counts)
    smooth_corner = mesh.smooth[face_of_corner]
    smooth_loops = mesh.loops[smooth_corner]
    corners, _ = corner_triangles(mesh.positions, mesh.loops, mesh.starts, raw)
    common = dict(
        n_faces=n_faces,
        n_verts=len(mesh.positions),
        loops=mesh.loops,
        loops_next=mesh.loops[_next_corner(mesh)],
        starts_head=mesh.starts[:-1].astype("i8"),
        smooth_corner=smooth_corner,
        smooth_loops=smooth_loops,
        foc_smooth=face_of_corner[smooth_corner],
    )

    if mesh.uv is not None:
        return RenderLayout(
            indices=corners.reshape(-1).astype("u4"),
            face_of_corner=face_of_corner,
            uv=mesh.uv,
            **common,
        )

    # Shared half: only vertices that a smooth face touches get an entry, and
    # each accumulates the area-weighted normals of its smooth faces alone.
    n_verts = len(mesh.positions)
    used = np.zeros(n_verts, dtype=bool)
    used[smooth_loops] = True
    n_shared = int(used.sum())
    remap = np.zeros(n_verts, dtype="i8")
    remap[used] = np.arange(n_shared, dtype="i8")

    # Split half: one vertex per corner of a flat face, carrying that face's
    # own normal.
    flat_corners = np.flatnonzero(~smooth_corner)
    n_flat = len(flat_corners)

    corner_vertex = np.zeros(len(mesh.loops), dtype="i8")
    corner_vertex[smooth_corner] = remap[smooth_loops]
    corner_vertex[flat_corners] = n_shared + np.arange(n_flat, dtype="i8")

    return RenderLayout(
        indices=corner_vertex[corners].reshape(-1).astype("u4"),
        used=used,
        flat_loops=mesh.loops[flat_corners],
        foc_flat=face_of_corner[flat_corners],
        **common,
    )


def face_of_every_corner(layout: RenderLayout) -> np.ndarray:
    """Which face each corner belongs to, for every corner.

    ``layout.face_of_corner`` carries this only on the per-corner (UV) path, so
    it is derived from ``starts_head`` here rather than stored twice -- the runs
    between consecutive starts *are* the faces, by construction.
    """
    counts = np.diff(np.append(layout.starts_head, len(layout.loops)))
    return np.repeat(np.arange(len(layout.starts_head), dtype="i8"), counts)


def _newell_some(
    positions: np.ndarray,
    layout: RenderLayout,
    previous: np.ndarray,
    moved: np.ndarray,
) -> np.ndarray:
    """``previous`` with the moved vertices' faces recomputed, and nothing else.

    **Bit-identical to a full :func:`_newell` pass, not merely close**, and the
    unit of reuse is why: a face's raw normal is the reduction over that face's
    own corners and nothing outside it, so a face no moved vertex touches has an
    identical input and therefore an identical output. Reusing a *vertex*
    accumulation would not have this property -- floating-point addition is not
    associative, so subtracting an old contribution and adding a new one lands
    somewhere a fresh sum does not -- which is why the accumulation above is
    still recomputed whole from this array. ``preview_primitives`` promises
    byte-identity with ``to_primitives`` and that promise is kept exactly.
    """
    touched = np.zeros(layout.n_verts, dtype=bool)
    touched[moved] = True
    corners = touched[layout.loops]
    if not corners.any():
        return previous
    faces = np.unique(face_of_every_corner(layout)[corners])
    wanted = np.isin(face_of_every_corner(layout), faces)
    picked = np.flatnonzero(wanted)
    # Re-based starts for the gathered corner run: the *first* corner of each
    # face within the subset. Each face's slice holds the same values in the
    # same order it does in the full array, so each reduction is the same sum.
    counts = np.diff(np.append(layout.starts_head, len(layout.loops)))[faces]
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype("i8")
    raw = previous.copy()
    raw[faces] = _newell(
        positions, layout.loops[picked], layout.loops_next[picked], starts
    )
    return raw


def render_from_layout(
    layout: RenderLayout,
    positions: np.ndarray,
    *,
    moved: np.ndarray | None = None,
    previous: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """:func:`render_arrays`' vertex half, over *positions* and a cached layout.

    ``positions`` must be the ``(V, 3)`` array of the mesh the layout was built
    from, with the same length -- moved, not re-indexed.

    ``moved`` and ``previous`` together are the drag optimisation: the vertex
    indices this call moved, and the raw face normals the last call produced.
    Given both, only the faces those vertices touch are recomputed -- which on a
    200k-triangle import dragging a handful of vertices is the difference
    between the whole mesh and a dozen faces. Given either alone, or neither,
    every face is recomputed; a caller that is unsure which vertices moved must
    pass nothing, because a ``moved`` that is missing a vertex leaves that
    vertex's faces holding last frame's normals with nothing to say so.

    The raw face normals are returned through ``previous``' own array shape, so
    a caller keeping the optimisation alive holds onto what it got back --
    see :func:`raw_face_normals` for reading it out.
    """
    if layout.n_faces == 0:
        return (
            np.zeros((0, 3), dtype="f4"),
            np.zeros((0, 3), dtype="f4"),
            layout.uv,
            np.zeros(0, dtype="u4"),
        )

    if previous is not None and moved is not None and len(previous) == layout.n_faces:
        raw = _newell_some(positions, layout, previous, np.asarray(moved, dtype="i8"))
    else:
        raw = _newell(positions, layout.loops, layout.loops_next, layout.starts_head)
    unit = _normalize(raw)
    accum = _accumulate(layout.smooth_loops, raw[layout.foc_smooth], layout.n_verts)

    if layout.uv is not None:
        # One vertex per corner: the normal is still the shared value at the
        # vertex for a smooth corner, only the buffer is not de-duplicated.
        assert layout.face_of_corner is not None
        normals = unit[layout.face_of_corner]
        normals[layout.smooth_corner] = _normalize(accum[layout.smooth_loops])
        _stash(layout, raw)
        return (
            positions[layout.loops].astype("f4"),
            normals.astype("f4"),
            np.array(layout.uv, dtype="f4", copy=True),
            layout.indices,
        )

    assert layout.used is not None and layout.flat_loops is not None
    assert layout.foc_flat is not None
    out_positions = np.concatenate(
        [positions[layout.used], positions[layout.flat_loops]]
    ).astype("f4")
    out_normals = np.concatenate(
        [_normalize(accum[layout.used]), unit[layout.foc_flat]]
    ).astype("f4")
    _stash(layout, raw)
    return out_positions, out_normals, None, layout.indices


#: The last raw face normals each layout produced, so a drag's next frame can
#: reuse the faces it did not touch.
#:
#: Keyed by the layout's own ``id`` and holding a **weak** reference to nothing
#: -- the value is the array and the key is checked against a stored identity,
#: so a recycled id cannot serve another layout's normals: the entry records the
#: layout object itself and is only read back when that same object is passed
#: in. The layout is rebuilt whenever the mesh changes (it is a pure function of
#: an immutable ``Mesh``), which is what makes "same layout object" the correct
#: revision stamp here rather than any array's identity.
_RAW_CACHE: dict[int, tuple[RenderLayout, np.ndarray]] = {}


def _stash(layout: RenderLayout, raw: np.ndarray) -> None:
    # One entry, not a growing table: a drag touches one layout at a time per
    # material group, and an unbounded cache of face-normal arrays over a
    # session of imports is megabytes nobody asked for. Bounded by clearing when
    # it grows past a handful.
    if len(_RAW_CACHE) > 8:
        _RAW_CACHE.clear()
    _RAW_CACHE[id(layout)] = (layout, raw)


def raw_face_normals(layout: RenderLayout) -> np.ndarray | None:
    """The raw face normals the last :func:`render_from_layout` on this exact
    layout produced, or ``None``.

    The identity check is against the *layout object*, never against an array's
    ``id``: a layout is a pure function of an immutable ``Mesh``, so the same
    object means the same topology, where a recycled array id would mean
    nothing at all.
    """
    found = _RAW_CACHE.get(id(layout))
    if found is None or found[0] is not layout:
        return None
    return found[1]


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

    loops, uv = mesh.loops, mesh.uv
    if np.linalg.det(m[:3, :3]) < 0.0:
        # The UV travels with its corner, so the same permutation reverses it:
        # a mirrored face's corners keep the texture coordinates they had, in
        # the order the reversed loop reads them.
        perm = reversed_corner_perm(mesh.starts)
        loops = mesh.loops[perm]
        if uv is not None:
            uv = uv[perm]

    return Mesh(
        positions=positions,
        loops=loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=uv,
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
