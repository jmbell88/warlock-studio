"""The parts every element op is assembled from.

An op that rewrites topology is nearly always the same four moves in a
different order -- keep some faces, renumber the vertices, splice corners into a
neighbour's loop, walk the border of a region -- and writing them out per op is
how two ops end up disagreeing about what happens to a uv. So they live here,
once, and every op funnels its result through :func:`rebuild`.

**:func:`rebuild` takes ``uv`` as a required keyword.** That is the whole reason
it exists rather than callers constructing a :class:`~.mesh.Mesh` directly: a
face-corner uv array is exactly as long as ``loops`` and indexed the same way,
so any rewrite of ``loops`` invalidates it unless the same permutation is
applied. Making the parameter required means an op cannot forget -- it has to
write ``uv=None`` and say in its docstring that it drops them, or produce the
array. Every op in Clay states its uv policy in one of three words: *preserved*
(the corner travels, so its uv travels), *interpolated* (a new corner's uv is a
lerp within the face that made it, which never reads another face's corners and
is therefore seam-safe), or *inherited* (a new face copies the corners it grew
from).

Nothing here imports imgui, moderngl, pygame or ``service``; nothing here knows
what a document or an undo step is. An op is a pure function of a mesh and a
selection, and this is the layer beneath it.
"""

from __future__ import annotations

import numpy as np

from .adjacency import adjacency
from .mesh import Mesh, reversed_corner_perm

__all__ = [
    "compact_vertices",
    "corner_spans",
    "flat_next",
    "rebuild",
    "region_boundary_corners",
    "reversed_corner_perm",
    "splice_corners",
    "starts_from_counts",
    "take_faces",
]


def starts_from_counts(counts: np.ndarray) -> np.ndarray:
    """CSR offsets from per-face corner counts -- the ``[0, cumsum]`` idiom."""
    counts = np.asarray(counts, dtype="i8").reshape(-1)
    return np.concatenate([[0], np.cumsum(counts)]).astype("i4")


def rebuild(
    positions: np.ndarray,
    loops: np.ndarray,
    starts: np.ndarray,
    material: np.ndarray,
    smooth: np.ndarray,
    *,
    uv: np.ndarray | None,
) -> Mesh:
    """Assemble a :class:`~.mesh.Mesh`, forcing an explicit uv decision.

    See the module docstring: ``uv`` is keyword-only and has no default, so
    every op has to have thought about it.
    """
    return Mesh(
        positions=np.asarray(positions, dtype="f4").reshape(-1, 3),
        loops=np.asarray(loops, dtype="i4").reshape(-1),
        starts=np.asarray(starts, dtype="i4").reshape(-1),
        material=np.asarray(material, dtype="i4").reshape(-1),
        smooth=np.asarray(smooth, dtype=bool).reshape(-1),
        uv=None if uv is None else np.asarray(uv, dtype="f4").reshape(-1, 2),
    )


def corner_spans(starts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The corner indices of *faces*, concatenated in the given face order.

    Vectorised because the alternative is a Python loop per face, which is the
    one thing this module exists to keep out of the ops.
    """
    starts = starts.astype("i8")
    faces = np.asarray(faces, dtype="i8").reshape(-1)
    if len(faces) == 0:
        return np.zeros(0, dtype="i8")
    counts = starts[faces + 1] - starts[faces]
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype="i8")
    which = np.repeat(np.arange(len(faces), dtype="i8"), counts)
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return starts[faces][which] + (np.arange(total, dtype="i8") - offs[which])


def flat_next(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(offsets, next, prev)`` for corners gathered face by face.

    The same wrap-within-a-face trick :mod:`.adjacency` applies to ``loops``,
    applied instead to a *gathered* corner array -- which is what an op working
    on a subset of faces needs, since the subset's corners are contiguous in the
    gather but not in the mesh.
    """
    counts = np.asarray(counts, dtype="i8").reshape(-1)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    total = int(offsets[-1])
    nxt = np.arange(1, total + 1, dtype="i8")
    if len(counts):
        nxt[offsets[1:] - 1] = offsets[:-1]
    prv = np.empty(total, dtype="i8")
    if total:
        prv[nxt] = np.arange(total, dtype="i8")
    return offsets, nxt, prv


def take_faces(mesh: Mesh, faces: np.ndarray) -> Mesh:
    """The sub-mesh of the given faces, in the given order. Vertices untouched.

    Vertices are deliberately *not* compacted here: an op that deletes faces and
    then adds new ones wants to keep the old vertex numbering while it works and
    compact once at the end, and an op that is only reordering faces wants the
    numbering left alone entirely. UV **preserved** -- the corners travel whole.
    """
    faces = np.asarray(faces, dtype="i8").reshape(-1)
    corners = corner_spans(mesh.starts, faces)
    counts = mesh.starts.astype("i8")[faces + 1] - mesh.starts.astype("i8")[faces]
    return rebuild(
        mesh.positions,
        mesh.loops[corners],
        starts_from_counts(counts),
        mesh.material[faces],
        mesh.smooth[faces],
        uv=None if mesh.uv is None else mesh.uv[corners],
    )


def compact_vertices(mesh: Mesh) -> tuple[Mesh, np.ndarray]:
    """Drop vertices no face uses. Returns ``(mesh, old_to_new)``.

    ``old_to_new`` is ``(V,)`` i4 with −1 for a dropped vertex, which is what a
    caller needs to carry a *selection* across the same edit -- an op that
    deletes faces has to renumber the vertices the user still has selected, and
    guessing the mapping afterwards is not possible.
    """
    used = np.zeros(len(mesh.positions), dtype=bool)
    used[mesh.loops] = True
    if used.all():
        return mesh, np.arange(len(mesh.positions), dtype="i4")
    old_to_new = np.full(len(mesh.positions), -1, dtype="i8")
    old_to_new[used] = np.arange(int(used.sum()), dtype="i8")
    out = rebuild(
        mesh.positions[used],
        old_to_new[mesh.loops],
        mesh.starts,
        mesh.material,
        mesh.smooth,
        uv=mesh.uv,
    )
    return out, old_to_new.astype("i4")


def splice_corners(
    loops: np.ndarray,
    starts: np.ndarray,
    uv: np.ndarray | None,
    *,
    after: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
    value_uv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Insert corners into the middle of existing face loops.

    ``after[i]`` is a corner index; ``counts[i]`` vertices are inserted directly
    after it, taken in order from the flat ``values`` array. A face grows by the
    number of corners spliced into it and keeps everything else.

    This is what stops a subdivision or a loop cut cracking. When one face is
    split at the midpoint of an edge, the *unselected* face on the other side of
    that edge still ends at the two original vertices, and the new midpoint sits
    on its edge without being one of its corners -- a T-junction, which renders
    as a hairline seam that moves when the camera does. Splicing the midpoint
    into the neighbour's loop turns its quad into a pentagon and the crack
    closes with no extra faces.

    UV is the caller's: pass ``value_uv`` when the mesh has uvs, and the same
    insertion applies to both arrays so corners and uvs stay in step.
    """
    after = np.asarray(after, dtype="i8").reshape(-1)
    counts = np.asarray(counts, dtype="i8").reshape(-1)
    values = np.asarray(values, dtype="i4").reshape(-1)
    if len(after) == 0 or len(values) == 0:
        return loops, starts, uv

    # np.insert wants its indices sorted to keep repeated ones in order, so the
    # insertion points are sorted and the flat ``values`` array is gathered to
    # match rather than assumed to arrive in that order.
    src = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.argsort(after, kind="stable")
    after, counts, src = after[order], counts[order], src[order]
    which = np.repeat(np.arange(len(after), dtype="i8"), counts)
    local = (
        np.arange(int(counts.sum()), dtype="i8")
        - np.concatenate([[0], np.cumsum(counts)[:-1]])[which]
    )
    take = src[which] + local
    # np.insert places values *before* the given index, so "after corner c" is
    # "before c + 1".
    at = np.repeat(after + 1, counts)
    new_loops = np.insert(loops, at, values[take])

    starts_i8 = starts.astype("i8")
    face_of = np.repeat(np.arange(len(starts_i8) - 1, dtype="i8"), np.diff(starts_i8))
    grown = np.bincount(face_of[after], weights=counts, minlength=len(starts_i8) - 1)
    new_starts = starts_from_counts(np.diff(starts_i8) + grown.astype("i8"))

    new_uv = uv
    if uv is not None:
        rows = np.zeros((len(values), 2), dtype="f4") if value_uv is None else value_uv
        new_uv = np.insert(uv, at, np.asarray(rows, dtype="f4")[take], axis=0)
    return new_loops.astype("i4"), new_starts, new_uv


def region_boundary_corners(mesh: Mesh, faces: np.ndarray) -> np.ndarray:
    """Corners of *faces* whose edge no other selected face shares.

    The border of a selected region, in corner form -- which is the form
    extrude wants, because a wall quad is built from the corner's own vertex,
    the next corner's vertex and the two new ones above them, and the corner is
    also where the uv to inherit lives. Ordered by corner index, so a region's
    border comes back grouped by face and in each face's own traversal order,
    which is what makes the wall quads' winding follow the region's.
    """
    faces = np.asarray(faces, dtype="i8").reshape(-1)
    a = adjacency(mesh)
    if len(faces) == 0 or len(mesh.loops) == 0:
        return np.zeros(0, dtype="i8")
    selected = np.zeros(len(mesh.starts) - 1, dtype=bool)
    selected[faces] = True
    mine = np.flatnonzero(selected[a.corner_face])
    per_edge = np.bincount(a.corner_edge[mine], minlength=a.n_edges)
    return mine[per_edge[a.corner_edge[mine]] == 1]
