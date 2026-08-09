"""Texture coordinates for authored geometry: projection, not solving.

``Mesh.uv`` has been a per-corner field since the CSR layout was written and
nothing has ever filled it in -- every generator produced ``None`` and every op
carried that ``None`` faithfully forward. This is the half that produces one,
and it is deliberately the *cheap* half: a planar projection per face, chosen by
which axis the face points along most.

**Not LSCM, and that is a decision rather than a shortfall.** A conformal solver
is the right answer for an organic mesh and the wrong shape for this codebase:
it is an eigenproblem per island, it needs seams the user has drawn, and it
fails in ways ("the solve did not converge") that a modelling panel has nothing
useful to say about. A box projection is predictable, instant, has no failure
mode, and is exactly right for the blockout geometry Clay produces. When
something here needs a real unwrap, it will be because a real surface arrived,
and that is the moment to write it.

**Islands may overlap, and that is what a cube projection is.** Faces pointing
+X and -X land in the same square, because the projection is by axis *pair*
rather than by direction -- which is what Blender's Cube Projection does and
what a user asking for a box unwrap is asking for. A layout with no overlap at
all is a packing problem, and the primitives that can answer it cheaply (a box's
six faces in a cross) answer it in their own generator instead.

Everything here is pure: numpy and :mod:`~.mesh`, nothing from ``service``,
``queue`` or ``studio``, so every rule about where a corner lands is assertable
from a hand-built mesh.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh, _face_normals, face_count

__all__ = ["box_unwrap", "planar_unwrap"]

# The two axes each dominant normal projects onto, and which of them is flipped
# so the result is not mirrored when seen from outside the face. Written out
# rather than derived from a cross product: the table *is* the convention, and
# deriving it would put the sign of every island at the mercy of an argument
# order nobody would think to test.
#
# Index is the dominant axis (0 = X, 1 = Y, 2 = Z); the entry is the pair of
# axes ``(u, v)`` the face's positions are read from. A face whose normal points
# the *negative* way has its u mirrored within the object's own footprint, which
# is what keeps lettering on the far side of a box the right way round.
_AXES: tuple[tuple[int, int], ...] = (
    (2, 1),  # X-facing: u from Z, v from Y
    (0, 2),  # Y-facing: u from X, v from Z
    (0, 1),  # Z-facing: u from X, v from Y
)


def dominant_axis(normals: np.ndarray) -> np.ndarray:
    """Which axis each normal points along most. ``(F,)`` of 0, 1 or 2."""
    return np.argmax(np.abs(np.asarray(normals, dtype="f8")), axis=1).astype("i4")


def planar_unwrap(mesh: Mesh, axis: int = 1) -> Mesh:
    """Project every face onto one axis. The degenerate case, and a useful one.

    A whole mesh flattened down one axis is what a decal, a floor or a signboard
    wants, and it is what :func:`box_unwrap` reduces to when every face happens
    to point the same way.
    """
    return _projected(mesh, np.full(face_count(mesh), int(axis) % 3, dtype="i4"))


def box_unwrap(mesh: Mesh) -> Mesh:
    """Project each face along whichever axis its normal is closest to.

    The result is normalised over the mesh's own bounding box, so a long thin
    object is not stretched into a square: one unit of UV is the *largest*
    extent of the mesh, and the shorter axes use proportionally less of the
    square. Scaling each island to fill the square independently is the obvious
    alternative and is wrong -- it makes texel density depend on how big a face
    happens to be, so a checker map comes out with different sized squares on
    every panel of one object.
    """
    if face_count(mesh) == 0:
        return mesh
    # Computed once and handed down. ``_projected`` needs the same normals to
    # decide which corners are mirrored, and asking for them again is a second
    # full pass over every loop in the mesh for an answer already in hand.
    normals = _face_normals(mesh)
    return _projected(mesh, dominant_axis(normals), normals=normals)


def _projected(mesh: Mesh, axes: np.ndarray, *, normals: np.ndarray | None = None) -> Mesh:
    """One planar projection per face, into a shared normalised square.

    ``normals`` is an optimisation and never a second opinion: a caller that has
    already measured them passes them in, and one that has not -- the planar
    unwrap, which picks its axis without looking at a normal -- leaves it None
    and gets the same array measured here.
    """
    positions = np.asarray(mesh.positions, dtype="f8")
    if len(positions) == 0:
        return mesh
    low = positions.min(axis=0)
    span = float(np.max(positions.max(axis=0) - low))
    # A flat or single-point mesh has no extent to normalise by. One is the
    # scale that leaves the coordinates as they are, which keeps a plane at
    # y = 0 unwrapping sensibly rather than dividing by zero.
    scale = 1.0 / span if span > 1e-12 else 1.0

    counts = np.diff(np.asarray(mesh.starts, dtype="i8"))
    face_of = np.repeat(np.arange(face_count(mesh), dtype="i8"), counts)
    corner_axis = axes[face_of]
    corner_pos = (positions[np.asarray(mesh.loops, dtype="i8")] - low) * scale

    extent = (positions.max(axis=0) - low) * scale
    uv = np.zeros((len(mesh.loops), 2), dtype="f4")
    if normals is None:
        normals = _face_normals(mesh)
    for axis in range(3):
        which = np.flatnonzero(corner_axis == axis)
        if not len(which):
            continue
        u_axis, v_axis = _AXES[axis]
        u = corner_pos[which, u_axis]
        # Mirrored within the object's own footprint rather than within the
        # unit square: the coordinates are normalised by the *largest* extent,
        # so a short axis does not reach 1 and ``1 - u`` would push that face's
        # island off its own geometry. The flip is per corner, taken from the
        # sign of its own face's normal, because opposite faces share the
        # square and without it one of the pair reads mirrored.
        facing = normals[face_of[which], axis] < 0.0
        uv[which, 0] = np.where(facing, extent[u_axis] - u, u)
        uv[which, 1] = corner_pos[which, v_axis]

    from dataclasses import replace

    return replace(mesh, uv=uv)
