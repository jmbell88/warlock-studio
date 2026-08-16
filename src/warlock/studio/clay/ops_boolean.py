"""Constructive solid geometry: the one op here that removes what it overlaps.

This is the counterpart :func:`.ops.join` names in its own docstring and
deliberately is not. A join *welds*: it concatenates the geometry, fuses points
that coincide, and keeps everything inside the overlap, because throwing that
away means classifying every face against every other solid and a wrong
classification silently deletes a surface the user can see. That reasoning has
not changed -- what changed is that the classifier is no longer ours to get
wrong. ``manifold3d`` is a published, tested CSG kernel; ``trimesh.boolean``
dispatches to it; so the op that could not be written by hand is one call.

Both survive, and the manual says which is which, because they answer different
questions. Two shapes that *touch* and should read as one surface is a join.
Two shapes that *interpenetrate* and should read as one solid is a union: the
walls buried inside the other body are exactly what has to go, and a join leaves
them there to z-fight, to be exported, and to make the result unwatertight.

Three costs are real and stated rather than hidden.

**UVs do not survive.** A boolean recuts every face it touches; there is no
correspondence between an output corner and any input corner, so the honest
answer is ``uv=None`` rather than coordinates invented from the nearest
survivor. Unwrap afterwards.

**N-gons do not survive either.** The kernel works in triangles and returns
triangles, so a union of two subdivision-friendly quad boxes comes back as a
triangle soup. That is a real downgrade to the topology and is why *Merge
Objects* is not being replaced by this.

**And every input must be a closed volume.** "Union" is only defined over
solids: an open surface has no inside, so there is nothing to say whether a
point is in it. The kernel refuses, and the refusal is passed through as an
:class:`~.elements.OpError` naming the remedy rather than the exception.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import mesh as bm
from .document import Obj


def union(objs: Sequence[Obj]) -> bm.Mesh:
    """The boolean union of several objects, in the **first** one's frame.

    The frame convention is :func:`.ops.join`'s exactly, and for its reasons:
    every object after the first is carried through ``inv(first) @ own`` so it
    lands where it was drawn, while the first object's mesh is passed through
    untouched rather than multiplied by an ``inv(M) @ M`` that is only identity
    to within a rounding error. The target keeps its transform, so nothing
    about it should move.

    Disjoint inputs are *not* an error and are not special-cased: the union of
    two solids that do not touch is a two-shell solid, which is exactly what a
    weld at ``eps=0`` would also have produced, and refusing it would mean this
    op behaved differently depending on where the user had dragged something.
    """
    from .elements import OpError
    from .ops import _into

    if len(objs) < 2:
        raise OpError("Select at least two objects to union.")
    target = objs[0]
    meshes = [target.mesh] + [bm.transformed(o.mesh, _into(target, o)) for o in objs[1:]]

    result = _run(meshes, [o.name for o in objs])
    return _to_csr(result, target.mesh)


def _run(meshes: Sequence[bm.Mesh], names: Sequence[str]):
    """Hand the triangles to the kernel. -> a ``trimesh.Trimesh``.

    ``trimesh`` and ``manifold3d`` are imported here rather than at module
    scope, under the same rule Pillow follows in ``serialize.py``: this package
    is imported to answer questions about what an extrude does to a UV, and a
    top-level import would put a CSG kernel behind every one of them.
    """
    from .elements import OpError

    try:
        import trimesh
    except ImportError as error:  # pragma: no cover - a broken install
        raise OpError(
            "Union needs the trimesh and manifold3d packages, which are not "
            "installed in this environment."
        ) from error

    solids = []
    for mesh, name in zip(meshes, names, strict=True):
        tris, _face = bm.triangulate(mesh)
        if len(tris) == 0:
            raise OpError(f"{name} has no faces, so there is nothing to union.")
        solids.append(
            trimesh.Trimesh(
                vertices=np.asarray(mesh.positions, dtype="f8"),
                faces=np.asarray(tris, dtype="i8"),
                process=False,
            )
        )

    try:
        return trimesh.boolean.union(solids, engine="manifold")
    except ImportError as error:
        # What a missing *backend* looks like: trimesh imports fine and only
        # raises when the engine is asked for. Named separately because the
        # remedy is different from a missing trimesh.
        raise OpError(
            "Union needs the manifold3d package, which is not installed in "
            "this environment."
        ) from error
    except ValueError as error:
        # The kernel's own refusal, most often "not all meshes are volumes".
        # Rewritten rather than passed through: ``check_volume`` phrases it for
        # a library caller, and what the user needs is the remedy.
        raise OpError(
            "Union needs every selected object to be a closed solid. One of "
            "them has holes or loose faces -- fill them first, or use Merge "
            "Objects, which keeps the geometry as it is."
        ) from error
    except Exception as error:  # the kernel's own internal failures
        raise OpError(f"The union could not be computed: {error}") from error


def _to_csr(result, target: bm.Mesh) -> bm.Mesh:
    """A ``trimesh.Trimesh`` back into this package's CSR form.

    ``material`` and ``smooth`` are taken from the *target*'s first face rather
    than invented. There is no correspondence between an output face and any
    input face, so there is no per-face answer to carry across -- and the
    target's is not an arbitrary choice among the inputs, because the target is
    already what a merge keeps the name, the transform and the frame of.

    ``uv`` is ``None``, which is the module docstring's stated cost.
    """
    from .elements import OpError

    faces = np.asarray(result.faces, dtype="i4")
    if len(faces) == 0:
        raise OpError(
            "The union came out empty. That happens when the objects do not "
            "enclose any volume between them."
        )
    slot = int(target.material[0]) if len(target.material) else 0
    smooth = bool(target.smooth[0]) if len(target.smooth) else False
    return bm.Mesh(
        positions=np.asarray(result.vertices, dtype="f4"),
        loops=faces.reshape(-1),
        starts=np.arange(len(faces) + 1, dtype="i4") * 3,
        material=np.full(len(faces), slot, dtype="i4"),
        smooth=np.full(len(faces), smooth, dtype=bool),
        uv=None,
    )
