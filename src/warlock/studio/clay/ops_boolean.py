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

#: The three boolean operations, and what each does to the *first* selected
#: object -- which is the target in all three, exactly as it is for a merge.
#:
#: Written out rather than derived from the kernel's own names because the
#: **order matters for two of them and not for the third**, and that is the one
#: thing a user has to be told: a union is the same whichever object was
#: selected first, a difference is the first one *minus* the rest, and an
#: intersection is again order-free. The outliner's order decides, which is the
#: rule Merge Objects already establishes.
KINDS: tuple[str, ...] = ("union", "difference", "intersection")

#: ``glbimport.MAX_TRIANGLES`` caps a single *imported* mesh at 2M triangles,
#: but a boolean has no cap on how many objects are selected at once --
#: ``manifold3d`` builds one arrangement over every input together, so N
#: objects at the import ceiling is N * 2M triangles handed to the kernel in
#: one call, allocated on the frame thread with no way to refuse partway
#: through. Set to the same 2M: one mesh at the ceiling is exactly what one
#: import already allows through, so the budget is "no worse than importing
#: the equivalent geometry as a single object," not a new, lower bar.
MAX_BOOLEAN_TRIANGLES = 2_000_000


def _refuse_complexity(meshes: Sequence[bm.Mesh], kind: str) -> None:
    """Refuse before the kernel runs, from the triangle count it would face.

    An ``n``-cornered face triangulates into ``n - 2`` triangles, so summing
    ``len(mesh.loops) - 2 * face_count(mesh)`` over every input gives the exact
    count :func:`_run` will hand to ``manifold3d`` -- without doing the
    triangulation just to check. Summed across *all* inputs, not the largest
    one, because the kernel's cost is the size of the combined arrangement.
    """
    from .elements import OpError

    total = sum(len(m.loops) - 2 * bm.face_count(m) for m in meshes)
    if total > MAX_BOOLEAN_TRIANGLES:
        raise OpError(
            f"This {kind} would need {total:,} triangles, past the "
            f"{MAX_BOOLEAN_TRIANGLES:,} Clay works with. Select fewer objects, "
            "or simplify them first."
        )


def union(objs: Sequence[Obj]) -> bm.Mesh:
    """The boolean union of several objects, in the **first** one's frame."""
    return boolean(objs, "union")


def difference(objs: Sequence[Obj]) -> bm.Mesh:
    """The first object with every other one cut out of it.

    The one boolean whose **order matters**, and the reason the whole family
    takes the outliner's order rather than a set: "the block minus the hole" and
    "the hole minus the block" are different shapes, and only one of them is
    ever what was meant.
    """
    return boolean(objs, "difference")


def intersection(objs: Sequence[Obj]) -> bm.Mesh:
    """Only what every selected object has in common."""
    return boolean(objs, "intersection")


def boolean(objs: Sequence[Obj], kind: str = "union") -> bm.Mesh:
    """A boolean of several objects, in the **first** one's frame.

    The frame convention is :func:`.ops.join`'s exactly, and for its reasons:
    every object after the first is carried through ``inv(first) @ own`` so it
    lands where it was drawn, while the first object's mesh is passed through
    untouched rather than multiplied by an ``inv(M) @ M`` that is only identity
    to within a rounding error. The target keeps its transform, so nothing
    about it should move.

    Disjoint inputs are *not* an error and are not special-cased **for a
    union**: the union of two solids that do not touch is a two-shell solid,
    which is what a weld at ``eps=0`` would also have produced, and refusing it
    would mean the op behaved differently depending on where the user had
    dragged something. A difference or an intersection over disjoint solids is
    equally well defined -- the whole of the first, and nothing at all -- and
    the empty answer is the one case worth naming, which :func:`.ops_boolean`'s
    caller does rather than this.
    """
    from .elements import OpError
    from .ops import _into

    if kind not in KINDS:
        raise OpError(f"{kind!r} is not one of {', '.join(KINDS)}.")
    if len(objs) < 2:
        # "to union" / "to subtract" / "to intersect": the verb rather than the
        # noun, because the sentence is an instruction.
        verb = {"union": "union", "difference": "subtract", "intersection": "intersect"}
        raise OpError(f"Select at least two objects to {verb[kind]}.")
    target = objs[0]
    meshes = [target.mesh] + [bm.transformed(o.mesh, _into(target, o)) for o in objs[1:]]

    _refuse_complexity(meshes, kind)
    result = _run(meshes, [o.name for o in objs], kind)
    return _to_csr(result, target.mesh, kind)


def _run(meshes: Sequence[bm.Mesh], names: Sequence[str], kind: str = "union"):
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
            f"A boolean {kind} needs the trimesh and manifold3d packages, "
            "which are not installed in this environment."
        ) from error

    solids = []
    for mesh, name in zip(meshes, names, strict=True):
        tris, _face = bm.triangulate(mesh)
        if len(tris) == 0:
            raise OpError(f"{name} has no faces, so there is nothing to {kind}.")
        solids.append(
            trimesh.Trimesh(
                vertices=np.asarray(mesh.positions, dtype="f8"),
                faces=np.asarray(tris, dtype="i8"),
                process=False,
            )
        )

    try:
        # ``trimesh.boolean`` names all three; the kind is validated in
        # :func:`boolean` so this cannot reach for one that is not there.
        return getattr(trimesh.boolean, kind)(solids, engine="manifold")
    except ImportError as error:
        # What a missing *backend* looks like: trimesh imports fine and only
        # raises when the engine is asked for. Named separately because the
        # remedy is different from a missing trimesh.
        raise OpError(
            f"A boolean {kind} needs the manifold3d package, which is not "
            "installed in this environment."
        ) from error
    except ValueError as error:
        # The kernel's own refusal, most often "not all meshes are volumes".
        # Rewritten rather than passed through: ``check_volume`` phrases it for
        # a library caller, and what the user needs is the remedy.
        raise OpError(
            f"A boolean {kind} needs every selected object to be a closed "
            "solid. One of them has holes or loose faces -- fill them first, "
            "or use Merge Objects, which keeps the geometry as it is."
        ) from error
    except Exception as error:  # the kernel's own internal failures
        raise OpError(f"The {kind} could not be computed: {error}") from error


def _to_csr(result, target: bm.Mesh, kind: str = "union") -> bm.Mesh:
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
            f"The {kind} came out empty. A union of solids that enclose no "
            "volume between them, an intersection of solids that do not "
            "overlap, or a difference that removes everything -- each is a "
            "correct answer and an object with nothing in it, which on screen "
            "is indistinguishable from the operation having failed."
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
