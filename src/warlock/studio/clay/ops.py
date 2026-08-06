"""Object-level operations: mirror, snap, bake and duplicate.

These sit beside :mod:`~warlock.studio.clay.document` rather than inside it
because they are *geometry*, not bookkeeping: each one takes an :class:`Obj`
and hands back a new one, and none of them knows that a document, a history or
a selection exists. The document layer is what turns a returned object into a
:class:`~.edits.MeshEdit` or a :class:`~.edits.TransformEdit`.

**Object-level, and that is the whole boundary.** The element ops -- extrude,
inset, bevel, dissolve, subdivide and the rest -- live in :mod:`.ops_topo`,
:mod:`.ops_dissolve`, :mod:`.ops_subdiv` and :mod:`.ops_bevel`, because they
take a mesh and a *selection* rather than an object, and they hand back the
selection the UI should show next. An op here moves or rebuilds a whole
object; an op there rewrites what is inside one.

**Mirror bakes into the mesh; it never sets a negative scale on the node.**
Writing ``scale.x = -1`` is one line and looks equivalent, and it is the single
most portable-looking way to ship an inside-out asset. glTF's own specification
says a negative determinant flips the winding order, and readers genuinely
disagree about whether to honour that: some flip their front-face convention,
some flip the indices, some ignore it. An asset that renders correctly in the
viewport and inside out in an engine has nothing in the file to explain itself.
:func:`~.mesh.transformed` already reverses every face loop when the linear
part's determinant is negative, so baking produces a mesh that is mirrored,
positively scaled and outward-wound -- true under every reader, with no
convention to agree on.

**Snapping at step zero is the identity.** It reads like a degenerate case to
guard against and it is really the *off* switch: the grid size comes straight
from a settings field, the user turns snapping off by clearing it, and a naive
``round(v / step) * step`` would divide by zero or quantise the whole scene
onto the origin. Every snap function here returns its input unchanged at zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from ..viewer import math3d as m3
from . import mesh as bm
from .document import Obj

_AXIS_NAMES = ("x", "y", "z")

# A duplicate's name counts up in a three-digit suffix, as Blender's does: the
# outliner sorts alphabetically, and "Box.010" sorting between "Box.001" and
# "Box.2" is the sort of thing nobody reports as a bug and everybody notices.
_SUFFIX = re.compile(r"^(?P<base>.*)\.(?P<n>\d{3,})$")


def mirror(obj: Obj, axis: int) -> Obj:
    """The object reflected across the plane through its own origin.

    ``axis`` is 0, 1 or 2 -- X, Y or Z in the object's *local* frame, which is
    the frame a modelling package's "Mirror X" works in and the only one that
    needs no second argument. Mirroring across a world plane is this composed
    with the object's transform, and belongs to whatever tool offers it rather
    than here.

    The mesh is rebuilt and the transform is left exactly as it was: see the
    module docstring for why a negative node scale is not an option. The uid is
    kept, because this edits one object rather than making another.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2 ({', '.join(_AXIS_NAMES)}), got {axis!r}")
    matrix = np.eye(4)
    matrix[axis, axis] = -1.0
    return replace(obj, mesh=bm.transformed(obj.mesh, matrix))


def snap_value(value: float, step: float) -> float:
    """``value`` to the nearest multiple of ``step``; unchanged at step zero.

    Half **away from zero**, not Python's ``round``, which breaks a tie by
    parity: ``round(0.5)`` is 0 and ``round(1.5)`` is 2, so a user dragging
    across the grid would find some half-way points sticking backwards and
    others forwards with nothing visible to say which. Away-from-zero is also
    what makes the grid symmetric -- ``-0.34`` snaps to ``-0.25`` exactly as
    ``0.34`` snaps to ``0.25`` -- and a user who has mirrored half a model
    across the origin is relying on that without ever thinking about it.
    """
    step = abs(float(step))
    if step == 0.0:
        return float(value)
    return math.copysign(math.floor(abs(value) / step + 0.5), value) * step


def snap_translation(vec: Iterable[float], step: float) -> np.ndarray:
    """Each component of a position onto the grid, as a fresh ``(3,)`` f8."""
    out = np.asarray(vec, dtype="f8")
    if step == 0.0:
        return out.copy()
    return np.array([snap_value(float(v), step) for v in out], dtype="f8")


def snap_rotation(quat: Iterable[float], degrees: float) -> np.ndarray:
    """The rotation's *angle* quantised, with its axis kept.

    Snapping the angle rather than decomposing to Euler angles and snapping
    each is the whole decision here. Euler snapping quantises three numbers
    that are not independent, so it moves the axis as well as the angle: an
    object tilted off the cardinal axes visibly swings when the user turns
    snapping on and touches nothing. Snapping the angle alone is what a gizmo
    drag means -- "rotate this by fifteen degrees about the ring I grabbed" --
    and it leaves an already-placed object where it was put.

    The identity rotation comes back unchanged, because a zero rotation has no
    axis to keep and any answer other than the identity would invent one.
    """
    q = np.asarray(quat, dtype="f8")
    if degrees == 0.0:
        return q.copy()
    length = float(np.linalg.norm(q[:3]))
    if length == 0.0:
        return m3.quat_identity()
    angle = 2.0 * math.atan2(length, float(q[3]))
    snapped = math.radians(snap_value(math.degrees(angle), degrees))
    return m3.quat_from_axis_angle(q[:3] / length, snapped)


def bake_transform(obj: Obj) -> Obj:
    """The object's TRS folded into its positions, and the TRS reset.

    What it is *for* is a mesh that has to be measured or exported in the frame
    it is drawn in -- a bounds check, a mirror across a world plane, the render
    that gets handed to trellis. Folding is a one-way operation: the transform
    the user typed is gone afterwards, replaced by the geometry it described.

    A negative scale reaches this function even though :func:`mirror` refuses
    to leave one on a node -- a user can type one into the properties panel --
    and it is handled for free, because :func:`~.mesh.transformed` reverses the
    loops on any negative determinant, not just on a mirror it produced itself.
    """
    matrix = m3.compose(obj.translation, obj.rotation, obj.scale)
    return replace(
        obj,
        mesh=bm.transformed(obj.mesh, matrix),
        translation=m3.vec3(),
        rotation=m3.quat_identity(),
        scale=m3.vec3(1.0, 1.0, 1.0),
    )


def _next_name(name: str, taken: Iterable[str] = ()) -> str:
    """``Box`` -> ``Box.001`` -> ``Box.002``, skipping any name already in use.

    Counting up rather than prefixing ("Copy of Copy of Box") keeps the name
    the same length however many times it is duplicated, and keeps the original
    name readable at the front where the outliner truncates from the right.
    """
    used = set(taken)
    match = _SUFFIX.match(name)
    base = match.group("base") if match else name
    start = int(match.group("n")) if match else 0
    # Bounded rather than a bare ``while True``: the loop cannot run past one
    # candidate per name already in use plus the source's own, and a bound is
    # cheaper than trusting that.
    for n in range(start + 1, start + len(used) + 3):
        candidate = f"{base}.{n:03d}"
        if candidate not in used and candidate != name:
            return candidate
    return f"{base}.{start + len(used) + 3:03d}"  # pragma: no cover - unreachable


def duplicate(obj: Obj, uid: int, *, taken: Iterable[str] = ()) -> Obj:
    """A copy of the object under a new uid, sharing its mesh.

    Sharing rather than copying the mesh is safe *because* ``Mesh`` is
    immutable and every op is ``Mesh -> Mesh``: a later edit to either object
    replaces its whole mesh rather than writing through the shared arrays, and
    until then the viewport's ``id(obj.mesh)``-keyed GPU cache uploads the
    geometry once for both. Copying it would double the memory of a scene of
    repeated props to buy nothing.

    Everything mutable *is* copied. ``params`` in particular: sharing the dict
    would make editing the copy's radius edit the original's, which is the one
    thing a duplicate must never do. The transform arrays are copied by
    ``Obj.__post_init__`` on the way through.

    ``taken`` is the names the document already holds. It is optional because
    the useful guarantee without it -- a name that differs from the source's --
    is the one a caller with no document can want; a caller that has an
    outliner should pass what is in it.
    """
    return replace(
        obj,
        uid=uid,
        name=_next_name(obj.name, taken),
        params=dict(obj.params),
    )
