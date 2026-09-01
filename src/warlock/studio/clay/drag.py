"""Axis constraints and typed values for a transform in progress.

A gizmo drag answers "how far" with the mouse, which is the right control for
placing something by eye and the wrong one for placing it *exactly*. Every
modelling package therefore lets the keyboard join a drag already under way:
``X`` locks it to an axis, digits type the number outright, and the pair
compose -- ``X`` then ``2`` is "two metres along X" with no dragging left in it
at all.

**It is a filter on the delta, not a second kind of drag.** Everything here is
``(quantity, lock) -> quantity``: the gizmo still produces a displacement, a
factor triple or a quaternion exactly as it did, and this narrows it on the way
past. That is what keeps the constraint out of the three gizmos, out of the
element-drag preview and out of the commit -- all of which go on knowing only
that they were handed a delta -- and it is what makes every rule below
assertable from three numbers rather than from a viewport.

**A typed value replaces the quantity; an axis lock only projects it.** The two
are deliberately different, and the difference is what a user means by each: a
lock is a statement about *direction* and leaves the mouse in charge of the
amount, while a number is the amount and leaves nothing. So a typed value with
no axis is still meaningful -- it sets the length of the displacement the mouse
chose the direction of, and the size of a uniform scale -- which is why the
lock is not a precondition for typing.

Pure: numpy plus ``viewer.math3d`` for the one quaternion conversion, which is
the same reason every other module in this package reaches for it -- an XYZW
quaternion is built in exactly one place in this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..viewer import math3d as m3

__all__ = [
    "AXES",
    "FALLOFF_CHUNK_PAIRS",
    "MAX_FALLOFF_PAIRS",
    "DragInput",
    "constrain_rotation",
    "constrain_scale",
    "constrain_translation",
    "falloff",
    "proportional_set",
    "readout",
]

AXES: tuple[str, str, str] = ("x", "y", "z")

#: What a number may be built out of. A bare ``-`` or a trailing ``.`` parses to
#: nothing and is *kept*, because both are states a half-typed number passes
#: through and discarding them would make the minus sign unreachable.
_NUMERIC = set("0123456789.-")


def _index(axis: str) -> int | None:
    return AXES.index(axis) if axis in AXES else None


def _unit(axis: str) -> np.ndarray:
    out = np.zeros(3)
    out[AXES.index(axis)] = 1.0
    return out


@dataclass
class DragInput:
    """What the keyboard has said about the drag currently under way.

    Held on the view rather than in the app's Clay state, because it is about
    one drag: it is created empty at the press and dropped at the release, and a
    value that outlived a drag would silently constrain the next one.
    """

    axis: str = ""
    typed: str = ""
    #: Set once a number has been typed *and* is parseable, so a caller can tell
    #: "the user is typing" from "the user has typed a value", which read the
    #: same on the string alone at the moment it holds only ``-``.
    _cache: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def active(self) -> bool:
        return bool(self.axis) or bool(self.typed)

    def value(self) -> float | None:
        """The typed number, or ``None`` while there is not one yet."""
        try:
            return float(self.typed)
        except ValueError:
            return None

    def key(self, name: str) -> bool:
        """Feed one key by pygame name. -> whether it was consumed.

        Pressing the locked axis again *clears* the lock rather than being
        ignored, which is the convention every package shares and the only one
        that leaves a mistyped ``X`` recoverable without abandoning the drag.
        """
        if name in AXES:
            self.axis = "" if self.axis == name else name
            return True
        if name == "backspace":
            self.typed = self.typed[:-1]
            return True
        if len(name) == 1 and name in _NUMERIC:
            self.typed += name
            return True
        return False


def constrain_translation(displacement: np.ndarray, drag: DragInput) -> np.ndarray:
    """A world-space displacement, narrowed by the axis lock and the typed value.

    With a number and no axis the *direction the mouse chose* is kept and only
    the length is replaced -- a zero-length displacement has no direction to
    keep, so it stays zero rather than inventing one.
    """
    out = np.asarray(displacement, dtype="f8").reshape(3).copy()
    value = drag.value()
    index = _index(drag.axis)
    if value is not None:
        if index is not None:
            return _unit(drag.axis) * value
        length = float(np.linalg.norm(out))
        return out / length * value if length > 1e-12 else out
    if index is not None:
        kept = np.zeros(3)
        kept[index] = out[index]
        return kept
    return out


def constrain_scale(factors: np.ndarray, drag: DragInput) -> np.ndarray:
    """A per-axis scale triple, narrowed the same way.

    An unlocked axis goes back to **one**, not to whatever the gizmo said: the
    point of locking is that nothing else moves, and leaving the other two at
    their dragged values would be a lock that only renamed the readout.
    """
    out = np.asarray(factors, dtype="f8").reshape(3).copy()
    value = drag.value()
    index = _index(drag.axis)
    if value is not None:
        if index is not None:
            kept = np.ones(3)
            kept[index] = value
            return kept
        return np.full(3, value)
    if index is not None:
        kept = np.ones(3)
        kept[index] = out[index]
        return kept
    return out


def constrain_rotation(quat: np.ndarray, drag: DragInput) -> np.ndarray:
    """A rotation, narrowed to one axis and/or to a typed angle in **degrees**.

    Locked with no number, the drag's angle is *projected* onto the axis rather
    than kept whole -- the component of the sweep about that axis, which is what
    a user turning a free rotation into a constrained one is asking for and what
    keeps the result continuous as the lock goes on.

    With a number and no lock the drag's own axis is kept and only the angle is
    replaced, which mirrors the translation rule exactly. The identity rotation
    has no axis, so a number typed against one leaves it identity rather than
    inventing an axis to turn about.
    """
    q = np.asarray(quat, dtype="f8").reshape(4)
    value = drag.value()
    index = _index(drag.axis)
    if value is None and index is None:
        return q
    length = float(np.linalg.norm(q[:3]))
    angle = 2.0 * float(np.arctan2(length, float(q[3])))
    axis = q[:3] / length if length > 1e-12 else None

    if index is not None:
        wanted = _unit(drag.axis)
        if value is not None:
            return m3.quat_from_axis_angle(wanted, np.radians(value))
        if axis is None:
            return m3.quat_identity()
        return m3.quat_from_axis_angle(wanted, angle * float(axis @ wanted))
    if axis is None:
        return m3.quat_identity()
    return m3.quat_from_axis_angle(axis, np.radians(value))


# --- proportional editing -----------------------------------------------------
#
# The falloff turns a selection into a *neighbourhood*: the selected vertices
# move fully, the ones near them move less, and the surface bends instead of
# tearing. It is what takes Clay from blockout-only to organic-adjustment-
# capable, and it is a weight per vertex and nothing else -- the drag maths, the
# preview and the commit are all untouched by it.

#: The largest ``selected x vertices`` product the distance search will attempt.
#: The search is a broadcast rather than a spatial index, which is right for the
#: blockout geometry Clay authors and quadratic on an imported reconstruction --
#: so past this it declines, and the drag is an ordinary one. Declining beats
#: both a stall at the press and a spatial index nothing else here needs. This
#: is the **time** cap; the memory cap is the chunk below, because at this
#: count the one-shot broadcast materialised ``pairs x 3 x float64`` -- close to
#: a gigabyte, on the frame thread, at the press of a drag.
MAX_FALLOFF_PAIRS = 40_000_000

#: How many pairs one chunk of the search materialises: ~48 MB of delta at
#: float64, and gone before the next chunk allocates. The result is
#: bit-identical to the one-shot broadcast -- every per-element value is
#: computed by the same expression in the same order, and the running
#: ``minimum`` fold only *selects* among them, which is the native-kernel bar
#: applied to a rewrite that never left numpy.
FALLOFF_CHUNK_PAIRS = 2_000_000


def _min_distance(positions: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Per-vertex distance to the nearest of *anchors*, in bounded memory.

    Chunked over the anchor rows under a running minimum rather than one
    broadcast over all of them: the peak temporary is one chunk's delta, not
    the whole ``anchors x positions`` block, and the answer cannot differ --
    ``min`` over chunk minima is ``min`` over the same values.
    """
    rows = max(1, FALLOFF_CHUNK_PAIRS // max(len(positions), 1))
    nearest = np.full(len(positions), np.inf)
    for start in range(0, len(anchors), rows):
        delta = positions[None, :, :] - anchors[start : start + rows][:, None, :]
        np.minimum(nearest, np.sqrt((delta**2).sum(axis=2)).min(axis=0), out=nearest)
    return nearest


def falloff(distance: np.ndarray, radius: float) -> np.ndarray:
    """Smooth weights from a distance to the selection. 1 at zero, 0 at *radius*.

    ``1 - smoothstep`` rather than a linear ramp: linear weights have a crease
    at the selection boundary and another at the radius, both of which show up
    as a visible ridge in the surface -- which is precisely the artefact
    proportional editing exists to avoid. Zero radius returns a hard selection,
    the ``snap_value`` convention: the off switch is the same control.
    """
    d = np.asarray(distance, dtype="f8")
    if radius <= 0.0:
        return (d <= 0.0).astype("f8")
    t = np.clip(d / float(radius), 0.0, 1.0)
    return 1.0 - (3.0 * t**2 - 2.0 * t**3)


def proportional_set(
    positions: np.ndarray, selected: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, weights)`` for a drag with a soft falloff around *selected*.

    Distance is to the **nearest selected vertex**, not to the selection's
    centroid: a centroid measures from a point the geometry may not even pass
    through, so dragging one end of a long selected strip would fade out along
    the strip itself.

    The set is the selected vertices -- all present, all at weight 1 -- plus
    every vertex strictly inside the radius, **in vertex-index order**: the
    caller pairs each entry with its weight and indexes the mesh with the lot,
    so no ordering beyond that alignment is promised. A vertex at exactly the
    radius weighs zero and is dropped, because carrying it means a drag that
    reports moving geometry it does not move. The two declining paths -- zero
    radius, or a mesh past ``MAX_FALLOFF_PAIRS`` -- hand the selection back in
    the order it arrived, at weight 1 throughout.
    """
    positions = np.asarray(positions, dtype="f8").reshape(-1, 3)
    selected = np.asarray(selected, dtype="i8").reshape(-1)
    if len(selected) == 0 or radius <= 0.0:
        return selected.astype("i4"), np.ones(len(selected))
    if len(selected) * len(positions) > MAX_FALLOFF_PAIRS:
        return selected.astype("i4"), np.ones(len(selected))

    distance = _min_distance(positions, positions[selected])
    weights = falloff(distance, radius)
    weights[selected] = 1.0
    keep = np.flatnonzero(weights > 0.0)
    return keep.astype("i4"), weights[keep]


# The unit each tool's readout is quoted in. Scale is a ratio and has none,
# which is stated here rather than by an empty branch in the formatter.
_UNITS = {"move": " m", "rotate": " deg", "scale": ""}


def readout(tool: str, quantity: np.ndarray, drag: DragInput) -> str:
    """The one line the HUD draws: what the drag currently amounts to.

    It shows the *result* rather than what was typed, so a locked axis with no
    number still reads as a number and a half-typed one reads as the value it
    will replace -- with the raw text appended while it is being entered, since
    a field you cannot see is not a field.
    """
    values = np.asarray(quantity, dtype="f8").reshape(-1)
    unit = _UNITS.get(tool, "")
    if tool == "rotate":
        body = f"{float(values[0]):.1f}{unit}"
    elif len(values) == 1:
        body = f"{float(values[0]):.3f}{unit}"
    else:
        body = "  ".join(f"{a.upper()} {float(v):.3f}" for a, v in zip(AXES, values, strict=True))
        body += unit
    parts = [f"[{drag.axis.upper()}]"] if drag.axis else []
    parts.append(body)
    if drag.typed:
        parts.append(f"= {drag.typed}")
    return "  ".join(parts)


def screen_angle(
    pivot: np.ndarray, normal: np.ndarray, was: np.ndarray, now: np.ndarray
) -> float:
    """The signed angle from ``was`` to ``now`` about ``normal``, in radians.

    Both points are projected into the plane through ``pivot`` first, so a ray
    hit that is a hair off the plane -- which every ray hit is, at float
    precision -- does not tilt the answer.

    Signed rather than absolute, and that is the whole of why it is a function:
    ``arccos`` of the dot product gives the angle and loses the direction, so a
    rotate drag turned the same way whichever way the mouse went round. The
    cross product against the axis is what carries the sign.

    Zero for a degenerate pair rather than a refusal: a pointer exactly on the
    pivot has no angle to report, and the caller is a live drag that must go on
    drawing.
    """
    pivot = np.asarray(pivot, dtype="f8")
    axis = np.asarray(normal, dtype="f8")
    first = np.asarray(was, dtype="f8") - pivot
    second = np.asarray(now, dtype="f8") - pivot
    first = first - axis * float(np.dot(first, axis))
    second = second - axis * float(np.dot(second, axis))
    len_first = float(np.linalg.norm(first))
    len_second = float(np.linalg.norm(second))
    if len_first < 1e-9 or len_second < 1e-9:
        return 0.0
    first = first / len_first
    second = second / len_second
    return float(
        math.atan2(float(np.dot(np.cross(first, second), axis)), float(np.dot(first, second)))
    )
