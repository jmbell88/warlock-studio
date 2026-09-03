"""What the viewport says without being asked: the hint line and the axis ball.

Two pieces of chrome that a modeller reads constantly and that Clay had neither
of. Both are **pure** -- numbers and strings, no imgui, no GL -- which is the
whole reason they are here rather than in the pane that draws them: "does edge
mode mention the loop shortcut" and "does the +X ball sit on the right when the
camera is at the front" are questions a headless test can ask, and they are
exactly the questions a screenshot cannot be made to fail on.

Nothing here imports outward. ``panes/clay_hud.py`` draws it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# --- the navigation widget ---------------------------------------------------

#: The six axis ends: the axis, its sign, the ``Camera.AXIS_VIEWS`` name a click
#: on it asks for, and the letter drawn in it.
#:
#: **The view names are the contract and are written out**, exactly as
#: ``AXIS_VIEWS`` writes out its angles: a ball says "put the camera on the +X
#: side" and must not have to know which way ``theta`` runs to say it.
#: ``test_every_ball_puts_the_camera_where_its_axis_points`` checks the six
#: against the camera rather than against this table, so a wrong pairing here is
#: a failure rather than a definition.
#:
#: Only the positive ends carry a letter. Blender's widget does the same and the
#: reason is legible rather than decorative: six labelled balls at 20 px are six
#: things to read, and the negative end of an axis is identified by being
#: opposite the one that is labelled.
AXIS_ENDS: tuple[tuple[int, int, str, str], ...] = (
    (0, 1, "right", "X"),
    (0, -1, "left", ""),
    (1, 1, "top", "Y"),
    (1, -1, "bottom", ""),
    (2, 1, "front", "Z"),
    (2, -1, "back", ""),
)


@dataclass(frozen=True)
class AxisBall:
    """One end of one axis, placed in the widget's own box.

    ``x``/``y`` are pixels from the box's top-left, so the caller adds its
    origin and nothing else. ``depth`` is positive toward the viewer, which is
    what the caller sorts on: the balls behind the centre are drawn first, so
    the near ones overlap them rather than the other way round.
    """

    view: str
    label: str
    x: float
    y: float
    depth: float
    positive: bool


def axis_layout(view_matrix: Any, size: float) -> list[AxisBall]:
    """The six balls, back to front, in a ``size`` x ``size`` box.

    ``view_matrix`` is the camera's own world-to-camera matrix, whose upper-left
    3x3 is the rotation: its rows are the camera's right, up and backward axes
    (``math3d.look_at`` writes ``s``, ``u``, ``-f``). So multiplying a world
    direction by it gives x to the right of the screen, y up it, and z *toward*
    the viewer -- and the only conversion left is that screen y grows downward.

    Sorted rather than left in table order because these overlap: at a
    front-on camera the +Z and -Z balls land on the same pixel, and which of
    them is on top is the whole of what tells the reader which way they are
    looking.
    """

    rotation = np.asarray(view_matrix, dtype="f8")[:3, :3]
    centre = size * 0.5
    # The balls sit inside the box rather than on its edge: a ball is drawn as a
    # disc of its own and one centred on the boundary would be half clipped.
    radius = size * 0.5 * 0.78
    out: list[AxisBall] = []
    for axis, sign, view, label in AXIS_ENDS:
        direction = np.zeros(3, dtype="f8")
        direction[axis] = float(sign)
        camera = rotation @ direction
        out.append(
            AxisBall(
                view=view,
                label=label,
                x=centre + float(camera[0]) * radius,
                y=centre - float(camera[1]) * radius,
                depth=float(camera[2]),
                positive=sign > 0,
            )
        )
    out.sort(key=lambda ball: ball.depth)
    return out


# --- the hint line -----------------------------------------------------------

#: What every mode says about picking, before the tool has its say. The element
#: modes get the selection verbs that only exist there; object mode gets the
#: two that only exist *outside* an element mode.
_PICK = {
    "object": "LMB select . Shift extend . Tab edit",
    "vertex": "LMB pick . drag marquee . L linked . Ctrl+/- grow/shrink . Tab object",
    "edge": (
        "LMB pick . Alt+click loop . Ctrl+Alt+click ring . L linked . Tab object"
    ),
    "face": "LMB pick . Alt+click loop . L linked . Ctrl+/- grow/shrink . Tab object",
}

#: What the tool in hand adds. Keyed on the tool rather than folded into the
#: mode line because the two vary independently: Move in face mode and Move in
#: object mode drag the same way and select differently.
_TOOL = {
    "select": "",
    "move": "G move . drag an arrow",
    "rotate": "E rotate . drag a ring",
    "scale": "S scale . drag a handle",
}

#: The line **while a keyboard drag is live**, which replaces everything above
#: it: mid-drag the only keys that mean anything are the ones that constrain,
#: commit or cancel it, and a line still offering "Tab object" would be offering
#: a key that is not listened to.
_DRAGGING = (
    "X/Y/Z lock (again: local, again: off) . type a number . "
    "Enter/LMB commit . Esc/RMB cancel . G/R/S switch"
)

#: Always true, and always last: the two mouse buttons that navigate. They are
#: the keys a newcomer to a 3D viewport asks about first and the ones a manual
#: is least likely to be open at.
_NAVIGATE = "Alt+LMB orbit . MMB pan . wheel zoom"


def hint(mode: str, tool: str, *, dragging: bool = False, drag_kind: str = "") -> str:
    """One line of what the mouse and the keyboard do right now.

    Clay's viewport had no such line, and the cost was specific rather than
    general: **every selection verb the mode offers is invisible**. Alt+click
    for a loop, L for linked, Ctrl+plus to grow -- none of them is a button, so
    a user who has not read chapter 30 has no way to discover that edge mode can
    do anything a vertex mode cannot.

    ``drag_kind`` names the live drag ("move"/"rotate"/"scale") and is used only
    to say which one is running; the keys are the same for all three, which is
    the point of the line.
    """

    if dragging:
        kind = drag_kind or "drag"
        return f"{kind.capitalize()} . {_DRAGGING}"
    parts = [_PICK.get(mode, _PICK["object"])]
    extra = _TOOL.get(tool, "")
    if extra:
        parts.append(extra)
    parts.append(_NAVIGATE)
    return " . ".join(parts)


def keys_named(text: str) -> set[str]:
    """Every key or chord the line mentions, for the parity test.

    Crude on purpose: a token is a key if it is one of the shapes this app's
    shortcut sheet writes -- a capital letter, a chord with a ``+``, or one of
    the named keys. Anything cleverer would be a second parser to keep in
    agreement with the sheet, and the point of this function is to *catch*
    disagreement.

    **A single letter counts only when it is capital**, which is the app's own
    convention throughout (``G``, ``L``, ``Tab``) and not a nicety: "drag a
    ring" reads its article as a binding otherwise, and the parity test then
    fails demanding that Clay implement the ``A`` key.
    """

    words = {
        token.strip(".,")
        for chunk in text.split(" . ")
        for token in chunk.split()
    }
    named = {"Tab", "Enter", "Esc", "LMB", "MMB", "RMB", "wheel"}
    out = set()
    for word in words:
        if word in named or "+" in word or (len(word) == 1 and word.isupper()):
            out.add(word)
        elif "/" in word and all(
            len(part) == 1 and part.isupper() for part in word.split("/") if part
        ):
            out.update(part for part in word.split("/") if part)
    return out


def angle_of(dx: float, dy: float) -> float:
    """The screen angle of a drag, in radians. Here because the hint line and
    the drag machinery both want one and neither should own it."""

    return math.atan2(float(dy), float(dx))


# --- the statistics overlay ---------------------------------------------------


def stats(doc: Any) -> str:
    """What the document holds, and how much of it is selected. One line.

    Blender's statistics overlay, and the reason it earns a place on a viewport
    that already has a hint line: **every number here was otherwise unavailable
    anywhere in Clay**. The outliner counts objects and nothing counted
    vertices, edges, faces or triangles -- so "is this mesh 500 triangles or
    50,000" was a question the app could not answer about the thing on screen,
    which is the question that decides whether a game asset is finished.

    Selected counts are shown only when there *is* a selection, and only for the
    element mode in hand: a face count while vertices are being picked is a
    number about a selection the user does not have.

    Pure, and derived per call rather than cached. It walks the meshes, which
    is O(objects) in numpy shape reads -- the arrays are not touched, only
    their lengths -- so there is nothing to invalidate and nothing to go stale.
    """

    objects = [obj for obj in doc.objects if getattr(obj, "visible", True)]
    verts = edges = faces = tris = 0
    for obj in objects:
        mesh = obj.mesh
        verts += int(len(mesh.positions))
        count = _faces_of(mesh)
        faces += count
        loops = getattr(mesh, "loops", None)
        # ``or ()`` is wrong on a numpy array -- truthiness of one with more
        # than one element raises -- so the absence is tested with ``is None``.
        corners = 0 if loops is None else int(len(loops))
        # A polygon of n corners fans into n-2 triangles, so the triangle count
        # is the corner count less twice the face count. The *edge* count is
        # not the corner count: a cube has 24 corners and 12 edges, because
        # every edge is shared by two faces, and reporting 24 would be a number
        # a reader can check against a cube and find wrong.
        edges += _unique_edges(mesh)
        tris += max(0, corners - 2 * count)
    parts = [
        f"{len(objects)} object{'' if len(objects) == 1 else 's'}",
        f"{verts:,} verts",
        f"{edges:,} edges",
        f"{faces:,} faces",
        f"{tris:,} tris",
    ]
    picked = _selected(doc)
    if picked:
        parts.append(picked)
    return "  ".join(parts)


#: Unique-edge counts, keyed on the mesh object and pinning it.
#:
#: Keyed on the ``Mesh`` itself rather than on an id or a revision, which is
#: ``ClayState.manifold``'s rule and its reason: a ``Mesh`` is immutable and
#: every op replaces it, so "this count is still about what is on screen" is
#: exactly ``mesh is measured``. An ``id()`` would be recycled by the allocator
#: onto a different mesh and silently report the last edit's edges.
#:
#: Bounded, because the pin keeps every measured mesh alive: this is a readout
#: and must not become a second undo stack. The whole cache is dropped rather
#: than evicted one by one -- a readout that recomputes once is a readout that
#: was free, and an LRU here would be machinery for nothing.
_EDGE_CACHE: dict[Any, int] = {}
_EDGE_CACHE_MAX = 64


def _unique_edges(mesh: Any) -> int:
    """How many distinct edges a mesh has. Memoised on the mesh.

    The pair per face corner, sorted within the pair so ``(a, b)`` and
    ``(b, a)`` are one edge, then counted distinct. O(L log L) and run once per
    mesh rather than once per frame -- a statistics overlay that rebuilt an
    adjacency sixty times a second would cost more than everything it reports.
    """
    hit = _EDGE_CACHE.get(mesh)
    if hit is not None:
        return hit
    loops = getattr(mesh, "loops", None)
    starts = getattr(mesh, "starts", None)
    if loops is None or starts is None or len(loops) == 0:
        return 0
    loops = np.asarray(loops)
    starts = np.asarray(starts)
    # The next corner within each face, which is the corner after it except at
    # a face's last corner, where it wraps to that face's first.
    nxt = np.arange(1, len(loops) + 1, dtype="i8")
    nxt[starts[1:] - 1] = starts[:-1]
    pairs = np.stack([loops, loops[nxt]], axis=1)
    pairs = np.sort(pairs, axis=1)
    count = int(len(np.unique(pairs, axis=0)))
    if len(_EDGE_CACHE) >= _EDGE_CACHE_MAX:
        _EDGE_CACHE.clear()
    _EDGE_CACHE[mesh] = count
    return count


def _faces_of(mesh: Any) -> int:
    """How many faces a mesh has, off its CSR offsets.

    ``starts`` is ``(F+1,)`` -- one offset per face plus the terminator -- which
    is the shape every op in ``clay/`` reads it as.
    """
    starts = getattr(mesh, "starts", None)
    if starts is None:
        return 0
    return max(0, int(len(starts)) - 1)


def _selected(doc: Any) -> str:
    """The selected count, in the mode it is a count of."""

    mode = getattr(doc, "element_mode", "object")
    if mode == "object":
        count = len(getattr(doc, "selection", ()) or ())
        return f"{count} selected" if count else ""
    total = 0
    for sel in (getattr(doc, "element_sel", {}) or {}).values():
        total += sel.count(mode)
    if not total:
        return ""
    word = {"vertex": "vert", "edge": "edge", "face": "face"}.get(mode, mode)
    return f"{total:,} {word}{'' if total == 1 else 's'} selected"
