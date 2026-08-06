"""What is selected inside a mesh, and the one type that says so.

There is exactly one selection type, holding all three element kinds at once,
rather than three types or a tagged union. The mode decides which field an
interaction *writes*, but a conversion reads two of them at once (vertices to
faces asks which faces have all their corners selected) and the undo
reconciliation has to drop a whole selection without caring what mode it was
made in. One frozen record with three arrays does all of that and keeps the
document's per-object dictionary uniform.

**Edges are vertex pairs, not indices into** :func:`~.mesh.edges`. Canonical
edge ids are positions in a lexsorted unique array, so inserting one edge
anywhere renumbers every edge after it -- and every op that touches topology
inserts edges. A pair ``(a, b)`` survives anything that does not delete the
vertices themselves, and mapping a pair back to an id is one ``searchsorted``
(:meth:`~.adjacency.Adjacency.edge_ids`) at the moment an edge-shaped table
actually has to be indexed. The cost is two integers per edge instead of one;
the alternative is a selection that silently points at different edges after
every operation.

**Arrays are copied, read-only and canonical**, the ``Mesh`` idiom, for the same
two reasons: a caller cannot write through one and change a selection somebody
else is holding, and a selection is *replaced whole* rather than mutated, so
``id(sel)`` is a sound cache key for the overlay's index buffers. Canonical
means sorted unique throughout -- vertices and faces ascending, edges
low-vertex-first and lexsorted -- so two selections of the same elements are
equal array-for-array however they were built, which is what makes the "did the
selection change" comparison in the overlay cache honest.

:class:`OpError` lives here rather than beside the ops because it is part of
this vocabulary: an op refuses a *selection*, and the refusal is a sentence
shown to the user as a toast with no edit recorded. It is a ``ValueError``
subclass so that a caller who forgets to catch it still fails loudly rather than
committing half a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .adjacency import adjacency
from .mesh import Mesh

__all__ = [
    "MODES",
    "ElementSel",
    "OpError",
    "affected_verts",
    "combine",
    "convert",
    "empty",
    "is_empty",
]

Mode = Literal["object", "vertex", "edge", "face"]
MODES: tuple[Mode, ...] = ("object", "vertex", "edge", "face")

How = Literal["replace", "add", "subtract"]


class OpError(ValueError):
    """A user-facing refusal: shown as a toast, with no edit recorded.

    The message is the whole user interface for it, so it names the element it
    refused and, where there is one, the thing to do instead.
    """


def _verts(a: np.ndarray | None) -> np.ndarray:
    arr = np.unique(np.asarray(a, dtype="i4").reshape(-1)) if a is not None else None
    out = arr if arr is not None else np.zeros(0, dtype="i4")
    return out.astype("i4")


def _pairs(a: np.ndarray | None) -> np.ndarray:
    if a is None:
        return np.zeros((0, 2), dtype="i4")
    rows = np.asarray(a, dtype="i4").reshape(-1, 2)
    if len(rows) == 0:
        return np.zeros((0, 2), dtype="i4")
    lo = np.minimum(rows[:, 0], rows[:, 1])
    hi = np.maximum(rows[:, 0], rows[:, 1])
    return np.unique(np.stack([lo, hi], axis=1), axis=0).astype("i4")


@dataclass(frozen=True, eq=False)
class ElementSel:
    """Selected vertices, edges and faces of one mesh. Canonical and frozen."""

    verts: np.ndarray = None  # type: ignore[assignment]  # (n,) i4 sorted unique
    edges: np.ndarray = None  # type: ignore[assignment]  # (m,2) i4 lexsorted unique
    faces: np.ndarray = None  # type: ignore[assignment]  # (k,) i4 sorted unique

    def __post_init__(self) -> None:
        object.__setattr__(self, "verts", _verts(self.verts))
        object.__setattr__(self, "edges", _pairs(self.edges))
        object.__setattr__(self, "faces", _verts(self.faces))
        for name in ("verts", "edges", "faces"):
            getattr(self, name).setflags(write=False)

    def __len__(self) -> int:
        return len(self.verts) + len(self.edges) + len(self.faces)

    def count(self, mode: str) -> int:
        return len(getattr(self, {"vertex": "verts", "edge": "edges"}.get(mode, "faces")))

    def same_as(self, other: ElementSel) -> bool:
        """Element-for-element equality. ``__eq__`` stays identity (``eq=False``)."""
        return (
            np.array_equal(self.verts, other.verts)
            and np.array_equal(self.edges, other.edges)
            and np.array_equal(self.faces, other.faces)
        )


def empty() -> ElementSel:
    return ElementSel()


def is_empty(sel: ElementSel | None) -> bool:
    return sel is None or len(sel) == 0


def affected_verts(mesh: Mesh, sel: ElementSel) -> np.ndarray:
    """Every vertex the selection touches, however it was expressed.

    The gizmo's centroid and the live drag both work on this: a face selection
    moves its corners, an edge selection its endpoints, and a mixed one the
    union, with no per-mode branch anywhere in the drag code.
    """
    parts = [sel.verts.astype("i8"), sel.edges.reshape(-1).astype("i8")]
    if len(sel.faces):
        starts = mesh.starts.astype("i8")
        for f in sel.faces.tolist():
            parts.append(mesh.loops[starts[f] : starts[f + 1]].astype("i8"))
    if not any(len(p) for p in parts):
        return np.zeros(0, dtype="i4")
    return np.unique(np.concatenate(parts)).astype("i4")


def _face_corner_mask(mesh: Mesh, verts: np.ndarray) -> np.ndarray:
    """A ``(F,)`` bool: faces every one of whose corners is in *verts*."""
    n_faces = len(mesh.starts) - 1
    if n_faces == 0:
        return np.zeros(0, dtype=bool)
    inside = np.isin(mesh.loops, verts)
    return np.minimum.reduceat(inside.astype("i1"), mesh.starts[:-1].astype("i8")) > 0


def convert(mesh: Mesh, sel: ElementSel, target: str) -> ElementSel:
    """Reinterpret a selection in another element mode.

    The rules are Wings3D's, and the asymmetry is deliberate. Going *down* --
    faces to edges to vertices -- is a union: everything the selection touches
    comes along, because every one of those elements is genuinely part of what
    was selected. Going *up* is a conjunction: a face is selected only when all
    of its corners (or all of its edges) are, because a face is not "partly
    selected" and lighting up every face that shares one vertex with a selected
    one would select most of the mesh from one click.
    """
    if target == "object":
        return empty()

    verts = affected_verts(mesh, sel)
    if target == "vertex":
        return ElementSel(verts=verts)

    a = adjacency(mesh)
    if target == "edge":
        pairs = [sel.edges.astype("i8")]
        if len(sel.faces):
            face_mask = np.isin(a.corner_face, sel.faces)
            pairs.append(a.edge_verts[np.unique(a.corner_edge[face_mask])].astype("i8"))
        if len(sel.verts) and a.n_edges:
            both = np.isin(a.edge_verts, sel.verts).all(axis=1)
            pairs.append(a.edge_verts[both].astype("i8"))
        rows = np.concatenate([p.reshape(-1, 2) for p in pairs]) if pairs else None
        return ElementSel(edges=rows)

    if target == "face":
        faces = [sel.faces.astype("i8")]
        if len(sel.verts) or len(sel.edges):
            faces.append(np.flatnonzero(_face_corner_mask(mesh, verts)))
        return ElementSel(faces=np.concatenate(faces) if faces else None)

    raise ValueError(f"unknown element mode {target!r}")


def _rows_minus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return a
    keep = ~(a[:, None, :] == b[None, :, :]).all(axis=2).any(axis=1)
    return a[keep]


def combine(a: ElementSel, b: ElementSel, how: How = "replace") -> ElementSel:
    """``replace`` | ``add`` | ``subtract`` -- the three click modifiers.

    Subtracting an element that is not selected is a no-op rather than an
    error: a Ctrl-drag marquee sweeps over whatever is under it, and most of
    that is usually not selected.
    """
    if how == "replace":
        return b
    if how == "add":
        return ElementSel(
            verts=np.concatenate([a.verts, b.verts]),
            edges=np.concatenate([a.edges, b.edges]),
            faces=np.concatenate([a.faces, b.faces]),
        )
    if how == "subtract":
        return ElementSel(
            verts=np.setdiff1d(a.verts, b.verts),
            edges=_rows_minus(a.edges, b.edges),
            faces=np.setdiff1d(a.faces, b.faces),
        )
    raise ValueError(f"unknown combine mode {how!r}")


def select_all(mesh: Mesh, mode: str) -> ElementSel:
    """Everything of the given kind."""
    if mode == "vertex":
        return ElementSel(verts=np.arange(len(mesh.positions), dtype="i4"))
    if mode == "edge":
        return ElementSel(edges=adjacency(mesh).edge_verts)
    if mode == "face":
        return ElementSel(faces=np.arange(len(mesh.starts) - 1, dtype="i4"))
    return empty()


def invert(mesh: Mesh, sel: ElementSel, mode: str) -> ElementSel:
    """Everything of the given kind that is *not* in *sel*."""
    return combine(select_all(mesh, mode), sel, "subtract")


def restrict(mesh: Mesh, sel: ElementSel) -> ElementSel:
    """Drop elements a mesh no longer has.

    An op can shrink a mesh under a selection that outlived it -- a delete
    followed by an undo lands here -- and an out-of-range face index would take
    down the overlay build rather than the op.
    """
    n_verts, n_faces = len(mesh.positions), len(mesh.starts) - 1
    edges = sel.edges
    if len(edges):
        edges = edges[(edges < n_verts).all(axis=1)]
    return replace(
        sel,
        verts=sel.verts[sel.verts < n_verts],
        edges=edges,
        faces=sel.faces[sel.faces < n_faces],
    )
