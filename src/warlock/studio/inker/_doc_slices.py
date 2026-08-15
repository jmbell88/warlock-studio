"""Slices on the document: the ops, the undo hooks, and the geometry mapping.

The model is in :mod:`.slices`; what is here is the half that pushes history and
the half that keeps a slice describing the pixels it described before a flip.

Two rules, both borrowed from neighbours rather than invented.

**A step that changes nothing is not pushed.** ``dirty`` is a comparison against
``history.head``, so setting a slice's name to its own name would make a saved
document ask to be saved -- the same check ``_push_tags`` and Packwright's
``rename_source`` make.

**Nothing here touches the composite.** Slices are not pixels: ``rev`` ticks so
a pane redraws, and the dirty rectangle, the frame stamps and the flatten caches
are all left exactly alone. A slice edit that invalidated the canvas would throw
away every cached frame in the document to move a rectangle nobody paints with.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import transform as tf
from .slices import (
    SLICE_PROPS,
    Slice,
    SliceAddEdit,
    SliceChangeEdit,
    SliceKey,
    SliceRemoveEdit,
    slice_props,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document

#: A canvas point mapper: ``(x, y) -> (x, y)``.
Point = Callable[[float, float], "tuple[float, float]"]


def _remap(entry: Slice, point: Point, canvas: tuple[int, int]) -> None:
    """Rewrite one slice, and every key on it, through a point mapper.

    The rectangle is mapped by its two corners rather than axis by axis, for
    ``transform.rect_from_points``' reason. The pivot and the centre are
    *relative to the bounds*, so each is lifted to an absolute canvas position,
    mapped, and put back against the new origin -- which is what makes a flip
    move a pivot to the mirrored side of the sprite rather than leaving it
    pinned to the same corner.
    """
    entry.bounds, entry.pivot, entry.center = _remapped(
        entry.bounds, entry.pivot, entry.center, point, canvas
    )
    entry.keys = {
        frame_uid: SliceKey(
            *_remapped(key.bounds, key.pivot, key.center, point, canvas)
        )
        for frame_uid, key in entry.keys.items()
    }


def _remapped(
    bounds: tuple[int, int, int, int],
    pivot: tuple[float, float] | None,
    center: tuple[int, int, int, int] | None,
    point: Point,
    canvas: tuple[int, int],
) -> tuple[
    tuple[int, int, int, int], tuple[float, float] | None, tuple[int, int, int, int] | None
]:
    x0, y0, x1, y1 = bounds
    moved = tf.clamp_rect(
        tf.rect_from_points(point(x0, y0), point(x1, y1)), canvas
    )
    nx0, ny0 = moved[0], moved[1]
    if pivot is not None:
        px, py = point(x0 + pivot[0], y0 + pivot[1])
        pivot = (px - nx0, py - ny0)
    if center is not None:
        cx0, cy0, cx1, cy1 = center
        a = point(x0 + cx0, y0 + cy0)
        b = point(x0 + cx1, y0 + cy1)
        rel = tf.rect_from_points((a[0] - nx0, a[1] - ny0), (b[0] - nx0, b[1] - ny0))
        # Clamped against the slice's own extent rather than the canvas: a
        # nine-slice centre is an inset of its slice, and one that escaped the
        # rectangle it insets would stretch a panel from outside itself.
        center = tf.clamp_rect(rel, (moved[2] - nx0, moved[3] - ny0))
    return moved, pivot, center


class SliceOps:
    """Slices, mixed into :class:`~.document.Document`."""

    # -- lookup --------------------------------------------------------------

    def slice_by_uid(self: Document, uid: int) -> Slice | None:
        for entry in self.slices:
            if entry.uid == uid:
                return entry
        return None

    def slice_index(self: Document, uid: int) -> int:
        for index, entry in enumerate(self.slices):
            if entry.uid == uid:
                return index
        raise KeyError(uid)

    # -- undo hooks ----------------------------------------------------------

    def _put_slice(self: Document, index: int, entry: Slice) -> None:
        at = max(0, min(int(index), len(self.slices)))
        self.slices.insert(at, entry)
        self.rev += 1

    def _drop_slice(self: Document, uid: int) -> None:
        self.slices = [entry for entry in self.slices if entry.uid != uid]
        self.rev += 1

    def _apply_slice(self: Document, uid: int, props: dict[str, Any]) -> None:
        """Install one slice's properties. Copies, for ``_set_tags``' reason:
        the edit holds these dictionaries and may replay any number of times,
        so handing the document the same ``keys`` object would let the next
        drag write through into the step meant to reverse it."""
        entry = self.slice_by_uid(uid)
        if entry is None:
            return
        for name in SLICE_PROPS:
            if name not in props:
                continue
            value = props[name]
            setattr(entry, name, dict(value) if name == "keys" else value)
        self.rev += 1

    # -- ops -----------------------------------------------------------------

    def add_slice(
        self: Document,
        bounds: tuple[int, int, int, int],
        *,
        name: str = "",
        pivot: tuple[float, float] | None = None,
        center: tuple[int, int, int, int] | None = None,
    ) -> Slice:
        """A new slice, clamped into the canvas. One undo step."""
        entry = Slice(
            name=name or self._next_slice_name(),
            bounds=tf.clamp_rect(bounds, self.size),
            pivot=pivot,
            center=center,
        )
        index = len(self.slices)
        self._put_slice(index, entry)
        self.history.push(SliceAddEdit(index, entry))
        return entry

    def _next_slice_name(self: Document) -> str:
        taken = {entry.name for entry in self.slices}
        index = len(self.slices) + 1
        while f"Slice {index}" in taken:
            index += 1
        return f"Slice {index}"

    def remove_slice(self: Document, uid: int) -> bool:
        try:
            index = self.slice_index(uid)
        except KeyError:
            return False
        entry = self.slices[index]
        self._drop_slice(uid)
        self.history.push(SliceRemoveEdit(index, entry))
        return True

    def set_slice(
        self: Document, uid: int, *, was: dict[str, Any] | None = None, **props: Any
    ) -> bool:
        """Change a slice, as one undo step, or do nothing.

        ``was`` is for the canvas: a drag mutates the live slice every frame so
        the overlay follows the cursor, and at release it hands back the
        properties as they stood at the press. Without it the "before" half of
        the step would be read *after* the drag and the whole gesture would
        undo to itself. Everything else passes ``props`` and lets the before be
        read here.
        """
        entry = self.slice_by_uid(uid)
        if entry is None:
            return False
        current = slice_props(entry)
        for name in SLICE_PROPS:
            if name in props:
                setattr(entry, name, props[name])
        # A drag writes floats straight off the cursor, and everything
        # downstream assumes the declared types; ``normalise`` is the one place
        # that coercion lives, so it cannot be spelled differently here.
        entry.normalise()
        entry.bounds = tf.clamp_rect(entry.bounds, self.size)
        after = slice_props(entry)
        # A ``was`` naming only some of the five leaves the rest *unchanged*,
        # which is what overlaying it on ``after`` says -- rather than "absent,
        # therefore different", which would push a step for a field nobody
        # touched.
        before = current if was is None else {**after, **_clean(was)}
        if before == after:
            return False
        self._apply_slice(uid, after)
        self.history.push(SliceChangeEdit(uid, before, after))
        return True

    def set_slice_key(
        self: Document,
        uid: int,
        frame_uid: int | None = None,
        *,
        key: SliceKey | None = None,
        clear: bool = False,
    ) -> bool:
        """Key this slice on one frame, or drop the key it has there.

        **Explicit, always.** Nothing else in this module writes a key: a drag
        moves the base rectangle, and a user who wants this frame to differ says
        so. The alternative -- keying whichever frame the playhead sat on when a
        corner was dragged -- produces an animation with a slightly different
        rectangle on every frame and no way to tell which were meant.

        The default ``key`` is the slice as it *resolves* on that frame, so
        "Key this frame" on an unkeyed frame changes nothing visible and gives
        the user something to then edit.
        """
        entry = self.slice_by_uid(uid)
        if entry is None or frame_uid is None:
            return False
        keys = dict(entry.keys)
        if clear:
            if frame_uid not in keys:
                return False
            del keys[frame_uid]
        else:
            keys[frame_uid] = key if key is not None else entry.at(frame_uid)
        return self.set_slice(uid, keys=keys)

    # -- geometry ------------------------------------------------------------
    #
    # Called from inside the closures ``_doc_geometry`` hands ``_replay``, and
    # *before* the planes are mapped: each of these reads ``self.size``, which is
    # the old canvas until ``_map_planes`` rebuilds the stack. Redo replays the
    # same closure over a restored document, so the order holds there too.

    def _map_slices(self: Document, point: Point, canvas: tuple[int, int]) -> None:
        for entry in self.slices:
            _remap(entry, point, canvas)

    def _slices_flip(self: Document, axis: str) -> None:
        size = self.size
        self._map_slices(lambda x, y: tf.flip_point((x, y), size, axis), size)

    def _slices_rotate90(self: Document, quarters: int = 1) -> None:
        size = self.size
        self._map_slices(
            lambda x, y: tf.rotate90_point((x, y), size, quarters),
            tf.rotate90_size(size, quarters),
        )

    def _slices_scale(self: Document, size: tuple[int, int]) -> None:
        old = self.size
        new = (max(1, int(size[0])), max(1, int(size[1])))
        self._map_slices(lambda x, y: tf.scale_point((x, y), old, new), new)

    def _slices_offset(
        self: Document, offset: tuple[int, int], size: tuple[int, int]
    ) -> None:
        """Crop and canvas resize, which are one operation: a translation into a
        canvas of a different size."""
        new = (max(1, int(size[0])), max(1, int(size[1])))
        self._map_slices(lambda x, y: tf.offset_point((x, y), offset), new)

    # -- what an export reads ------------------------------------------------

    def sprite_meta_for_frame(self: Document, frame_uid: int | None = None) -> dict[str, Any]:
        """This document's slices on one frame, as plain data in canvas coords.

        Plain dicts and tuples, and that is the contract rather than a
        convenience: Packwright duck-types this method and must never learn what
        a frame, a key or a ``Slice`` is -- the resolving happens here, on the
        side that owns the model, and ``packwright.sources`` coerces the result
        into its own frozen types.

        Absolute coordinates, where the model stores the pivot and the centre
        relative to the bounds. Converting at the boundary is the whole reason
        the boundary is here: every consumer downstream (the sheet sidecar, the
        TexturePacker sidecar, ``.wpack``) wants source-image space, and each of
        them converting for itself is three chances to disagree.
        """
        out = []
        for entry in self.slices:
            key = entry.at(frame_uid)
            x0, y0, x1, y1 = key.bounds
            out.append(
                {
                    "name": entry.name,
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "pivot": (
                        None
                        if key.pivot is None
                        else (x0 + key.pivot[0], y0 + key.pivot[1])
                    ),
                    "center": (
                        None
                        if key.center is None
                        else (
                            x0 + key.center[0],
                            y0 + key.center[1],
                            x0 + key.center[2],
                            y0 + key.center[3],
                        )
                    ),
                }
            )
        # The first slice carrying a pivot, in document order. One rule, stated
        # once: a sprite has one pivot and a document may have many slices, so
        # something has to choose -- and "the first one that has an opinion" is
        # the only choice a user can predict from the list they are looking at.
        pivot = next((entry["pivot"] for entry in out if entry["pivot"] is not None), None)
        return {"pivot": pivot, "slices": out}


def _clean(props: dict[str, Any]) -> dict[str, Any]:
    """A caller's ``was`` reduced to the five known keys, detached."""
    out = {name: props[name] for name in SLICE_PROPS if name in props}
    if "keys" in out:
        out["keys"] = dict(out["keys"])
    return out
