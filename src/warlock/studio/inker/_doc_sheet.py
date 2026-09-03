"""Character-sheet corrections: one write funnel, five verbs.

``sheetscope`` decides *which* frames a correction reaches; this mixin is the
one door that writes them. Every verb below resolves to :meth:`SheetOps.map_frames`,
which is ``filter_range`` (``_doc_ranges``) narrowed to one track and an
explicit frame list rather than a rectangle: ``commit_floating`` first, the
distinct cels of the named slots (deduped by ``id()``, ``_cels_in``'s rule),
a tilemap refused by name before anything is written, ``masked_apply`` per cel
with the track's alpha lock, the mode-aware edit from ``_patch_edit_for``, and
one ``_push_range`` for the lot -- so a correction sent to forty cells is one
``Ctrl+Z``, and a correction that changes nothing pushes nothing.

No new ``Edit`` class: a patch edit addressed by layer uid is exactly what a
range op already leaves on the stack, and what makes undo hold after a frame
is inserted or a track reordered.

The verbs take *frame indices* because that is what ``sheetscope`` hands back
and what the timeline shows; they are resolved to uids at the door, once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from . import filters, mirror, sheetscope
from ._doc_ranges import masked_apply
from .tiles import TilemapCel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document

__all__ = ["SheetOps"]

PixelFn = Callable[[np.ndarray], np.ndarray]


class SheetOps:
    """Sheet-scoped edits, mixed into :class:`~.document.Document`."""

    # -- addressing ----------------------------------------------------------

    def sheet_runs(self: Document) -> list[sheetscope.Run]:
        anim = self.anim
        return [] if anim is None else sheetscope.runs(anim.tags)

    def has_sheet(self: Document) -> bool:
        return bool(self.sheet_runs())

    def _track_by_uid(self: Document, track_uid: int) -> Any:
        anim = self.anim
        if anim is None:
            raise ValueError("this document has no timeline")
        for track in anim.tracks:
            if track.uid == int(track_uid):
                return track
        raise ValueError("that track is no longer in the document")

    def _cel_at(self: Document, track_uid: int, frame: int) -> Any:
        """The cel in a slot, or None -- never a placeholder, for
        ``filter_range``'s reason: these verbs do not autovivify."""
        anim = self.anim
        if anim is None or not (0 <= int(frame) < len(anim.frames)):
            return None
        return anim.cels.get((int(track_uid), anim.frames[int(frame)].uid))

    def _box(self: Document, box: tuple[int, int, int, int] | None):
        width, height = self.size
        return self.clip(box or (0, 0, width, height))

    # -- the funnel ----------------------------------------------------------

    def _map_cels(
        self: Document,
        track: Any,
        pairs: Sequence[tuple[Any, PixelFn, np.ndarray | None]],
        box: tuple[int, int, int, int],
    ) -> bool:
        """Write ``fn(before)`` into each ``(cel, fn, weight)`` as one step.

        The per-cel loop of :meth:`map_frames`, factored out so
        :meth:`mirror_run` can pair a different source with every target and
        still push once. ``pairs`` is already deduped by the caller.
        """
        if any(isinstance(layer, TilemapCel) for layer, _fn, _w in pairs):
            raise ValueError("a sheet correction of a tilemap layer is not yet modeled")
        x0, y0, x1, y1 = box
        edits: list[Any] = []
        for layer, fn, weight in pairs:
            before = layer.pixels[y0:y1, x0:x1].copy()
            produced = fn(before)
            if produced.shape != before.shape:
                raise ValueError("a sheet correction must keep the cell's shape")
            clipped = None if weight is None else weight[y0:y1, x0:x1]
            layer.pixels[y0:y1, x0:x1] = masked_apply(
                before, produced, clipped, alpha_lock=track.alpha_lock
            )
            edit = self._patch_edit_for(layer, box, before)
            if edit is None:
                continue
            self._stamp_layer(layer.uid)
            edits.append(edit)
        if not edits:
            return False
        pushed = self._push_range(edits)
        self.invalidate_all()
        return pushed

    def map_frames(
        self: Document,
        track_uid: int,
        frames: Sequence[int],
        fn: PixelFn,
        *,
        weight: np.ndarray | None = None,
        box: tuple[int, int, int, int] | None = None,
        exclude: Any = None,
    ) -> bool:
        """Apply ``fn`` to the distinct cels of ``track`` at ``frames``.

        ``weight`` is a full-canvas ``masked_apply`` weight or None for the
        whole cell; ``box`` bounds the write (the weight's bounds, typically).
        ``exclude`` is a cel object that must not be written even if it is
        linked into a target slot -- the source of a propagation, which
        already carries the correction.
        """
        if self.anim is None:
            return False
        track = self._track_by_uid(track_uid)
        self.commit_floating()
        clipped = self._box(box)
        if clipped is None:
            return False
        seen: set[int] = set() if exclude is None else {id(exclude)}
        pairs: list[tuple[Any, PixelFn, np.ndarray | None]] = []
        for frame in frames:
            layer = self._cel_at(track.uid, frame)
            if layer is None or id(layer) in seen:
                continue
            seen.add(id(layer))
            pairs.append((layer, fn, weight))
        if not pairs:
            return False
        return self._map_cels(track, pairs, clipped)

    # -- the verbs -----------------------------------------------------------

    def propagate_patch(
        self: Document,
        track_uid: int,
        source_frame: int,
        frames: Sequence[int],
        weight: np.ndarray,
    ) -> bool:
        """Copy the weighted pixels of the source cel onto every target."""
        source = self._cel_at(track_uid, source_frame)
        if source is None:
            raise ValueError("the source cell is empty")
        if weight.shape != source.pixels.shape[:2]:
            raise ValueError("the mark does not match the canvas")
        pixels = source.pixels.copy()
        bounds = _weight_bounds(weight)
        if bounds is None:
            return False
        return self.map_frames(
            track_uid,
            frames,
            lambda _before: pixels[bounds[1] : bounds[3], bounds[0] : bounds[2]],
            weight=weight,
            box=bounds,
            exclude=source,
        )

    def replace_colour_frames(
        self: Document,
        track_uid: int,
        frames: Sequence[int],
        old: tuple[int, int, int, int],
        new: tuple[int, int, int, int],
        tolerance: float = 0.0,
    ) -> bool:
        """``filters.replace_colour`` over the frames, inside the selection
        if there is one -- the same recolour the filter panel offers, sent to
        cells that are not on screen."""
        weight = None if self.mask is None else self.mask.mask
        bounds = None if self.mask is None else self.mask.bounds
        return self.map_frames(
            track_uid,
            frames,
            partial(filters.replace_colour, old=old, new=new, tolerance=float(tolerance)),
            weight=weight,
            box=bounds,
        )

    def shift_frames(
        self: Document,
        track_uid: int,
        frames: Sequence[int],
        dx: int,
        dy: int,
        weight: np.ndarray | None = None,
    ) -> bool:
        """Translate the selected pixels by whole pixels on every frame.

        The weight is the selection (``doc.mask``) unless given; a document
        with neither is refused, because moving *everything* is
        ``shift_range``'s job and it already exists.
        """
        if weight is None:
            if self.mask is None:
                raise ValueError("select the pixels to move first")
            weight = self.mask.mask
        if not (int(dx) or int(dy)):
            return False
        # The whole canvas: a translation's destination is outside the
        # selection's own bounds by definition.
        return self.map_frames(
            track_uid,
            frames,
            partial(mirror.translate_within, weight=weight, dx=int(dx), dy=int(dy)),
        )

    def mirror_to(
        self: Document,
        track_uid: int,
        source_frame: int,
        target_frame: int,
        face_fraction: float = mirror.FACE_FRACTION,
    ) -> bool:
        """The mirror of the source cel onto the target, face excluded."""
        source = self._cel_at(track_uid, source_frame)
        if source is None:
            raise ValueError("the source cell is empty")
        flipped = mirror.mirrored(source.pixels)
        weight = mirror.face_weight(
            flipped.shape[:2], mirror.face_box(flipped, face_fraction)
        )
        return self.map_frames(
            track_uid, [int(target_frame)], lambda _b: flipped, weight=weight, exclude=source
        )

    def mirror_run(
        self: Document,
        track_uid: int,
        run: sheetscope.Run,
        face_fraction: float = mirror.FACE_FRACTION,
    ) -> bool:
        """Every frame of ``run`` mirrored onto its counterpart, as one step."""
        if self.anim is None:
            return False
        sheet = self.sheet_runs()
        track = self._track_by_uid(track_uid)
        self.commit_floating()
        box = self._box(None)
        if box is None:
            return False
        seen: set[int] = set()
        pairs: list[tuple[Any, PixelFn, np.ndarray | None]] = []
        for frame in range(run.start, run.end + 1):
            target_frame = sheetscope.counterpart(sheet, frame)
            if target_frame is None:
                continue
            source = self._cel_at(track.uid, frame)
            target = self._cel_at(track.uid, target_frame)
            if source is None or target is None or target is source:
                continue
            if id(target) in seen:
                continue
            seen.add(id(target))
            flipped = mirror.mirrored(source.pixels)
            weight = mirror.face_weight(
                flipped.shape[:2], mirror.face_box(flipped, face_fraction)
            )
            pairs.append((target, _constant(flipped), weight))
        if not pairs:
            return False
        return self._map_cels(track, pairs, box)


def _constant(plane: np.ndarray) -> PixelFn:
    return lambda _before: plane


def _weight_bounds(weight: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(weight.any(axis=1))
    cols = np.flatnonzero(weight.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
