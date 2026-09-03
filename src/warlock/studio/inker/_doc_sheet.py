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
from .undo import SheetBaseEdit

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
        edits = self._sheet_cel_edits(track, pairs, box)
        if not edits:
            return False
        pushed = self._push_range(edits)
        self.invalidate_all()
        return pushed

    def _sheet_cel_edits(
        self: Document,
        track: Any,
        pairs: Sequence[tuple[Any, PixelFn, np.ndarray | None]],
        box: tuple[int, int, int, int],
    ) -> list[Any]:
        """:meth:`_map_cels` without the push, so a caller can add to the step.

        Named for this mixin because ``RangeOps`` already has a ``_cel_edits`` of
        its own and ``Document`` inherits both -- a collision the MRO resolves
        silently and in the other direction. Split out for :meth:`merge_render`,
        which changes the document's recorded render in the *same* step as the
        pixels -- a merge undone
        without its base restores the picture and leaves the document's idea of
        what the renderer last gave it in the future, and the next merge would
        then read every restored edit as untouched. The five existing verbs go
        on calling :meth:`_map_cels` and behave exactly as they did.
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
        return edits

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


    def merge_render(
        self: Document,
        track_uid: int,
        incoming: Sequence[np.ndarray],
    ) -> Any:
        """Land a re-rendered sheet on this document without losing hand edits.

        The sixth verb, and the only one that reads a cell before deciding
        whether it may write it. :mod:`.sheetmerge` holds the comparison and
        the rule; this holds the document work.

        **Nothing painted is overwritten silently.** A cell the user changed
        and the renderer also changed is a *conflict*: the edit stands, the
        cell is flagged, and a person decides. That default is not
        configurable, because the two mistakes do not cost the same -- a cell
        wrongly kept is one click to re-take, and a cell wrongly taken is work
        that is gone.

        **The base advances to the incoming render**, for every cell including
        the ones that kept an edit, because that is now what the renderer last
        gave us. It rides in the same undo step as the pixels; see
        :class:`~.undo.SheetBaseEdit` for what goes wrong if it does not.

        Cells outside the re-rendered runs need no special case: their incoming
        pixels are the previous atlas's own, copied through by the worker, so
        they compare equal to the base and classify themselves. The merge never
        has to be told what the subset was.

        Returns :class:`~.sheetmerge.MergeCounts`.
        """
        from . import sheetmerge

        if self.anim is None:
            raise ValueError("this document has no timeline to merge into")
        base = getattr(self, "sheet_base", None)
        if base is None:
            raise ValueError(
                "this document has no recorded render to merge against -- open the "
                "sheet from Troupe rather than as a plain image"
            )
        frames = self.anim.frames
        if len(incoming) != len(frames):
            raise ValueError(
                f"this document has {len(frames)} frames and that sheet has "
                f"{len(incoming)}"
            )
        width, height = self.size
        for cell in incoming:
            if cell.shape[:2] != (height, width):
                raise ValueError(
                    f"that sheet's cells are {cell.shape[1]}x{cell.shape[0]} and this "
                    f"document is {width}x{height}"
                )
        track = self._track_by_uid(track_uid)
        if track.alpha_lock:
            # Refused, never bypassed. ``_sheet_cel_edits`` passes the lock into
            # ``masked_apply``, so a locked track would keep the *old*
            # silhouette while this function reported the render as taken --
            # a merge that lies about what it did.
            raise ValueError(
                "turn off this track's alpha lock: a merged render replaces the "
                "whole cell rather than painting inside it"
            )
        self.commit_floating()

        counts = {verdict: 0 for verdict in sheetmerge.VERDICTS}
        pairs: list[tuple[Any, PixelFn, np.ndarray | None]] = []
        conflicts: set[int] = set()
        digests: dict[int, str] = {}
        for index, frame in enumerate(frames):
            layer = self._cel_at(track.uid, index)
            fresh = sheetmerge.cell_digest(incoming[index])
            digests[frame.uid] = fresh
            if layer is None:
                counts["unknown"] += 1
                continue
            if self.anim.is_linked(track.uid, frame.uid):
                # One cel serving two slots cannot take two different renders,
                # and ``_sheet_cel_edits`` dedupes by identity -- so one of them
                # would be dropped without a word. An imported sheet has no
                # linked cels; this only fires if somebody linked them by hand.
                raise ValueError(
                    "unlink this sheet's cels before merging: a linked cel cannot "
                    "take two different renders"
                )
            verdict = sheetmerge.classify(
                base.digests.get(frame.uid),
                sheetmerge.cell_digest(layer.pixels),
                fresh,
            )
            counts[verdict] += 1
            if verdict == "take":
                pairs.append((layer, _constant(incoming[index]), None))
            elif verdict == "conflict":
                conflicts.add(frame.uid)

        edits = self._sheet_cel_edits(track, pairs, self._box(None)) if pairs else []
        after = sheetmerge.SheetBase(
            digests=digests,
            conflicts=conflicts,
            source=dict(base.source),
            algorithm=base.algorithm,
        )
        edits.append(SheetBaseEdit(before=base.copy(), after=after))
        self.sheet_base = after
        self._push_range(edits)
        self.invalidate_all()
        return sheetmerge.MergeCounts(
            taken=counts["take"],
            kept=counts["keep"],
            agreed=counts["agreed"],
            conflicts=counts["conflict"],
            unknown=counts["unknown"],
        )


def _constant(plane: np.ndarray) -> PixelFn:
    return lambda _before: plane


def _weight_bounds(weight: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(weight.any(axis=1))
    cols = np.flatnonzero(weight.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
