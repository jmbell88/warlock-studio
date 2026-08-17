"""Selections, floating pixels, the clipboard, and the free transform.

The four are one concern because they are one lifecycle: a selection names a
region, a lift turns that region into floating pixels, the clipboard is the same
buffer parked somewhere, and a transform is what a floating buffer can do before
it lands. ``_masked_alpha`` is the invariant they share -- pixels that carry
their own coverage -- and it lives here because both its callers do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import composite as cp
from . import transform as tf
from .anim_edits import TrackAddEdit
from .animation import Track
from .brush import MAX_STAMP, Stamp
from .layers import Layer
from .selection import FloatingBuffer, SelectionMask, magic_wand, render_transform
from .undo import CompoundEdit, LayerAddEdit, PatchEdit, SelectionEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document


def _masked_alpha(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A copy of ``pixels`` with a selection mask folded into its alpha.

    The one invariant shared by a lifted buffer and the clipboard: pixels that
    carry their own coverage, so whatever composites them later needs no mask
    at all -- and a feathered edge stays feathered through both.
    """
    out = pixels.copy()
    out[..., 3] = (out[..., 3].astype(np.float32) * mask / 255.0).astype(np.uint8)
    return out


class SelectionOps:
    """Selection, floating pixels and the clipboard, mixed into
    :class:`~.document.Document`."""

    # -- selection ----------------------------------------------------------

    def select(self: Document, mask: SelectionMask | None, op: str = "replace") -> None:
        before = None if self.mask is None else self.mask.mask
        if mask is None:
            self.mask = None
        elif op == "replace" or (self.mask is None and op == "add"):
            self.mask = mask.copy()
        elif self.mask is None:
            # Subtracting from nothing, or intersecting with it, is nothing.
            # Adopting the new mask instead would make a subtract-drag on an
            # empty canvas *create* the selection it was meant to cut away.
            self.mask = None
        else:
            self.mask = self.mask.combined(mask, op)
        if self.mask is not None and self.mask.is_empty:
            self.mask = None
        after = None if self.mask is None else self.mask.mask
        # Nothing changed, so nothing to undo. Pushing anyway -- which
        # subtracting from an empty selection did -- moved the head, and dirty
        # is a comparison against the head: an Alt-drag on a canvas with no
        # selection made a saved document ask to be saved again, and spent a
        # Ctrl+Z doing nothing. The both-None case was the one that reproduced;
        # re-selecting the same region (a marquee redrawn over itself, Select
        # All twice, feathering by zero) is the same no-op and was still
        # pushing.
        if before is None and after is None:
            return
        if before is not None and after is not None and np.array_equal(before, after):
            return
        self.history.push(SelectionEdit(before, after))
        self.rev += 1

    def select_all(self: Document) -> None:
        width, height = self.size
        self.select(SelectionMask.full(width, height))

    def deselect(self: Document) -> None:
        self.commit_floating()
        if self.mask is not None:
            # Remembered here and nowhere else. Undo's own route back to no
            # selection (``set_selection_mask``) deliberately does not record
            # one: a Ctrl+Z is the user putting the selection *back*, so there
            # is nothing for a reselect to restore and overwriting the memory
            # would throw away the mask they actually meant.
            self._last_mask = self.mask.mask.copy()
            self.select(None)

    def reselect(self: Document) -> bool:
        """Bring back the selection the last :meth:`deselect` took away.

        An ordinary ``select``, so it pushes an ordinary step and is undone by
        an ordinary Ctrl+Z -- the memory is the only new thing here.

        A mask whose shape no longer matches the canvas is refused rather than
        placed or rescaled. It is a per-pixel statement about a canvas that no
        longer exists, and the two honest answers to "reselect after a crop"
        are "nothing" and "guess"; guessing would put a selection somewhere the
        user never drew one.
        """
        mask = self._last_mask
        if mask is None:
            return False
        width, height = self.size
        if mask.shape != (height, width):
            return False
        self.select(SelectionMask(mask.copy()))
        return True

    def invert_selection(self: Document) -> None:
        width, height = self.size
        current = self.mask or SelectionMask(np.zeros((height, width), dtype=np.uint8))
        self.select(current.inverted())

    def feather_selection(self: Document, radius: float) -> None:
        if self.mask is not None:
            self.select(self.mask.feathered(radius))

    def grow_selection(self: Document, radius: int) -> None:
        if self.mask is not None:
            self.select(self.mask.grown(radius))

    def shrink_selection(self: Document, radius: int) -> None:
        if self.mask is not None:
            self.select(self.mask.shrunk(radius))

    def border_selection(self: Document, width: int) -> None:
        if self.mask is not None:
            self.select(self.mask.bordered(width))

    def select_layer_alpha(
        self: Document, index: int | None = None, op: str = "replace"
    ) -> None:
        """Select what is painted on a layer, at the coverage it is painted at.

        The alpha channel *is* a selection mask -- both are 8-bit per-pixel
        coverage over the canvas -- so this is a copy rather than a threshold,
        and a soft brush edge becomes a soft selection edge. Thresholding would
        turn every antialiased drawing into a jagged one the first time somebody
        asked to select it.

        It reads the layer's own alpha, not the composite's: "select this
        layer's pixels" is a question about one layer, and its opacity and blend
        mode are about how it is *shown*. That is the same split
        ``eyedrop(layer_only=True)`` makes.
        """
        index = self.stack.active_index if index is None else index
        layer = self.stack[index]
        self.select(SelectionMask(layer.pixels[..., 3].copy()), op)

    def select_wand(
        self: Document, xy: tuple[int, int], *, tolerance: int = 32, op: str = "replace",
        contiguous: bool = True, wrap: str | tuple[bool, bool] = "off",
    ) -> None:
        self.select(
            magic_wand(
                self._composite, xy, tolerance=tolerance, contiguous=contiguous, wrap=wrap
            ),
            op,
        )

    # -- floating pixels ----------------------------------------------------

    def lift(self: Document, mask: SelectionMask | None = None) -> bool:
        """Cut the selection out of the active layer and float it. One step.

        Refused on a content-locked layer: a lift is a *cut*, and the hole it
        leaves is exactly the write the lock exists to prevent. ``copy`` stays
        legal beside it -- reading pixels off a locked layer changes nothing --
        and so does ``paste_as_layer``, which writes to a layer of its own.
        """
        if self.write_locked():
            return False
        self.commit_floating()
        mask = mask or self.mask
        if mask is None:
            return False
        bounds = mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return False
        self._ensure_active_cel()
        layer = self.stack.active
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        crop = mask.mask[y0:y1, x0:x1]

        # The floating pixels keep their own alpha *multiplied* by the mask, so
        # a feathered lift floats a feathered chunk rather than a hard one.
        pixels = _masked_alpha(before, crop)
        # And what stays behind is the *remainder*, subtracted rather than
        # computed from (1 - mask). A lift is a partition of alpha: whatever
        # floats away must be exactly what is no longer there. Computed
        # independently, both halves truncate toward zero and the two floors
        # sum to a - 1 wherever a * m / 255 is not a whole number -- so every
        # feathered edge lost one level of alpha per lift, and a lift-and-drop
        # that changed nothing else still darkened its own outline a step at a
        # time. Safe in uint8 without a clip: the lifted alpha is a floor of
        # a * m / 255 with m <= 255, so it never exceeds ``before``.
        cut = before.copy()
        cut[..., 3] = before[..., 3] - pixels[..., 3]
        layer.pixels[y0:y1, x0:x1] = cut

        after = layer.pixels[y0:y1, x0:x1].copy()
        edit: Any = PatchEdit(layer.uid, box, before, after)
        # The buffer names the step that is actually *on the stack*, because
        # ``cancel_floating`` revokes it by identity. Wrapping the patch in a
        # compound and then naming the bare patch would leave the revoke
        # searching for something the stack does not hold -- the alpha-cut would
        # stay and the lifted pixels would be dropped.
        pending, self._pending_cels = self._pending_cels, []
        if pending:
            edit = CompoundEdit([*pending, edit])
        self.floating = FloatingBuffer(
            pixels=pixels, mask=crop.copy(), offset=(x0, y0), layer_uid=layer.uid,
            lift_edit=edit,
            # Where these pixels were cut from. A ranged commit re-cuts the
            # same rectangle out of every other cel in the range, which is what
            # makes the gesture repeatable rather than a stamp of this one cel.
            source_box=box,
        )
        self.history.push(edit)
        self.invalidate(box, layer_uid=layer.uid)
        return True

    def move_floating(self: Document, dx: int, dy: int) -> None:
        if self.floating is not None:
            self.floating.moved(dx, dy)
            self.rev += 1

    def commit_floating(self: Document) -> bool:
        """Write the floating pixels where they now sit and stop floating.

        **Canonical-only, tiled or not** -- stated here rather than left to be
        discovered. The write is clipped to the canvas, so a buffer left hanging
        over the edge of a tiled document loses its overhang instead of landing
        on the far side. Everything that *paints* wraps (brush, fill, shapes,
        wand); this does not, because a floating buffer is a rectangle being
        positioned rather than coverage being laid down, and wrapping it would
        have to answer what the transform box and its handles mean when the
        subject is in four places at once. Deliberate for T-B v1.

        **Deliberately not refused by the content lock.** A buffer floats for as
        long as the user leaves it floating -- across layer switches, frame
        steps and the panel's lock checkbox -- and every save, every geometry op
        and every structural op commits it first. So refusing here would not
        protect the pixels; it would wedge the document, leaving a buffer that
        can neither land nor be saved. See ``LayerOps.write_locked``.
        """
        floating, self.floating = self.floating, None
        if floating is None:
            return False
        ox, oy = floating.offset
        fw, fh = floating.size
        box = self.clip((ox, oy, ox + fw, oy + fh))
        if box is None:
            self.rev += 1
            return True
        # Keyed by the buffer's own layer, not the active one: a paste chooses
        # its target when it is made, and the user may have selected another row
        # while it floated.
        self._ensure_cel_for(floating.layer_uid)
        layer = self.stack.by_uid(floating.layer_uid)
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        crop = floating.pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
        merged = cp.over(cp.to_float(before), cp.to_float(crop))
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8(merged)
        self._commit_patch(layer, box, before)
        return True

    def commit_floating_range(
        self: Document, t0: int, t1: int, f0: int, f1: int
    ) -> bool:
        """Commit the floating buffer to every cel of a timeline rect.

        **The preview only ever showed the active cel** -- the buffer holds
        that cel's pixels and no others -- so commit is where the range takes
        effect. What is replayed is the *gesture*, not the buffer: cut the
        lift's rectangle with the lift's mask, run the recorded flips in order,
        render one scale-then-shear-then-rotate through
        :func:`~.selection.render_transform`, composite at the offset the user
        dragged to. Every cel therefore shows its own content moved, which is
        what a timeline-target transform means; pasting the buffer into each
        slot would make N copies of one drawing.

        Three falls back to the plain commit, because for each of them a range
        means nothing: no animation, a rect that misses the grid, and a
        **paste** -- which has no ``source_box``, so there is no rectangle to
        re-cut out of anybody else. The call site can therefore stay dumb and
        always ask for the ranged one.

        The active cel replays like every other: its lift is reversed first
        (by identity, ``revoke``'s rule -- the lift is routinely no longer the
        newest step) so the replay reads pristine pixels. If that lift had
        autovivified the cel it came from, revoking takes the cel back out and
        the slot is simply skipped below -- there was nothing there to
        transform.
        """
        floating = self.floating
        if floating is None:
            return False
        rect = self._range(t0, t1, f0, f1)
        if rect is None or floating.source_box is None:
            return self.commit_floating()
        # The whole recipe, captured before the buffer is torn down.
        flips = list(floating.flips)
        angle, scale, shear = floating.angle, floating.scale, floating.shear
        resample = floating.resample
        dest = floating.offset
        source_box = floating.source_box
        src_mask = (
            floating.source_mask if floating.source is not None else floating.mask
        ).copy()
        lift_edit = floating.lift_edit
        active_uid = floating.layer_uid

        self.floating = None
        if lift_edit is not None:
            self.history.revoke(self, lift_edit)

        targets = [cel for _track, cel in self._cels_in(rect)]
        active = self.stack.by_uid(active_uid)
        if active is not None and not any(cel is active for cel in targets):
            # The cel the user actually watched transform. In even when the
            # rect excludes it: dropping the one change they could see because
            # of where the marquee happened to stop would be indefensible.
            targets.append(active)
        if not targets:
            return True
        edits: list[Any] = []
        for layer in targets:
            edit = self._replay_transform_on(
                layer, source_box, src_mask, flips, angle, scale, shear, resample, dest
            )
            if edit is None:
                continue
            self._stamp_layer(layer.uid)
            edits.append(edit)
        if edits:
            self._push_range(edits)
        self.invalidate_all()
        return True

    def _replay_transform_on(
        self: Document,
        layer: Any,
        source_box: tuple[int, int, int, int],
        src_mask: np.ndarray,
        flips: list[str],
        angle: float,
        scale: tuple[float, float],
        shear: tuple[float, float],
        resample: str,
        dest: tuple[int, int],
    ) -> Any:
        """One cel through the recorded gesture: cut, render, composite.

        The cut is ``lift``'s own partition of alpha -- what floats away is
        subtracted from what stays, rather than each half being computed
        independently, because two independent floors lose a level of alpha on
        every feathered pixel. One patch over the union of the two rectangles,
        so a single undo puts back both the hole and the landing.

        Every cel is cut from the same ``source_box`` with the same mask, so
        every render has the same shape, so the active buffer's final offset
        places every replay exactly. That is what lets ``dest`` be one number
        for the whole range rather than one per cel.
        """
        sx0, sy0, sx1, sy1 = source_box
        region = layer.pixels[sy0:sy1, sx0:sx1]
        lifted = _masked_alpha(region, src_mask)
        source, mask = lifted, src_mask.copy()
        for axis in flips:
            source, mask = tf.flip(source, axis), tf.flip(mask, axis)
        moved, _moved_mask = render_transform(
            source, mask, angle, scale, shear, resample
        )
        dest_h, dest_w = moved.shape[:2]
        landing = self.clip((dest[0], dest[1], dest[0] + dest_w, dest[1] + dest_h))
        box = (
            (sx0, sy0, sx1, sy1)
            if landing is None
            else (
                min(sx0, landing[0]),
                min(sy0, landing[1]),
                max(sx1, landing[2]),
                max(sy1, landing[3]),
            )
        )
        bx0, by0, bx1, by1 = box
        before = layer.pixels[by0:by1, bx0:bx1].copy()
        # ``region`` is a view, so this is the hole, cut in place -- and it
        # happens after ``before`` is read, or the patch would record the state
        # halfway through its own gesture.
        region[..., 3] = region[..., 3] - lifted[..., 3]
        if landing is not None:
            lx0, ly0, lx1, ly1 = landing
            crop = moved[
                ly0 - dest[1] : ly1 - dest[1], lx0 - dest[0] : lx1 - dest[0]
            ]
            base = layer.pixels[ly0:ly1, lx0:lx1]
            layer.pixels[ly0:ly1, lx0:lx1] = cp.to_uint8(
                cp.over(cp.to_float(base), cp.to_float(crop))
            )
        return self._patch_edit_for(layer, box, before)

    def cancel_floating(self: Document) -> bool:
        """Put the pixels back where they were lifted from, exactly.

        Exact because it is the lift's own undo step rather than a re-paste --
        a feathered lift cannot be re-pasted without rounding. That step is
        reversed *and forgotten*: the user asked for the lift not to have
        happened, so leaving it redoable would let Ctrl+Y replay the alpha-cut
        with no buffer left to put back.

        The lift's *own* step, named on the buffer, and not simply the newest
        one. A buffer floats for as long as the user leaves it floating, and
        every selection op pushes a step meanwhile -- so "the newest step" is
        routinely something else, and reversing that instead dropped the
        lifted pixels, kept the alpha-cut and destroyed an unrelated edit, all
        three unrecoverably.

        A pasted buffer owns no step, and reversing one anyway would undo
        whatever the user did before pasting.
        """
        floating, self.floating = self.floating, None
        if floating is None:
            return False
        if not floating.lifted or not self.history.revoke(self, floating.lift_edit):
            # Nothing to reverse: a paste, or a lift whose step has since been
            # undone or evicted by the byte budget.
            self.rev += 1
        return True

    def delete_floating(self: Document) -> bool:
        """Throw the floating pixels away; the hole is already cut."""
        if self.floating is None:
            return False
        self.floating = None
        self.rev += 1
        return True

    def _cut_selection_patch(self: Document) -> list[Any] | None:
        """Cut the selection out of the active layer, *returning* the steps.

        Split out of :meth:`delete_selection` because ``layer_from_selection``
        needs the same write folded into a bigger ``CompoundEdit`` -- the cut
        and the new layer are one gesture and must be one Ctrl+Z.

        Three answers, and the distinction is what keeps the callers' contracts
        exact: ``None`` for "there was nothing to cut" (no selection, or one
        entirely off canvas), ``[]`` for a cut that changed no pixel, and a list
        of steps otherwise.

        It does not go through ``_commit_patch``, so it does not snap an indexed
        document -- and does not need to. This write touches alpha alone, the
        RGB it leaves behind was already snapped when it was written, and
        ``snap`` never touches alpha in any case.
        """
        if self.mask is None:
            return None
        bounds = self.mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return None
        self._ensure_active_cel()
        layer = self.stack.active
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        keep = 1.0 - self.mask.mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        layer.pixels[y0:y1, x0:x1, 3] = (before[..., 3].astype(np.float32) * keep).astype(
            np.uint8
        )
        after = layer.pixels[y0:y1, x0:x1].copy()
        if np.array_equal(before, after):
            self._discard_pending_cel()
            return []
        pending, self._pending_cels = self._pending_cels, []
        self.invalidate(box, layer_uid=layer.uid)
        return [*pending, PatchEdit(layer.uid, box, before, after)]

    def delete_selection(self: Document) -> bool:
        """Cut the selection out of the active layer without floating it."""
        if self.floating is not None:
            return self.delete_floating()
        if self.write_locked():
            return False
        edits = self._cut_selection_patch()
        if edits is None:
            return False
        if edits:
            self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        return True

    # -- clipboard ----------------------------------------------------------

    def copy(self: Document) -> bool:
        """Copy the selection (or the floating buffer) to the app clipboard.

        The pixels carry the mask in their alpha, which is exactly the
        invariant :meth:`lift` gives a :class:`FloatingBuffer` -- and it has to
        be, because ``paste`` floats what it takes verbatim and
        ``commit_floating`` composites a buffer without consulting its mask.
        Putting the *rectangular* crop on the clipboard instead meant an
        ordinary Ctrl+C over a lasso, ellipse, wand or feathered selection
        pasted its full bounding box, while hit-testing (which does read the
        mask) still only let the user grab the shape they selected.

        The mask travels beside the pixels regardless: a paste has to know
        which of them are its own for that hit-test.
        """
        if self.floating is not None:
            self.clipboard.put(self.floating.pixels, self.floating.mask)
            return True
        if self.mask is None:
            return False
        bounds = self.mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return False
        x0, y0, x1, y1 = box
        crop = self.mask.mask[y0:y1, x0:x1]
        self.clipboard.put(_masked_alpha(self.stack.active.pixels[y0:y1, x0:x1], crop), crop)
        return True

    def cut(self: Document) -> bool:
        return self.copy() and self.delete_selection()

    def capture_stamp(self: Document, mask: SelectionMask | None = None) -> Stamp | None:
        """The selection, lifted as a brush tip. ``None`` when there is none.

        Aseprite's "new brush from selection", and it is :meth:`copy` rather
        than :meth:`lift`: capturing a brush *reads* the drawing, so it cuts
        nothing, pushes no undo step, autovivifies no cel and is legal on a
        content-locked layer. The pixels come through ``_masked_alpha`` -- the
        one invariant a lift, the clipboard and a promoted layer all share -- so
        a feathered or lasso selection captures a feathered tip rather than its
        bounding box, and the tip's own alpha *is* the dab's coverage with no
        second mask to carry.

        The **active layer**, not the composite, for the reason
        ``eyedrop(layer_only=True)`` exists and the reason ``copy`` reads the
        same plane: "make a brush out of this" is a question about the drawing
        the user is working on, and a capture that folded in the layers under it
        would pick up a background nobody selected.

        Two answers are ``None`` and the caller has to tell them apart, which it
        can: no selection at all, and a selection whose bounds exceed
        :data:`.brush.MAX_STAMP` a side. Refused rather than cropped or scaled --
        both would hand back a brush that is not the thing that was selected.
        """
        mask = mask or self.mask
        if mask is None:
            return None
        bounds = mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return None
        x0, y0, x1, y1 = box
        if max(x1 - x0, y1 - y0) > MAX_STAMP:
            return None
        crop = mask.mask[y0:y1, x0:x1]
        return Stamp(_masked_alpha(self.stack.active.pixels[y0:y1, x0:x1], crop))

    # -- free transform -----------------------------------------------------

    def begin_transform(self: Document) -> bool:
        """Lift whatever is being transformed, so it can be moved freely.

        A transform acts on floating pixels and nothing else: it needs to
        rotate a chunk out of alignment with the pixel grid, which cannot be
        expressed as a write back into the layer until it is committed. With no
        selection, the whole layer is lifted -- "rotate this layer" is the
        common case and requiring a select-all first would be busywork.
        """
        if self.floating is not None:
            return True
        if self.mask is None:
            width, height = self.size
            self.select(SelectionMask.full(width, height))
        return self.lift()

    def transform_floating(
        self: Document,
        *,
        angle: float | None = None,
        scale: tuple[float, float] | None = None,
        shear: tuple[float, float] | None = None,
        resample: str = "smooth",
    ) -> bool:
        if self.floating is None:
            return False
        self.floating.transform(
            angle=angle, scale=scale, shear=shear, resample=resample
        )
        self.rev += 1
        return True

    def flip_floating(self: Document, axis: str) -> bool:
        if self.floating is None:
            return False
        self.floating.flip(axis)
        self.rev += 1
        return True

    def rotate_floating(self: Document, degrees: float) -> bool:
        if self.floating is None:
            return False
        return self.transform_floating(angle=self.floating.angle + degrees)

    def paste(self: Document, at: tuple[int, int] | None = None) -> bool:
        """Paste as a floating buffer, so it can be positioned before it lands.

        Canonical-only on a tiled document, for :meth:`commit_floating`'s
        reason and because it is the same landing: the paste floats, and what
        it eventually writes goes through that method.

        Refused on a content-locked layer, because the buffer it makes is bound
        to that layer and would land on it. ``paste_as_layer`` is the way to
        paste beside a locked layer and stays legal.
        """
        if self.write_locked():
            return False
        taken = self.clipboard.take()
        if taken is None:
            return False
        self.commit_floating()
        pixels, mask = taken
        if at is None:
            bounds = self.mask.bounds if self.mask is not None else None
            at = (bounds[0], bounds[1]) if bounds else (0, 0)
        self.floating = FloatingBuffer(
            pixels=pixels, mask=mask, offset=(int(at[0]), int(at[1])),
            layer_uid=self.stack.active.uid,
        )
        # A paste is a new branch of history even though it pushes no step of
        # its own; without this, Ctrl+V then Ctrl+Y redoes an unrelated edit.
        self.history.forget_redo()
        self.rev += 1
        return True

    def _add_layer_edit(
        self: Document, pixels: np.ndarray, name: str, *, offset: tuple[int, int] = (0, 0)
    ) -> Any:
        """Put ``pixels`` on a new layer above the active one, and *return* the
        step rather than pushing it.

        The tail of ``paste_as_layer``, extracted so ``layer_from_selection``
        can fold the same work into a compound with the cut that precedes it.
        Returning the edit rather than pushing it is the whole of the
        refactor: two entry points, one of which is half of a bigger gesture.

        **It does not carry group membership, and both callers must.** This
        inserts above the active row exactly as ``add_layer`` does, so a row
        added while the active one is inside a folder lands in the *middle* of
        that folder's span; leaving it out of the group splits the span and
        makes the contiguity invariant false with nothing to say so. The
        callers pair it with ``inherit_group_edits`` -- which has to be a
        separate call because one of them folds the result into a bigger
        compound and the other pushes it on its own.
        """
        width, height = self.size
        placed = tf.resize_canvas(pixels, (width, height), offset)
        layer = Layer(pixels=placed, name=name)
        if self.anim is not None:
            # A track carrying one cel, on the current frame only: the new
            # pixels are a thing that happens at a moment, not a thing that is
            # true for the whole clip.
            index = self.stack.active_index + 1
            track = Track(name=layer.name)
            cels = {self.anim.frame.uid: layer}
            self._put_track(index, track, cels)
            return TrackAddEdit(index, track, cels, pinned=True)
        index = self.stack.insert(self.stack.active_index + 1, layer)
        self.invalidate_all()
        return LayerAddEdit(index, layer)

    def paste_as_layer(self: Document, pixels: np.ndarray | None = None) -> bool:
        """Paste onto a layer of its own, placed at the top-left.

        A separate entry point rather than a flag on ``paste``: this one lands
        immediately and is undone by removing the layer, where an ordinary
        paste floats until it is put down. It stays legal beside a content-
        locked layer, because it writes to a layer of its own.
        """
        if pixels is None:
            taken = self.clipboard.take()
            if taken is None:
                return False
            # No second multiply: what the clipboard holds already carries its
            # mask in alpha, and applying it twice darkens every feathered edge.
            pixels, _mask = taken
        self.commit_floating()
        name = f"Pasted {len(self.stack) + 1}"
        # Read before the insert, for ``add_layer``'s reason: pasting a layer
        # while working inside a folder puts it in that folder, and a row that
        # stayed outside would sit in the middle of the folder's span.
        parent = self._parent_of_active()
        edit = self._add_layer_edit(pixels, name)
        self._push_with_inheritance(edit, parent)
        return True

    def layer_from_selection(self: Document, *, cut: bool = False) -> bool:
        """Promote the selection onto a layer of its own.

        ``Ctrl+J`` copies (``cut=False``, the default) and ``Ctrl+Shift+J``
        moves -- Photoshop's way round, with the plain chord the
        non-destructive one.

        One ``CompoundEdit``, because it is one gesture: the cut half (when
        asked for) and the new layer undo together or the user sees a state
        that never existed -- a hole with nothing above it.

        The pixels carry the mask in their alpha, through the same
        ``_masked_alpha`` the clipboard and a lift both use, so a feathered
        selection makes a feathered layer rather than a hard-edged crop of one.
        They are placed back at the selection's own offset rather than at the
        top-left, which is what makes this different from a paste: the new
        layer lines up with what it came from.

        The copy half is legal on a content-locked layer -- it reads. The cut
        half is not, and refuses the whole operation rather than half of it.
        """
        if self.mask is None:
            return False
        bounds = self.mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return False
        if cut and self.write_locked():
            return False
        self.commit_floating()
        x0, y0, x1, y1 = box
        crop = self.mask.mask[y0:y1, x0:x1]
        taken = _masked_alpha(self.stack.active.pixels[y0:y1, x0:x1], crop)
        # Read before the insert, and before the cut moves nothing: the new
        # layer joins whatever folder the one it came from is in, so promoting
        # part of a drawing keeps it beside the drawing.
        parent = self._parent_of_active()

        edits: list[Any] = []
        if cut:
            # Before the add, and it has to be: the cut is against the *active*
            # layer, and adding first would make the new one active and cut a
            # hole in the copy that was just made.
            edits.extend(self._cut_selection_patch() or [])
        edits.append(
            self._add_layer_edit(taken, f"Layer {len(self.stack) + 1}", offset=(x0, y0))
        )
        edits.extend(self.inherit_group_edits(parent))
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        return True

    def put_clipboard(self: Document, pixels: np.ndarray) -> None:
        """Load the app clipboard from outside -- an OS clipboard image.

        Everything pasted carries a mask, because a paste has to know which of
        its pixels are part of the selection; an image from elsewhere is fully
        selected by definition.
        """
        mask = np.full(pixels.shape[:2], 255, dtype=np.uint8)
        self.clipboard.put(pixels, mask)

    def float_pixels(self: Document, pixels: np.ndarray, at: tuple[int, int]) -> bool:
        """Float an array made somewhere else, at ``at``. The paste rule, exactly.

        The delivery route for everything that *generates* pixels and then
        wants them **placed**: the text stamp (C14) is the one that does. Such a
        generator wants the same four things -- put it down where I clicked, let
        me move it, let me commit it, let me take it back -- and a floating
        buffer is that, already written and already tested; writing into the
        layer directly would mean reinventing placement, the drag, the transform
        box and the undo step, with a different answer for each.

        **Image brushes deliberately do not come through here**, though they
        were expected to. A captured tip is not one picture being positioned: it
        is a *dab*, emitted repeatedly along a stroke, and it needs the spacing
        walk, symmetry, the spray's emission, tiled wrapping and per-dab
        coverage accumulation -- none of which a floating buffer has and all of
        which ``StrokeState`` already does. See :meth:`.brush.StrokeState._place`.

        It is ``paste`` in every respect that can be observed:

        * **Refused on a content-locked layer.** The buffer is bound to the
          active layer and would land on it, which is the write the lock exists
          to prevent. ``commit_floating`` is deliberately *not* refused -- see
          its docstring -- so the door has to be here, at the moment the buffer
          is made.
        * **Whatever is already floating commits first**, rather than being
          dropped: a second stamp while the first is still up means "put that
          one down", not "throw it away".
        * **``forget_redo``.** A float pushes no step of its own, so without it
          the redo branch survives and Ctrl+Y after a stamp replays an
          unrelated edit.
        * **Canonical-only on a tiled document**, because what it eventually
          writes goes through ``commit_floating``.

        The mask is solid rather than the pixels' own alpha, which is what
        ``put_clipboard`` does and for the same reason: the mask is what
        hit-tests a grab (``FloatingBuffer.contains``), and a mask cut to the
        glyphs would mean the user has to click *on a letter's stem* to move
        the word they just placed.
        """
        if self.write_locked():
            return False
        array = np.array(pixels, dtype=np.uint8, copy=True)
        if array.ndim != 3 or array.shape[2] != 4 or array.shape[0] < 1 or array.shape[1] < 1:
            return False
        self.commit_floating()
        self.floating = FloatingBuffer(
            pixels=array,
            mask=np.full(array.shape[:2], 255, dtype=np.uint8),
            offset=(int(at[0]), int(at[1])),
            layer_uid=self.stack.active.uid,
        )
        self.history.forget_redo()
        self.rev += 1
        return True
