"""Everything that puts colour onto a layer: strokes, fills, shapes, gradients,
filters -- and the eyedropper that reads it back.

``SHAPES`` and ``normalise_rect`` live here rather than in ``document.py``
because ``shape()`` is their only runtime user, and leaving them behind would
make this module import ``document`` at runtime -- the one edge the split is
built to avoid. ``document.py`` re-exports both, so ``document.SHAPES`` and
``inker.normalise_rect`` are unchanged.

Every method below ends at ``_commit_patch``, which stays in ``document.py``:
it is the single place an indexed document is snapped and the single place an
undo step is pushed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import brush as brush_mod
from . import composite as cp
from . import dither, filters
from . import gradient as grad
from .brush import DEFAULT_SPACING, StrokeState, clamp_brush
from .layers import Layer
from .selection import magic_wand

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import RGBA, Document

SHAPES = ("line", "rect", "ellipse")


def normalise_rect(p0: tuple[int, int], p1: tuple[int, int]) -> tuple[int, int, int, int]:
    """Two corners in any order -> (x0, y0, x1, y1) with x0 <= x1."""
    x0, y0 = p0
    x1, y1 = p1
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class PaintOps:
    """Writes into a layer's pixels, mixed into :class:`~.document.Document`."""

    def eyedrop(
        self: Document, xy: tuple[int, int], *, layer_only: bool = False
    ) -> RGBA | None:
        """The colour at a point: the composite by default, one layer on ask.

        The composite is the default because the colour a user is pointing at
        is the one they can see, and that is what they mean nine times in ten.
        ``layer_only`` is the tenth: picking up a line colour off a lineart
        layer that has a colour layer under it reads the *blend* otherwise,
        which is a colour that exists nowhere in the document.

        It reads the layer's own stored pixels, so it is unaffected by the
        layer's opacity, its blend mode and whether it is even visible -- all
        three are properties of how the layer is shown rather than of what was
        painted into it.
        """
        if not self.in_bounds(xy):
            return None
        source = self.stack.active.pixels if layer_only else self._composite
        r, g, b, a = source[int(xy[1]), int(xy[0])]
        return (int(r), int(g), int(b), int(a))

    # -- writing to a layer -------------------------------------------------

    def _weights(
        self: Document, rect: tuple[int, int, int, int], weight: np.ndarray
    ) -> np.ndarray:
        """Clip a write to the selection. The same multiply as a brush stamp --
        one rule, so a feathered selection softens every tool identically."""
        if self.mask is None:
            return weight
        x0, y0, x1, y1 = rect
        return weight * (self.mask.mask[y0:y1, x0:x1].astype(np.float32) / 255.0)

    def write_colour(
        self: Document, rect: tuple[int, int, int, int], colour: RGBA, weight: np.ndarray
    ) -> bool:
        """Composite a flat colour into a region of the active layer."""
        box = self.clip(rect)
        if box is None:
            return False
        self._ensure_active_cel()
        layer = self.stack.active
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        out = cp.paint_colour(
            before.astype(np.float32), colour, self._weights(box, weight)
        )
        if layer.alpha_lock:
            # The lock, in one line and after the formula rather than inside
            # it: "preserve transparency" is exactly *the alpha does not
            # change*, so restoring the channel is the definition rather than
            # an approximation of it. Colour written where alpha is zero is
            # invisible, which is what makes this enough on its own.
            out[..., 3] = before[..., 3]
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(out)
        self._commit_patch(layer, box, before)
        return True

    # -- filters ------------------------------------------------------------

    def begin_filter(self: Document) -> tuple[int, int, int, int] | None:
        """Open a filter session over the selection, or the whole layer.

        Returns the rect it will write, or None when there is nothing to
        filter. Nothing is pushed and nothing is changed: this only takes the
        copy that :meth:`preview_filter` recomputes from and
        :meth:`cancel_filter` restores.
        """
        self.end_filter()
        width, height = self.size
        bounds = self.mask.bounds if self.mask is not None else None
        box = self.clip(bounds or (0, 0, width, height))
        if box is None:
            return None
        self._ensure_active_cel()
        x0, y0, x1, y1 = box
        layer = self.stack.active
        self._filter = (box, layer.pixels[y0:y1, x0:x1].copy(), layer.uid)
        self._filter_memo = None
        return box

    def _filter_layer(self: Document) -> Layer | None:
        """The layer an open filter session belongs to, or None if it is gone.

        ``layer_by_uid`` and not ``stack.by_uid``, for exactly the reason
        ``PatchEdit._put`` gives: on an animated document the cel being filtered
        may be on a frame the playhead has since moved off, and it is still the
        cel these pixels came from and the one they must go back to. The frame
        therefore needs no recording of its own -- a cel's uid names it on every
        frame at once, and that is the whole point of addressing by uid.

        Two answers are None rather than a layer, and both mean "cancel this
        cleanly instead of writing somewhere". A uid nothing carries any more is
        a layer (or a whole track) deleted while the popup was up. A uid that
        now resolves to a *placeholder* is an autovivified cel undone while the
        popup was up: placeholder uids are stable per slot, so the empty slot
        takes the same number back -- and its plane is the shared read-only one,
        which raises on the first write.
        """
        if self._filter is None:
            return None
        try:
            layer = self.layer_by_uid(self._filter[2])
        except KeyError:
            return None
        if self.anim is not None and self.anim.is_placeholder(layer):
            return None
        return layer

    def _abandon_filter(self: Document) -> bool:
        """Drop a session whose layer went, restoring nothing and pushing
        nothing. There is no target left to put the pixels back into, and the
        cel this session brought into existence has to go with it."""
        self._filter = None
        self._filter_memo = None
        self._discard_pending_cel()
        return False

    def preview_filter(self: Document, name: str, **params: Any) -> bool:
        """Show the filter without recording it. Cheap to call every frame.

        The selection is honoured as a *weight*, not as a rectangle: a
        feathered edge fades the filter in, which is the same rule every other
        write in this class follows and the reason feathering means one thing
        across the whole app.

        **The filter itself is memoised for the life of the session; the write
        below it is not.** ``inker_bridge`` calls this on every frame the popup
        is up, deliberately, because the combo can switch filters -- but
        ``before`` is the snapshot :meth:`begin_filter` took and never changes,
        so ``apply_named`` is a pure function of ``(name, params)`` within one
        session and recomputing it sixty times a second is the whole of what
        made a 2048 square blur unusable (measured at 1.1 s per call, i.e. per
        frame). Only the expensive half is cached: the blend, the alpha lock
        and the invalidate below still run every frame, so switching filters
        still repaints exactly as before.
        """
        if self._filter is None:
            return False
        layer = self._filter_layer()
        if layer is None:
            return self._abandon_filter()
        box, before, _uid = self._filter
        x0, y0, x1, y1 = box
        key = (name, tuple(sorted(params.items())))
        if self._filter_memo is not None and self._filter_memo[0] == key:
            filtered = self._filter_memo[1]
        else:
            filtered = filters.apply_named(name, before, **params)
            self._filter_memo = (key, filtered)
        if self.mask is None:
            layer.pixels[y0:y1, x0:x1] = filtered
        else:
            weight = self.mask.mask[y0:y1, x0:x1].astype(np.float32)[..., None] / 255.0
            blended = before.astype(np.float32) + (
                filtered.astype(np.float32) - before.astype(np.float32)
            ) * weight
            layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(blended)
        if layer.alpha_lock:
            layer.pixels[y0:y1, x0:x1, 3] = before[..., 3]
        self.invalidate(box, layer_uid=layer.uid)
        return True

    def commit_filter(self: Document) -> bool:
        """Turn whatever is previewed into exactly one undo step."""
        if self._filter is None:
            return False
        layer = self._filter_layer()
        if layer is None:
            return self._abandon_filter()
        box, before, _uid = self._filter
        self._filter = None
        self._filter_memo = None
        # ``_commit_patch`` compares before against after and pushes nothing
        # when they match, which is what makes opening a filter, moving nothing
        # and pressing Apply a no-op rather than a step that dirties the file.
        self._commit_patch(layer, box, before)
        return True

    def cancel_filter(self: Document) -> bool:
        """Put the pixels back. Nothing was ever pushed, so nothing is undone."""
        if self._filter is None:
            return False
        layer = self._filter_layer()
        if layer is None:
            return self._abandon_filter()
        box, before, _uid = self._filter
        self._filter = None
        self._filter_memo = None
        x0, y0, x1, y1 = box
        layer.pixels[y0:y1, x0:x1] = before
        self._discard_pending_cel()
        self.invalidate(box, layer_uid=layer.uid)
        return True

    def end_filter(self: Document) -> None:
        """Abandon a session left open by a pane that went away.

        Cancels rather than commits: an unanswered question is not a yes, and
        the pixels on screen are a preview the user never approved.
        """
        self.cancel_filter()

    # -- palette conversion -------------------------------------------------
    #
    # The filter session's mechanism, generalised one dimension: a filter writes
    # one rectangle of one layer, and a conversion rewrites every plane of the
    # document. Everything else is deliberately the same -- the snapshot is
    # taken once and every preview recomputes from it (never from the last
    # preview, which would compound), each plane is addressed by uid, and
    # committing is the ordinary one-undo op rather than anything the session
    # knows how to push.
    #
    # **The preview covers the current frame only, and the commit covers the
    # document.** Previewing forty frames would mean converting forty planes per
    # parameter change to show one, and the frames the user cannot see are the
    # ones they are not judging. The popup says so.

    def begin_convert(self: Document) -> bool:
        """Open a conversion preview over the current frame's real layers.

        Nothing is pushed and nothing is changed: this only takes the copies
        that :meth:`preview_convert` recomputes from and :meth:`cancel_convert`
        restores.

        **Empty cels are skipped rather than autovivified**, which is the one
        place this deliberately parts company with ``begin_filter``. A filter is
        a write to the layer you have selected, so bringing that cel into
        existence is what the user asked for. A conversion is a document-wide
        mode change that nobody would expect to *populate* the timeline -- and
        the cels it would create are exactly the empty ones, where a conversion
        has nothing to do anyway.
        """
        self.end_convert()
        anim = self.anim
        planes = [
            layer
            for layer in self.stack
            if anim is None or not anim.is_placeholder(layer)
        ]
        if not planes:
            return False
        self._convert = [(layer.uid, layer.pixels.copy()) for layer in planes]
        self._convert_memo = None
        return True

    def _convert_target(self: Document, uid: int, before: np.ndarray) -> Layer | None:
        """The layer one recorded plane belongs to, or None to skip it.

        Three answers are None, and all three mean "this plane is not ours to
        write any more" rather than "cancel the session": the uid names nothing
        (a layer or a whole track deleted under the popup), it names a
        *placeholder* (a cel undone under the popup -- placeholder uids are
        stable per slot, and the plane behind one is the shared read-only
        blank), or the plane has been resized under the popup and the pixels no
        longer describe it.

        Skipping rather than abandoning is the difference from ``_filter``,
        which has exactly one layer and nothing left to do when it goes. Here
        the other planes are still previewing and still have to be put back.
        """
        try:
            layer = self.layer_by_uid(uid)
        except KeyError:
            return None
        if self.anim is not None and self.anim.is_placeholder(layer):
            return None
        if layer.pixels.shape != before.shape:
            return None
        return layer

    def preview_convert(
        self: Document, colours: Any, method: str = "nearest"
    ) -> bool:
        """Show the conversion without recording it. Safe to call every frame.

        Memoised per ``(table, method)`` for ``preview_filter``'s reason and
        more so: ``inker_bridge`` calls this on every frame the popup is up
        because the controls above it can change either, and Floyd-Steinberg is
        a Python loop over every pixel of every layer. The snapshot never
        changes, so the converted planes are a pure function of the two.
        """
        if self._convert is None:
            return False
        wanted = [tuple(c) for c in colours]
        if not wanted or method not in dither.METHODS:
            return False
        key = (tuple(wanted), method)
        if self._convert_memo is None or self._convert_memo[0] != key:
            self._convert_memo = (
                key,
                [
                    dither.convert(before, wanted, method)
                    for _uid, before in self._convert
                ],
            )
        for (uid, before), converted in zip(
            self._convert, self._convert_memo[1], strict=True
        ):
            layer = self._convert_target(uid, before)
            if layer is not None:
                layer.pixels[:, :] = converted
        self.invalidate_all()
        return True

    def commit_convert(self: Document, colours: Any, method: str = "nearest") -> bool:
        """Put the preview back, then convert for real as one undo step.

        The restore is not wasted work and not belt-and-braces: the preview has
        already written the converted pixels onto the current frame, so without
        it ``convert_to_palette``'s snapshot would record *those* as the state
        to undo to -- and one Ctrl+Z would land on a document that was never
        the user's.
        """
        if self._convert is None:
            return False
        self._restore_convert()
        return self.convert_to_palette(colours, method)

    def cancel_convert(self: Document) -> bool:
        """Put the pixels back. Nothing was ever pushed, so nothing is undone."""
        if self._convert is None:
            return False
        self._restore_convert()
        return True

    def _restore_convert(self: Document) -> None:
        """Undo the preview and close the session, writing no history."""
        recorded, self._convert = self._convert, None
        self._convert_memo = None
        for uid, before in recorded or ():
            layer = self._convert_target(uid, before)
            if layer is not None:
                layer.pixels[:, :] = before
        self.invalidate_all()

    def end_convert(self: Document) -> None:
        """Abandon a session left open by a pane that went away. Cancels rather
        than commits, for :meth:`end_filter`'s reason."""
        self.cancel_convert()

    # -- strokes ------------------------------------------------------------

    def begin_stroke(
        self: Document,
        point: tuple[float, float],
        colour: RGBA,
        *,
        size: int = 8,
        hardness: float = 0.8,
        opacity: float = 1.0,
        spacing: float | None = None,
        mode: str = "paint",
        strength: float = 0.5,
        nib: str = "soft",
        pixel_perfect: bool = False,
        symmetry: str = "none",
        axis: tuple[float, float] | None = None,
        radial: int = brush_mod.DEFAULT_RADIAL,
        stabilise: float = 0.0,
        speed_taper: float = 0.0,
    ) -> None:
        self.end_stroke()
        self._ensure_active_cel()
        layer = self.stack.active
        self._stroke = StrokeState(
            layer_uid=layer.uid,
            size=self.size,
            before=layer.pixels.copy(),
            colour=tuple(colour),
            diameter=clamp_brush(size),
            hardness=hardness,
            opacity=opacity,
            spacing=DEFAULT_SPACING if spacing is None else spacing,
            mode=mode,
            strength=strength,
            nib=nib,
            pixel_perfect=pixel_perfect,
            symmetry=symmetry,
            axis=axis,
            radial=radial,
            stabilise=stabilise,
            speed_taper=speed_taper,
            clip=self.mask,
            alpha_lock=layer.alpha_lock,
        )
        self._stroke.begin(point, layer.pixels)
        self._touch_stroke()

    def stroke_to(self: Document, point: tuple[float, float]) -> None:
        if self._stroke is None:
            return
        self._stroke.to(point, self.stack.by_uid(self._stroke.layer_uid).pixels)
        self._touch_stroke()

    def _touch_stroke(self: Document) -> None:
        """Recomposite what the last dab touched, without pushing history: the
        undo step is the whole stroke, and it is pushed once at release."""
        assert self._stroke is not None
        if self._stroke.dirty is not None:
            self.invalidate(self._stroke.dirty)

    def end_stroke(self: Document) -> bool:
        """Close the stroke and make it exactly one undo step."""
        stroke, self._stroke = self._stroke, None
        if stroke is not None and stroke.pending:
            # The pixel-perfect filter holds one pixel back to see whether the
            # next makes it an elbow, and at release there is no next: without
            # this flush a click marks nothing and every stroke is one pixel
            # short. Before the ``dirty is None`` test, since for a click the
            # flush is the only thing that makes it non-None.
            stroke.finish(self.stack.by_uid(stroke.layer_uid).pixels)
        if stroke is None or stroke.dirty is None:
            # A brush-down with no dab. ``begin_stroke`` may have autovivified a
            # cel for it, and nothing below will reach ``_commit_patch`` to
            # decide the cel's fate, so it is decided here.
            self._discard_pending_cel()
            return False
        box = self.clip(stroke.dirty)
        if box is None:
            self._discard_pending_cel()
            return False
        layer = self.stack.by_uid(stroke.layer_uid)
        x0, y0, x1, y1 = box
        self._commit_patch(layer, box, stroke.before[y0:y1, x0:x1])
        return True

    # -- fill, shapes, gradients -------------------------------------------

    def fill(
        self: Document,
        xy: tuple[int, int],
        colour: RGBA,
        *,
        thresh: int = 32,
        contiguous: bool = True,
    ) -> bool:
        """Flood fill from a point, on the composite's colours.

        The threshold is what makes this usable on generated art: an SDXL image
        has no flat regions, only nearly-flat ones behind an antialiased edge,
        and an exact-match fill stops after a few hundred pixels every time.
        The region comes from the same predicate the magic wand uses, so the
        two can never disagree about what "similar" means.
        """
        if not self.in_bounds(xy):
            return False
        region = magic_wand(
            self._composite, xy, tolerance=thresh, contiguous=contiguous
        )
        bounds = region.bounds
        if bounds is None:
            return False
        x0, y0, x1, y1 = bounds
        weight = region.mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        return self.write_colour(bounds, colour, weight)

    def shape(
        self: Document,
        kind: str,
        p0: tuple[int, int],
        p1: tuple[int, int],
        colour: RGBA,
        size: int,
        *,
        filled: bool = False,
    ) -> bool:
        """Line, rectangle or ellipse, antialiased through a coverage mask."""
        if kind not in SHAPES:
            raise ValueError(f"unknown shape {kind!r}")
        from PIL import Image, ImageDraw

        width, height = self.size
        size = clamp_brush(size)
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        if kind == "line":
            draw.line([tuple(p0), tuple(p1)], fill=255, width=size, joint="curve")
        else:
            x0, y0, x1, y1 = normalise_rect(p0, p1)
            method = draw.rectangle if kind == "rect" else draw.ellipse
            if filled:
                method((x0, y0, x1, y1), fill=255, outline=255, width=size)
            else:
                method((x0, y0, x1, y1), fill=None, outline=255, width=size)

        coverage = np.asarray(canvas, dtype=np.uint8)
        rows = np.flatnonzero(coverage.any(axis=1))
        cols = np.flatnonzero(coverage.any(axis=0))
        if rows.size == 0:
            return False
        rect = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
        weight = coverage[rect[1] : rect[3], rect[0] : rect[2]].astype(np.float32) / 255.0
        return self.write_colour(rect, colour, weight)

    def gradient(
        self: Document,
        p0: tuple[float, float],
        p1: tuple[float, float],
        start: RGBA | None = None,
        end: RGBA | None = None,
        *,
        kind: str = "linear",
        stops: Any = None,
        dither: str | None = None,
    ) -> bool:
        """Fill the selection (or the whole layer) with a ramp.

        ``stops`` is the general form; ``start``/``end`` is the two-stop
        shorthand every existing caller uses. Both go through one interpolator.

        ``dither`` thresholds the ramp onto its own stops instead of blending
        between them -- see :func:`gradient.dithered`. It is applied inside the
        render, never afterwards: dithering the *result* would be quantising a
        blend that has already been made, which is a different picture and one
        that no longer lands on the stop colours exactly.

        The **selection weight is never dithered**, which is why it comes back
        from the render separately and is multiplied in below. A feathered
        selection means one thing across every tool in this class, and a soft
        edge chopped into a chequer is not that thing.
        """
        width, height = self.size
        rect = self.mask.bounds if self.mask is not None else None
        box = self.clip(rect or (0, 0, width, height))
        if box is None:
            return False
        rgba, weight = grad.render(
            (width, height), p0, p1, start, end, kind, stops=stops, dither=dither
        )
        x0, y0, x1, y1 = box
        self._ensure_active_cel()
        layer = self.stack.active
        before = layer.pixels[y0:y1, x0:x1].copy()
        crop = rgba[y0:y1, x0:x1]
        clipped = self._weights(box, weight[y0:y1, x0:x1])

        # Per-pixel colour, so ``paint_colour``'s flat-colour form does not fit;
        # this is the same arithmetic with the colour varying.
        src_a = clipped
        dst_a = before[..., 3].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        share = np.divide(src_a, out_a, out=np.zeros_like(src_a), where=out_a > 0.0)
        out = np.empty(before.shape, dtype=np.float32)
        rgb = before[..., :3].astype(np.float32)
        out[..., :3] = rgb + (crop[..., :3] * 255.0 - rgb) * share[..., None]
        out[..., 3] = out_a * 255.0
        if layer.alpha_lock:
            # The one write in this class that was not honouring the lock.
            # ``write_colour`` restores the channel after its formula and says
            # why; a ramp is the same composite with the colour varying, so it
            # is the same one line. Without it, "preserve transparency" held for
            # every tool except the gradient, which filled the transparent part
            # of the layer in.
            out[..., 3] = before[..., 3]
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(out)
        self._commit_patch(layer, box, before)
        return True
