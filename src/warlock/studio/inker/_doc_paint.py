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
from . import filters, tiling
from . import gradient as grad
from ._doc_ranges import masked_apply
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


def _translated(pixels: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """A canvas-sized plane shifted by (dx, dy), cropped, zero-filled behind.

    Not ``np.roll``: a layer is canvas-sized and its edge *is* the canvas edge,
    so what leaves one side is gone rather than arriving on the other. Rolling
    would put a sprite's head on the far side of the frame the moment somebody
    nudged it past the border, which reads as corruption.
    """
    out = np.zeros_like(pixels)
    height, width = pixels.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = pixels[sy0:sy1, sx0:sx1]
    return out


def _content_box(pixels: np.ndarray) -> tuple[int, int, int, int] | None:
    """The half-open bbox of every pixel that is not all zeros, or None.

    Any channel, not alpha alone -- and that is the difference between an undo
    that restores the layer and one that nearly does. The eraser cuts alpha and
    leaves RGB where it was, so an erased region is ``(r, g, b, 0)``: invisible,
    outside any alpha-only box, and *zeroed* by the translate below. A patch
    measured on alpha would not record that, and the colours would be gone with
    no step to bring them back -- silently, because nothing on screen changed.
    """
    rows = np.flatnonzero(pixels.any(axis=(1, 2)))
    cols = np.flatnonzero(pixels.any(axis=(0, 2)))
    if rows.size == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _union(
    a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    if a is None or b is None:
        return a or b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


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
        """Composite a flat colour into a region of the active layer.

        The door the fill and the three shapes come through as well, which is
        why the content lock is checked here rather than three times over.
        """
        box = self.clip(rect)
        if box is None:
            return False
        if self.write_locked():
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
        filter -- which includes a content-locked layer, refused here at the
        top so no preview is ever shown for a filter that cannot be applied.
        Nothing is pushed and nothing is changed: this only takes the
        copy that :meth:`preview_filter` recomputes from and
        :meth:`cancel_filter` restores.
        """
        self.end_filter()
        if self.write_locked():
            return None
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
        # ``masked_apply`` rather than the blend spelled out here: the range
        # filter (``_doc_ranges.filter_range``) has to fade a filter into a
        # selection by exactly the same rule, and two copies of it is how
        # "feathering" comes to mean two things in one editor.
        layer.pixels[y0:y1, x0:x1] = masked_apply(
            before,
            filtered,
            None if self.mask is None else self.mask.mask[y0:y1, x0:x1],
            alpha_lock=layer.alpha_lock,
        )
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

    # -- moving a layer's pixels --------------------------------------------
    #
    # The move tool's third arm: with no floating buffer and no selection, a
    # drag translates what is *on* the layer. Translate-with-crop rather than
    # ``np.roll``: a layer is canvas-sized by invariant and its edges are the
    # canvas's edges, so pixels pushed off one side are gone rather than
    # arriving on the other. (A tiled-wrap variant of exactly this would be one
    # ``tiling.pieces`` call, and it is deliberately unwired -- moving a layer
    # is not a paint stroke, and "the drawing wrapped round while I nudged it"
    # is not something a user asks for by pressing an arrow key.) A cel-offset
    # model, where the layer keeps its pixels and grows an origin, was
    # considered and rejected: it is a second coordinate space through every op
    # in the package, which is the trade ``layers.py`` already refused.
    #
    # The session mirrors the filter session exactly -- begin snapshots, preview
    # re-renders *from the snapshot* so a drag cannot compound, commit pushes
    # one patch, cancel puts the pixels back and pushes nothing -- with one
    # deliberate difference at the abandon door: see :meth:`end_layer_move`.

    def begin_layer_move(self: Document) -> bool:
        """Open a move session on the active layer. -> whether it opened.

        Refused on a content-locked layer, which is the same door every other
        write goes through. Read with ``getattr`` because the flag is a layer
        property landing beside this work rather than under it, and a move that
        silently ignored a lock is worse than one that reads a default of False
        until the flag exists.
        """
        self.end_layer_move()
        # The lock is read *before* the float is committed: a refusal must
        # leave the document exactly as it found it, and committing first would
        # land a floating buffer as the side effect of a move that never
        # happened.
        if getattr(self.stack.active, "content_lock", False):
            return False
        self.commit_floating()
        self._ensure_active_cel()
        layer = self.stack.active
        self._move = (layer.pixels.copy(), layer.uid)
        return True

    def _move_layer(self: Document) -> Layer | None:
        """The layer an open move session belongs to, or None if it is gone.

        ``_filter_layer``'s rule and for its reasons: a session lives across
        frames, a cel is addressed by uid on every frame at once, and a uid that
        now resolves to a placeholder is an autovivified cel undone underneath
        us -- whose plane is the shared read-only one.
        """
        if self._move is None:
            return None
        try:
            layer = self.layer_by_uid(self._move[1])
        except KeyError:
            return None
        if self.anim is not None and self.anim.is_placeholder(layer):
            return None
        return layer

    def _abandon_move(self: Document) -> bool:
        self._move = None
        self._discard_pending_cel()
        return False

    def preview_layer_move(self: Document, dx: int, dy: int) -> bool:
        """Show the layer at a **total** offset from where the session opened.

        Total, not incremental: re-rendering from the snapshot every frame is
        what stops a slow drag and a fast one producing different results, and
        it is the same anti-compounding rule ``preview_filter`` follows.
        """
        if self._move is None:
            return False
        layer = self._move_layer()
        if layer is None:
            return self._abandon_move()
        source, _uid = self._move
        # In place, so a cel linked across several frames stays *one object*
        # and the move shows on every frame it appears in. Rebinding
        # ``layer.pixels`` would silently break every link in the row.
        layer.pixels[:] = _translated(source, int(dx), int(dy))
        width, height = self.size
        self.invalidate((0, 0, width, height), layer_uid=layer.uid)
        return True

    def commit_layer_move(self: Document) -> bool:
        """Turn whatever is previewed into exactly one undo step.

        The patch covers the union of where the content *was* and where it
        *is*, rather than the whole canvas: a 16-pixel nudge of a sprite on a
        2048 square would otherwise cost 32 MiB of history for a change that
        touches a few thousand pixels. See :func:`_content_box` for why "the
        content" is not "the non-transparent pixels".
        """
        if self._move is None:
            return False
        layer = self._move_layer()
        if layer is None:
            return self._abandon_move()
        source, _uid = self._move
        self._move = None
        box = _union(_content_box(source), _content_box(layer.pixels))
        if box is None:
            # Both empty: an empty layer was moved, which changed nothing.
            self._discard_pending_cel()
            return False
        x0, y0, x1, y1 = box
        self._commit_patch(layer, box, source[y0:y1, x0:x1])
        return True

    def cancel_layer_move(self: Document) -> bool:
        """Put the pixels back. Nothing was pushed, so nothing is undone.

        The user-initiated half of the pair, and the only one that restores:
        reachable from Escape and from the canvas, both of which act on a
        session that is still live. Every *automatic* path -- anything that
        finds a session somebody else left open -- goes through
        :meth:`end_layer_move` instead, because restoring from a snapshot of
        unknown age is how an unrelated edit gets wiped.
        """
        if self._move is None:
            return False
        layer = self._move_layer()
        if layer is None:
            return self._abandon_move()
        source, _uid = self._move
        self._move = None
        layer.pixels[:] = source
        self._discard_pending_cel()
        width, height = self.size
        self.invalidate((0, 0, width, height), layer_uid=layer.uid)
        return True

    def end_layer_move(self: Document) -> bool:
        """Close a session left open by a gesture that went away.

        ``end_filter``'s role, with the opposite answer, and the difference is
        the point. A filter session is a *question* -- a popup with an Apply
        button the user never pressed -- so abandoning it cancels: an unanswered
        question is not a yes. A move session is a *drag*: the pixels on screen
        are where the user put them, and there is no unpressed button.

        So this commits, and committing is also the only safe answer. Cancelling
        restores ``layer.pixels`` wholesale from the snapshot taken at
        ``begin_layer_move``, which is correct while the session is live and
        catastrophic once it is stale -- anything painted in between is
        overwritten by a snapshot older than it, while those strokes' own
        ``PatchEdit``s stay in history describing pixels that are no longer
        there. Committing can never lose a pixel: it records what actually
        happened as one step and leaves the layer exactly as it looks.
        """
        return self.commit_layer_move()

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
        wrap: str | tuple[bool, bool] = "off",
        scatter: float = 0.0,
        seed: int = 0,
    ) -> bool:
        """Open a stroke on the active layer. False when the layer is locked.

        ``wrap`` is one of :data:`.tiling.TILED_AXES` and is passed in by the
        canvas from the tab's own view state -- tiling is never read off the
        document, because it is a property of how a tab is being looked at and
        must not travel in a file. ``scatter``/``seed`` are the spray tool's
        emission; both default to the values that make this the stroke the
        editor has always opened.

        The refusal comes *before* ``_ensure_active_cel``, so a brush-down on a
        locked track of an animated document does not autovivify a cel that
        nothing may then write to.
        """
        self.end_stroke()
        if self.write_locked():
            return False
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
            wrap_axes=tiling.axes_of(wrap),
            scatter=scatter,
            seed=seed,
        )
        self._stroke.begin(point, layer.pixels)
        self._touch_stroke()
        return True

    def stroke_to(self: Document, point: tuple[float, float]) -> None:
        if self._stroke is None:
            return
        self._stroke.to(point, self.stack.by_uid(self._stroke.layer_uid).pixels)
        self._touch_stroke()

    def spray_at(self: Document, point: tuple[float, float], count: int) -> None:
        """Scatter ``count`` dabs about a point, inside the open stroke.

        The spray tool's ``stroke_to``: it does *not* walk the spacing, because
        an airbrush is not a line -- the canvas calls it every frame the button
        is held, so a stationary cursor keeps emitting, which is the whole
        behaviour. One press-to-release is still one ``PatchEdit``, since the
        dabs go through the same stroke buffer everything else does.
        """
        if self._stroke is None:
            return
        self._stroke.spray(
            point, count, self.stack.by_uid(self._stroke.layer_uid).pixels
        )
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
        wrap: str | tuple[bool, bool] = "off",
    ) -> bool:
        """Flood fill from a point, on the composite's colours.

        The threshold is what makes this usable on generated art: an SDXL image
        has no flat regions, only nearly-flat ones behind an antialiased edge,
        and an exact-match fill stops after a few hundred pixels every time.
        The region comes from the same predicate the magic wand uses, so the
        two can never disagree about what "similar" means -- ``wrap`` included,
        which is why it is threaded straight through rather than reimplemented.
        """
        if not self.in_bounds(xy):
            return False
        region = magic_wand(
            self._composite, xy, tolerance=thresh, contiguous=contiguous, wrap=wrap
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
        wrap: str | tuple[bool, bool] = "off",
    ) -> bool:
        """Line, rectangle or ellipse, antialiased through a coverage mask.

        With ``wrap`` on, the coverage is rasterised over a plane big enough to
        hold the whole gesture *and* the brush width that spills past the canvas
        edge, and then folded home by :func:`.tiling.fold_coverage`. Rasterising
        into the canvas and wrapping afterwards is the only order that works:
        Pillow clips to its image, so anything drawn past the edge would be gone
        before there was anything to fold.

        With it off the plane *is* the canvas at the origin, so the fold is the
        identity and the bounding box is the one this method always computed.
        """
        if kind not in SHAPES:
            raise ValueError(f"unknown shape {kind!r}")
        from PIL import Image, ImageDraw

        size = clamp_brush(size)
        axes = tiling.axes_of(wrap)
        (ox, oy), (pw, ph) = self._shape_plane(p0, p1, size, axes)
        canvas = Image.new("L", (pw, ph), 0)
        draw = ImageDraw.Draw(canvas)
        q0 = (p0[0] - ox, p0[1] - oy)
        q1 = (p1[0] - ox, p1[1] - oy)
        if kind == "line":
            draw.line([q0, q1], fill=255, width=size, joint="curve")
        else:
            x0, y0, x1, y1 = normalise_rect(q0, q1)
            method = draw.rectangle if kind == "rect" else draw.ellipse
            if filled:
                method((x0, y0, x1, y1), fill=255, outline=255, width=size)
            else:
                method((x0, y0, x1, y1), fill=None, outline=255, width=size)

        folded = tiling.fold_coverage(
            np.asarray(canvas, dtype=np.uint8), (ox, oy), self.size, axes
        )
        if folded is None:
            return False
        rect, coverage = folded
        return self.write_colour(rect, colour, coverage.astype(np.float32) / 255.0)

    def _shape_plane(
        self: Document,
        p0: tuple[int, int],
        p1: tuple[int, int],
        size: int,
        axes: tuple[bool, bool],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """The rasterisation plane a shape needs: ``(origin, (w, h))``.

        The canvas exactly, on an unwrapped axis -- so nothing about the
        untiled path changes. On a wrapped one it grows to hold the gesture and
        the brush width around it, clamped to the 3x3 neighbourhood the tiled
        view actually shows: a drag cannot land further out than one tile away,
        and a clamp is cheaper than trusting a number that arrives from a mouse.
        """
        width, height = self.size
        pad = size + 1
        origin, extent = [], []
        for wrapped, lo_p, hi_p, span in (
            (axes[0], min(p0[0], p1[0]), max(p0[0], p1[0]), width),
            (axes[1], min(p0[1], p1[1]), max(p0[1], p1[1]), height),
        ):
            if not wrapped:
                origin.append(0)
                extent.append(span)
                continue
            lo = max(-span, min(0, int(lo_p) - pad))
            hi = min(2 * span, max(span, int(hi_p) + pad + 1))
            origin.append(lo)
            extent.append(hi - lo)
        return (origin[0], origin[1]), (extent[0], extent[1])

    def gradient(
        self: Document,
        p0: tuple[float, float],
        p1: tuple[float, float],
        start: RGBA | None = None,
        end: RGBA | None = None,
        *,
        kind: str = "linear",
        stops: Any = None,
    ) -> bool:
        """Fill the selection (or the whole layer) with a ramp.

        ``stops`` is the general form; ``start``/``end`` is the two-stop
        shorthand every existing caller uses. Both go through one interpolator.

        **A gradient deliberately does not wrap**, and takes no ``wrap``
        argument to say so. A ramp has two endpoints: wrapping one puts the last
        stop against the first, which is a hard edge at the seam -- the one
        thing tiled mode exists to remove. A seamless ramp is a different
        object (a periodic one), not this one with a flag.
        """
        if self.write_locked():
            return False
        width, height = self.size
        rect = self.mask.bounds if self.mask is not None else None
        box = self.clip(rect or (0, 0, width, height))
        if box is None:
            return False
        rgba, weight = grad.render((width, height), p0, p1, start, end, kind, stops=stops)
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
            # The same one line ``write_colour`` and ``preview_filter`` already
            # had, and the one write path that was missing it: a ramp run over
            # a layer with "preserve transparency" on painted straight through
            # the edge of the shape it was meant to stay inside, because the
            # composite above sets ``out_a`` from the ramp's own coverage.
            # Restoring the channel *is* the definition of the lock rather than
            # an approximation of it, which is why it goes after the formula.
            out[..., 3] = before[..., 3]
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(out)
        self._commit_patch(layer, box, before)
        return True
