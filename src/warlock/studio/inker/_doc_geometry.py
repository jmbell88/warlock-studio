"""Whole-canvas geometry: flip, rotate, scale, crop, canvas resize.

Every one of these is a canvas-level operation recorded by snapshot rather than
by patch, so they all take the same two-line shape -- commit whatever is
floating, then hand ``_replay`` a closure over ``_map_planes``. The machinery
they lean on (``_replay``, ``_grid_snapshot``, ``_map_planes``) stays in
``document.py``: it is shared with the indexed-colour ops and with anything else
that rewrites every plane at once.

**Four of the five are exact on an indexed document and say so.** A flip, a
quarter turn, a crop and a canvas resize move pixels without inventing any, so
they hand ``_map_planes`` an ``index_fn`` -- the same permutation, applied to
the index plane -- and two palette slots holding the same colour come out the
other side still two slots. ``scale`` is the exception and cannot be anything
else: a smooth resample produces colours that were never in the table, so its
indices are re-resolved from the result. That is the one stated place in the
whole indexed design where indices are inferred rather than permuted.

**A tilemap layer survives exactly one of the five.** ``resize_canvas`` moves
the picture by whole cells, so it re-grids ``refs`` (``_doc_tiles._tile_regrid``,
passed down as ``_map_planes``' ``refs_fn``) and refuses a non-tile-aligned
offset by name. The other four still refuse the document outright, because
each would have to permute every cell's *flag bits* as well as its position
and no such algebra is written yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import transform as tf

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document


class GeometryOps:
    """Canvas-level geometry, mixed into :class:`~.document.Document`."""

    # Each ``run`` maps the *slices* first and the planes second, and the order
    # is load-bearing rather than tidy: every slice mapper reads ``self.size``,
    # which is the old canvas only until ``_map_planes`` rebuilds the stack. It
    # holds on redo too, because redo replays this same closure over a document
    # ``restore_snapshot`` has already put back.

    def flip(self: Document, axis: str) -> None:
        self._refuse_tilemaps("flip")
        self.commit_floating()

        def run() -> None:
            self._slices_flip(axis)
            self._map_planes(
                lambda plane: tf.flip(plane, axis),
                index_fn=lambda plane: tf.flip(plane, axis),
            )

        self._replay(run)

    def rotate90(self: Document, quarters: int = 1) -> None:
        self._refuse_tilemaps("rotation")
        self.commit_floating()

        def run() -> None:
            self._slices_rotate90(quarters)
            self._map_planes(
                lambda plane: tf.rotate90(plane, quarters),
                index_fn=lambda plane: tf.rotate90(plane, quarters),
            )

        self._replay(run)

    def scale(self: Document, size: tuple[int, int], *, resample: str = "smooth") -> None:
        self._refuse_tilemaps("scale")
        self.commit_floating()

        def run() -> None:
            self._slices_scale(size)
            self._map_planes(lambda plane: tf.scale(plane, size, resample=resample))

        self._replay(run)

    def offset_layer(self: Document, dx: int, dy: int) -> bool:
        """Roll the active layer's contents, wrapping at every edge.

        The seamless-tile move: put the seam in the middle of the canvas, look
        at it, paint over it. :func:`transform.translate` already does the whole
        job -- ``wrap=True`` is an exact permutation, so no pixel is invented and
        none is lost, and its docstring already argues the index-plane case --
        so what was missing was the funnel and the button.

        **Wrap is always on.** An un-wrapped offset is the move tool, which
        exists; this is the tiling operation, and an offset that dropped content
        off one edge would be a different thing wearing the same name.

        The selection mask is deliberately **not** moved: this offsets the
        *layer*, not the selection, and a marquee that travelled with it would
        make "offset, then paint inside the selection" mean something nobody
        asked for. Content-lock refusal like every structural op, and one undo
        step through the standard patch funnel -- so the colour mode is applied,
        an indexed document records an index patch, and a zero offset pushes
        nothing at all.
        """
        if self.write_locked(self.stack.active):
            return False
        width, height = self.size
        dx, dy = int(dx) % width if width else 0, int(dy) % height if height else 0
        if not dx and not dy:
            return False
        self.commit_floating()
        # Autovivify like every other write path: on an animated document the
        # active row may be a placeholder over the shared read-only blank
        # plane, and rolling that in place raises mid-gesture. The pending cel
        # rides into ``_commit_patch``'s step, a stroke's own machinery.
        self._ensure_active_cel()
        layer = self.stack.active
        box = (0, 0, width, height)
        before = layer.pixels.copy()
        layer.pixels[:, :] = tf.translate(layer.pixels, dx, dy, wrap=True)
        if layer.indices is not None:
            # The permutation, applied to the plane rather than re-inferred
            # from the colours: two slots holding one colour stay two slots.
            layer.indices[:, :] = tf.translate(layer.indices, dx, dy, wrap=True)
        self._commit_patch(layer, box, before)
        return True

    def descale_to_grid(self: Document, scale: int, phase: tuple[int, int]) -> None:
        """Reduce the document onto its own pixel lattice, one pixel per cell.

        The remedy for an SDXL "pixel art" render at 1024 squared whose real art
        is 64 squared. A plain smooth scale to that size gives a small
        photograph of pixel art; a nearest scale ignores the *phase* and samples
        wherever the arithmetic happens to land, so half the authored pixels
        come back as their neighbours' edges. This samples each cell's centre on
        the measured lattice, which is what makes it exact.

        Shaped like :meth:`scale` and for the same reason: one selection applies
        to the pixels, the index planes and the mask alike, because the sampling
        is pure element selection and knows nothing about what a plane means.
        One undo step.
        """
        self._refuse_tilemaps("descale")
        size = tf.descale_size(self.size, scale, phase)
        if not size[0] or not size[1]:
            raise ValueError("that grid leaves no cells in this document")
        self.commit_floating()

        def run() -> None:
            self._slices_scale(size)
            self._map_planes(
                lambda plane: tf.descale(plane, scale, phase),
                index_fn=lambda plane: tf.descale(plane, scale, phase),
            )

        self._replay(run)

    def crop(self: Document, rect: tuple[int, int, int, int]) -> bool:
        self._refuse_tilemaps("crop")
        box = self.clip(rect)
        if box is None:
            return False
        self.commit_floating()

        def run() -> None:
            self._slices_offset((-box[0], -box[1]), (box[2] - box[0], box[3] - box[1]))
            self._map_planes(
                lambda plane: tf.crop(plane, box),
                index_fn=lambda plane: tf.crop(plane, box),
            )

        self._replay(run)
        return True

    def crop_to_selection(self: Document) -> bool:
        bounds = self.mask.bounds if self.mask is not None else None
        return self.crop(bounds) if bounds else False

    def resize_canvas(
        self: Document,
        size: tuple[int, int],
        offset: tuple[int, int] | None = None,
        *,
        anchor: str = "top-left",
    ) -> None:
        """Grow or crop the canvas, placing the old image by *anchor*.

        An explicit ``offset`` still wins, because it is the general form and
        the anchor is a name for nine of its values -- and because every caller
        that already computed one should keep working unchanged.

        **The one geometry op a tilemap layer survives.** Every other method
        here still refuses one outright, because a flip or a turn would have to
        permute each cell's flag bits as well as its position; a resize
        translates by whole cells and pads or crops, so ``_tile_regrid`` is a
        pure pad/crop of the refs plane. It refuses by name when the offset is
        not a whole number of tiles -- before ``commit_floating``, so a refused
        resize has changed nothing at all.
        """
        if offset is None:
            offset = tf.anchor_offset(self.size, size, anchor)
        where = offset
        regrid = self._tile_regrid(size, where)
        self.commit_floating()

        def run() -> None:
            self._slices_offset(where, size)
            self._map_planes(
                lambda plane: tf.resize_canvas(plane, size, where),
                # The new room is the *transparent index*, which is only slot 0
                # by coincidence -- see ``transform.resize_canvas``'s ``fill``.
                index_fn=lambda plane: tf.resize_canvas(
                    plane, size, where, self.transparent_index
                ),
                refs_fn=regrid,
            )

        self._replay(run)
