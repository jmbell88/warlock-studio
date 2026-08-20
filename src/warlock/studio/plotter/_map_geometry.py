"""Changing the grid itself: the one operation no rectangle describes.

A resize is a **snapshot** step rather than a patch, and that is why it is its
own module rather than a third method in ``_map_paint``: every layer's *shape*
changes at once, so there is no dirty rect to record and nothing the patch
machinery can carry. Objects move with the content, because their coordinates
are absolute pixels and leaving them put would silently detach every trigger
volume from the geometry it was drawn around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..tilegrid import gid as gidlib
from ._map_model import (
    MAX_DIMENSION,
    ObjectLayer,
    Shape,
    TileLayer,
    _dimension,
    scaled_shape,
)
from .edits import CompoundEdit, Edit, InfiniteEdit, ResizeEdit, TileSizeEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


class GeometryOps:
    """Grid resizing, mixed into :class:`~.tilemap.MapDoc`."""

    def resize(
        self: MapDoc, width: int, height: int, *, offset_x: int = 0, offset_y: int = 0
    ) -> bool:
        """Change the grid, anchoring the old content at ``(offset_x, offset_y)``.

        **On an infinite map the origin slides by the same offset**, and it does
        so here rather than at the two call sites that need it. The offset moves
        the content within the window; an infinite map's cells have a *true*
        coordinate that a window change must not move, so the window's own
        corner absorbs it. One rule in one place: growth
        (:meth:`grow_to_hold`), a crop (:meth:`autocrop`) and any future caller
        all get it, and it rides the same undo step as the shape.
        """
        width, height = _dimension(width, "width"), _dimension(height, "height")
        dx, dy = int(offset_x), int(offset_y)
        if (width, height) == (self.width, self.height) and (dx, dy) == (0, 0):
            return False
        before_origin = (int(self.origin_x), int(self.origin_y))
        after_origin = (
            (before_origin[0] - dx, before_origin[1] - dy)
            if self.infinite
            else before_origin
        )

        before = {layer.uid: layer.data for layer in self.tile_layers()}
        after: dict[int, np.ndarray] = {}
        for uid, data in before.items():
            grown = gidlib.empty_layer(width, height)
            # The overlap of the old rectangle, shifted, with the new one.
            sx0, sy0 = max(0, -dx), max(0, -dy)
            tx0, ty0 = max(0, dx), max(0, dy)
            span_w = min(data.shape[1] - sx0, width - tx0)
            span_h = min(data.shape[0] - sy0, height - ty0)
            if span_w > 0 and span_h > 0:
                grown[ty0 : ty0 + span_h, tx0 : tx0 + span_w] = data[
                    sy0 : sy0 + span_h, sx0 : sx0 + span_w
                ]
            after[uid] = grown

        shift_x, shift_y = dx * self.tile_w, dy * self.tile_h
        before_objects: dict[int, list[tuple[float, float]]] = {}
        after_objects: dict[int, list[tuple[float, float]]] = {}
        # ``all_layers`` rather than ``self.layers``: an object layer inside a
        # group is still on the grid being resized, and leaving its objects put
        # is exactly the detachment this whole shift exists to prevent.
        for layer in self.all_layers():
            if not isinstance(layer, ObjectLayer):
                continue
            before_objects[layer.uid] = [(o.x, o.y) for o in layer.objects]
            after_objects[layer.uid] = [(o.x + shift_x, o.y + shift_y) for o in layer.objects]

        self.history.push(
            ResizeEdit(
                before_size=(self.width, self.height),
                after_size=(width, height),
                before=before,
                after=after,
                before_objects=before_objects,
                after_objects=after_objects,
                before_origin=before_origin,
                after_origin=after_origin,
            )
        )
        self._apply_resize((width, height), after, after_objects, after_origin)
        return True

    def offset(
        self: MapDoc,
        dx: int,
        dy: int,
        *,
        wrap: bool = True,
        scope: str = "map",
    ) -> bool:
        """Move cells by whole cells. Tiled's Map -> Offset Map.

        ``wrap`` rolls, which is an exact permutation -- no cell is invented and
        none is lost, so offsetting back is the identity. Objects wrap with the
        cells, modulo the map's pixel size, which is Tiled's answer and what
        keeps the identity holding for them too. Without ``wrap`` the vacated
        cells take gid 0, which is the honest answer for a shift that genuinely
        pushed content off the edge -- and objects then ride the shift off the
        map with the content they annotate.

        ``scope`` is ``"map"`` (every tile leaf) or ``"layer"`` (the active one).
        Whole-map scope moves objects by the pixel equivalent, which is
        :meth:`resize`'s own rule and for its reason: object coordinates are
        absolute pixels, and leaving them put silently detaches every trigger
        volume from the geometry it was drawn around. A single-layer offset moves
        no objects -- the objects are not on that layer.

        One snapshot edit, ``ResizeEdit``'s shape, because every layer's contents
        change at once and there is no dirty rect the patch machinery could carry.
        """
        dx, dy = int(dx), int(dy)
        if scope not in ("map", "layer"):
            raise ValueError("an offset scope is 'map' or 'layer'")
        if wrap and self.width and self.height:
            dx, dy = dx % self.width, dy % self.height
        if (dx, dy) == (0, 0):
            return False

        if scope == "layer":
            layer = self.active()
            targets = [layer] if isinstance(layer, TileLayer) else []
        else:
            targets = list(self.tile_layers())
        if not targets:
            return False

        before = {layer.uid: layer.data for layer in targets}
        after: dict[int, np.ndarray] = {}
        for uid, data in before.items():
            if wrap:
                after[uid] = np.ascontiguousarray(np.roll(data, (dy, dx), axis=(0, 1)))
                continue
            moved = gidlib.empty_layer(self.width, self.height)
            sx0, sy0 = max(0, -dx), max(0, -dy)
            tx0, ty0 = max(0, dx), max(0, dy)
            span_w = min(data.shape[1] - sx0, self.width - tx0)
            span_h = min(data.shape[0] - sy0, self.height - ty0)
            if span_w > 0 and span_h > 0:
                moved[ty0 : ty0 + span_h, tx0 : tx0 + span_w] = data[
                    sy0 : sy0 + span_h, sx0 : sx0 + span_w
                ]
            after[uid] = moved

        before_objects: dict[int, list[tuple[float, float]]] = {}
        after_objects: dict[int, list[tuple[float, float]]] = {}
        if scope == "map":
            shift_x, shift_y = dx * self.tile_w, dy * self.tile_h
            # Under ``wrap`` the objects wrap too, modulo the map's pixel size --
            # Tiled's own answer, and the only one that keeps the docstring's
            # identity claim true for them: the cells were normalized by modulo
            # above, so a shift applied *un*-wrapped moved objects by the
            # normalized amount -- ``offset(-1)`` on an 8-wide map pushed every
            # object 7 tiles right, and ``offset(+1)`` then ``offset(-1)`` left
            # them displaced a full map width from the geometry they annotate.
            pixel_w, pixel_h = self.width * self.tile_w, self.height * self.tile_h
            for layer in self.all_layers():
                if not isinstance(layer, ObjectLayer):
                    continue
                before_objects[layer.uid] = [(o.x, o.y) for o in layer.objects]
                after_objects[layer.uid] = [
                    (
                        ((o.x + shift_x) % pixel_w, (o.y + shift_y) % pixel_h)
                        if wrap
                        else (o.x + shift_x, o.y + shift_y)
                    )
                    for o in layer.objects
                ]

        self.history.push(
            ResizeEdit(
                before_size=(self.width, self.height),
                after_size=(self.width, self.height),
                before=before,
                after=after,
                before_objects=before_objects,
                after_objects=after_objects,
            )
        )
        self._apply_resize((self.width, self.height), after, after_objects)
        return True

    def grow_to_hold(self: MapDoc, x0: int, y0: int, x1: int, y1: int) -> bool:
        """Make room for an inclusive cell rect on an infinite map. Grew?

        Painting past the edge of an infinite map grows it, and the growth is an
        ordinary :meth:`resize` -- which is already one undoable step, already
        moves objects by the pixel equivalent, and already has a corpus. The
        origin absorbs the shift so a cell's *true* coordinate never moves: the
        stored grid slides under it.

        A finite map never grows; painting past its edge is clipped by every
        tool exactly as it always was.
        """
        if not self.infinite:
            return False
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        left = max(0, -min(x0, 0))
        top = max(0, -min(y0, 0))
        right = max(0, x1 - self.width + 1)
        bottom = max(0, y1 - self.height + 1)
        if not (left or top or right or bottom):
            return False
        width = self.width + left + right
        height = self.height + top + bottom
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            # The extent cap, and it is the *engine's* own ``MAX_DIMENSION``
            # rather than the new-map form's cap: this package is pinned against
            # reaching into ``studio``, and the argument is the same one
            # ``_dimension`` already makes -- an infinite map does not mean an
            # unbounded allocation.
            raise ValueError(
                f"an infinite map's painted extent stops at {MAX_DIMENSION} cells a side"
            )
        # ``resize`` slides the origin itself on an infinite map, so this is a
        # plain resize and the growth is one undoable step.
        return self.resize(width, height, offset_x=left, offset_y=top)

    def content_bounds(self: MapDoc) -> tuple[int, int, int, int] | None:
        """The inclusive cell rect holding every non-empty cell, or ``None``.

        Across *every* tile leaf, not the active one: a map is cropped as a
        whole, and a background layer that runs wider than the foreground is
        still content.
        """
        lo_x = lo_y = None
        hi_x = hi_y = None
        for layer in self.tile_layers():
            ys, xs = np.nonzero(np.asarray(layer.data) != gidlib.EMPTY)
            if not xs.size:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            lo_x = x0 if lo_x is None else min(lo_x, x0)
            hi_x = x1 if hi_x is None else max(hi_x, x1)
            lo_y = y0 if lo_y is None else min(lo_y, y0)
            hi_y = y1 if hi_y is None else max(hi_y, y1)
        if lo_x is None:
            return None
        return lo_x, lo_y, hi_x, hi_y  # type: ignore[return-value]

    def autocrop(self: MapDoc) -> bool:
        """Shrink the grid to the cells that hold something.

        Delegated to :meth:`resize` verbatim rather than reimplemented, so the
        objects travel by the rule that already exists and undo records the step
        the shape it already records. A wholly empty map is refused: cropping it
        to nothing is not a map, and there is no size that would be right.
        """
        found = self.content_bounds()
        if found is None:
            return False
        x0, y0, x1, y1 = found
        return self.resize(
            x1 - x0 + 1, y1 - y0 + 1, offset_x=-x0, offset_y=-y0
        )

    def set_infinite(self: MapDoc, infinite: bool, *, crop: bool = True) -> bool:
        """Convert the map between fixed-size and infinite. Changed?

        **Finite to infinite is a flag.** The document already holds its cells
        as one dense window (see ``MapDoc.infinite``), so there is nothing to
        re-chunk: the rectangle it has becomes the window it starts with, its
        origin is (0, 0) because that is where a finite map's cells already are,
        and the first stroke past the edge grows it.

        **Infinite to finite crops to content**, because that is the only
        honest fixed rectangle to pick: the window may carry slack an erase left
        behind, and the cells sit at true coordinates that a finite map has no
        way to express. ``crop=False`` keeps the window as-is, for a caller that
        has already chosen the rectangle. A map with nothing painted is not
        cropped either way -- there is no content to crop to, and a 1x1 map is
        not what "make this finite" meant.

        One undo step whichever way it goes.
        """
        infinite = bool(infinite)
        if infinite == bool(self.infinite):
            return False
        before_origin = (int(self.origin_x), int(self.origin_y))
        cropped: Edit | None = None
        if not infinite and crop and self.content_bounds() is not None and self.autocrop():
            # Taken off the stack and carried inside the compound below, rather
            # than left as a step of its own.
            cropped = self.history.top
            self.history.drop()
        flag = InfiniteEdit(
            before=not infinite,
            after=infinite,
            before_origin=before_origin,
            # A finite map's origin is (0, 0) by definition; going the other way
            # the crop has already put the origin where it belongs.
            after_origin=(int(self.origin_x), int(self.origin_y)) if infinite else (0, 0),
        )
        self.history.push(
            CompoundEdit(edits=[cropped, flag]) if cropped is not None else flag
        )
        self._apply_infinite(infinite, flag.after_origin)
        return True

    def _apply_infinite(self: MapDoc, infinite: bool, origin: tuple[int, int]) -> None:
        self.infinite = bool(infinite)
        self.origin_x, self.origin_y = int(origin[0]), int(origin[1])

    def set_tile_size(self: MapDoc, tile_w: int, tile_h: int) -> bool:
        """Change the size of a cell, keeping every painted tile where it is.

        **This is allowed on a map with content, and the projection is not.**
        The two look alike and are not: a projection decides the *lattice*, so a
        set drawn for one paints the wrong shape into every cell already painted
        for the other, while a cell's size decides only how large that cell is
        drawn. A gid names a tile either way, and ``render`` already treats a
        tileset whose tiles do not match the grid as ordinary -- a 32px map with
        48px trees, anchored bottom left -- so nothing is re-sliced, nothing is
        renumbered and nothing is lost. What it *does* mean is that a plain
        image added later is sliced at the new size, because that is the size
        ``plotter_tilesets`` slices at; tilesets already attached keep the
        slicing they arrived with, which is exactly Tiled's model.
        """
        tile_w = _dimension(tile_w, "tile width")
        tile_h = _dimension(tile_h, "tile height")
        if (tile_w, tile_h) == (self.tile_w, self.tile_h):
            return False

        # A ratio rather than a cell walk: an object is not on the grid, it is
        # at a pixel, and the pixel that was two cells across is still two cells
        # across afterwards only if it moves by the same factor the cell did.
        scale_x = tile_w / float(self.tile_w)
        scale_y = tile_h / float(self.tile_h)
        before_objects: dict[int, list[tuple[float, float, Shape]]] = {}
        after_objects: dict[int, list[tuple[float, float, Shape]]] = {}
        for layer in self.all_layers():
            if not isinstance(layer, ObjectLayer):
                continue
            # The whole ``shape``, not a ``(w, h)`` pair: ``w``/``h`` are
            # *derived* from the shape and read-only, and three of the seven
            # shapes carry their extent as vertices instead. Replaying the
            # object it was is also what makes the undo exact rather than
            # approximately the same rectangle.
            before_objects[layer.uid] = [(o.x, o.y, o.shape) for o in layer.objects]
            after_objects[layer.uid] = [
                (o.x * scale_x, o.y * scale_y, scaled_shape(o.shape, scale_x, scale_y))
                for o in layer.objects
            ]

        self.history.push(
            TileSizeEdit(
                before_size=(self.tile_w, self.tile_h),
                after_size=(tile_w, tile_h),
                before_objects=before_objects,
                after_objects=after_objects,
            )
        )
        self._apply_tile_size((tile_w, tile_h), after_objects)
        return True

    def _apply_tile_size(
        self: MapDoc,
        size: tuple[int, int],
        objects: dict[int, list[tuple[float, float, Shape]]],
    ) -> None:
        self.tile_w, self.tile_h = int(size[0]), int(size[1])
        for uid, placed in objects.items():
            layer = self.layer(uid)
            if isinstance(layer, ObjectLayer):
                # ``strict`` for ``_apply_resize``'s reason: a length mismatch
                # means the list changed between the record and the replay, at
                # which point these rectangles describe other objects.
                for obj, (x, y, shape) in zip(layer.objects, placed, strict=True):
                    obj.x, obj.y, obj.shape = float(x), float(y), shape

    def _apply_resize(
        self: MapDoc,
        size: tuple[int, int],
        data: dict[int, np.ndarray],
        objects: dict[int, list[tuple[float, float]]],
        origin: tuple[int, int] | None = None,
    ) -> None:
        self.width, self.height = int(size[0]), int(size[1])
        if origin is not None:
            self.origin_x, self.origin_y = int(origin[0]), int(origin[1])
        for uid, array in data.items():
            layer = self.layer(uid)
            if isinstance(layer, TileLayer):
                # A copy, or undo and the document would share one array and
                # the next stamp would write into the history.
                layer.data = array.copy()
        for uid, positions in objects.items():
            layer = self.layer(uid)
            if isinstance(layer, ObjectLayer):
                # ``strict``, like every other zip in this package. The two
                # lists are recorded from one walk of the same layer and
                # replayed in LIFO order, so a length mismatch means an object
                # arrived or left between the record and the replay -- at which
                # point the positions no longer describe these objects and
                # truncating quietly leaves the tail sitting on the old grid,
                # detached from the geometry it was drawn around. That is the
                # thing the resize moves objects to prevent.
                for obj, (x, y) in zip(layer.objects, positions, strict=True):
                    obj.x, obj.y = float(x), float(y)
