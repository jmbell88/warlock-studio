"""The map as one flat RGBA image.

Pure, and deliberately *not* the thing the canvas draws. The canvas issues one
textured quad per visible cell through imgui's draw list, which is O(what is on
screen) and needs no image at all; this composites the *whole* map, which is
what an export wants and what nothing per-frame should ever ask for.

Having both is not duplication of the interesting part: the interesting part is
where a cell lands and which way round it is, and both answers come from
:mod:`.gid` and :attr:`~.tileset.Tileset.tile_rect`. What differs is only the
surface being drawn onto.

**The three flags are applied transpose-then-mirror**, which is Tiled's own
order and is what makes the eight combinations the eight square symmetries. The
transpose is what a diagonal flip *is*, so a non-square tile flipped diagonally
comes out with its axes swapped -- that is the format's behaviour, not a bug
here, and the blit clips rather than refusing.
"""

from __future__ import annotations

import numpy as np

from . import gid as gidlib
from .tilemap import MapDoc, TileLayer


def orient(tile: np.ndarray, flip_h: bool, flip_v: bool, flip_d: bool) -> np.ndarray:
    """One tile's pixels, with its transform flags applied.

    A view where it can be one -- every branch here is a slice or a transpose,
    so an unflipped tile costs nothing at all and the common case is free.
    """
    if flip_d:
        tile = np.transpose(tile, (1, 0, 2))
    if flip_h:
        tile = tile[:, ::-1]
    if flip_v:
        tile = tile[::-1, :]
    return tile


def _over(dst: np.ndarray, src: np.ndarray, opacity: float) -> None:
    """Source-over, in place, on one cell-sized region.

    Float rather than integer arithmetic: a map is routinely a stack of layers
    with soft edges, and repeated 8-bit rounding through four or five of them
    shows as a visible dark fringe along every antialiased tile edge.
    """
    sa = (src[..., 3:4].astype(np.float32) / 255.0) * float(opacity)
    if not sa.any():
        return
    da = dst[..., 3:4].astype(np.float32) / 255.0
    out_a = sa + da * (1.0 - sa)
    # Where the result is fully transparent there is no colour to divide by;
    # leaving those pixels alone is the same answer without the warning.
    safe = np.where(out_a > 0.0, out_a, 1.0)
    rgb = (
        src[..., :3].astype(np.float32) * sa + dst[..., :3].astype(np.float32) * da * (1.0 - sa)
    ) / safe
    dst[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    dst[..., 3:4] = np.clip(np.rint(out_a * 255.0), 0, 255).astype(np.uint8)


def render_layer(doc: MapDoc, layer: TileLayer, out: np.ndarray) -> None:
    """Composite one tile layer onto ``out``, which is the map's pixel size."""
    tile_w, tile_h = doc.tile_w, doc.tile_h
    ids = gidlib.tile_ids(layer.data)
    flags = gidlib.flags(layer.data)
    # A per-tileset cache of resolved tiles is pointless -- ``tile_pixels`` is
    # already a view -- but resolving the *tileset* is a linear scan of the
    # list, so the answer for each distinct id is worked out once.
    # A *failed* lookup is memoised too, as None -- which is why the membership
    # test is ``not in`` rather than a falsy ``get``. A gid no loaded tileset
    # carries is the one answer that costs a full scan of the list every time,
    # and a map that has lost a tileset is made entirely of them.
    resolved: dict[int, tuple | None] = {}
    for row in range(min(layer.height, doc.height)):
        for column in range(min(layer.width, doc.width)):
            tile_id = int(ids[row, column])
            if not tile_id:
                continue
            if tile_id not in resolved:
                resolved[tile_id] = doc.resolve(tile_id)
            entry = resolved[tile_id]
            if entry is None:
                continue
            tileset, local = entry
            mask = int(flags[row, column])
            pixels = orient(
                tileset.tile_pixels(local),
                bool(mask & gidlib.FLIP_H),
                bool(mask & gidlib.FLIP_V),
                bool(mask & gidlib.FLIP_D),
            )
            # A tileset whose tiles are larger than the map's grid is ordinary
            # -- a 32px map with 48px trees -- and Tiled anchors such a tile by
            # its *bottom* left, so it grows upward out of its cell. Clipped
            # with the same idiom ``tools.stamp`` uses rather than refused.
            height, width = pixels.shape[0], pixels.shape[1]
            x0 = column * tile_w
            y0 = row * tile_h + tile_h - height
            sx0, sy0 = max(0, -x0), max(0, -y0)
            dx0, dy0 = max(0, x0), max(0, y0)
            span_w = min(width - sx0, out.shape[1] - dx0)
            span_h = min(height - sy0, out.shape[0] - dy0)
            if span_w <= 0 or span_h <= 0:
                continue
            _over(
                out[dy0 : dy0 + span_h, dx0 : dx0 + span_w],
                pixels[sy0 : sy0 + span_h, sx0 : sx0 + span_w],
                layer.opacity,
            )


def render_map(doc: MapDoc, *, include_hidden: bool = False) -> np.ndarray:
    """The whole map, bottom layer first, as an RGBA array.

    A hidden layer is not drawn, for the reason Clay's hidden object is not
    exported: one flag decides what you see and what comes out, and two answers
    to that are how an export starts disagreeing with the screen.
    """
    out = np.zeros((doc.pixel_height, doc.pixel_width, 4), dtype=np.uint8)
    for layer in doc.layers:
        if not isinstance(layer, TileLayer):
            # Object layers carry no pixels: they are metadata an engine reads,
            # and drawing the editor's handles into an export would be drawing
            # the ruler onto the drawing.
            continue
        if not layer.visible and not include_hidden:
            continue
        if layer.opacity <= 0.0:
            continue
        render_layer(doc, layer, out)
    return out
