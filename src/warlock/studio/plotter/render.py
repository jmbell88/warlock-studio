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
    alpha = src[..., 3]
    if float(opacity) >= 1.0 and bool(((alpha == 255) | (alpha == 0)).all()):
        # A source whose alpha is *binary*, at full opacity, is a masked copy --
        # and exactly one, not an approximation. That is the shape a tile
        # actually has (an opaque body with a transparent surround), which is
        # why the test is this rather than the tempting all-255: an all-255 tile
        # is rare, so an early-out gated on it fires almost never and measures
        # beautifully on a synthetic benchmark. Roughly 20x off an export.
        #
        # Where src is opaque: sa is 1, out_a is 1, safe is 1, and rgb reduces
        # to src's own channels -- uint8 widened to float32 and back, so rint
        # and clip are identities on them.
        #
        # Where src is clear: sa is 0, so out_a is dst's own alpha and rgb is
        # dst * da / da, which rint returns to dst exactly. The one pixel that
        # is *not* left alone is where dst is clear too -- there safe is 1, so
        # the full path writes rgb 0, discarding whatever colour was stored
        # under a zero alpha. Reproduced rather than tidied up: bit-identity is
        # the bar, and `dead` is computed before the copy for that reason.
        opaque = alpha == 255
        dead = ~opaque & (dst[..., 3] == 0)
        np.copyto(dst, src, where=opaque[..., None])
        if dead.any():
            dst[..., :3][dead] = 0
        return
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
    tile_h = doc.tile_h
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
    # The *oriented* tile is memoised beside the lookup, on (id, flags), because
    # orientation is a pure function of that pair and a map is the same handful
    # of tiles repeated thousands of times -- 64 tiles by 8 symmetries is a few
    # megabytes against one numpy call per cell. Read-only by construction: the
    # entries are only ever passed to ``_over`` as its source.
    oriented: dict[tuple[int, int], np.ndarray] = {}
    # ``draw_order`` rather than a nested range, because for an isometric map
    # row-major is not monotone in screen depth -- cell ``(width - 1, 0)`` sits
    # in front of ``(0, 1)`` and would be painted under it. It *is* row-major
    # for an orthogonal one, so nothing changes there.
    for column, row in doc.draw_order():
        if row >= layer.height or column >= layer.width:
            continue
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
        pixels = oriented.get((tile_id, mask))
        if pixels is None:
            pixels = orient(
                tileset.tile_pixels(local),
                bool(mask & gidlib.FLIP_H),
                bool(mask & gidlib.FLIP_V),
                bool(mask & gidlib.FLIP_D),
            )
            oriented[(tile_id, mask)] = pixels
        # A tileset whose tiles are larger than the map's grid is ordinary
        # -- a 32px map with 48px trees -- and Tiled anchors such a tile by
        # its *bottom* left, so it grows upward out of its cell. Clipped
        # with the same idiom ``tools.stamp`` uses rather than refused.
        height, width = pixels.shape[0], pixels.shape[1]
        origin_x, origin_y = doc.cell_origin(column, row)
        x0 = int(origin_x)
        y0 = int(origin_y) + tile_h - height
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
