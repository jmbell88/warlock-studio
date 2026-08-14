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

from typing import Any

import numpy as np

from . import gid as gidlib
from . import scene
from .tilemap import OPAQUE_WHITE, ImageLayer, MapDoc, TileLayer

# The flat composite is ``pixel_height * pixel_width * 4`` bytes in one
# allocation, and both factors are the document's own numbers -- a 4096-square
# map of 64-pixel tiles is 68 gigapixels and asks for a quarter of a terabyte
# before anything here gets a turn. 2^28 pixels is a gigabyte of RGBA: far past
# any map anyone renders on purpose (a 512-square map at 32px is 4 megapixels)
# and far short of a machine's RAM. **New rather than corpus-keyed** -- nothing
# stored is measured against it, so it needs no ``docs/measurements/`` document;
# it is a refusal about arithmetic, not a threshold about quality. Read from
# module globals at call time so a test can lower it.
MAX_RENDER_PIXELS = 1 << 28


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


def _tinted(pixels: np.ndarray, tint: Any) -> np.ndarray:
    """One image multiplied by a layer tint, or the same array untouched.

    The early-out is not an optimisation but the bit-identity guarantee: an
    untinted map -- which is every map today, and every map a ``.wmap`` can
    currently store -- has to composite through exactly the arithmetic it did
    before tints existed, or every pinned export byte moves. ``OPAQUE_WHITE`` is
    the multiplicative identity, so skipping it is the same answer.
    """
    if tuple(tint) == OPAQUE_WHITE:
        return pixels
    factor = np.asarray(tuple(tint), dtype=np.float32) / 255.0
    return np.clip(np.rint(pixels.astype(np.float32) * factor), 0, 255).astype(np.uint8)


def _blit_over(out: np.ndarray, pixels: np.ndarray, x0: int, y0: int, opacity: float) -> None:
    """``_over`` one image onto another at a place that may be off the edge.

    The clip idiom ``tools.stamp`` uses, shared by the tile loop and the image
    layer rather than written twice: a tile taller than its cell and an image
    layer nudged by a negative offset are the same arithmetic.
    """
    height, width = pixels.shape[0], pixels.shape[1]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    span_w = min(width - sx0, out.shape[1] - dx0)
    span_h = min(height - sy0, out.shape[0] - dy0)
    if span_w <= 0 or span_h <= 0:
        return
    _over(
        out[dy0 : dy0 + span_h, dx0 : dx0 + span_w],
        pixels[sy0 : sy0 + span_h, sx0 : sx0 + span_w],
        opacity,
    )


def render_layer(
    doc: MapDoc,
    layer: TileLayer,
    out: np.ndarray,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    opacity: float | None = None,
    tint: Any = OPAQUE_WHITE,
) -> None:
    """Composite one tile layer onto ``out``, which is the map's pixel size.

    The three keyword arguments are what :func:`scene.resolve` worked out --
    they default to "no ancestor said anything", so a caller holding one loose
    layer still gets what it always got.
    """
    tile_h = doc.tile_h
    opacity = float(layer.opacity if opacity is None else opacity)
    shift_x, shift_y = int(round(offset[0])), int(round(offset[1]))
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
            # Tinted here rather than at the blit, so the multiply is paid once
            # per distinct (tile, orientation) instead of once per cell -- a map
            # is the same handful of tiles repeated thousands of times, which is
            # the memo's whole reason. The tint is constant for this call, so it
            # is not part of the key.
            pixels = _tinted(
                orient(
                    tileset.tile_pixels(local),
                    bool(mask & gidlib.FLIP_H),
                    bool(mask & gidlib.FLIP_V),
                    bool(mask & gidlib.FLIP_D),
                ),
                tint,
            )
            oriented[(tile_id, mask)] = pixels
        # A tileset whose tiles are larger than the map's grid is ordinary
        # -- a 32px map with 48px trees -- and Tiled anchors such a tile by
        # its *bottom* left, so it grows upward out of its cell. Clipped
        # rather than refused.
        height = pixels.shape[0]
        origin_x, origin_y = doc.cell_origin(column, row)
        _blit_over(
            out,
            pixels,
            int(origin_x) + shift_x,
            int(origin_y) + tile_h - height + shift_y,
            opacity,
        )


def repeats(origin: int, step: int, span: int, repeat: bool) -> list[int]:
    """Where one axis of a repeating image layer starts, in ``out``'s space.

    ``[origin]`` when it does not repeat -- Tiled's own behaviour, and it means
    the caller needs no branch. When it does, the run is walked back to the
    first copy that is still partly on the surface (which may begin off the near
    edge) and forward to the last, so a tiny picture on a large map costs the
    copies it draws and no more.
    """
    if not repeat or step <= 0:
        return [origin]
    first = origin % step
    if first > 0:
        first -= step
    return list(range(first, span, step))


def render_image(
    doc: MapDoc,
    layer: ImageLayer,
    out: np.ndarray,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    opacity: float | None = None,
    tint: Any = OPAQUE_WHITE,
) -> None:
    """Composite one image layer onto ``out``.

    Through the same :func:`_over` the tile loop uses, so an image layer and a
    tile layer blend identically and an export's stack is one arithmetic rather
    than two. A layer with no picture draws nothing rather than refusing --
    Tiled writes an ``<imagelayer>`` with no source and it is a legal, empty
    layer.
    """
    pixels = _tinted(layer.pixels, tint)
    if pixels.size == 0:
        return
    opacity = float(layer.opacity if opacity is None else opacity)
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    for y0 in repeats(int(round(offset[1])), height, out.shape[0], layer.repeat_y):
        for x0 in repeats(int(round(offset[0])), width, out.shape[1], layer.repeat_x):
            _blit_over(out, pixels, x0, y0, opacity)


def render_map(doc: MapDoc, *, include_hidden: bool = False) -> np.ndarray:
    """The whole map, bottom layer first, as an RGBA array.

    A hidden layer is not drawn, for the reason Clay's hidden object is not
    exported: one flag decides what you see and what comes out, and two answers
    to that are how an export starts disagreeing with the screen. A layer inside
    a hidden *group* is hidden too, which is :func:`scene.resolve`'s answer
    rather than one taken here.

    **Parallax is treated as 1.0, deliberately.** A parallax factor is a
    relationship between a layer and a *camera*, and a still has no camera:
    there is no scroll position an export could honour it against, so the only
    honest thing to draw is the map at rest. The canvas, which does have one,
    applies it against the view origin.
    """
    width, height = int(doc.pixel_width), int(doc.pixel_height)
    ceiling = MAX_RENDER_PIXELS
    if width * height > ceiling:
        raise ValueError(
            f"this map renders to {width * height} pixels, past the {ceiling} "
            "this build will composite in one image"
        )
    out = np.zeros((height, width, 4), dtype=np.uint8)
    for entry in scene.resolve(doc, include_hidden=include_hidden):
        if entry.opacity <= 0.0:
            continue
        if isinstance(entry.layer, TileLayer):
            render_layer(
                doc,
                entry.layer,
                out,
                offset=entry.offset,
                opacity=entry.opacity,
                tint=entry.tint,
            )
        elif isinstance(entry.layer, ImageLayer):
            render_image(
                doc,
                entry.layer,
                out,
                offset=entry.offset,
                opacity=entry.opacity,
                tint=entry.tint,
            )
        # Object layers carry no pixels: they are metadata an engine reads, and
        # drawing the editor's handles into an export would be drawing the ruler
        # onto the drawing.
    return out


# One mean colour per tile, memoised on the identity of the tileset's pixel
# array. Safe because ``Tileset.__post_init__`` freezes that array: the object
# is immutable, so an id that matches is the same art, and a *replaced* tileset
# is a new array with a new id.
_LUT_CACHE: dict[int, np.ndarray] = {}


def tile_colours(tileset: Any) -> np.ndarray:
    """``(tile_count, 4)`` of each tile's alpha-weighted mean colour.

    Weighted, because an unweighted mean over a tile that is mostly transparent
    is dominated by whatever colour happens to be stored under the zero alpha --
    which is routinely black, and would draw a sparse map as a dark smear.
    """
    key = id(tileset.pixels)
    cached = _LUT_CACHE.get(key)
    if cached is not None and len(cached) == tileset.tile_count:
        return cached
    out = np.zeros((tileset.tile_count, 4), dtype=np.uint8)
    for local in range(tileset.tile_count):
        tile = tileset.tile_pixels(local).astype(np.float32)
        alpha = tile[..., 3]
        total = float(alpha.sum())
        if total <= 0.0:
            continue
        out[local, :3] = np.clip(
            np.rint((tile[..., :3] * alpha[..., None]).sum(axis=(0, 1)) / total), 0, 255
        )
        out[local, 3] = int(round(float(alpha.mean())))
    _LUT_CACHE[key] = out
    return out


def minimap(doc: MapDoc) -> np.ndarray:
    """The whole map as one pixel per cell, ``(height, width, 4)``.

    **Not a downscale of ``render_map``.** That composites at full size and is
    guarded by ``MAX_RENDER_PIXELS`` precisely because the full size is enormous
    -- a 512-square map of 32px tiles is a 268-megapixel blend, and asking for
    one every time the head moves is not something a frame can afford. One mean
    colour per tile makes it a handful of array operations over the *cell* grid
    instead, which is four orders of magnitude smaller.

    Composited bottom-first through the same ``_over`` the flat renderer uses
    and over the same :func:`scene.resolve`, so visibility, opacity and tint
    cascade through groups exactly as they do everywhere else and the picture
    agrees with the canvas about what is showing. Flip flags are ignored on
    purpose: every one of the eight transforms is a permutation of a tile's
    pixels, and a mean is invariant under all of them.

    **Layer offsets and parallax are ignored, and image layers are absent.**
    This is one pixel per *cell*, not a scaled picture: a pixel offset is a
    fraction of a cell and has nowhere to land, and an image layer has no cells
    at all. What the minimap is for is "where am I on the map", which is a
    question about the grid.
    """
    out = np.zeros((int(doc.height), int(doc.width), 4), dtype=np.uint8)
    luts = [(ref, tile_colours(ref.tileset)) for ref in doc.tilesets]
    for entry in scene.resolve(doc):
        layer = entry.layer
        if not isinstance(layer, TileLayer) or entry.opacity <= 0.0:
            continue
        cells = np.zeros_like(out)
        ids = np.asarray(layer.data) & np.uint32(gidlib.GID_MASK)
        for ref, lut in luts:
            if not len(lut):
                continue
            picked = (ids >= ref.firstgid) & (ids <= ref.last_gid)
            if not picked.any():
                continue
            cells[picked] = lut[ids[picked] - np.uint32(ref.firstgid)]
        _over(out, _tinted(cells, entry.tint), float(entry.opacity))
    return out
