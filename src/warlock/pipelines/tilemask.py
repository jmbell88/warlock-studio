"""A blob-47 terrain set is *composited*, not generated.

**Because both materials are seamless, colour is continuous across every tile
boundary for free.** A tile standing at map cell ``(cx, cy)`` samples both
materials at ``(u, v) = position mod T``, so two adjacent tiles sample two
adjacent phases of one periodic texture and the join between them is a join the
texture already tiles across. Nothing here has to match colours, blend edges or
feather one picture into another. The only thing two neighbouring tiles must
agree about is **which pixels belong to the inner terrain** -- so the whole
problem is a scalar field, and not a drawing problem at all.

The field is a signed distance thresholded against wrap-periodic noise::

    d_out(p) = min over NON-member cell rects of dist(p, rect)   # 0 inside one
    d_in(p)  = min over MEMBER     cell rects of dist(p, rect)   # 0 inside one
    D(p)     = d_out(p) - d_in(p)                       # signed, + inside
    alpha(p) = clip((D(p) - (inset + amplitude*noise(p))) / feather + 0.5, 0, 1)

``dist`` to an axis-aligned box is one ``hypot`` of two clamped differences,
vectorised over the whole grid of pixel centres and min-reduced over the rects.
Exact, and no scipy, no distance transform, no marching anything: the rects are
cells of a lattice and there are at most nine of them per tile.

**The locality argument is what makes a 47-tile atlas equal to the whole map.**
A cell outside the 3x3 block around a tile is at distance at least ``T`` from
every pixel *in* that tile, and :func:`coverage` refuses a configuration where
``inset + amplitude + feather`` exceeds ``T/2`` -- so at distance ``T`` alpha is
already saturated at 1 and a far cell cannot move it. Which is why that bound is
raised rather than commented: it is not a taste limit, it is the hypothesis of
the theorem the atlas relies on. ``tests/test_tilemask.py`` renders a 12x12
membership field twice, once as one 384px field and once by blitting atlas
columns, and pins the two byte-for-byte.

**The eight bit constants and the 47-case collapse are restated here, not
imported.** ``studio.tilegrid.blob`` owns them for the editor;
``tests/tilegrid/test_tilegrid_imports.py`` pins that package as a leaf, and no
module under ``pipelines/`` imports ``studio`` -- a pipeline runs inside worker
and Blender processes where ``studio`` is not importable at all. So this is the
restate-and-pin pattern the repo already uses twice (``tilesheet.MAX_SEED``
against ``service.validation.MAX_SEED``; ``service.tilesheets.TILE_SIZES``
against ``tilesheet.TILE_SIZES``): the copy is derived the same way from the
same rule, and a *test* imports both and pins them equal, because tests are not
layered. The table is derived at import and never typed, for the reason
``blob.py`` gives -- a hand-written list of 47 masks is a list somebody
eventually renumbers by hand, and the atlas layout is keyed on the order.

**The wiggle repeats every tile.** One noise field of period ``T`` serves all 47
cases -- it has to, or the boundary would not line up where two tiles meet --
and the visible consequence is that a long straight run of coastline carries the
same wobble over and over. The fix is phase variants: ``Tileset.phases`` already
carries ``k*k`` sub-rows per terrain, and a phase-``k`` set wants noise of period
``k*T`` sliced into ``k*k`` tiles. That is what :func:`wrap_noise`'s
``period_tiles`` is for. Only ``period_tiles=1`` ships; the parameter exists so
the second version is a parameter rather than a rewrite.

Pure in this package's sense: stdlib at module scope, ``numpy`` inside the
functions that need it (``seam.py`` and ``material.py``'s convention), no torch,
no ``service``, no ``queue``, no ``studio``.
"""

from __future__ import annotations

from typing import Any

#: Every byte of every generated set depends on the field below -- the distance,
#: the noise, the thresholds and the rounding in :func:`composite`. Bumped when
#: any of them changes what a given ``(inner, outer, size, seed)`` produces, so a
#: set's sidecar can say which field drew it. It is not a file format version:
#: nothing here reads anything.
MASK_VERSION = 1

#: How far the coverage boundary sits *inside* the shared cell edge, as a
#: fraction of the tile size. It cannot be zero. At zero the boundary runs along
#: the cell edge, which puts half the feather -- and every negative excursion of
#: the noise -- into the neighbouring cell, and a non-member cell is then no
#: longer pure B. That is the one thing the map round-trip needs: a cell nobody
#: is a member of must be exactly the outer material, because that is what the
#: painter blits there without consulting this module at all.
#:
#: 3/16. At the house 32px tile that is 6px of inner terrain held back from the
#: edge, which reads as a coastline rather than as a cell boundary; at
#: :data:`MIN_TILE` it is 3px, which is where it stops reading as anything.
BLOB_INSET_RATIO = 0.1875

#: How far the noise is allowed to push the boundary either way, as a fraction
#: of the tile size. Strictly less than :data:`BLOB_INSET_RATIO` (0.09 against
#: 0.1875, a margin of a little over two to one) because the noise must never
#: push the boundary out past the cell edge -- if it could, the guarantee above
#: would hold only for most seeds, which is the worst kind of guarantee.
NOISE_AMPLITUDE_RATIO = 0.09

#: The width of the alpha ramp, as a fraction of the tile size. About one pixel
#: at 32px. Deliberately narrow: these are pixel-art tiles and a wide ramp is a
#: grey halo, not an antialiased edge. It is not zero because a hard threshold
#: on a distance field produces a visibly stepped diagonal.
FEATHER_RATIO = 0.03

#: The smallest tile this field is worth drawing on. Below 16px the inset is
#: under 3px -- there is no longer enough inner terrain between the wiggle and
#: the cell edge for "inside" and "outside" to read as different roles -- and
#: the finest noise octave has fallen to one lattice cell per pixel, where the
#: detail is not detail any more. Raised rather than clamped, because a caller
#: asking for an 8px terrain set wants to be told the answer is a smudge.
MIN_TILE = 16

#: The eight neighbour bits, clockwise from north. ``studio.tilegrid.blob``'s,
#: restated: the values are positional, the atlas order derives from them, and
#: they are not free to renumber. ``tests/test_tilemask.py`` pins them against
#: the other copy.
N = 1
NE = 2
E = 4
SE = 8
S = 16
SW = 32
W = 64
NW = 128

#: ``(dx, dy, bit)`` clockwise from north, in the y-down convention every array
#: in this repo uses -- north is ``dy = -1``, which is row zero of a tile.
NEIGHBOURS: tuple[tuple[int, int, int], ...] = (
    (0, -1, N),
    (1, -1, NE),
    (1, 0, E),
    (1, 1, SE),
    (0, 1, S),
    (-1, 1, SW),
    (-1, 0, W),
    (-1, -1, NW),
)

#: ``(corner, flank, flank)`` -- the two edges a corner is only visible past.
CORNER_FLANKS: tuple[tuple[int, int, int], ...] = (
    (NE, N, E),
    (SE, E, S),
    (SW, S, W),
    (NW, W, N),
)

#: One rectangle of the cell lattice, in pixels: ``(x0, y0, x1, y1)``, half-open
#: in neither direction because a distance does not care.
Rect = tuple[float, float, float, float]

#: Lattice cells per period at the coarsest octave, doubling from there: 4, 8,
#: 16. Four is the coarsest that still varies within one tile; sixteen at a 32px
#: tile is a two-pixel cell, which is the finest thing a pixel-art edge can
#: carry. A fourth octave would be lattice cells smaller than a pixel.
NOISE_LATTICE = 4

#: Three octaves at halving amplitude. Enough for the boundary to read as
#: organic -- one octave is a slow bulge, two is a bulge with a bump -- and the
#: third is already at the pixel floor described above.
NOISE_OCTAVES = 3


def normalise(mask: int) -> int:
    """Drop every diagonal bit whose two flanking edges are not both set.

    The whole 47-case collapse, and ``blob.normalise`` restated. A cell with a
    north-east neighbour but no north and no east is a cell whose north-east
    corner is open regardless -- the diagonal is behind the corner, not in it.

    **This module does not need it to build a tile.** :func:`blob_rects` takes
    the raw mask, and the collapse falls out of the geometry instead: a
    diagonal's nearest point to any pixel of the tile is the shared corner, and
    whenever a flank is absent that flank's own rect is strictly closer, so the
    diagonal never wins the minimum. ``tests/test_tilemask.py`` asserts exactly
    that, over all 256 raw masks. What the rule is *for* here is the atlas
    order -- 47 columns and which case each one is.
    """
    value = int(mask) & 0xFF
    for corner, first, second in CORNER_FLANKS:
        if not (value & first and value & second):
            value &= ~corner
    return value


#: The 47 canonical masks, ascending. Derived, so the count is an observation
#: rather than a promise, and the order is a property of the integers: the
#: isolated tile at index 0, the interior fill at index 46. That is the order
#: every published blob sheet uses and the order ``Tileset.local_for`` reads
#: columns in, so :func:`blob_atlas` emits its columns in it and nothing
#: downstream reorders anything.
BLOB_MASKS: tuple[int, ...] = tuple(sorted({normalise(m) for m in range(256)}))

TILE_COUNT: int = len(BLOB_MASKS)

#: Cached backing store for :data:`BLOB_INDEX`, which is built on first access.
_BLOB_INDEX: Any = None


def __getattr__(name: str) -> Any:
    """``BLOB_INDEX``, built on first read rather than at import.

    A ``uint8`` array of 256 entries mapping every raw mask to its index in
    :data:`BLOB_MASKS` -- ``blob.BLOB_INDEX``, restated, and read-only for its
    reason: the hot caller indexes it with a whole ``(h, w)`` mask field at
    once, and a dict would put a Python loop in the middle of that.

    It is the one published name here that *is* an ndarray, and this module's
    rule is that numpy is imported inside the functions that need it -- a
    pipeline is imported by workers that may never touch it. So it is lazy
    rather than module-level, which keeps both promises instead of choosing.
    """
    global _BLOB_INDEX
    if name != "BLOB_INDEX":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _BLOB_INDEX is None:
        import numpy as np

        table = np.array(
            [BLOB_MASKS.index(normalise(m)) for m in range(256)], dtype=np.uint8
        )
        table.setflags(write=False)
        _BLOB_INDEX = table
    return _BLOB_INDEX


# -- the field ----------------------------------------------------------------


def wrap_noise(
    size: int,
    *,
    seed: int,
    octaves: int = NOISE_OCTAVES,
    period_tiles: int = 1,
) -> Any:
    """Seeded value noise that is exactly periodic. -> float32 in ``[-1, 1]``.

    Returns a ``(span, span)`` field where ``span = size * period_tiles``, and
    ``span`` is also its exact period in both axes. **Periodicity is free rather
    than enforced**: each octave lays a lattice of ``k`` cells across one period
    and takes its cell indices ``mod k``, so the cell to the right of the last
    one *is* the first one and there is no seam to fix afterwards. Nothing here
    blends, mirrors or tapers an edge, which is what a wrapped-noise
    implementation usually has to do and what usually goes subtly wrong.

    Value noise on a lattice with smoothstep interpolation, not a sum of
    sinusoids. Sinusoids of harmonic periods are also exactly periodic and much
    shorter to write, and they read as *ripples* -- a regular beat along the
    coastline. The entire point of the noise is that the boundary should not
    look designed.

    Normalised by its own extremes so the range is exactly ``[-1, 1]``. The
    alternative -- dividing by the sum of the octave amplitudes -- is bounded by
    ``[-1, 1]`` too but never reaches it, which would silently make
    :data:`NOISE_AMPLITUDE_RATIO` mean a different displacement for every octave
    count. The bound itself is load-bearing either way: :func:`coverage`'s
    precondition assumes ``|noise| <= 1``.

    ``period_tiles`` is the door to phase variants and nothing generated ships
    with it yet -- see the module docstring.
    """
    import numpy as np

    span_tiles = int(period_tiles)
    count = int(octaves)
    side = int(size)
    if side < 1:
        raise ValueError(f"noise needs at least a 1px field; got {side}")
    if span_tiles < 1:
        raise ValueError(f"a noise period spans at least one tile; got {span_tiles}")
    if count < 1:
        raise ValueError(f"noise needs at least one octave; got {count}")
    span = side * span_tiles

    rng = np.random.default_rng(seed)
    index = np.arange(span, dtype=np.float64)
    total = np.zeros((span, span), dtype=np.float64)
    weight = 1.0
    for octave in range(count):
        cells = NOISE_LATTICE << octave
        values = rng.random((cells, cells))
        # One period of pixels spans exactly ``cells`` lattice cells, so the
        # position of pixel ``p`` is ``p * cells / span`` and the wrap is a
        # modulo on the integer part. Both ends of the interpolation are taken
        # mod ``cells``, which is the whole of the periodicity.
        position = index * (cells / span)
        floor = np.floor(position)
        low = floor.astype(np.int64) % cells
        high = (low + 1) % cells
        frac = position - floor
        # Smoothstep, so the field has zero derivative at every lattice node --
        # including the node the wrap lands on, which is why the seam is not
        # merely continuous but flat.
        ramp = frac * frac * (3.0 - 2.0 * frac)
        wx = ramp[None, :]
        wy = ramp[:, None]
        top_left = values[np.ix_(low, low)]
        top_right = values[np.ix_(low, high)]
        bottom_left = values[np.ix_(high, low)]
        bottom_right = values[np.ix_(high, high)]
        top = top_left + (top_right - top_left) * wx
        bottom = bottom_left + (bottom_right - bottom_left) * wx
        total += weight * (top + (bottom - top) * wy)
        weight *= 0.5

    lowest, highest = float(total.min()), float(total.max())
    if highest - lowest < 1e-12:
        # A degenerate draw has no variation to stretch, and a flat zero is the
        # honest answer: the boundary is then exactly the inset circle.
        return np.zeros((span, span), dtype=np.float32)
    return (((total - lowest) / (highest - lowest)) * 2.0 - 1.0).astype(np.float32)


def blob_rects(mask: int, size: int) -> tuple[list[Rect], list[Rect]]:
    """The 3x3 cell block around one tile, split into ``(member, outer)``.

    Tile-local pixel coordinates: cell ``(dx, dy)`` occupies
    ``[dx*T, (dx+1)*T] x [dy*T, (dy+1)*T]``, so the tile itself is
    ``[0, T] x [0, T]`` and north is ``dy = -1``.

    **The centre cell is always a member**, which is what a tile *is* -- the
    picture of a cell that belongs to the terrain. The consequence is worth
    stating because it removes half the arithmetic downstream: every pixel of
    the tile is inside a member rect, so ``d_in`` is zero throughout and the
    signed distance is just ``d_out``.

    Takes the **raw** mask, not the normalised one. See :func:`normalise`: the
    collapse is a fact about this geometry rather than an input to it.
    """
    side = int(size)
    if side < 1:
        raise ValueError(f"a tile is at least one pixel across; got {side}")
    value = int(mask) & 0xFF
    member: list[Rect] = [(0.0, 0.0, float(side), float(side))]
    outer: list[Rect] = []
    for dx, dy, bit in NEIGHBOURS:
        rect: Rect = (
            float(dx * side),
            float(dy * side),
            float((dx + 1) * side),
            float((dy + 1) * side),
        )
        (member if value & bit else outer).append(rect)
    return member, outer


def _centres(size: int) -> tuple[Any, Any]:
    """The ``(py, px)`` grids of pixel *centres*, as float64.

    Centres rather than corners, because the field is sampled per pixel and a
    corner sample puts the boundary half a pixel off in each axis -- which on a
    16px tile is a visible bias towards the north-west.
    """
    import numpy as np

    coords = np.arange(int(size), dtype=np.float64) + 0.5
    return np.meshgrid(coords, coords, indexing="ij")


def _nearest(py: Any, px: Any, rects: list[Rect]) -> Any:
    """Distance from every pixel centre to the nearest of ``rects``, or inf.

    Distance to an axis-aligned box is ``hypot`` of the per-axis overshoot,
    clamped at zero so a pixel inside the box is at distance zero. Min-reduced
    in place over the rects rather than stacked, because the map-wide caller
    passes every cell of the field and a stacked ``(cells, h, w)`` array of
    those is hundreds of megabytes for an answer that is one plane.

    An empty list is ``+inf``, not an error: no outer rect means nothing to be
    outside of, which is exactly the interior-fill case, and ``+inf`` carries
    through the threshold to a saturated alpha of 1 on its own.
    """
    import numpy as np

    if not rects:
        return np.full(px.shape, np.inf, dtype=np.float64)
    best: Any = None
    for x0, y0, x1, y1 in rects:
        dx = np.maximum(np.maximum(x0 - px, 0.0), px - x1)
        dy = np.maximum(np.maximum(y0 - py, 0.0), py - y1)
        distance = np.hypot(dx, dy)
        best = distance if best is None else np.minimum(best, distance, out=best)
    return best


def signed_distance(
    member_rects: list[Rect], outer_rects: list[Rect], size: int
) -> Any:
    """``d_out - d_in`` over a ``(size, size)`` grid of pixel centres. float32.

    Positive inside the member region, negative outside it, and zero on the
    boundary between two rects that touch. ``size`` is the side of the pixel
    grid, which is ``T`` for one tile and the whole map's pixel side for a
    whole-map render -- the same function serves both, which is what lets the
    round-trip test compare them rather than compare two implementations.

    Both lists empty raises. There is no field to describe, and the honest
    arithmetic would be ``inf - inf``.
    """
    import numpy as np

    side = int(size)
    if side < 1:
        raise ValueError(f"a field is at least one pixel across; got {side}")
    if not member_rects and not outer_rects:
        raise ValueError("a field with no cells at all has no boundary to measure")
    py, px = _centres(side)
    return (_nearest(py, px, outer_rects) - _nearest(py, px, member_rects)).astype(
        np.float32
    )


def _require_room(size: int, inset: float, amplitude: float, feather: float) -> None:
    """The two preconditions, raised with the numbers that broke them.

    ``inset + amplitude + feather <= size/2`` is not a taste limit. It is the
    hypothesis of the locality argument in the module docstring: a cell outside
    the 3x3 block is at least ``size`` away, so if the whole ramp fits inside
    half a tile then alpha is already saturated there and one tile of the atlas
    is the same picture as the corresponding cell of the whole map. Relax it and
    the 47-column atlas silently stops being able to express the map.
    """
    side = int(size)
    if side < MIN_TILE:
        raise ValueError(
            f"a {side}px terrain tile is below the {MIN_TILE}px floor; at that "
            f"size the {BLOB_INSET_RATIO:g} inset is under 3px and inside and "
            f"outside stop reading as different roles"
        )
    if feather <= 0.0:
        raise ValueError(f"the feather is a ramp width and must be positive; got {feather:g}")
    if inset < 0.0 or amplitude < 0.0:
        raise ValueError(
            f"inset {inset:g} and amplitude {amplitude:g} are distances and "
            f"cannot be negative"
        )
    total = float(inset) + float(amplitude) + float(feather)
    if total > side / 2.0:
        raise ValueError(
            f"inset {inset:g} + amplitude {amplitude:g} + feather {feather:g} = "
            f"{total:g} does not fit in half of a {side}px tile ({side / 2.0:g}); "
            f"past that a cell two away can move this tile's coverage and the "
            f"47-column atlas no longer describes the map"
        )


def coverage(
    member_rects: list[Rect],
    outer_rects: list[Rect],
    size: int,
    *,
    noise: Any,
    inset: float,
    amplitude: float,
    feather: float,
) -> Any:
    """How much of each pixel is inner terrain. -> float32 ``(size, size)``, 0..1.

    The signed distance thresholded against ``inset + amplitude*noise`` and
    ramped over ``feather``. One documented return dtype -- float32, always --
    because two callers comparing this against each other must compare like with
    like, and a float64 return that a store rounded to float32 would make every
    such comparison disagree for no reason at all.

    ``+inf`` (no outer rects) saturates to 1 and ``-inf`` (no member rects) to 0,
    both by the clip, so the interior fill and the empty case need no branch.
    """
    import numpy as np

    side = int(size)
    _require_room(side, inset, amplitude, feather)
    field = np.asarray(noise, dtype=np.float32)
    if field.shape != (side, side):
        raise ValueError(
            f"the noise is {field.shape} and this field is ({side}, {side})"
        )
    distance = signed_distance(member_rects, outer_rects, side)
    threshold = np.float32(inset) + np.float32(amplitude) * field
    ramped = (distance - threshold) / np.float32(feather) + np.float32(0.5)
    return np.clip(ramped, 0.0, 1.0).astype(np.float32)


def _resolve(
    size: int, inset: float | None, amplitude: float | None, feather: float | None
) -> tuple[float, float, float]:
    """The three distances in pixels, from the ratios when not given.

    Ratios rather than pixels by default, so the boundary is the same *shape* at
    every tile size -- an absolute 6px inset is a coastline at 32px and a solid
    band at 16px. A caller that passes pixels means pixels.
    """
    side = int(size)
    return (
        float(BLOB_INSET_RATIO * side if inset is None else inset),
        float(NOISE_AMPLITUDE_RATIO * side if amplitude is None else amplitude),
        float(FEATHER_RATIO * side if feather is None else feather),
    )


def blob_coverages(
    size: int,
    *,
    seed: int,
    inset: float | None = None,
    amplitude: float | None = None,
    feather: float | None = None,
) -> Any:
    """All 47 coverage masks. -> float32 ``(47, size, size)``, in atlas order.

    **One noise field serves every case**, and that is not an optimisation. Two
    tiles that meet must agree about the boundary pixel by pixel, and they only
    do so if the same noise value stands at the same ``(u, v)`` in both -- which
    is the same argument that makes the materials line up, applied to the mask.
    Per-case noise would produce 47 individually pretty tiles that do not join.
    """
    import numpy as np

    side = int(size)
    inset, amplitude, feather = _resolve(side, inset, amplitude, feather)
    _require_room(side, inset, amplitude, feather)
    noise = wrap_noise(side, seed=seed)
    out = np.empty((TILE_COUNT, side, side), dtype=np.float32)
    for index, mask in enumerate(BLOB_MASKS):
        member, outer = blob_rects(mask, side)
        out[index] = coverage(
            member,
            outer,
            side,
            noise=noise,
            inset=inset,
            amplitude=amplitude,
            feather=feather,
        )
    return out


def composite(inner: Any, outer: Any, alpha: Any) -> Any:
    """A over B through the mask. -> uint8 ``(h, w, 4)``.

    A straight lerp of all four channels, not Porter-Duff ``over``. Both
    operands are seamless *materials* -- opaque tiles of ground -- and the alpha
    here is a coverage mask rather than either picture's own transparency, so
    the answer at coverage ``a`` is the material that is there ``a`` of the
    time. Lerping the alpha channel too means an inner material that is itself
    partly transparent stays partly transparent instead of being made opaque by
    a mask that never claimed to.

    Round-half-up on an integer-valued lerp, so the ends are exact: at ``a = 0``
    the result is ``outer`` bit for bit and at ``a = 1`` it is ``inner``. The
    painter blits the outer material into a non-member cell without calling this
    at all, and those two paths have to agree.
    """
    import numpy as np

    front = np.asarray(inner, dtype=np.uint8)
    back = np.asarray(outer, dtype=np.uint8)
    if front.ndim != 3 or front.shape[2] != 4:
        raise ValueError("an inner material must be RGBA, shaped (h, w, 4)")
    if back.shape != front.shape:
        raise ValueError(
            f"the two materials are {front.shape} and {back.shape}; they must match"
        )
    mask = np.asarray(alpha, dtype=np.float32)
    if mask.shape != front.shape[:2]:
        raise ValueError(
            f"the coverage is {mask.shape} and the materials are {front.shape[:2]}"
        )
    front_f = front.astype(np.float32)
    back_f = back.astype(np.float32)
    mixed = back_f + (front_f - back_f) * mask[:, :, None]
    return np.clip(mixed + 0.5, 0.0, 255.0).astype(np.uint8)


def blob_atlas(
    inner: Any,
    outer: Any,
    size: int,
    *,
    seed: int,
    inset: float | None = None,
    amplitude: float | None = None,
    feather: float | None = None,
) -> Any:
    """One terrain's 47 tiles. -> uint8 ``(size, 47*size, 4)``.

    **Forty-seven columns by one row, in ascending** :data:`BLOB_MASKS` **order.**
    That is the layout ``Tileset.__post_init__`` enforces (a terrain set is
    ``blob.TILE_COUNT`` columns wide, one blob case per column) and the one
    ``Tileset.local_for`` indexes with, so nothing downstream reorders anything
    -- a set of these rows stacked is already a terrain set's atlas.

    Both materials are ``(size, size, 4)`` uint8 and both are assumed seamless.
    Nothing here checks that -- ``seam.report`` is the thing that measures it,
    and it is advisory there for a reason -- but every claim in the module
    docstring is conditional on it.
    """
    import numpy as np

    side = int(size)
    front = np.asarray(inner, dtype=np.uint8)
    back = np.asarray(outer, dtype=np.uint8)
    for name, array in (("inner", front), ("outer", back)):
        if array.shape != (side, side, 4):
            raise ValueError(
                f"the {name} material is {array.shape} and a {side}px tile needs "
                f"({side}, {side}, 4)"
            )
    masks = blob_coverages(
        side, seed=seed, inset=inset, amplitude=amplitude, feather=feather
    )
    atlas = np.empty((side, TILE_COUNT * side, 4), dtype=np.uint8)
    for index in range(TILE_COUNT):
        atlas[:, index * side : (index + 1) * side] = composite(
            front, back, masks[index]
        )
    return atlas
