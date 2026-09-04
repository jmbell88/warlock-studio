"""Where every sprite lands: the two modes, and the settings that decide.

**Grid** puts every sprite in a uniform cell the size of the largest sprite,
row-major in key order. It is the mode that produces a *tileset* -- a regular
atlas an engine can slice by arithmetic -- which is why the grid's geometry is
spelled as Tiled's own margin and spacing: the outer border and the gutter are
both ``padding``, so :mod:`.tsxout` can hand the numbers straight to
``tilegrid.tileset.Tileset`` rather than approximating them.

**A grid pack never trims, whatever ``trim`` says.** Trimming moves each
sprite's content to its own bounding box, and in a grid it was then blitted at
the *cell's* top-left -- so a 16px tile whose art sat four pixels in came out
four pixels up and left of where it belongs, and every tile was re-registered
to a different origin from its neighbours. The atlas was smaller and the
tileset was wrong: a `.tsx` slices by arithmetic and cannot know that cell 7
was nudged. The two things the setting could mean here -- "make the atlas
smaller" and "keep the tiles aligned" -- cannot both be had, and a mode whose
entire purpose is arithmetic alignment answers to the second (the 2026-09-02
review, section 7). MaxRects still trims: nothing there is addressed by
position, and the sidecar records each frame's ``trim`` for the consumer that
wants the original box back.

**MaxRects** packs tightly and irregularly, which is what an atlas for
individually-addressed sprites wants. Its size search is deterministic and
stated: the smallest power-of-two square whose area covers the total, then
double the shorter side until it fits or the ceiling is reached.

**Power-of-two defaults to off for a grid pack, on for MaxRects.** A grid's
cells are a fixed size regardless of the atlas around them, so rounding the
atlas up buys nothing but dead space past the last column/row -- and near a
size boundary that dead space is the whole atlas again: measured over a sweep
of sprite counts and cell sizes, 1.61x the tight area on average, 3.65x worst
case. MaxRects keeps rounding up by default because its atlas has no unused
margin to speak of and pow2 is still what most engines expect loading a
non-tileset texture. See :class:`PackSettings.power_of_two`.

``padding >= extrude * 2`` is validated here and it is not arbitrary. Extrude
replicates each sprite's border pixels outward into the gutter so a filtered
texture cannot sample its neighbour; two adjacent sprites each extruding into a
shared gutter therefore need twice the room. Getting this wrong produces bleed
that only shows on a GPU with filtering on, at some zoom levels.

``max_size`` is ceilinged by ``pipelines.sheet.MAX_ATLAS_PX`` -- imported, not
restated -- because "how big may an atlas be before an engine refuses it" is one
question and it already has an answer here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...pipelines import sheet as sheetlib
from .maxrects import order, pack
from .sources import SliceSpec, Sprite
from .trim import trim_rect

MODES = ("grid", "maxrects")

#: The two TexturePacker JSON shapes :mod:`.texturepacker` can write. Named
#: here rather than in that module because :class:`PackSettings` validates
#: against it and :mod:`.texturepacker` already imports from this module, not
#: the other way round -- a second copy of the tuple there would be the two
#: places one of them is free to drift from the other.
JSON_SCHEMAS = ("array", "hash")

DEFAULT_PADDING = 2
DEFAULT_MAX_SIZE = 2048

# A gutter wider than the largest atlas tile is not a gutter, and the sliders
# offer 16 and 8 -- so this refuses only a hand-edited manifest, where a padding
# of a billion is a grid whose first cell is past the size ceiling and whose
# arithmetic overflows nothing but the user's patience.
MAX_PADDING = 256

# The most sprites one atlas may hold. MaxRects is super-quadratic in practice
# -- `_prune` is O(F^2) per placement and the free-rect count F grows with the
# items placed -- so a document claiming a million sources is not a slow pack
# but one that never returns.
#
# **1024, measured.** This said 4096 and called it "short of where the search
# stops answering"; that half was never measured and was wrong. One pack of
# 4096 random 8-64px items takes 190 seconds on the reference machine, and
# `maxrects_layout` can call pack once per candidate size. 1024 is the largest
# count whose single pack stays under five seconds. See
# docs/measurements/2026-08-31-packwright-max-sprites.md for the table and for
# what would lift it again (bucketing the free-rect list so neither `_score`
# nor `_prune` scans all of it).
MAX_SPRITES = 1024


def next_pot(value: int) -> int:
    """The smallest power of two at or above ``value``. ``1`` for anything
    below it, since a zero-sized texture is not a texture."""
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


@dataclass(frozen=True)
class PackSettings:
    mode: str = "grid"
    padding: int = DEFAULT_PADDING
    extrude: int = 0
    trim: bool = True
    max_size: int = DEFAULT_MAX_SIZE
    #: ``None`` is the mode-resolved default, applied in ``__post_init__``:
    #: ``False`` for a grid pack, ``True`` for MaxRects (unchanged). A grid's
    #: cells are already a fixed size -- there is nothing "tighter" about
    #: rounding one up, only dead space past the last column/row, and it is
    #: what silently doubles a grid atlas near a size boundary (measured:
    #: 1.61x mean, 3.65x worst case, over the same (count, cell) sweep
    #: :func:`grid_layout` documents). MaxRects keeps rounding up by default
    #: because its whole atlas is used corner to corner and pow2 is what an
    #: engine loading it into a non-addressed texture slot usually still
    #: wants. Passing an explicit ``True``/``False`` always wins -- this is a
    #: default, not a per-mode override that fights the user. The sentinel
    #: only resolves *once*, at construction: once a document holds a
    #: settings object the field is a plain bool, and ``dataclasses.replace``
    #: would otherwise carry a grid pack's resolved ``False`` straight onto
    #: a MaxRects switch. ``document.py::set_settings`` is what re-arms the
    #: sentinel on a bare mode change -- see it for the mode-switch case.
    power_of_two: bool | None = None
    #: Trailing and defaulted, so an older ``.wpack`` -- or a caller that never
    #: heard of it -- keeps auto behaviour. Grid mode only: a MaxRects pack has
    #: no uniform cell for a column count to describe, so setting one there is
    #: refused by name rather than silently ignored -- the same taste every
    #: other combination on this dataclass is held to (``padding``/``extrude``,
    #: ``max_size``). ``None`` is auto -- the near-square search below.
    columns: int | None = None
    #: Which TexturePacker JSON shape :func:`export_files` writes. Settings-
    #: level rather than an argument to the export call, the same seam
    #: ``mode``/``trim``/``power_of_two`` already use: it is a per-document
    #: choice that belongs beside them in the ``.wpack`` and in undo, not a
    #: one-off passed at the moment of export and forgotten. ``"array"`` is
    #: what every export wrote before this field existed, so it stays default.
    json_schema: str = "array"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if self.power_of_two is None:
            # Resolved once, here, rather than read as ``None`` downstream:
            # every consumer of ``power_of_two`` -- ``grid_layout``,
            # ``maxrects_layout``, the settings pane's toggle -- wants a
            # plain bool, and a sentinel that leaked past construction would
            # be a second "is it set" question every one of them would have
            # to ask.
            object.__setattr__(self, "power_of_two", self.mode != "grid")
        if self.padding < 0 or self.extrude < 0:
            raise ValueError("padding and extrude cannot be negative")
        if self.padding > MAX_PADDING:
            raise ValueError(f"padding must be at most {MAX_PADDING}px")
        if self.extrude > MAX_PADDING:
            raise ValueError(f"extrude must be at most {MAX_PADDING}px")
        if self.padding < self.extrude * 2:
            raise ValueError(
                f"padding must be at least twice extrude "
                f"({self.extrude} x 2 = {self.extrude * 2}, padding is {self.padding}) "
                "-- two neighbours extrude into one gutter"
            )
        if self.max_size < 1:
            raise ValueError("max size must be positive")
        if self.max_size > sheetlib.MAX_ATLAS_PX:
            raise ValueError(
                f"max size must be at most {sheetlib.MAX_ATLAS_PX}px, "
                "which is where engines start refusing a texture outright"
            )
        if self.columns is not None:
            if self.columns < 1:
                raise ValueError("columns must be at least 1")
            if self.mode != "grid":
                raise ValueError(
                    "columns only applies to a grid pack -- a MaxRects pack has "
                    "no uniform cell for a column count to describe"
                )
        if self.json_schema not in JSON_SCHEMAS:
            raise ValueError(f"json_schema must be one of {list(JSON_SCHEMAS)}")


@dataclass(frozen=True)
class Frame:
    """One sprite's place in the atlas.

    ``x``/``y``/``w``/``h`` is the rectangle in the atlas and is always the
    *trimmed* size. ``trim`` is where that rectangle sat inside the original
    image, and ``source_w``/``source_h`` is what the original was -- which is
    what lets a consumer put the sprite back where the artist drew it rather
    than flush against its own bounding box.
    """

    key: str
    name: str
    x: int
    y: int
    w: int
    h: int
    trim: tuple[int, int, int, int]
    source_w: int
    source_h: int
    empty: bool = False
    #: The sprite's metadata, carried through so a sidecar writer reads one
    #: object. Trailing and defaulted, and neither is read by any packer: the
    #: layout is decided by rectangles alone, so determinism is untouched by
    #: what a sprite happens to be called or where its pivot is.
    pivot: tuple[float, float] | None = None
    slices: tuple[SliceSpec, ...] = ()

    @property
    def trimmed(self) -> bool:
        """Whether anything was actually cut off. Reported rather than assumed:
        a sprite that fills its own canvas is untrimmed even with trimming on,
        and saying otherwise makes every consumer do offset arithmetic for a
        zero offset."""
        return self.trim != (0, 0, self.source_w, self.source_h)


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    mode: str
    padding: int
    extrude: int
    frames: tuple[Frame, ...] = ()
    # Grid mode only; zero in maxrects, where there is no cell.
    cell_w: int = 0
    cell_h: int = 0
    columns: int = 0
    rows: int = 0

    @property
    def is_grid(self) -> bool:
        return self.mode == "grid" and self.cell_w > 0 and self.cell_h > 0

    def frame(self, key: str) -> Frame | None:
        for entry in self.frames:
            if entry.key == key:
                return entry
        return None


@dataclass(frozen=True)
class _Measured:
    sprite: Sprite
    trim: tuple[int, int, int, int]
    empty: bool = field(default=False)

    @property
    def w(self) -> int:
        return self.trim[2]

    @property
    def h(self) -> int:
        return self.trim[3]


def _measure(
    sprites: list[Sprite], settings: PackSettings, *, trim: bool | None = None
) -> list[_Measured]:
    """Every sprite's packing rectangle, in canonical key order.

    Sorted here rather than left to the caller because the *order is part of
    the layout*: grid mode places row-major in it, and maxrects sorts from it.
    A caller's list order comes from a document the user reorders, which is not
    an order a file format should depend on.

    The sprite-count ceiling is asked here rather than in either packer, which
    is what makes it one door: every caller of either mode comes through this.

    ``trim`` overrides the setting for a caller that has its own answer, which
    is :func:`grid_layout` and only it: a grid is addressed by arithmetic, so
    moving a sprite to its own bounding box would misregister the cell. The
    module docstring carries the argument.
    """
    if len(sprites) > MAX_SPRITES:
        raise ValueError(
            f"this pack holds {len(sprites)} sprites; {MAX_SPRITES} is the most one "
            "atlas will take -- split it into several packs"
        )
    out = []
    for sprite in sorted(sprites, key=lambda s: s.key):
        x, y, w, h, empty = trim_rect(
            sprite.pixels, enabled=settings.trim if trim is None else trim
        )
        out.append(_Measured(sprite=sprite, trim=(x, y, w, h), empty=empty))
    return out


def _frame(entry: _Measured, x: int, y: int) -> Frame:
    return Frame(
        key=entry.sprite.key,
        name=entry.sprite.name,
        x=x,
        y=y,
        w=entry.w,
        h=entry.h,
        trim=entry.trim,
        source_w=entry.sprite.width,
        source_h=entry.sprite.height,
        empty=entry.empty,
        pivot=entry.sprite.meta.pivot,
        slices=entry.sprite.meta.slices,
    )


# --- grid ---------------------------------------------------------------------


def grid_layout(sprites: list[Sprite], settings: PackSettings) -> Layout:
    """Uniform cells, row-major, with Tiled's margin-and-spacing geometry.

    The atlas is ``padding + columns * (cell + padding)`` across, so the outer
    border and every gutter are one ``padding`` -- which is exactly
    ``margin == spacing == padding`` in a ``.tsx``. That equivalence is the
    whole reason grid mode can emit a tileset at all.

    **Auto** (``settings.columns is None``) picks a near-square column count
    and, when ``power_of_two`` rounds the width up, *re-derives* columns from
    the rounded width -- the same span Tiled's own formula would compute from
    the finished image -- and rows from the rounded height the same way. That
    keeps this layout and :func:`~.tsxout.grid_tileset`'s independent read of
    the exported PNG in permanent agreement, at the cost of leaving the exact
    column count up to wherever the rounding lands.

    **Explicit** (``settings.columns`` set) is the opposite trade: a tileset
    author who asked for a specific column count -- to match an existing tile
    index, say -- gets exactly that count, never silently widened because
    rounding bought room for one more. The atlas may still round up for
    ``power_of_two``; the grid does not follow it there, so the rounded
    atlas can carry dead margin past the last column/row. That margin is
    honest padding-contract territory (the *content* area is still exactly
    ``padding + columns * (cell + padding)``) but it is no longer what Tiled's
    own formula would derive from the full image, so :func:`~.tsxout.grid_tileset`
    checks the two agree before it will write a ``.tsx`` -- and refuses,
    naming the mismatch, rather than emit one that slices wrong.
    """
    # ``trim=False``, always: see the module docstring. Passed as an override
    # rather than read off ``settings`` so the document keeps the user's answer
    # for the day they switch back to MaxRects.
    entries = _measure(sprites, settings, trim=False)
    if not entries:
        raise ValueError("there is nothing to pack")

    pad = settings.padding
    cell_w = max(entry.w for entry in entries)
    cell_h = max(entry.h for entry in entries)
    step_w, step_h = cell_w + pad, cell_h + pad

    if settings.columns is not None:
        columns = settings.columns
        width = pad + columns * step_w
        if settings.power_of_two:
            width = next_pot(width)
        rows = -(-len(entries) // columns)
        height = pad + rows * step_h
        if settings.power_of_two:
            height = next_pot(height)
    else:
        columns = max(1, math.isqrt(len(entries) - 1) + 1)
        width = pad + columns * step_w
        if settings.power_of_two:
            width = next_pot(width)
        # Re-derive the column count from the *final* width, so a power-of-two
        # rounding that bought room for another column is a column the .tsx and
        # this layout agree about -- rather than a strip of the atlas that a
        # tileset reader slices into tiles nothing here knows exist.
        columns = max(1, (width - pad) // step_w)
        rows = -(-len(entries) // columns)
        height = pad + rows * step_h
        if settings.power_of_two:
            height = next_pot(height)
        # And rows from the final height, for the same reason columns came from
        # the final width: ``columns``/``rows`` here have to be the grid a
        # ``.tsx`` reader derives from the image, not the number of cells that
        # happen to be occupied. Trailing cells are legitimately empty -- a
        # tileset with blank tiles at the end is ordinary -- but a layout and
        # its own tileset disagreeing about the shape of the grid is not.
        rows = max(1, (height - pad) // step_h)

    _check_size(width, height, settings)
    frames = tuple(
        _frame(entry, pad + (index % columns) * step_w, pad + (index // columns) * step_h)
        for index, entry in enumerate(entries)
    )
    return Layout(
        width=width,
        height=height,
        mode="grid",
        padding=pad,
        extrude=settings.extrude,
        frames=frames,
        cell_w=cell_w,
        cell_h=cell_h,
        columns=columns,
        rows=rows,
    )


# --- maxrects -----------------------------------------------------------------


def _candidate_sizes(area: int, floor_w: int, floor_h: int, limit: int) -> list[tuple[int, int]]:
    """The size search, written out so it can be read and tested.

    Start at the smallest power-of-two square whose area covers the total (and
    which is at least large enough for the biggest single item), then double
    the shorter side each time. Doubling the *shorter* side keeps the atlas
    near-square, which is what a GPU wants and what keeps the wasted corner
    small.

    **The limit itself is the last candidate**, and it has to be, because it
    need not be a power of two. With a 1500 px ceiling the doubling walked
    1024 and then 2048, which is past it -- so the loop ended and a set that
    fits in 1500 square was refused as "does not fit in a 1500px atlas". The
    search is over *working* sizes rather than over final ones either way: a
    pack that is not power-of-two shrinks to its used extent afterwards, so an
    oversized candidate costs nothing but the attempt.
    """
    side = max(next_pot(math.isqrt(max(area - 1, 0)) + 1), next_pot(floor_w), next_pot(floor_h))
    sizes: list[tuple[int, int]] = []
    width, height = side, side
    while width <= limit and height <= limit:
        sizes.append((width, height))
        if width <= height:
            width *= 2
        else:
            height *= 2
    if limit >= max(floor_w, floor_h) and (limit, limit) not in sizes:
        sizes.append((limit, limit))
    return sizes


def maxrects_layout(sprites: list[Sprite], settings: PackSettings) -> Layout:
    entries = _measure(sprites, settings)
    if not entries:
        raise ValueError("there is nothing to pack")

    pad = settings.padding
    by_key = {entry.sprite.key: entry for entry in entries}
    # Each item carries its own gutter, and the whole pack is then offset by
    # one padding -- so a sprite has ``padding`` free on every side, whether its
    # neighbour is another sprite or the edge of the atlas. Extrude needs that
    # to be true at the edges too, which is what an un-offset pack gets wrong.
    items = order([(entry.sprite.key, entry.w + pad, entry.h + pad) for entry in entries])
    area = sum(w * h for _key, w, h in items)
    floor_w = max(w for _key, w, _h in items)
    floor_h = max(h for _key, _w, h in items)

    for width, height in _candidate_sizes(area, floor_w + pad, floor_h + pad, settings.max_size):
        placed = pack(items, width - pad, height - pad)
        if placed is None:
            continue
        frames = tuple(
            _frame(by_key[p.key], p.x + pad, p.y + pad)
            for p in sorted(placed, key=lambda p: p.key)
        )
        if not settings.power_of_two:
            width = max(f.x + f.w for f in frames) + pad
            height = max(f.y + f.h for f in frames) + pad
        _check_size(width, height, settings)
        return Layout(
            width=width,
            height=height,
            mode="maxrects",
            padding=pad,
            extrude=settings.extrude,
            frames=frames,
        )

    raise ValueError(
        f"these {len(entries)} sprites do not fit in a {settings.max_size}px atlas "
        "-- raise the max size, trim them, or split the pack"
    )


def _check_size(width: int, height: int, settings: PackSettings) -> None:
    if max(width, height) > settings.max_size:
        raise ValueError(
            f"that atlas would be {width}x{height}px; this pack's limit is "
            f"{settings.max_size}px -- raise the max size or use fewer sprites"
        )
    # And the engine ceiling on top, which ``PackSettings`` already bounds
    # ``max_size`` by. Belt and braces, but the message is the useful one and it
    # is the sprite-sheet pipeline's own.
    sheetlib.check_atlas_size(width, height)


def layout(sprites: list[Sprite], settings: PackSettings) -> Layout:
    """The dispatch. One entry point, so a caller never picks the packer."""
    if settings.mode == "grid":
        return grid_layout(sprites, settings)
    return maxrects_layout(sprites, settings)
