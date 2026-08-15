"""Where every sprite lands: the two modes, and the settings that decide.

**Grid** puts every sprite in a uniform cell the size of the largest trimmed
sprite, row-major in key order. It is the mode that produces a *tileset* -- a
regular atlas an engine can slice by arithmetic -- which is why the grid's
geometry is spelled as Tiled's own margin and spacing: the outer border and the
gutter are both ``padding``, so :mod:`.tsxout` can hand the numbers straight to
``plotter.tileset.Tileset`` rather than approximating them.

**MaxRects** packs tightly and irregularly, which is what an atlas for
individually-addressed sprites wants. Its size search is deterministic and
stated: the smallest power-of-two square whose area covers the total, then
double the shorter side until it fits or the ceiling is reached.

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
from typing import Any

from ...pipelines import sheet as sheetlib
from .maxrects import order, pack
from .sources import Sprite
from .trim import trim_rect

MODES = ("grid", "maxrects")

DEFAULT_PADDING = 2
DEFAULT_MAX_SIZE = 2048

# A gutter wider than the largest atlas tile is not a gutter, and the sliders
# offer 16 and 8 -- so this refuses only a hand-edited manifest, where a padding
# of a billion is a grid whose first cell is past the size ceiling and whose
# arithmetic overflows nothing but the user's patience.
MAX_PADDING = 256

# The most sprites one atlas may hold. MaxRects is quadratic in the free-rect
# list, so a document claiming a million sources is not a slow pack but a hung
# frame thread; 4096 is well past any real sheet (a 64x64 grid at the 8192px
# ceiling) and short of where the search stops answering.
MAX_SPRITES = 4096


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
    power_of_two: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
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
    slices: tuple[Any, ...] = ()

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


def _measure(sprites: list[Sprite], settings: PackSettings) -> list[_Measured]:
    """Every sprite's packing rectangle, in canonical key order.

    Sorted here rather than left to the caller because the *order is part of
    the layout*: grid mode places row-major in it, and maxrects sorts from it.
    A caller's list order comes from a document the user reorders, which is not
    an order a file format should depend on.

    The sprite-count ceiling is asked here rather than in either packer, which
    is what makes it one door: every caller of either mode comes through this.
    """
    if len(sprites) > MAX_SPRITES:
        raise ValueError(
            f"this pack holds {len(sprites)} sprites; {MAX_SPRITES} is the most one "
            "atlas will take -- split it into several packs"
        )
    out = []
    for sprite in sorted(sprites, key=lambda s: s.key):
        x, y, w, h, empty = trim_rect(sprite.pixels, enabled=settings.trim)
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
    """
    entries = _measure(sprites, settings)
    if not entries:
        raise ValueError("there is nothing to pack")

    pad = settings.padding
    cell_w = max(entry.w for entry in entries)
    cell_h = max(entry.h for entry in entries)
    step_w, step_h = cell_w + pad, cell_h + pad

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
    # And rows from the final height, for the same reason columns came from the
    # final width: ``columns``/``rows`` here have to be the grid a ``.tsx``
    # reader derives from the image, not the number of cells that happen to be
    # occupied. Trailing cells are legitimately empty -- a tileset with blank
    # tiles at the end is ordinary -- but a layout and its own tileset
    # disagreeing about the shape of the grid is not.
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
