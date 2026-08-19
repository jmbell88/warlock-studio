"""A tileset: one image, sliced into a grid.

Frozen, and its pixels are copied and made read-only on construction, for the
reason ``clay.mesh.Mesh`` is immutable -- the UI invalidates its GPU texture
upload only when the document's tileset list changes (``tileset_epoch``), so an
in-place edit anywhere would move no counter, leave the cache holding stale
pixels, and the map would render last week's tiles forever with nothing in the
data to say why.

**The slicing is validated here, not at the first draw.** A geometry that does
not fit the image -- a margin larger than the image, a spacing that leaves a
partial final column -- is a file or a form the user got wrong, and the useful
moment to say so is when the tileset is made. Deferring it produces a tileset
that exists, appears in the list, and draws garbage.

Only *image* tilesets exist here. Tiled's "collection of images" variant, where
each tile is its own file, is refused by :mod:`.tsx` and :mod:`.tmx` rather than
half-supported: the whole model below assumes one atlas and a tile id that is an
index into its grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import blob

RGBA = tuple[int, int, int, int]


def rgba_colour(colour: Any, what: str) -> RGBA:
    """Four channels of 0..255 as the hashable tuple every colour here is.

    Public for :func:`frozen_rgba`'s reason and no other: a second caller
    turned up. :class:`~._map_model.TileLayer` and its three siblings each
    carry a ``tint``, and a private copy of these four lines beside them is how
    one of the two comes to accept a channel the other refuses. ``what`` names
    the thing in the refusal, because "a terrain fill must be four channels" is
    the wrong sentence to show someone tinting a layer.
    """
    values = tuple(int(part) for part in colour)
    if len(values) != 4 or any(part < 0 or part > 255 for part in values):
        raise ValueError(f"{what} must be four channels of 0..255")
    return values  # type: ignore[return-value]


def colour_text(value: Any, what: str) -> str | None:
    """Tiled's ``#RRGGBB``/``#AARRGGBB`` *spelling*, validated. ``None`` if unset.

    :func:`rgba_colour`'s sibling for the two fields that are stored as Tiled's
    own text rather than as channels -- a map's ``backgroundcolor`` and an
    object layer's ``color``. They are kept as text because they are round-trip
    values nothing in the editor computes with, and the cost of that is exactly
    this: without a check at the setter, an arbitrary string reaches the TMX
    writer verbatim and produces a file Tiled cannot parse, from an editor that
    had no complaint. ``backgroundcolor`` grew this check inline; ``color`` had
    none at all, which is what a second private copy always turns into.
    """
    if value in (None, ""):
        return None
    text = str(value)
    digits = text[1:] if text.startswith("#") else ""
    if len(digits) not in (6, 8):
        raise ValueError(f"{what} is #RRGGBB or #AARRGGBB")
    try:
        int(digits, 16)
    except ValueError as exc:
        raise ValueError(f"{what} is #RRGGBB or #AARRGGBB") from exc
    return text


# --- per-tile metadata -------------------------------------------------------
#
# Six of ``docs/PLOTTER_COMPAT.md``'s refused rows fall to one model: a tile can
# carry a class, custom properties, a random-pick probability, an animation and
# a set of collision shapes. Stored **sparsely** -- Tiled's own shape, and the
# right one here because most tiles in an atlas carry nothing at all and a dense
# table would be one empty record per tile per tileset.
#
# The collision shapes are **tilegrid's own frozen records**, small and dumb, and
# not ``plotter._map_model``'s. This package imports nothing under ``warlock``
# and a shared leaf that reached into one of the engines it serves would turn
# "shared vocabulary" into a dependency cycle; the plotter converts to and from
# its own shapes at the codec instead. Packwright and Inker see only additive
# fields and never these types.


@dataclass(frozen=True)
class TileRect:
    """A rectangle of collision, in pixels within the tile."""

    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


@dataclass(frozen=True)
class TileEllipse:
    """An ellipse inscribed in the same box a :class:`TileRect` describes."""

    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


@dataclass(frozen=True)
class TilePolygon:
    """A closed outline, as points relative to ``(x, y)``."""

    x: float = 0.0
    y: float = 0.0
    points: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points",
            tuple((float(px), float(py)) for px, py in self.points),
        )


#: Every collision shape, for an ``isinstance`` that cannot go stale.
COLLISION_SHAPES = (TileRect, TileEllipse, TilePolygon)


@dataclass(frozen=True)
class TileFrame:
    """One frame of a tile's animation: which local id, and for how long."""

    local_id: int = 0
    duration_ms: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_id", int(self.local_id))
        object.__setattr__(self, "duration_ms", max(1, int(self.duration_ms)))


@dataclass(frozen=True)
class TileMeta:
    """Everything one tile carries beyond its picture.

    ``probability`` is the weight a *random* brush gives this tile, Tiled's own
    field. **Zero means never chosen at random and always placeable by hand** --
    Tiled's rule, and the useful one: it is how a set marks a tile that belongs
    to the palette but not to the scatter.

    ``animation`` is ordered and its frames name *local* ids within this
    tileset, which is what makes a tileset self-contained: a firstgid is a
    property of the map that loaded it, not of the atlas.
    """

    class_name: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    animation: tuple[TileFrame, ...] = ()
    collision: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_name", str(self.class_name))
        object.__setattr__(self, "properties", dict(self.properties))
        object.__setattr__(self, "probability", float(self.probability))
        object.__setattr__(self, "animation", tuple(self.animation))
        object.__setattr__(self, "collision", tuple(self.collision))
        if self.probability < 0.0:
            raise ValueError("a tile probability cannot be negative")
        for shape in self.collision:
            if not isinstance(shape, COLLISION_SHAPES):
                raise ValueError(f"{shape!r} is not a tile collision shape")

    @property
    def is_empty(self) -> bool:
        """Whether this record says nothing, so a writer can drop it.

        The sparse store's own rule: a tile whose metadata is all defaults is a
        tile with no metadata, and keeping the record would make a round trip
        grow a ``<tile>`` element every time somebody opened a class field and
        closed it again.
        """
        return (
            not self.class_name
            and not self.properties
            and self.probability == 1.0
            and not self.animation
            and not self.collision
        )


@dataclass(frozen=True)
class TerrainSpec:
    """One terrain a tileset declares: what it is called and what it is made of.

    Here rather than in :mod:`.terrain` because it is a *field* of
    :class:`Tileset` -- the roles are a fact about the atlas, not about any map
    that loads it, so the same generated set embedded in two ``.wmap`` files
    carries them in both. Putting it next door would make the two modules import
    each other.

    **A terrain's position in the tuple is its precedence**, which is the whole
    of how a cell with three terrains around it picks one picture: a cell's blob
    membership is every neighbour ranked at or above its own. So this is a
    tuple, and every serialisation of it is an ordered list.
    """

    name: str
    fill: RGBA
    outline: RGBA

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "fill", rgba_colour(self.fill, "a terrain fill"))
        object.__setattr__(self, "outline", rgba_colour(self.outline, "a terrain outline"))


def frozen_rgba(pixels: Any, what: str = "a tileset image") -> np.ndarray:
    """A private RGBA copy nothing can write through.

    Public, and shared with ``packwright.sources``, which held a byte-identical
    copy differing only in the noun in its error message. The rule is one rule:
    the UI keys a texture upload on the array's identity, so an in-place edit
    would leave the cache holding a live key over stale pixels -- and two
    spellings of it is how one of them comes to be relaxed alone.

    ``what`` names the thing in the refusal, because "a tileset image must be
    RGBA" is the wrong sentence to show someone packing a sprite.
    """
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"{what} must be RGBA, shaped (h, w, 4)")
    array = array.copy()
    array.setflags(write=False)
    return array


#: Tiled's nine anchors plus its "let the projection decide" default. Written
#: out rather than derived, for ``TOOLS``' reason: the validator and every
#: writer read one list, so a tenth value cannot be accepted by one and refused
#: by the other.
OBJECT_ALIGNMENTS = (
    "unspecified",
    "topleft",
    "top",
    "topright",
    "left",
    "center",
    "right",
    "bottomleft",
    "bottom",
    "bottomright",
)

#: Whether an oversized tile draws at its own size or is fitted to the cell.
RENDER_SIZES = ("tile", "grid")

#: How a fitted tile fills the cell it was fitted to.
FILL_MODES = ("stretch", "preserve-aspect-fit")


@dataclass(frozen=True)
class Tileset:
    """One sliced atlas. ``name`` is what the map calls it and what a ``.tsx``
    is written under; it is not an identity -- two tilesets may share a name and
    the map distinguishes them by position."""

    name: str
    pixels: np.ndarray
    tile_w: int
    tile_h: int
    spacing: int = 0
    margin: int = 0
    # Tiled's custom class name for the tileset itself. This is independent of
    # per-tile classes (which remain a named refusal at the reader door).
    class_name: str = ""
    # Coordinate grid used when authoring per-tile collision objects. It is
    # still document metadata when an atlas carries no collision objects.
    grid_orientation: str = "orthogonal"
    grid_width: int = 0
    grid_height: int = 0
    # hflip, vflip, rotate, prefer-untransformed: the choices a terrain/random
    # brush is allowed to derive from this tileset.
    transformations: tuple[bool, bool, bool, bool] = (False, False, False, False)
    # Preserved verbatim from a .tsx so a round trip does not quietly drop it.
    properties: dict[str, Any] = field(default_factory=dict)
    # Empty for an ordinary atlas. Non-empty makes this a *terrain set*: one row
    # per terrain, one column per blob case, so a tile's role is its position.
    terrains: tuple[TerrainSpec, ...] = ()
    # Phase variants per axis for a terrain set: each terrain owns phases**2
    # consecutive sub-rows, row-major by phase, and the painter places phase
    # (x mod k, y mod k) at cell (x, y). 1 is the classic single-row set.
    phases: int = 1
    # Per-tile metadata by local id, sparse: most tiles carry nothing.
    tiles: dict[int, TileMeta] = field(default_factory=dict)
    # Foreign Wang sets, kept as data.
    #
    # **Beside ``terrains``, not instead of it.** ``terrains`` is the *blob
    # preset*: a positional 47-column layout whose bar is byte-identity on the
    # whole existing corpus, and which ``terrain_of``/``local_for`` read a cell's
    # role off directly. A set that is not that preset -- corner-only,
    # edge-only, or mixed with a table of its own -- is recognised as data and
    # painted by constraint matching instead. A tileset carrying one is not
    # thereby a terrain set: ``is_terrain_set`` still means the preset.
    wangsets: tuple[Any, ...] = ()
    # --- presentation ---------------------------------------------------------
    #
    # Four fields that change how a tile is *drawn* and nothing about which tile
    # it is. Every one of them is Tiled's, read, written, stored and rendered.
    #
    # ``offset`` is a draw displacement in pixels, which is how a tall tree tile
    # is anchored to the cell its trunk stands in. The **minimap ignores it**, by
    # the one-pixel-per-cell rule already stated for layer offsets: at that scale
    # a sub-cell displacement has nowhere to go.
    offset_x: int = 0
    offset_y: int = 0
    # Where a *tile object* is anchored to its own position. "unspecified" is
    # Tiled's own default and means the map's projection decides -- bottom-left
    # for orthogonal, bottom-centre for isometric -- which is what
    # ``project.object_to_pixels`` already computes.
    object_alignment: str = "unspecified"
    # How an oversized tile is fitted to the grid: "tile" draws it at its own
    # size (so a 48px tree on a 32px map overhangs, anchored by the rule above),
    # "grid" fits it to the cell. ``fill_mode`` decides whether that fit
    # stretches or preserves the aspect.
    render_size: str = "tile"
    fill_mode: str = "stretch"
    # The palette's own backdrop, as Tiled's ``#RRGGBB`` text. Presentation
    # only, and preserved on the trip.
    background: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixels", frozen_rgba(self.pixels))
        object.__setattr__(self, "class_name", str(self.class_name))
        for name in ("tile_w", "tile_h", "spacing", "margin"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "terrains", tuple(self.terrains))
        object.__setattr__(self, "wangsets", tuple(self.wangsets))
        # Empty records dropped on the way in, so "is there metadata on tile 7"
        # has one answer and a round trip cannot grow elements nobody authored.
        object.__setattr__(
            self,
            "tiles",
            {
                int(local): meta
                for local, meta in dict(self.tiles).items()
                if not meta.is_empty
            },
        )
        if self.tile_w < 1 or self.tile_h < 1:
            raise ValueError("a tile must be at least one pixel across")
        object.__setattr__(self, "grid_orientation", str(self.grid_orientation))
        object.__setattr__(self, "grid_width", int(self.grid_width or self.tile_w))
        object.__setattr__(self, "grid_height", int(self.grid_height or self.tile_h))
        if self.grid_orientation not in ("orthogonal", "isometric"):
            raise ValueError("a tileset object grid is orthogonal or isometric")
        for name in ("offset_x", "offset_y"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "object_alignment", str(self.object_alignment))
        object.__setattr__(self, "render_size", str(self.render_size))
        object.__setattr__(self, "fill_mode", str(self.fill_mode))
        object.__setattr__(
            self, "background", colour_text(self.background, "a tileset background")
        )
        if self.object_alignment not in OBJECT_ALIGNMENTS:
            raise ValueError(
                f"a tileset object alignment is one of {list(OBJECT_ALIGNMENTS)}"
            )
        if self.render_size not in RENDER_SIZES:
            raise ValueError(f"a tileset render size is one of {list(RENDER_SIZES)}")
        if self.fill_mode not in FILL_MODES:
            raise ValueError(f"a tileset fill mode is one of {list(FILL_MODES)}")
        if self.grid_width < 1 or self.grid_height < 1:
            raise ValueError("a tileset object grid must be at least one pixel across")
        transforms = tuple(bool(value) for value in self.transformations)
        if len(transforms) != 4:
            raise ValueError("tileset transformations hold four flags")
        object.__setattr__(self, "transformations", transforms)
        if self.spacing < 0 or self.margin < 0:
            raise ValueError("spacing and margin cannot be negative")
        if self.columns < 1 or self.rows < 1:
            raise ValueError(
                f"a {self.image_w}x{self.image_h} image holds no "
                f"{self.tile_w}x{self.tile_h} tiles at margin {self.margin}, "
                f"spacing {self.spacing}"
            )
        # Validated here for the reason the slicing above is: a terrain set
        # whose geometry does not match its declaration is a file or a form
        # somebody got wrong, and the useful moment to say so is now rather than
        # when a paint stroke indexes past the end of a row.
        object.__setattr__(self, "phases", int(self.phases))
        if self.phases not in (1, 2, 4):
            raise ValueError("a terrain set's phase count per axis is 1, 2 or 4")
        if self.terrains:
            if self.columns != blob.TILE_COUNT:
                raise ValueError(
                    f"a terrain set is {blob.TILE_COUNT} columns wide, one blob case "
                    f"per column; this one is {self.columns}"
                )
            needed = len(self.terrains) * self.phases * self.phases
            if self.rows < needed:
                raise ValueError(
                    f"{len(self.terrains)} terrains at {self.phases} phases need "
                    f"{needed} rows, and this image holds {self.rows}"
                )

    # -- geometry ------------------------------------------------------------

    @property
    def image_w(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def image_h(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def columns(self) -> int:
        """How many whole tiles fit across.

        Tiled's own formula: the margin is taken off both edges, and each tile
        after the first costs its own width plus one spacing. A partial tile at
        the end is not counted -- it is not a tile.
        """
        span = self.image_w - 2 * self.margin + self.spacing
        step = self.tile_w + self.spacing
        return max(0, span // step) if step else 0

    @property
    def rows(self) -> int:
        span = self.image_h - 2 * self.margin + self.spacing
        step = self.tile_h + self.spacing
        return max(0, span // step) if step else 0

    @property
    def tile_count(self) -> int:
        return self.columns * self.rows

    def tile_rect(self, local_id: int) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` of one tile within the image, by local id.

        Local ids run row-major from zero, which is what ``firstgid`` is added
        to. Out of range raises: a caller asking for a tile this set does not
        have is a bug upstream, and returning the first tile would draw a
        plausible wrong picture.
        """
        index = int(local_id)
        if index < 0 or index >= self.tile_count:
            raise IndexError(f"tile {index} is outside this tileset (0..{self.tile_count - 1})")
        column, row = index % self.columns, index // self.columns
        return (
            self.margin + column * (self.tile_w + self.spacing),
            self.margin + row * (self.tile_h + self.spacing),
            self.tile_w,
            self.tile_h,
        )

    def tile_pixels(self, local_id: int) -> np.ndarray:
        """One tile's pixels. A read-only *view*, since the base is read-only:
        a caller that wants to write copies, and one that does not pays
        nothing."""
        x, y, w, h = self.tile_rect(local_id)
        return self.pixels[y : y + h, x : x + w]

    # -- terrain -------------------------------------------------------------

    @property
    def is_terrain_set(self) -> bool:
        return bool(self.terrains)

    def meta_of(self, local_id: int) -> TileMeta:
        """One tile's metadata, or an empty record.

        Never ``None``: every caller wants to read a field off it, and an
        ``if meta is not None`` at each of them is one place for the check to be
        forgotten. The empty record is a fresh default rather than a shared
        singleton, because it is frozen but its ``properties`` dict is not.
        """
        return self.tiles.get(int(local_id)) or TileMeta()

    def with_meta(self, local_id: int, meta: TileMeta | None) -> Tileset:
        """A copy of this tileset with one tile's metadata replaced.

        A copy, because a ``Tileset`` is frozen for :func:`frozen_rgba`'s
        reason: the UI keys its texture upload on identity, and an in-place edit
        anywhere would move no counter. ``None`` -- or an empty record -- removes
        the entry.
        """
        tiles = dict(self.tiles)
        if meta is None or meta.is_empty:
            tiles.pop(int(local_id), None)
        else:
            tiles[int(local_id)] = meta
        return replace(self, tiles=tiles)

    def terrain_of(self, local_id: int) -> int | None:
        """Which terrain a tile belongs to, or ``None`` if it is not one.

        **This is the only place a cell's terrain comes from.** There is no
        parallel per-cell terrain array anywhere in the package: the gid names a
        tileset and a local id, the local id's row names the terrain, and one
        answer cannot disagree with itself. A stored field would be a second
        source of truth that ``.wmap``, undo and every Tiled round trip would
        each have to keep honest.
        """
        if not self.terrains:
            return None
        index = int(local_id)
        if index < 0 or index >= self.tile_count:
            return None
        row = (index // blob.TILE_COUNT) // (self.phases * self.phases)
        return row if row < len(self.terrains) else None

    def local_for(self, terrain: int, blob_index: int, phase: int = 0) -> int:
        """The local id of one terrain's one blob case. The layout, in one line.

        ``phase`` is ``py * phases + px``, the row-major index into the
        terrain's ``phases**2`` sub-rows; 0 is the only phase of a classic set.
        """
        row, column, sub = int(terrain), int(blob_index), int(phase)
        if row < 0 or row >= len(self.terrains):
            raise IndexError(f"terrain {row} is outside this set (0..{len(self.terrains) - 1})")
        if column < 0 or column >= blob.TILE_COUNT:
            raise IndexError(f"blob case {column} is outside 0..{blob.TILE_COUNT - 1}")
        if sub < 0 or sub >= self.phases * self.phases:
            raise IndexError(
                f"phase {sub} is outside 0..{self.phases * self.phases - 1}"
            )
        return (row * self.phases * self.phases + sub) * blob.TILE_COUNT + column

    def uv(self, local_id: int) -> tuple[float, float, float, float]:
        """``(u0, v0, u1, v1)`` for one tile, for a draw-list quad.

        Here rather than in the pane because it is arithmetic over the slicing
        this class owns, and because a preview, a canvas and a palette grid all
        need the same answer.
        """
        x, y, w, h = self.tile_rect(local_id)
        return (
            x / self.image_w,
            y / self.image_h,
            (x + w) / self.image_w,
            (y + h) / self.image_h,
        )


@dataclass(frozen=True)
class TilesetRef:
    """A tileset as the *map* holds it: the set, plus where its ids begin.

    ``firstgid`` is a property of the map's tileset list rather than of the
    tileset, which is why it lives out here -- the same ``.tsx`` used by two
    maps has a different firstgid in each, and baking it into
    :class:`Tileset` would make that one shared object two.
    """

    firstgid: int
    tileset: Tileset
    # Where the tileset came from, when it came from a file. Carried so a
    # ``.tmx`` export can reference the external ``.tsx`` a user already has
    # rather than writing a second copy of it under a new name.
    source: str = ""

    @property
    def last_gid(self) -> int:
        """The highest gid this reference answers for."""
        return self.firstgid + self.tileset.tile_count - 1

    def holds(self, tile_id: int) -> bool:
        return self.firstgid <= int(tile_id) <= self.last_gid

    def local(self, tile_id: int) -> int:
        return int(tile_id) - self.firstgid


def repolish(original: Tileset, pixels: np.ndarray) -> Tileset:
    """The same terrain set wearing new art.

    The Inker round-trip's other half: a tileset's atlas is opened as an
    ordinary document, painted over, and handed back through
    ``plotter_tilesets.tileset_from_inker``. Lives here rather than in a
    generator because it *makes no pixels* -- it only re-wraps somebody else's,
    keeping the terrain roles the map's gids already point at. (It arrived with
    the procedural ground generator, which was retired with the AI one; this
    survived because editing an atlas is not generating one.)

    Refused by name when the image size changed, because **the roles are
    positional**: an atlas cropped or padded in a paint program is a set whose
    tile ninety-three is no longer sand's interior, and accepting it would
    repaint an entire map wrong while every gid in it stayed valid. The message
    says the size to come back with, because a refusal that does not is a
    refusal that sends someone to a forum.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[:2] != original.pixels.shape[:2]:
        got = "x".join(str(part) for part in array.shape[1::-1]) if array.ndim >= 2 else "?"
        raise ValueError(
            f"this atlas is {got}, and the tileset it replaces is "
            f"{original.image_w}x{original.image_h}; resize it and try again"
        )
    # ``replace`` rather than mutation because ``Tileset`` is frozen: the
    # result goes back in through ``replace_tileset``, whose epoch bump is what
    # tells the pane's texture cache to re-upload -- an in-place edit would
    # move no counter and redraw nothing.
    return replace(original, pixels=array)
