"""Tiled Wang sets, as data rather than as one recognised preset.

Terrain painting here has always been the **blob preset**: a two-value collapse
into 47 cases, laid out positionally in 47 columns, with ``Tileset.terrain_of``
reading a cell's role straight off its local id. That model is kept exactly --
it is what the whole existing terrain corpus is written against, and its bar is
byte-identity -- and this module is the *general* case beside it: a set that
says, per tile, which colour sits at each of eight positions.

**The blob preset is derivable from here** (:func:`blob_wangset`), which is the
point of putting the two together: the 47-case collapse stops being a special
kind of thing and becomes one particular table of wangids, so a Tiled user
opening a set this editor generated sees exactly what a Tiled user authoring one
by hand would produce.

**Membership is still derived from the gid and never stored per cell.** A wangid
is looked up by tile id and the map goes on storing gids alone -- the standing
rule ``terrain.py`` opens with, and the reason a hand-stamped cell joins a field
for free.

The eight positions are Tiled's own order, clockwise from the top, and index 0
means *unset* rather than a colour: a corner-only set leaves every edge slot 0,
and an edge-only set leaves every corner slot 0. That is what ``kind`` records,
and it is what lets one matcher serve all three kinds without asking which it
has.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from . import blob

#: The eight ``wangid`` slots, in Tiled's own order: top, top-right, right,
#: bottom-right, bottom, bottom-left, left, top-left. The same clockwise-from-
#: north order :data:`blob.NEIGHBOURS` uses, which is why translating between
#: the two is arithmetic rather than a lookup table.
POSITIONS = 8

#: Which slots are *edges* and which are *corners*, by index into a wangid.
EDGE_SLOTS: tuple[int, ...] = (0, 2, 4, 6)
CORNER_SLOTS: tuple[int, ...] = (1, 3, 5, 7)

#: The three kinds Tiled declares. ``mixed`` uses every slot; the other two
#: leave their unused half at 0, and a reader must not "helpfully" fill it in.
WANG_KINDS = ("corner", "edge", "mixed")

#: Which *other* cells share each slot, as ``((dx, dy), that cell's slot)``.
#:
#: **An edge is shared with one neighbour; a corner is shared with three**, and
#: getting that wrong is the whole difficulty of this table. A corner is a
#: lattice *point*, and four cells meet at every lattice point -- so this cell's
#: top-right corner is also the right-hand neighbour's top-left, the cell
#: above's bottom-right, and the up-and-right diagonal's bottom-left. Consulting
#: only the diagonal (the one whose slot name is simply reversed) looks right
#: and is the case that almost never fires: the diagonal is empty far more often
#: than the orthogonal neighbours are, so a set painted left-to-right would
#: reconcile against nothing at all.
#:
#: Listed orthogonal-first, because those are the cells a stroke has usually
#: already painted, and the first neighbour with an opinion is the one taken.
SHARERS: tuple[tuple[tuple[tuple[int, int], int], ...], ...] = (
    (((0, -1), 4),),                                  # 0 top edge
    (((1, 0), 7), ((0, -1), 3), ((1, -1), 5)),        # 1 top-right corner
    (((1, 0), 6),),                                   # 2 right edge
    (((1, 0), 5), ((0, 1), 1), ((1, 1), 7)),          # 3 bottom-right corner
    (((0, 1), 0),),                                   # 4 bottom edge
    (((-1, 0), 3), ((0, 1), 7), ((-1, 1), 1)),        # 5 bottom-left corner
    (((-1, 0), 2),),                                  # 6 left edge
    (((-1, 0), 1), ((0, -1), 5), ((-1, -1), 3)),      # 7 top-left corner
)

#: The slot directly across from each one. Meaningful for an *edge* -- this
#: cell's top is the cell above's bottom -- and kept because the edge half of
#: :data:`SHARERS` is exactly this plus the offset, which is worth being able to
#: state and check rather than only to read off a table.
OPPOSITE: tuple[int, ...] = (4, 5, 6, 7, 0, 1, 2, 3)

#: Where each slot sits *on the tile*, as a fraction of its width and height,
#: in :data:`POSITIONS` order. An edge slot is the middle of that side and a
#: corner slot is the corner itself, which is not a drawing convenience: it is
#: what the slot *means*, since a corner is the lattice point four cells meet
#: at and an edge is the line two share.
#:
#: Here rather than in the pane that draws it, because the same eight points
#: are what the pane's click regions are keyed on -- a second copy is how a
#: marker comes to be drawn somewhere it cannot be clicked, which is exactly
#: the defect ``picking`` was factored out to prevent for collision handles.
SLOT_FRACTIONS: tuple[tuple[float, float], ...] = (
    (0.5, 0.0),  # 0 top edge
    (1.0, 0.0),  # 1 top-right corner
    (1.0, 0.5),  # 2 right edge
    (1.0, 1.0),  # 3 bottom-right corner
    (0.5, 1.0),  # 4 bottom edge
    (0.0, 1.0),  # 5 bottom-left corner
    (0.0, 0.5),  # 6 left edge
    (0.0, 0.0),  # 7 top-left corner
)


@dataclass(frozen=True)
class WangColour:
    """One colour a Wang set paints with.

    ``probability`` is Tiled's own tie-break weight, used when two tiles match a
    constraint equally well. Zero is legal and means "never chosen automatically"
    -- the same reading a tile's own probability has.
    """

    name: str = ""
    colour: str = "#ffffff"
    probability: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "colour", str(self.colour))
        object.__setattr__(self, "probability", float(self.probability))
        if self.probability < 0.0:
            raise ValueError("a wang colour probability cannot be negative")


@dataclass(frozen=True)
class WangSet:
    """A named table of per-tile wangids over a set of colours.

    ``tiles`` maps a **local** tile id to its eight slots, each 0 (unset) or a
    1-based index into ``colours``. Local, because a firstgid is a property of
    the map that loaded the tileset rather than of the tileset.
    """

    name: str = "Terrain"
    kind: str = "mixed"
    colours: tuple[WangColour, ...] = ()
    tiles: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "colours", tuple(self.colours))
        if self.kind not in WANG_KINDS:
            raise ValueError(f"a wang set is one of {list(WANG_KINDS)}, not {self.kind!r}")
        cleaned: dict[int, tuple[int, ...]] = {}
        for local, wangid in dict(self.tiles).items():
            values = tuple(int(slot) for slot in wangid)
            if len(values) != POSITIONS:
                raise ValueError(f"a wangid has {POSITIONS} slots, not {len(values)}")
            if any(slot < 0 or slot > len(self.colours) for slot in values):
                raise ValueError("a wangid slot names a colour this set does not have")
            cleaned[int(local)] = values
        object.__setattr__(self, "tiles", cleaned)

    @property
    def slots(self) -> tuple[int, ...]:
        """Which of the eight positions this set actually uses.

        Read off ``kind`` rather than off the data: a corner set every one of
        whose tiles happens to have an unset edge is still a corner set, and
        inferring the kind from the table would make an all-blank tile change
        what the set *is*.
        """
        if self.kind == "corner":
            return CORNER_SLOTS
        if self.kind == "edge":
            return EDGE_SLOTS
        return tuple(range(POSITIONS))

    def wangid_of(self, local_id: int) -> tuple[int, ...]:
        """One tile's wangid, or all-unset for a tile the set says nothing about.

        No studio caller yet -- the wang tests exercise it directly. It is the
        per-tile read a ``field_of`` callback for :func:`constraints_from` is
        built from, which is why it stays published rather than folding into
        ``matching``.
        """
        return self.tiles.get(int(local_id), (0,) * POSITIONS)

    def matching(self, wanted: dict[int, int]) -> list[int]:
        """Every tile whose wangid satisfies ``{slot: colour}``, best first.

        **Exact match on the constrained slots only.** A slot the caller left
        out is a slot nothing has an opinion about yet -- the edge of the map, or
        a neighbour holding a tile this set does not describe -- and demanding a
        particular colour there would refuse every tile for no reason.

        Ordered by probability descending and then by lowest local id, so a tie
        resolves the same way twice. The caller takes the first, or leaves the
        cell alone when the list is empty: **a near-miss is never written**,
        because a wrong tile is the silent half-read in picture form.
        """
        found = [
            local
            for local, wangid in self.tiles.items()
            if all(wangid[slot] == colour for slot, colour in wanted.items())
        ]
        return sorted(
            found,
            key=lambda local: (-self._weight(local), local),
        )

    def _weight(self, local: int) -> float:
        """A tile's tie-break weight: the mean of its slots' colour weights.

        The mean rather than the first slot's, so a tile made of two colours is
        weighted by both -- and an all-unset tile weighs 1.0 rather than 0, which
        keeps a set whose colours all weigh 1 in pure lowest-id order.
        """
        wangid = self.tiles.get(local, ())
        used = [slot for slot in wangid if slot]
        if not used:
            return 1.0
        return sum(self.colours[slot - 1].probability for slot in used) / len(used)


def slot_points(
    tile_w: int, tile_h: int, slots: Any = None
) -> dict[int, tuple[float, float]]:
    """``{slot: (x, y)}`` in *tile pixels*, for the slots ``slots`` names.

    ``slots`` defaults to all eight; pass a set's own :attr:`WangSet.slots` to
    get only the half it uses, which is what keeps a corner-only set from
    growing edge values nobody stores and nothing reads.

    The shape a ``picking.nearest_region`` call wants -- a key and a point --
    because that is the whole reason the marker positions are data: the tile
    view converts them to screen once and the picker answers off the same dict
    the drawing used.
    """
    width, height = float(max(1, int(tile_w))), float(max(1, int(tile_h)))
    wanted = range(POSITIONS) if slots is None else [int(slot) for slot in slots]
    return {
        slot: (SLOT_FRACTIONS[slot][0] * width, SLOT_FRACTIONS[slot][1] * height)
        for slot in wanted
        if 0 <= slot < POSITIONS
    }


# --- authoring ---------------------------------------------------------------
#
# Five pure edits, each returning a **new** frozen set. Pure for ``picking``'s
# reason exactly: the caller decides what the result goes through, which for
# the tileset editor is ``MapDoc.replace_tileset`` -- the one undoable door a
# tileset's own content changes by -- and never a second write path invented
# beside it.


def with_slot(wangset: WangSet, local_id: int, slot: int, colour: int) -> WangSet:
    """One tile's one slot set to ``colour``. 0 clears it.

    A wangid that ends up all-zero drops out of ``tiles`` rather than being
    stored as eight noughts. That is the sparse rule ``set_tile_meta`` already
    keeps for tile metadata, and it is what makes "does this set describe tile
    seven" one question with one answer -- a stored blank would round-trip out
    to a ``<wangtile>`` element saying nothing.
    """
    at = int(slot)
    if not 0 <= at < POSITIONS:
        raise IndexError(f"a wangid has slots 0..{POSITIONS - 1}, not {at}")
    value = int(colour)
    if value < 0 or value > len(wangset.colours):
        raise ValueError(f"this set has no colour {value}")
    wangid = list(wangset.wangid_of(local_id))
    wangid[at] = value
    tiles = dict(wangset.tiles)
    if any(wangid):
        tiles[int(local_id)] = tuple(wangid)
    else:
        tiles.pop(int(local_id), None)
    return replace(wangset, tiles=tiles)


def with_colour(wangset: WangSet, colour: WangColour) -> WangSet:
    """``wangset`` with one more colour, appended. Existing wangids are
    untouched: a colour's number is its position, and appending moves none."""
    return replace(wangset, colours=(*wangset.colours, colour))


def with_colour_at(wangset: WangSet, index: int, colour: WangColour) -> WangSet:
    """``wangset`` with colour ``index`` (0-based) replaced -- a rename, a new
    swatch or a new probability. The wangids name *positions*, so none moves."""
    at = int(index)
    if not 0 <= at < len(wangset.colours):
        raise IndexError(f"no colour {at} in this set")
    colours = list(wangset.colours)
    colours[at] = colour
    return replace(wangset, colours=tuple(colours))


def without_colour(wangset: WangSet, index: int) -> WangSet:
    """``wangset`` minus colour ``index``, **with every wangid renumbered**.

    The renumbering is the whole of this function and the reason it is not a
    one-line ``replace``. A wangid slot is a 1-based position in ``colours``, so
    dropping the second of three does two things to the table at once: every
    slot naming it becomes unset, and every slot naming a *later* colour now
    names a colour one place to the left. Leaving either undone is not a
    cosmetic error -- :meth:`WangSet.__post_init__` refuses a slot pointing past
    the end, so the set would not even construct, and a set that constructed
    with the numbers unshifted would have quietly repainted every tile in it.
    """
    at = int(index)
    if not 0 <= at < len(wangset.colours):
        raise IndexError(f"no colour {at} in this set")
    gone = at + 1
    tiles: dict[int, tuple[int, ...]] = {}
    for local, wangid in wangset.tiles.items():
        edited = tuple(
            0 if slot == gone else (slot - 1 if slot > gone else slot) for slot in wangid
        )
        if any(edited):
            tiles[int(local)] = edited
    colours = tuple(c for pos, c in enumerate(wangset.colours) if pos != at)
    return replace(wangset, colours=colours, tiles=tiles)


def blob_wangset(names: list[str], colours: list[str]) -> WangSet:
    """The 47-case blob preset, as an ordinary Wang set.

    This is the derivation the general model exists to make possible: the
    collapse stops being a special kind of thing and becomes one particular
    table. One terrain per colour, laid out in :data:`blob.BLOB_MASKS` order and
    ``blob.TILE_COUNT`` columns apart -- exactly the positional layout
    ``Tileset.local_for`` computes, stated once more as data.

    Its ``kind`` is ``mixed`` because a blob tile constrains all eight slots: the
    collapse is precisely the rule that a corner only counts when both its
    flanking edges do.

    No studio caller yet -- the wang tests exercise it directly. It exists as
    the proof the module docstring leans on (the blob preset *is* one
    particular table), and it is what a Tiled ``.tsx`` export of the preset
    will serialize when that door is built.
    """
    entries = [
        WangColour(name=name or f"Terrain {index + 1}", colour=colour)
        for index, (name, colour) in enumerate(zip(names, colours, strict=True))
    ]
    tiles: dict[int, tuple[int, ...]] = {}
    for index in range(len(entries)):
        base = index * blob.TILE_COUNT
        for case, mask in enumerate(blob.BLOB_MASKS):
            tiles[base + case] = tuple(
                index + 1 if mask & bit else 0
                for bit in (
                    blob.N,
                    blob.NE,
                    blob.E,
                    blob.SE,
                    blob.S,
                    blob.SW,
                    blob.W,
                    blob.NW,
                )
            )
    return WangSet(name="Terrain", kind="mixed", colours=tuple(entries), tiles=tiles)


def constraints_from(
    field_of: Any, x: int, y: int, wangset: WangSet
) -> dict[int, int]:
    """What this cell's neighbours already say about its eight slots.

    ``field_of(x, y)`` returns a neighbour's wangid, or ``None`` when there is
    nothing there to ask -- off the map, an empty cell, or a tile this set does
    not describe. Those contribute no constraint at all, which is what lets a
    field grow outward instead of refusing at its own edge.

    Only the slots this set uses are asked about, so a corner-only set is never
    constrained by an edge nobody stores.
    """
    wanted: dict[int, int] = {}
    for slot in wangset.slots:
        for (dx, dy), theirs in SHARERS[slot]:
            neighbour = field_of(int(x) + dx, int(y) + dy)
            if neighbour is None:
                continue
            colour = neighbour[theirs]
            if colour:
                # First opinion wins. In a consistent field they all agree, and
                # where they do not, a fixed order is what makes the answer the
                # same twice rather than dependent on dict iteration.
                wanted[slot] = colour
                break
    return wanted
