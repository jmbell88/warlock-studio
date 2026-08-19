"""Which blob role each cell of an already-gridded sheet plays.

Nothing in Warlock has produced a 47-column terrain set since the procedural and
AI ground generators were retired on 2026-08-18, so the whole terrain feature is
reachable only through files authored elsewhere -- and Tiled's Wang import is
recognise-or-refuse, which does not help a user who drew a blob sheet by hand or
asked a model for one. This measures such a sheet and offers to reorder it.

**Detection is a suggestion; the caller confirms it with the user.** The same
clause :mod:`.slicing` states, and for the same reason: this runs only at the
door where a file nobody in this program made arrives, and what it returns is
offered in a popup with an "import as plain tiles" button beside it. A
silhouette coincidence that happens to look like a complete terrain set is
mitigated by that button, not by tightening a threshold.

This is the fourth measurement in the tree and the four must not be merged:

* :mod:`warlock.pipelines.pixel` -- the luma lattice period and phase of an
  upscaled render, PIL, on the export path.
* :mod:`warlock.studio.inker.transform` -- the same lattice measure inside the
  editor, on the open document.
* :mod:`warlock.studio.tilegrid.slicing` -- dark separator *bands* between the
  cells of a ruled sheet, at the import doors.
* this module -- which blob *role* each cell of an already-gridded sheet plays.

Different questions at different doors.

**The two ratios below are import-door heuristics behind a confirm popup and no
``docs/measurements/`` document is owed for them.** That rule (CLAUDE.md) is
about constants the *stored corpus is keyed on* -- ``trellis_band``,
``SEAM_MAX``, the grade scale -- where a change silently reinterprets everything
already recorded. Nothing is stored keyed on these: they decide what a popup
offers, once, and the user says yes or no.

**v1 infers a single terrain per sheet and ``phases = 1``.** A sheet with two
terrains in it is two sheets as far as this is concerned, and phase variants are
what the extras count is for: a v2 that uses them has the number it needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import blob

#: How much of a tile's edge strip must be background for that side to be open.
#: A side is either drawn to the edge or it is not, so this sits well clear of a
#: half: a strip that is 60% background is a side nothing runs into.
EDGE_OPEN_RATIO = 0.6

#: How deep the edge strips and corner squares are, as a fraction of the tile.
#: An eighth of a 32px tile is 4px, which is thick enough to survive a soft edge
#: and thin enough that a tile's own interior detail never reaches it.
_STRIP_DIVISOR = 8


@dataclass(frozen=True)
class SheetRoles:
    """Which source cell plays each of the 47 canonical blob cases.

    ``order`` is length 47, in ascending :data:`blob.BLOB_MASKS` order, holding
    the source cell index (reading order) that plays each case. ``extras``
    counts cells whose role was already taken -- not used in v1, and the number
    a phase-variant v2 would need. ``background`` records which of the two
    background rules fired, so a popup and a test can both say which.
    """

    order: tuple[int, ...]
    extras: int
    background: str

    @property
    def tile_count(self) -> int:
        return len(self.order)


#: How many candidate backdrop colours the opaque path tries before giving up.
#: Small because the answer is nearly always the first or second, and because
#: each candidate costs a full pass over the cells.
_BACKGROUND_CANDIDATES = 4


def _background_masks(
    pixels: np.ndarray, tile_w: int, tile_h: int
) -> list[tuple[np.ndarray, str]]:
    """The candidate background rules to try, best first.

    Transparency when the sheet carries any, and then nothing else: that is
    unambiguous and is what an exported blob set looks like.

    **The opaque case tries several candidates rather than one, and this is a
    deliberate departure from a single most-common-colour rule.** On a canonical
    blob sheet the tiles' outer rings are *not* mostly backdrop: a third of the
    47 cases are closed on every side, so their whole ring is fill, and which of
    the two colours wins a straight count is close to a coin flip. Picking wrong
    inverts every silhouette and the sheet is refused with nothing to say why.

    Trying the few commonest ring colours in turn is safe precisely because the
    caller's acceptance test is so strict: a wrong background essentially never
    produces all 47 distinct roles, so the first candidate that does is the right
    one. The rule validates itself instead of needing to be lucky.
    """
    array = np.asarray(pixels)
    if array.shape[2] == 4 and (array[:, :, 3] == 0).any():
        return [(array[:, :, 3] == 0, "alpha")]

    rgb = array[:, :, :3].astype(np.uint32)
    packed = rgb[:, :, 0] << 16 | rgb[:, :, 1] << 8 | rgb[:, :, 2]
    ring = np.zeros(packed.shape, dtype=bool)
    depth = max(1, min(tile_w, tile_h) // _STRIP_DIVISOR)
    for y in range(0, packed.shape[0] - tile_h + 1, tile_h):
        for x in range(0, packed.shape[1] - tile_w + 1, tile_w):
            ring[y:y + depth, x:x + tile_w] = True
            ring[y + tile_h - depth:y + tile_h, x:x + tile_w] = True
            ring[y:y + tile_h, x:x + depth] = True
            ring[y:y + tile_h, x + tile_w - depth:x + tile_w] = True
    if not ring.any():
        return []
    values, counts = np.unique(packed[ring], return_counts=True)
    order = np.argsort(-counts, kind="stable")[:_BACKGROUND_CANDIDATES]
    return [(packed == values[int(index)], "border") for index in order]


def _cell_mask(background: np.ndarray, tile_w: int, tile_h: int) -> int:
    """The eight-bit neighbour mask one tile's silhouette implies.

    ``background`` is the ``(tile_h, tile_w)`` bool mask of that one cell.

    A *closed* side means a neighbour is present there -- the tile is drawn out
    to that edge, so whatever is next to it continues the field. So the bit
    vocabulary reads inverted from the mask this builds: an open side clears its
    bit. That inversion is the whole translation between "what the picture
    looks like" and "what the blob table is indexed by", and getting it backwards
    produces a set whose every tile is its own opposite.
    """
    depth = max(1, min(tile_w, tile_h) // _STRIP_DIVISOR)

    def ratio(region: np.ndarray) -> float:
        return float(region.mean()) if region.size else 1.0

    sides = {
        blob.N: background[:depth, :],
        blob.S: background[tile_h - depth:, :],
        blob.W: background[:, :depth],
        blob.E: background[:, tile_w - depth:],
    }
    corners = {
        blob.NE: background[:depth, tile_w - depth:],
        blob.SE: background[tile_h - depth:, tile_w - depth:],
        blob.SW: background[tile_h - depth:, :depth],
        blob.NW: background[:depth, :depth],
    }
    mask = 0xFF
    for bit, region in {**sides, **corners}.items():
        if ratio(region) >= EDGE_OPEN_RATIO:
            mask &= ~bit
    return blob.normalise(mask)


def infer_roles(pixels: np.ndarray, tile_w: int, tile_h: int) -> SheetRoles | None:
    """The blob role of every cell, or ``None`` when this is not a terrain set.

    ``None`` unless **every one of the 47** canonical masks is matched by at
    least one tile. A partial set is refused silently rather than offered with a
    warning: the popup only ever *offers*, and a half-set that cannot be painted
    with is not an offer worth making.

    The first tile in reading order wins a contested mask; the rest are counted
    as extras, which is the number a phase-variant v2 would use.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("a sheet is an (h, w, 3) or (h, w, 4) array")
    tile_w, tile_h = int(tile_w), int(tile_h)
    if tile_w < 1 or tile_h < 1:
        raise ValueError("a tile must be at least one pixel across")
    rows = array.shape[0] // tile_h
    cols = array.shape[1] // tile_w
    if rows * cols < blob.TILE_COUNT:
        return None

    for background, mode in _background_masks(array, tile_w, tile_h):
        claimed: dict[int, int] = {}
        extras = 0
        for index in range(rows * cols):
            y = (index // cols) * tile_h
            x = (index % cols) * tile_w
            mask = _cell_mask(background[y:y + tile_h, x:x + tile_w], tile_w, tile_h)
            if mask in claimed:
                extras += 1
            else:
                claimed[mask] = index
        if len(claimed) < blob.TILE_COUNT:
            continue
        order = tuple(claimed[mask] for mask in blob.BLOB_MASKS)
        return SheetRoles(order=order, extras=extras, background=mode)
    return None


def reorder(
    pixels: np.ndarray, roles: SheetRoles, tile_w: int, tile_h: int
) -> np.ndarray:
    """The atlas rebuilt as 47 columns by one row, in canonical blob order.

    **The reorder is not cosmetic; it *is* the role assignment.** A terrain
    set's layout is positional -- ``Tileset.terrain_of`` and ``local_for`` are
    the only source of a cell's role, and exactly 47 columns are enforced at
    construction -- so importing without reordering would give every tile a
    wrong role, silently, and every stroke would draw the wrong picture.

    Fresh array, never a view onto the source: :mod:`.slicing`'s discipline.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("a sheet is an (h, w, 3) or (h, w, 4) array")
    tile_w, tile_h = int(tile_w), int(tile_h)
    cols = array.shape[1] // tile_w
    if cols < 1:
        raise ValueError("that sheet holds no whole tile")
    out = np.zeros((tile_h, len(roles.order) * tile_w, 4), dtype=np.uint8)
    for column, source in enumerate(roles.order):
        y = (source // cols) * tile_h
        x = (source % cols) * tile_w
        cell = array[y:y + tile_h, x:x + tile_w]
        target = out[:, column * tile_w:(column + 1) * tile_w]
        if cell.shape[2] == 3:
            target[:, :, :3] = cell
            target[:, :, 3] = 255
        else:
            target[:, :] = cell
    return out
