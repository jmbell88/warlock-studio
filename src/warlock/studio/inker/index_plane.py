"""Index planes: the arithmetic of a truly indexed document, and nothing else.

:mod:`.indexed` constrains RGBA writes onto a table. This module is the other
thing -- an ``(H, W) uint8`` plane where the *number* is the picture and the
table is a lookup applied afterwards -- and the difference between the two is
**identity**. Two palette slots holding the same RGB are indistinguishable in an
RGBA plane: recolouring one of them repaints both, sorting the table cannot know
which pixels belonged to which, and a ``.aseprite`` file that legitimately holds
duplicate swatches loses the distinction the moment it is read. An index plane
keeps it, and that is the whole reason this module exists.

Both mechanisms stay. A document in ``color_mode == "rgb"`` with a palette is
*palette-constrained RGB* and goes on snapping through :mod:`.indexed`, because
legacy documents legally contain soft alpha that one-index-per-pixel cannot
represent. Entering true indexed mode is an explicit, undoable conversion.

**Pure and small on purpose.** No document, no undo, no imgui, no service --
this package's pin test holds it to importing nothing under ``warlock``. Every
function takes arrays and tables and returns arrays and tables, so the funnel in
``document._commit_patch`` and the three new edit types in :mod:`.undo` can all
lean on one definition of what resolving and materialising mean.

Two rules run through everything here and are worth stating once:

*The transparent index is reached only through alpha.* It is never a
nearest-match candidate. A palette routinely carries a black in slot 0 as its
transparent entry and a *drawn* black elsewhere; if slot 0 could win a distance
comparison, every black the user painted would silently become a hole.

*Nearest is measured in straight RGB*, for ``indexed.nearest``'s reason -- the
two must agree pixel for pixel or a document converted one way and painted the
other moves under the brush.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "MAX_COLOURS",
    "OPAQUE_THRESHOLD",
    "apply_remap",
    "histogram",
    "lut",
    "materialize",
    "merge_table",
    "permutation_tables",
    "resolve",
    "tables_from_order",
]

RGBA = tuple[int, int, int, int]

#: The alpha at which a pixel stops being a hole and starts being a colour.
#: 128 -- the midpoint -- rather than 1 or 255, because what arrives at the
#: funnel is routinely a *soft* edge: an antialiased shape, a feathered
#: selection, a soft nib. Thresholding at 1 makes a one-percent halo opaque and
#: fattens every drawn edge by a pixel; at 255 it makes a 99% edge a hole and
#: eats one. The midpoint is the only choice that rounds a coverage ramp to the
#: shape the user drew, and it is what every indexed editor does with a soft
#: brush it cannot represent.
OPAQUE_THRESHOLD = 128

#: The most entries an indexed palette may hold. The cap PNG-P, GIF and
#: ``.aseprite`` all share, and the reason a plane can be ``uint8`` at all --
#: which is in turn what makes the ORA representation a plain mode-P PNG rather
#: than a format of our own. A 257th colour is refused by name, never truncated.
MAX_COLOURS = 256


def lut(palette: Sequence[RGBA], transparent: int = 0) -> np.ndarray:
    """The palette as a ``(P, 4) uint8`` lookup table.

    The transparent slot comes back as ``(0, 0, 0, 0)`` regardless of what the
    table says its colour is, which is the one place the *materialisation* is
    allowed to disagree with the stored palette. Aseprite files routinely give
    the transparent index an opaque colour (it is displayed in the palette
    editor like any other), and honouring that literally would paint the hole.
    The stored entry is left alone -- it is what a ``.aseprite`` writer has to
    put back -- so this is a projection, not an edit.

    Every other entry keeps its alpha. A palette here is RGBA rather than RGB
    because ``.aseprite`` palettes carry alpha and ``.gpl`` cannot, and dropping
    it at the door would make the ``.gpl`` projection the record.
    """
    if not palette:
        raise ValueError("an indexed document needs at least one colour")
    if len(palette) > MAX_COLOURS:
        raise ValueError(f"an indexed palette holds at most {MAX_COLOURS} colours")
    table = np.zeros((len(palette), 4), dtype=np.uint8)
    for i, colour in enumerate(palette):
        entry = tuple(colour)
        table[i, : len(entry[:4])] = np.asarray(entry[:4], dtype=np.uint8)
        if len(entry) < 4:
            table[i, 3] = 255
    if 0 <= transparent < len(palette):
        table[transparent] = 0
    return table


def materialize(indices: np.ndarray, table: np.ndarray) -> np.ndarray:
    """``(H, W) uint8`` indices through a ``(P, 4)`` table -> ``(H, W, 4)``.

    A fresh, C-contiguous array every time: the result is assigned straight onto
    ``Layer.pixels``, which the compositor slices, the texture uploader reads
    and ``PatchEdit`` copies out of, and a fancy-indexed view would surprise all
    three.

    Out-of-range indices are **clipped rather than raised on**. A plane can
    outlive its table by one operation -- undo restores indices before the
    compound's palette edit restores the table on documents built by hand in
    tests -- and a materialisation that raises out of the middle of a history
    walk leaves the document in a state nothing can repair. Clipping shows the
    user a wrong colour for one frame; raising loses the drawing.
    """
    if indices.dtype != np.uint8 or indices.ndim != 2:
        raise ValueError("an index plane is (H, W) uint8")
    if table.dtype != np.uint8 or table.ndim != 2 or table.shape[1] != 4:
        raise ValueError("a lookup table is (P, 4) uint8")
    if not table.shape[0]:
        raise ValueError("a lookup table has at least one entry")
    safe = indices if table.shape[0] >= 256 else np.minimum(indices, table.shape[0] - 1)
    return np.ascontiguousarray(table[safe])


def resolve(
    pixels: np.ndarray,
    table: np.ndarray,
    transparent: int | None = 0,
    *,
    prefer: tuple[RGBA, int] | None = None,
) -> np.ndarray:
    """``(H, W, 4)`` RGBA -> the ``(H, W) uint8`` index plane that best holds it.

    The one inference in the whole design, and it happens at exactly two places:
    the commit funnel (every tool paints RGBA, as it always has) and a smooth
    scale (which resamples and so cannot carry indices through). Everywhere else
    -- flips, rotations, crops, palette remaps -- the plane is *permuted*, never
    re-inferred, which is what makes duplicate-slot identity survive geometry.

    Alpha decides first: below :data:`OPAQUE_THRESHOLD` the pixel is the
    transparent index, full stop. Above it, the nearest entry in straight RGB
    among the slots that are **not** the transparent one.

    ``prefer`` is ``(colour, index)`` -- the palette slot the user is painting
    with. Pixels whose RGB matches that colour exactly take that slot rather
    than the lowest-numbered duplicate of it, which is what makes "I painted
    with slot 7" true in a table where slots 3 and 7 are the same brown. It
    changes nothing when the table has no duplicates, and everything the moment
    it does. Output that no gesture authored -- a filter, a paste, a gradient --
    has no preference to express and falls to lowest-index-wins, deterministically.

    ``transparent=None`` says *this table has no transparent slot*: every entry
    is a candidate and sub-threshold pixels land on slot 0. One caller has that
    shape -- :func:`dither.convert_indices`, which resolves onto the candidate
    sub-table and re-labels the transparent pixels itself -- and it is spelled
    as a mode rather than reached by passing an out-of-range index, so an
    arithmetic slip producing ``-1`` cannot silently mean it.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("resolve takes (H, W, 4) uint8")
    if table.dtype != np.uint8 or table.ndim != 2 or table.shape[1] != 4:
        raise ValueError("a lookup table is (P, 4) uint8")
    count = table.shape[0]
    if not count:
        raise ValueError("a lookup table has at least one entry")
    if transparent is None:
        hole, excluded = 0, -1
    else:
        hole = int(transparent) if 0 <= int(transparent) < count else 0
        excluded = hole
    out = np.full(pixels.shape[:2], hole, dtype=np.uint8)
    if pixels.size == 0:
        return out

    visible = pixels[..., 3] >= OPAQUE_THRESHOLD
    if not visible.any():
        return out

    # Every slot but the transparent one. A palette of exactly one colour --
    # which is also the transparent slot -- leaves nothing to match against, so
    # the whole table is the candidate set and every pixel lands on slot 0. A
    # degenerate document rather than a refusal: it has one colour and one
    # meaning for it, and there is no wrong answer to give.
    candidates = np.array(
        [i for i in range(count) if i != excluded] or list(range(count)),
        dtype=np.intp,
    )
    entries = table[candidates, :3].astype(np.int32)

    rgb = pixels[..., :3][visible]
    # ``indexed.snap``'s packed-uint32 unique: the cost is set by the number of
    # distinct colours in the region and the palette size, never by the pixel
    # count. A full-canvas commit on a pixel-art document is a few dozen
    # distinct colours against a few dozen swatches.
    packed = rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
    keys, inverse = np.unique(packed, return_inverse=True)
    colours = np.stack([(keys >> 16) & 0xFF, (keys >> 8) & 0xFF, keys & 0xFF], axis=1).astype(
        np.int32
    )
    delta = colours[:, None, :] - entries[None, :, :]
    picks = candidates[np.argmin((delta * delta).sum(axis=2), axis=1)]

    if prefer is not None:
        colour, slot = prefer
        slot = int(slot)
        if 0 <= slot < count and slot != excluded:
            r, g, b = (int(c) for c in tuple(colour)[:3])
            picks = np.where(keys == (r << 16 | g << 8 | b), np.uint8(slot), picks)

    out[visible] = picks[inverse].astype(np.uint8)
    return out


def apply_remap(indices: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """*indices* rewritten through a slot map. A new array; the input stands.

    ``forward[old] == new``. This is what a palette reorder, a slot merge and an
    ``.aseprite`` import's mask remap all reduce to -- a table of at most 256
    bytes applied to the plane -- and it is exact by construction. There is no
    colour comparison anywhere in it, which is precisely why duplicate slots
    survive: nothing ever asks what colour a slot holds.
    """
    if indices.dtype != np.uint8 or indices.ndim != 2:
        raise ValueError("an index plane is (H, W) uint8")
    if forward.dtype != np.uint8 or forward.ndim != 1:
        raise ValueError("a remap is a (P,) uint8 table")
    if not forward.shape[0]:
        raise ValueError("a remap has at least one entry")
    safe = indices if forward.shape[0] >= 256 else np.minimum(indices, forward.shape[0] - 1)
    return np.ascontiguousarray(forward[safe])


def tables_from_order(order: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(forward, inverse)`` from "the new table's slot *p* holds old ``order[p]``".

    The single definition of what a palette reorder does to a plane, expressed in
    the only terms every reorder in the app already speaks: ``sort_palette``
    builds exactly this list, and ``move_slot``'s
    ``table.insert(to, table.pop(index))`` is one case of it. Derived from the
    order rather than written out per operation, so a reorder cannot rearrange
    the table one way and the pixels another.

    ``inverse`` *is* the order (old slot of each new position); ``forward`` is
    its argsort (new position of each old slot), which is what a plane gets.
    Both are returned because undo needs the other one and here is the only
    place where the permutation is known exactly.
    """
    count = len(order)
    if count < 1:
        raise ValueError("a palette has at least one slot")
    if sorted(int(i) for i in order) != list(range(count)):
        raise ValueError("a reorder is a permutation of every slot")
    inverse = np.asarray([int(i) for i in order], dtype=np.uint8)
    forward = np.zeros(count, dtype=np.uint8)
    forward[inverse] = np.arange(count, dtype=np.uint8)
    return forward, inverse


def permutation_tables(count: int, index: int, to: int) -> tuple[np.ndarray, np.ndarray]:
    """``(forward, inverse)`` for moving slot *index* to position *to*.

    The exact inverse of the list operation ``table.insert(to, table.pop(index))``
    performs, and derived by *performing that operation on the positions* rather
    than by writing the arithmetic out a second time -- so the two cannot drift.
    """
    count = int(count)
    if count < 1:
        raise ValueError("a palette has at least one slot")
    index = max(0, min(int(index), count - 1))
    to = max(0, min(int(to), count - 1))
    order = list(range(count))
    order.insert(to, order.pop(index))
    return tables_from_order(order)


def merge_table(count: int, gone: int, into: int) -> np.ndarray:
    """The forward map for deleting slot *gone*, its pixels landing on *into*.

    Non-invertible on purpose -- two slots become one, and no table puts them
    back -- which is why the op that uses it undoes by replay rather than by an
    inverse remap. Every slot above the removed one shifts down by one, because
    the table it is applied to has had an entry taken out of it.

    *into* is given in **pre-removal** numbering (it is the slot the caller
    found by a nearest search over the surviving table) and is shifted here, so
    callers never have to reason about two numbering schemes at once.
    """
    count = int(count)
    if count < 2:
        raise ValueError("the last slot cannot be removed")
    gone = max(0, min(int(gone), count - 1))
    into = max(0, min(int(into), count - 1))
    if into == gone:
        raise ValueError("a slot cannot merge into itself")
    shifted = into - 1 if into > gone else into
    forward = np.zeros(count, dtype=np.uint8)
    for slot in range(count):
        if slot == gone:
            forward[slot] = shifted
        else:
            forward[slot] = slot - 1 if slot > gone else slot
    return forward


def histogram(indices: np.ndarray, count: int) -> list[int]:
    """How many pixels sit in each slot. Exact, and finally per-*slot*.

    :func:`indexed.histogram` counts by colour and so cannot tell two identical
    swatches apart -- it reports both as holding every pixel of either. This
    counts the plane, so "is slot 7 used" has an answer even when slot 3 is the
    same brown, which is what makes deleting an unused duplicate safe.

    The transparent slot is counted like any other. It is a real slot holding
    real pixels; whether the user considers those pixels "used" is a question
    for the pane, which knows which index is transparent.
    """
    if indices.dtype != np.uint8 or indices.ndim != 2:
        raise ValueError("an index plane is (H, W) uint8")
    count = int(count)
    if count < 1:
        raise ValueError("a palette has at least one slot")
    if indices.size == 0:
        return [0] * count
    return np.bincount(indices.ravel(), minlength=count)[:count].astype(np.int64).tolist()
