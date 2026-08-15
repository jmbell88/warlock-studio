"""A generated sprite atlas as an editable document: one cell per frame.

``sheetout``'s inverse, and deliberately its opposite in one respect: that one
reaches outside the package for the sheet *format*, and this one reaches for
nothing at all. The cells arrive as plain data -- rectangles and a kind, read
from a draft's sidecar by whoever is doing the opening -- so the slicer never
learns what a job directory is, what a service is, or where the atlas came
from. The whole of what it knows is "these rectangles, in this order".

That is not tidiness for its own sake. The grid is never re-detected from
pixels: a candidate atlas whose content sits a few pixels off its rectangles is
still sliced on the rectangles it was generated on, and the alternative --
finding the seams -- turns one mis-registered generation into a document nobody
can open. Slicing on given rectangles is what makes that impossible.

The tags are the other half of the trick. A four-direction walk needs to play
one row at a time, and a tag per row is enough: ``Animation.loop_range`` and
``play_direction`` already restrict playback to the tag containing the
playhead, so a walk sheet loops per direction with no change to the animation
engine whatsoever.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .animation import (
    DEFAULT_DURATION_MS,
    DIRECTION_ORDER,
    Animation,
    DirectionalLayout,
    Frame,
    Tag,
    Track,
)
from .layers import Layer, LayerStack

__all__ = ["document_from_atlas", "document_from_grid", "grid_rects", "walk_tags"]


def walk_tags() -> list[Tag]:
    """One looping tag per direction: ``walk_front`` 0-3, ``walk_left`` 4-7...

    Named with the direction rather than left as bare ``front``/``left``,
    because a document can carry several tagged spans and "left" alone stops
    meaning anything the moment a second cycle is added to it.
    """
    per = 4
    return [
        Tag(
            name=f"walk_{direction}",
            start=index * per,
            end=index * per + per - 1,
            loop=True,
        )
        for index, direction in enumerate(DIRECTION_ORDER)
    ]


def _cell_rect(cell: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (int(cell["x"]), int(cell["y"]), int(cell["w"]), int(cell["h"]))


def document_from_atlas(
    atlas_rgba: np.ndarray,
    cells: Sequence[Mapping[str, Any]],
    kind: str,
    *,
    track_name: str = "Sprite",
) -> Any:
    """One document whose frames are ``cells``, sliced out of ``atlas_rgba``.

    Refuses rather than repairs, in every case: an unknown kind, a cell count
    that does not match the grid, a rectangle off the edge of the atlas, or
    cells of differing sizes. Each of those means the sidecar and the PNG
    disagree, and a document assembled out of a disagreement is one the user
    edits for ten minutes before finding out.

    The result is *unsaved but clean*: it has no path, so the first Ctrl+S is a
    Save As, and an empty undo stack, so closing it immediately prompts about
    nothing. A draft on disk is not this document's file -- it is where this
    document came from.
    """
    layout = DirectionalLayout.of(kind)
    if layout is None:
        raise ValueError(f"{kind!r} is not a sprite sheet layout this build knows")
    if len(cells) != layout.frame_count:
        raise ValueError(
            f"a {layout.kind} sheet is {layout.frame_count} cells and this "
            f"sidecar has {len(cells)}"
        )

    return _document_from_rects(
        atlas_rgba,
        [_cell_rect(cell) for cell in cells],
        # A turnaround is four still views, not a cycle: tagging it would put
        # four one-frame loops in the timeline that mean nothing to play.
        tags=walk_tags() if layout.kind == "walk" else [],
        layout=layout,
        track_name=track_name,
    )


def _document_from_rects(
    atlas_rgba: np.ndarray,
    rects: Sequence[tuple[int, int, int, int]],
    *,
    tags: Sequence[Tag] = (),
    layout: DirectionalLayout | None = None,
    track_name: str = "Sprite",
) -> Any:
    """The slicing itself, shared by both doors into this module.

    Extracted rather than duplicated because the *refusals* are the valuable
    part and they are the same either way: cells of differing sizes and a
    rectangle off the edge of the atlas mean the same thing whether the
    rectangles came from a sidecar or from a grid the user typed. What differs
    between the callers is only how the rectangles were arrived at.
    """
    from .document import Document, matte_for
    from .undo import UNDO_BYTES, UndoStack

    if not rects:
        raise ValueError("a sprite sheet needs at least one cell")
    height, width = atlas_rgba.shape[:2]
    sizes = {(w, h) for _, _, w, h in rects}
    if len(sizes) != 1:
        raise ValueError("every cell of a sprite sheet is the same size")
    cell_w, cell_h = sizes.pop()
    if cell_w < 1 or cell_h < 1:
        raise ValueError("a cell has a positive size")
    for x, y, w, h in rects:
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(
                f"cell ({x}, {y}, {w}, {h}) is outside the {width}x{height} atlas"
            )

    track = Track(name=track_name)
    frames = [Frame(duration_ms=DEFAULT_DURATION_MS) for _ in rects]
    cel_map: dict[tuple[int, int], Layer] = {}
    for frame, (x, y, w, h) in zip(frames, rects, strict=True):
        # ``.copy()`` and not a view: a view would keep the whole atlas alive
        # behind every frame and, far worse, make two cells that happened to
        # overlap the same pixels -- so a stroke on one frame would appear on
        # another with no link to explain it.
        pixels = np.ascontiguousarray(atlas_rgba[y : y + h, x : x + w]).copy()
        cel_map[(track.uid, frame.uid)] = Layer(pixels=pixels, name=track_name)

    anim = Animation(
        tracks=[track],
        frames=frames,
        cels=cel_map,
        tags=list(tags),
        current=0,
        layout=layout,
    )
    doc = Document(
        stack=LayerStack(anim.layers_for(anim.frames[0], (cell_w, cell_h)), 0),
        history=UndoStack(UNDO_BYTES),
        anim=anim,
    )
    doc.matte = matte_for(doc.composite)
    # ``ora`` and no path: the document is an animation, and the still-image
    # formats cannot hold one -- so the format it *would* be saved as is the
    # one that can, and the dialog opens on it.
    doc.file_format = "ora"
    doc.path = None
    return doc


def grid_rects(
    size: tuple[int, int],
    cell: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
    padding: tuple[int, int] = (0, 0),
    count: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Row-major cell rectangles for a plain grid. Pure, so the popup can count.

    **The last column and the last row carry no trailing padding**, and that
    off-by-one is the whole of what makes this worth a function: a 4-across
    sheet of 32px cells with 2px between them is 4*32 + 3*2 = 134 wide, not
    4*34 = 136 -- so dividing by ``cell + padding`` finds three columns and
    silently drops the fourth. The gaps are counted separately below.

    Every refusal names what is wrong with the numbers rather than returning a
    short list: a grid that produced two frames from a sixteen-frame sheet is a
    document the user edits for ten minutes before noticing.
    """
    width, height = int(size[0]), int(size[1])
    cell_w, cell_h = int(cell[0]), int(cell[1])
    off_x, off_y = int(offset[0]), int(offset[1])
    pad_x, pad_y = int(padding[0]), int(padding[1])
    if cell_w < 1 or cell_h < 1:
        raise ValueError("a cell has a positive size")
    if off_x < 0 or off_y < 0:
        raise ValueError("an offset is zero or more pixels in from the top left")
    if pad_x < 0 or pad_y < 0:
        raise ValueError("padding is zero or more pixels between cells")
    # ``+ pad`` before the divide is the trailing-gap correction: it lends the
    # last cell the padding it does not have so the division can assume every
    # cell carries one.
    columns = (width - off_x + pad_x) // (cell_w + pad_x)
    rows = (height - off_y + pad_y) // (cell_h + pad_y)
    if columns < 1 or rows < 1:
        raise ValueError(
            f"a {cell_w}x{cell_h} cell does not fit in the {width}x{height} image "
            f"at that offset"
        )
    capacity = int(columns * rows)
    total = capacity if count is None else int(count)
    if total < 1:
        raise ValueError("a sheet needs at least one frame")
    if total > capacity:
        raise ValueError(
            f"that grid holds {capacity} cells and {total} were asked for"
        )
    return [
        (
            off_x + (index % columns) * (cell_w + pad_x),
            off_y + (index // columns) * (cell_h + pad_y),
            cell_w,
            cell_h,
        )
        for index in range(total)
    ]


def document_from_grid(
    atlas_rgba: np.ndarray,
    cell: tuple[int, int],
    offset: tuple[int, int] = (0, 0),
    padding: tuple[int, int] = (0, 0),
    count: int | None = None,
    *,
    track_name: str = "Sprite",
) -> Any:
    """A plain image sliced on a typed grid, row-major, as an animation.

    ``document_from_atlas``'s sibling for a sheet that arrived from anywhere
    else -- a download, another tool, a scan. The layout is **None** and that
    is deliberate: a ``DirectionalLayout`` is a claim that these cells are four
    named directions in a fixed grid, which is something the generator knows
    and a user typing a cell size does not. Without it the document is an
    ordinary animation, which is exactly what an arbitrary sheet is.
    """
    height, width = atlas_rgba.shape[:2]
    rects = grid_rects(
        (int(width), int(height)), cell, offset, padding, count
    )
    return _document_from_rects(atlas_rgba, rects, track_name=track_name)
