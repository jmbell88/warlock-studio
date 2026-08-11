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

__all__ = ["document_from_atlas", "walk_tags"]


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
    from .document import Document, matte_for
    from .undo import UNDO_BYTES, UndoStack

    layout = DirectionalLayout.of(kind)
    if layout is None:
        raise ValueError(f"{kind!r} is not a sprite sheet layout this build knows")
    if len(cells) != layout.frame_count:
        raise ValueError(
            f"a {layout.kind} sheet is {layout.frame_count} cells and this "
            f"sidecar has {len(cells)}"
        )

    height, width = atlas_rgba.shape[:2]
    rects = [_cell_rect(cell) for cell in cells]
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
        # A turnaround is four still views, not a cycle: tagging it would put
        # four one-frame loops in the timeline that mean nothing to play.
        tags=walk_tags() if layout.kind == "walk" else [],
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
