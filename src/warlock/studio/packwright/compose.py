"""Painting the atlas, and the gutter around each sprite.

**Extrude replicates, it does not smear.** Each sprite's outermost row and
column are copied outward into the gutter so that a filtered texture sampling
just past a sprite's edge finds that sprite's own colour rather than its
neighbour's. The four corners come free by extruding vertically *after* the side
columns are already written, which is why the order in :func:`_extrude` matters
and is not tidiness.

The room this needs is guaranteed upstream rather than checked here:
``PackSettings`` refuses ``padding < extrude * 2``, and both layouts leave at
least one ``padding`` on every side of every sprite -- including against the
edge of the atlas, which is the case an un-offset MaxRects pack gets wrong and
which only shows as a one-pixel artefact at the atlas border.
"""

from __future__ import annotations

import io

import numpy as np

from .layout import Frame, Layout
from .sources import Sprite


def _extrude(atlas: np.ndarray, frame: Frame, margin: int) -> None:
    x, y, w, h = frame.x, frame.y, frame.w, frame.h
    # Sides first: one-pixel-wide slices broadcast across the gutter's width.
    atlas[y : y + h, x - margin : x] = atlas[y : y + h, x : x + 1]
    atlas[y : y + h, x + w : x + w + margin] = atlas[y : y + h, x + w - 1 : x + w]
    # Then top and bottom across the *widened* span, which carries the columns
    # just written into the corners with no separate corner case.
    span = slice(x - margin, x + w + margin)
    atlas[y - margin : y, span] = atlas[y : y + 1, span]
    atlas[y + h : y + h + margin, span] = atlas[y + h - 1 : y + h, span]


def compose(sprites: list[Sprite], layout: Layout) -> np.ndarray:
    """The finished RGBA atlas.

    A frame naming a sprite that is not present raises rather than leaving a
    hole: an atlas with an invisible gap looks like an art problem and sends the
    user looking in the wrong place -- the rule ``pipelines.sheet.pack`` already
    follows for a missing rendered frame.
    """
    by_key = {sprite.key: sprite for sprite in sprites}
    atlas = np.zeros((layout.height, layout.width, 4), dtype=np.uint8)
    for frame in layout.frames:
        sprite = by_key.get(frame.key)
        if sprite is None:
            raise ValueError(f"no sprite for packed frame {frame.key!r}")
        tx, ty, tw, th = frame.trim
        atlas[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w] = sprite.pixels[
            ty : ty + th, tx : tx + tw
        ]
        if layout.extrude:
            _extrude(atlas, frame, layout.extrude)
    return atlas


def png_bytes(pixels: np.ndarray) -> bytes:
    """The atlas as a PNG. Pillow imported here rather than at module scope,
    the rule the whole package follows."""
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8), "RGBA").save(out, "PNG")
    return out.getvalue()
