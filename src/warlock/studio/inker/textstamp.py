"""Text, rasterised into pixels and then forgotten.

**There are no text objects and no text layers**, and that is a decision rather
than a first instalment (``docs/INVARIANTS.md``, Aseprite divergence 17). A live
text object is not a drawing feature, it is a second document model: the glyphs
have to survive a save, a crop, a scale, a flip and an undo, every filter has to
decide whether it applies to them, the exporters have to flatten them, and the
font a file was authored with has to still exist on the machine that opens it.
What a pixel editor is actually asked for is "put this word on the canvas so I
can paint around it", and that is a stamp -- one array of pixels, delivered
through the floating buffer (``_doc_selection.float_pixels``), positioned and
committed by the machinery a paste already uses. Re-editing text is retyping it,
which is also how the eraser, the brush and every other tool here work.

So this module is a pure function and holds no state at all. It takes a plain
font *path* rather than anything from ``studio.fonts`` -- the engine may not
import the UI's type ramp, and a path is the whole of what a rasteriser needs.
The caller chooses the file; this decodes it or says it could not.

Two things about the rendering are load-bearing.

**The mask is rendered at the requested fidelity rather than post-processed
into it.** ``antialias=False`` draws into a Pillow image of mode ``"1"``, which
switches FreeType's own rasteriser to monochrome: the glyph outline is filled by
pixel-centre coverage, which is what a bitmap font looks like and what pixel art
wants. Thresholding an antialiased mask instead would be a *different* shape --
a rounded stem lands half a pixel either side of where the monochrome
rasteriser puts it, and the result is the lumpy outline every "pixel font"
plugin that takes the cheap route produces. It also keeps the promise the pixel
nibs make: with ``antialias=False`` and an opaque colour every alpha in the
result is 0 or 255, so a stamp on an indexed document adds no colours.

**The box is measured, padded and then cropped back to the ink.** A glyph is
allowed to draw outside its advance -- an italic overhang, an accent, the tail
of a *j* -- and ``multiline_textbbox`` reports a box that a hinted outline can
still exceed by a fraction of a pixel at either edge. So the render surface is
the measured box plus one pixel of :data:`SLACK` on every side, and the pixels
that come back are cropped to whatever actually got ink. Without the slack a
descender loses its last row; without the crop every stamp carries a
transparent margin whose width depends on the font, which the user then has to
work around when positioning it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import pixelguard

__all__ = ["MAX_SIZE", "MIN_SIZE", "SLACK", "text_stamp"]

#: Point sizes the stamp will attempt. The floor is where a hinted outline
#: stops being legible at all; the ceiling is a guard rather than a judgement --
#: a mistyped size must not ask FreeType for a 40,000 px glyph and a gigabyte
#: of mask.
MIN_SIZE = 4
MAX_SIZE = 512

#: Pixels of room left around the measured box before the glyphs are drawn into
#: it. One, on every side; see the module docstring.
SLACK = 1


def _rgba(colour: Sequence[int]) -> tuple[int, int, int, int]:
    """The ink colour as four ints. A three-tuple is taken as opaque, which is
    what every caller that has one means by it."""
    parts = [int(c) for c in colour][:4]
    while len(parts) < 4:
        parts.append(255)
    return (parts[0], parts[1], parts[2], parts[3])


def text_stamp(
    text: str,
    font_path: Any,
    size_px: int,
    colour: Sequence[int],
    antialias: bool = True,
) -> np.ndarray | None:
    """``text`` rendered in ``colour``, as an RGBA array cropped to its ink.

    ``None`` -- never an exception and never a blank array -- for every way this
    can decline: a font file that is missing, unreadable or not a font, a size
    outside :data:`MIN_SIZE`..:data:`MAX_SIZE`, and text that produces no ink at
    all (empty, or nothing but whitespace and newlines). One answer, because
    every one of them is the same thing from the caller's side -- there is
    nothing to float -- and the pane turns it into one toast. Raising instead
    would put a font the user picked from a system directory, which may be a
    broken file they have never opened, on the frame thread's exception path.

    The RGB is written across the whole array rather than only under the ink, so
    the transparent margin carries the ink's own colour: compositing straight
    alpha over a background reads RGB wherever alpha is non-zero, and a
    zero-alpha *black* fringe around white text is exactly the halo that shows
    up the first time somebody scales the stamp.
    """
    from PIL import Image, ImageDraw, ImageFont

    size = int(size_px)
    if size < MIN_SIZE or size > MAX_SIZE:
        return None
    body = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if not body.strip():
        return None
    try:
        font = ImageFont.truetype(str(font_path), size)
    except (OSError, ValueError, TypeError):
        # Missing, unreadable, not a font, or a face index this file does not
        # have. Pillow spells all four differently and the caller cares about
        # none of the differences.
        return None

    # Measured on a scratch draw rather than on the surface being rendered to,
    # because the surface's size is what is being measured.
    scratch = ImageDraw.Draw(Image.new("L", (1, 1)))
    try:
        x0, y0, x1, y1 = scratch.multiline_textbbox((0, 0), body, font=font)
    except (OSError, ValueError):
        return None
    width = int(x1 - x0) + 2 * SLACK
    height = int(y1 - y0) + 2 * SLACK
    if width <= 0 or height <= 0:
        return None
    # The surface below is allocated from a *measured* string at a size the
    # user typed: a 4000-point font, or a paragraph pasted into the field, and
    # nothing between the two and ``Image.new``. The same ceiling every other
    # allocation in this package answers to -- and a refusal by name, since
    # this one is reachable by typing rather than by opening a hostile file.
    pixelguard.check(width, height, "this text at this size")

    # Mode "1" is the whole of the ``antialias=False`` implementation: Pillow
    # sets ``ImageDraw.fontmode`` from the image it is drawing into, and "1"
    # is what puts FreeType in monochrome.
    mode = "L" if antialias else "1"
    surface = Image.new(mode, (width, height), 0)
    draw = ImageDraw.Draw(surface)
    draw.multiline_text(
        (SLACK - x0, SLACK - y0), body, font=font, fill=(255 if antialias else 1)
    )
    coverage = np.asarray(surface)
    if coverage.dtype == np.bool_:
        coverage = np.where(coverage, 255, 0).astype(np.uint8)
    coverage = coverage.astype(np.uint8, copy=False)

    rows = np.flatnonzero(coverage.any(axis=1))
    cols = np.flatnonzero(coverage.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        # Ink-free after all: a string of spaces, or a font with no glyph for
        # any character in it.
        return None
    coverage = coverage[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    red, green, blue, alpha = _rgba(colour)
    out = np.empty((*coverage.shape, 4), dtype=np.uint8)
    out[..., 0] = red
    out[..., 1] = green
    out[..., 2] = blue
    # The same multiply ``_doc_selection._masked_alpha`` does, for the same
    # reason: coverage and the colour's own alpha are both 8-bit coverage, and
    # a fully opaque colour leaves the mask exactly as it was rendered -- which
    # is what keeps the ``antialias=False`` promise of 0-or-255.
    out[..., 3] = (coverage.astype(np.float32) * alpha / 255.0).astype(np.uint8)
    return out
