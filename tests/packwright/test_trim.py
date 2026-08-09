"""Where the alpha stops -- and the parity test that justifies a second answer.

``pipelines.sheet.measure_trim`` has been the repo's definition of "trim" since
sprite sheets got a sidecar. This package computes the same thing on a numpy
array because a packer measures hundreds of sprites and decoding each into a PIL
image is the wrong price. Without the parity assertion below that is a fork, and
a packer whose idea of the subject differs by a pixel from the sidecar's
produces atlases that are correct and metadata that is not.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import sheet as sheetlib
from warlock.studio.packwright import trim


def _blank(w: int = 12, h: int = 9) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _theirs(pixels: np.ndarray):
    box = sheetlib.measure_trim(Image.fromarray(np.ascontiguousarray(pixels), "RGBA"))
    return None if box is None else (box["x"], box["y"], box["w"], box["h"])


@pytest.mark.parametrize(
    "mark",
    [
        (slice(0, 1), slice(0, 1)),        # one corner pixel
        (slice(3, 6), slice(2, 9)),        # an interior block
        (slice(0, 9), slice(0, 12)),       # the whole canvas
        (slice(8, 9), slice(11, 12)),      # the opposite corner
        (slice(4, 5), slice(0, 12)),       # a full row
    ],
)
def test_it_agrees_with_the_sprite_sheet_pipelines_answer(mark):
    pixels = _blank()
    pixels[mark[0], mark[1]] = (255, 0, 0, 255)
    assert trim.alpha_bbox(pixels) == _theirs(pixels)


def test_a_fully_transparent_sprite_has_no_box_in_either():
    pixels = _blank()
    assert trim.alpha_bbox(pixels) is None
    assert _theirs(pixels) is None


def test_a_pixel_with_colour_but_no_alpha_is_not_content():
    """Both answers key on alpha alone, which is what makes an erased region
    genuinely gone rather than invisible-but-packed."""
    pixels = _blank()
    pixels[4, 4] = (255, 0, 0, 0)
    assert trim.alpha_bbox(pixels) is None
    assert _theirs(pixels) is None


def test_a_single_unit_of_alpha_counts():
    pixels = _blank()
    pixels[2, 3, 3] = 1
    assert trim.alpha_bbox(pixels) == (3, 2, 1, 1)


# --- trim_rect ----------------------------------------------------------------


def test_an_empty_sprite_packs_as_a_one_by_one_and_is_marked():
    """A blank frame in the middle of a clip is a real frame -- it is the pause
    -- and silently dropping it renumbers every frame after it. One pixel
    rather than zero, because a zero-area rectangle is a special case for the
    packer, the compositor and every engine downstream."""
    x, y, w, h, empty = trim.trim_rect(_blank())
    assert (x, y, w, h) == (0, 0, trim.EMPTY_SIZE, trim.EMPTY_SIZE)
    assert empty is True


def test_trimming_off_returns_the_whole_image():
    """So the caller needs no branch: an untrimmed sprite is one whose trim
    rectangle happens to be its own bounds, and every consumer of
    ``spriteSourceSize`` works unchanged."""
    pixels = _blank()
    pixels[4, 4] = (255, 0, 0, 255)
    assert trim.trim_rect(pixels, enabled=False) == (0, 0, 12, 9, False)
    assert trim.trim_rect(pixels, enabled=True) == (4, 4, 1, 1, False)


def test_an_empty_sprite_is_empty_whether_or_not_trimming_is_on():
    """There is nothing to measure either way, and a caller that turned
    trimming off has not asked for a blank 200x200 rectangle in its atlas."""
    assert trim.trim_rect(_blank(), enabled=False)[4] is True
