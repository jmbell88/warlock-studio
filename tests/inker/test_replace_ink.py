"""The copy-colour ink: what "replace" writes, and where it lives.

Two things make this a mode rather than a flag. It writes the foreground RGBA
*verbatim* at full coverage -- alpha included, so it can paint transparency down
-- and it does that from the pre-stroke pixels, which is what stops a stroke
crossing itself from compounding. Both are asserted on exact bytes, because
"nearly the colour I picked" is precisely the failure a copy ink exists to
remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import brush

SIZE = (16, 16)
BACKDROP = (200, 100, 50, 255)
INK = (10, 20, 30, 40)


def _doc(fill=BACKDROP):
    doc = inker.Document.blank(*SIZE)
    doc.stack.active.pixels[:] = np.asarray(fill, dtype=np.uint8)
    doc.invalidate_all()
    return doc


def _narrow(value: float) -> int:
    """``composite.to_uint8_255``, spelled out: the editor narrows with
    ``clip(x + 0.5)`` everywhere, and an expected value computed with a
    different rounding would be testing Python's ``int`` instead of the ink."""
    return int(min(255.0, max(0.0, value + 0.5)))


def _stroke(doc, path, **kw):
    options = {"mode": "replace", "nib": "square", "size": 3, "colour": INK}
    options.update(kw)
    colour = options.pop("colour")
    doc.begin_stroke((float(path[0][0]), float(path[0][1])), colour, **options)
    for point in path[1:]:
        doc.stroke_to((float(point[0]), float(point[1])))
    doc.end_stroke()
    return doc.stack.active.pixels


def test_replace_is_the_fifth_mode_and_the_others_are_untouched():
    """The first five in order rather than the whole tuple: modes are appended
    (``shade`` is the sixth), and what this pins is that adding one neither
    reorders nor redefines the five that were already there."""
    assert brush.MODES[:5] == ("paint", "erase", "blur", "smudge", "replace")


def test_at_full_coverage_it_writes_the_colour_verbatim():
    """Including alpha 40 over an opaque backdrop: a normal paint would leave
    the pixel opaque and merely tinted, which is the whole distinction."""
    pixels = _stroke(_doc(), [(8, 8)])
    assert tuple(int(v) for v in pixels[8, 8]) == INK


def test_paint_over_the_same_pixel_leaves_it_composited_rather_than_copied():
    """The control: without it this file would pass against a paint mode that
    happened to land near the same numbers."""
    pixels = _stroke(_doc(), [(8, 8)], mode="paint")
    assert tuple(int(v) for v in pixels[8, 8]) != INK
    assert int(pixels[8, 8, 3]) == 255


def test_a_stroke_crossing_itself_is_one_dab_and_not_two():
    """The reason ``replace`` is on the ``_resolve`` side of the blur/smudge
    branch: coverage is the whole record of the stroke and is applied once, so
    a half-opacity replace does not creep towards the ink wherever the stroke
    doubles back."""
    once = _stroke(_doc(), [(8, 8)], opacity=0.5)[8, 8].copy()
    twice = _stroke(_doc(), [(8, 8), (10, 8), (8, 8)], opacity=0.5)[8, 8]
    assert np.array_equal(once, twice)


def test_it_paints_alpha_down_and_a_second_pass_does_not_take_it_further():
    """Only a mode reading ``before`` can do this. A live-pixels mode -- blur
    and smudge are the two -- would eat the alpha again on the second pass."""
    doc = _doc()
    pixels = _stroke(doc, [(8, 8), (10, 8), (8, 8)], opacity=0.5)
    # 255 + (40 - 255) * 0.5 = 147.5, and the shared narrowing is
    # ``clip(x + 0.5)``, so 148 -- not 147.5 applied twice, which is the point.
    assert int(pixels[8, 8, 3]) == 148


def test_partial_coverage_lerps_rather_than_snapping_to_a_hard_edge():
    """The documented divergence from Aseprite, asserted so it stays
    deliberate: this repository's doctrine is that feathering means one thing,
    so opacity, a soft nib and a feathered selection all soften a replace stroke
    exactly as they soften a paint one."""
    pixels = _stroke(_doc(), [(8, 8)], opacity=0.25)
    expected = tuple(
        _narrow(b + (c - b) * 0.25) for b, c in zip(BACKDROP, INK, strict=True)
    )
    assert tuple(int(v) for v in pixels[8, 8]) == expected


def test_a_feathered_selection_softens_it_the_same_way():
    doc = _doc()
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
    mask[:, :] = 128
    doc.select(inker.SelectionMask(mask))
    pixels = _stroke(doc, [(8, 8)])
    expected = tuple(
        _narrow(b + (c - b) * (128 / 255.0))
        for b, c in zip(BACKDROP, INK, strict=True)
    )
    assert tuple(int(v) for v in pixels[8, 8]) == expected


def test_a_selection_still_scopes_it():
    doc = _doc()
    doc.select(inker.SelectionMask.from_rect(SIZE, (0, 0, SIZE[0], 4)))
    pixels = _stroke(doc, [(8, 8)])
    assert tuple(int(v) for v in pixels[8, 8]) == BACKDROP


def test_the_alpha_lock_survives_the_new_branch():
    """Applied after the branch rather than inside it, which is what makes it
    free for every mode -- and the one thing a new mode could quietly bypass."""
    doc = _doc()
    doc.stack.active.alpha_lock = True
    pixels = _stroke(doc, [(8, 8)])
    assert int(pixels[8, 8, 3]) == 255
    assert tuple(int(v) for v in pixels[8, 8, :3]) == INK[:3]


def test_an_indexed_document_snaps_the_colour_and_keeps_the_alpha():
    """The snap runs at commit and is RGB-only, so a replace stroke can still
    paint alpha down on an indexed document -- which is the interaction that
    would otherwise silently undo half of what the mode is for."""
    doc = _doc()
    doc.palette = [(0, 0, 0, 255), (255, 255, 255, 255)]
    pixels = _stroke(doc, [(8, 8)])
    assert tuple(int(v) for v in pixels[8, 8, :3]) == (0, 0, 0)
    assert int(pixels[8, 8, 3]) == 40


def test_it_is_one_undo_step_that_restores_the_backdrop_exactly():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    _stroke(doc, [(6, 6), (10, 10)])
    assert doc.history.can_undo
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, before)
    assert not doc.history.can_undo


def test_it_wraps_with_the_rest_of_the_paint_modes():
    """Free: the wrap is in ``_stamp`` and the mode is in ``_resolve``, so the
    two compose without either knowing about the other."""
    doc = _doc()
    pixels = _stroke(doc, [(0, 8)], size=5, wrap="x")
    assert tuple(int(v) for v in pixels[8, SIZE[0] - 1]) == INK


@pytest.mark.parametrize("mode", ["blur", "smudge"])
def test_the_filter_modes_are_not_reached_through_the_replace_branch(mode):
    """A guard on the branch order in ``_stamp``: blur and smudge read the live
    layer and must never fall into the coverage path, whatever was added beside
    it."""
    doc = _doc()
    pixels = _stroke(doc, [(8, 8)], mode=mode, strength=1.0)
    assert tuple(int(v) for v in pixels[8, 8]) != INK
