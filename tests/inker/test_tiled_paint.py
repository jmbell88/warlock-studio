"""Tiled painting, asserted on the torus rather than on the plumbing.

The property that matters to a tile artist is not "the brush also draws at
x - width". It is that **the document behaves like a torus**: a stroke and the
same stroke moved by a whole tile produce the same drawing, up to a roll. That
is one assertion covering the stamp, the spacing walk, the coverage buffer, the
selection clip and the undo rect at once, and it is the one that fails if any of
them wraps with arithmetic of its own.

Everything here is parameterized over the three wrapping modes, and each of them
carries its own negative control: an axis that is *not* wrapped must clamp
exactly as it did before this feature existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import tiling

SIZE = (24, 24)
RED = (255, 0, 0, 255)

#: A whole-tile-ish delta per mode: only the wrapped axes may be shifted, since
#: rolling an unwrapped one is not a symmetry of the document.
DELTAS = {"x": (7, 0), "y": (0, 7), "both": (7, 5)}


def _painted(points, *, wrap="off", canvas=SIZE, colour=RED, doc=None, **kw):
    """One stroke through the public API. -> the active layer's pixels.

    ``canvas`` rather than ``size``, because ``size`` is the *brush* diameter
    that several of these pass through to ``begin_stroke``.
    """
    doc = doc if doc is not None else inker.Document.blank(*canvas)
    doc.begin_stroke((float(points[0][0]), float(points[0][1])), colour, wrap=wrap, **kw)
    for point in points[1:]:
        doc.stroke_to((float(point[0]), float(point[1])))
    doc.end_stroke()
    return doc.stack.active.pixels


def _shift(points, delta):
    return [(x + delta[0], y + delta[1]) for x, y in points]


# --- translation equivariance -----------------------------------------------


@pytest.mark.parametrize("wrap", ["x", "y", "both"])
@pytest.mark.parametrize("nib", ["soft", "pixel"])
def test_a_tiled_stroke_is_translation_equivariant_on_the_torus(wrap, nib):
    """The one assertion the whole feature is worth. If the stamp, the walk or
    the coverage buffer wrapped with its own arithmetic, one of the deltas below
    would come out a pixel off -- and a pixel at the seam is the only pixel
    anybody making a tile is looking at."""
    delta = DELTAS[wrap]
    path = [(4, 6), (14, 9), (11, 17)]
    here = _painted(path, wrap=wrap, nib=nib)
    there = _painted(_shift(path, delta), wrap=wrap, nib=nib)
    rolled = np.roll(here, (delta[1], delta[0]), axis=(0, 1))
    assert np.array_equal(rolled, there)


@pytest.mark.parametrize("wrap", ["x", "y", "both"])
def test_a_seam_crossing_stroke_is_the_centre_stroke_rolled(wrap):
    """Stated the other way round: a stroke deliberately drawn *over* the seam
    is exactly the same drawing as one in the middle, moved."""
    delta = DELTAS[wrap]
    centre = [(12, 12), (16, 14)]
    over = _shift(centre, (delta[0] + SIZE[0] // 2 - 12, delta[1]))
    a = _painted(centre, wrap=wrap)
    b = _painted(over, wrap=wrap)
    shift = (over[0][1] - centre[0][1], over[0][0] - centre[0][0])
    assert np.array_equal(np.roll(a, shift, axis=(0, 1)), b)


# --- the off path -----------------------------------------------------------


@pytest.mark.parametrize("wrap", ["off", "x", "y", "both"])
def test_an_interior_stroke_is_identical_however_the_flag_is_set(wrap):
    """Off-path parity, and the reason ``tiling.spans`` returns the plain clip
    rather than something equivalent to it: a stroke nowhere near an edge must
    be byte-identical, or every existing pixel test becomes a test of whether
    the toggle happens to be off."""
    path = [(8, 8), (14, 12), (10, 15)]
    assert np.array_equal(_painted(path), _painted(path, wrap=wrap))


def test_wrapping_off_still_clips_a_stroke_at_the_edge():
    pixels = _painted([(1, 1)], wrap="off", nib="square", size=9)
    alpha = pixels[..., 3]
    assert int(alpha[:, -1].max()) == 0 and int(alpha[-1, :].max()) == 0


# --- per-axis negative controls ---------------------------------------------


def test_x_only_tiling_wraps_sideways_and_clamps_at_the_top():
    """The per-axis control C15 exists for: a ground tile repeats horizontally
    and must not repeat vertically, so a stroke off the top edge is simply
    cropped."""
    pixels = _painted([(1, 1)], wrap="x", nib="square", size=9)
    alpha = pixels[..., 3]
    assert int(alpha[:, -1].max()) > 0  # wrapped round to the right edge
    assert int(alpha[-1, :].max()) == 0  # nothing arrived at the bottom


def test_y_only_tiling_is_the_mirror_of_that():
    pixels = _painted([(1, 1)], wrap="y", nib="square", size=9)
    alpha = pixels[..., 3]
    assert int(alpha[-1, :].max()) > 0
    assert int(alpha[:, -1].max()) == 0


# --- what deliberately does not wrap ----------------------------------------


def test_smudge_falls_back_to_clamped_even_with_tiling_on():
    """Deliberate: the pickup buffer trails the brush, and "the pixels it just
    passed over" has no answer when the brush is in two places at once."""
    doc = inker.Document.blank(*SIZE)
    doc.stack.active.pixels[:] = 255
    doc.stack.active.pixels[:, :3] = (0, 0, 0, 255)  # something to push along
    doc.invalidate_all()
    before = doc.stack.active.pixels.copy()
    doc.begin_stroke((2.0, 12.0), RED, mode="smudge", size=9, wrap="both", strength=1.0)
    doc.stroke_to((7.0, 12.0))
    doc.end_stroke()
    after = doc.stack.active.pixels
    assert not np.array_equal(after[:, 3:8], before[:, 3:8])  # it did something
    assert np.array_equal(after[:, -3:], before[:, -3:])  # and nothing wrapped


def test_a_gradient_takes_no_wrap_argument_at_all():
    """A ramp has endpoints; wrapping one puts the last stop against the first,
    which is a hard edge at the seam -- the thing tiled mode removes."""
    doc = inker.Document.blank(*SIZE)
    with pytest.raises(TypeError):
        doc.gradient((0, 0), (10, 10), RED, (0, 0, 0, 255), wrap="both")


# --- the alpha lock ---------------------------------------------------------


def test_the_alpha_lock_holds_on_every_wrapped_piece():
    """Free by construction -- the lock is applied inside ``_resolve``, which
    runs per piece -- and asserted because "free by construction" is a claim
    about code that has just been restructured."""
    doc = inker.Document.blank(*SIZE)
    layer = doc.stack.active
    layer.pixels[..., 3] = 255
    layer.pixels[..., :3] = 40
    layer.alpha_lock = True
    doc.invalidate_all()
    before_alpha = layer.pixels[..., 3].copy()
    _painted([(0.5, 12.0), (0.5, 13.0)], wrap="both", size=9, doc=doc)
    assert np.array_equal(layer.pixels[..., 3], before_alpha)
    # And the colour really did wrap, so the lock is not passing vacuously.
    assert int(layer.pixels[:, -1, 0].max()) > 40


# --- undo -------------------------------------------------------------------


def test_a_seam_crossing_stroke_is_exactly_one_undo_step():
    """One ``PatchEdit`` over the union of the pieces, not one per piece: a tile
    is a small canvas and a multi-rect edit type would need eviction accounting
    of its own to save kilobytes."""
    doc = inker.Document.blank(*SIZE)
    blank = doc.stack.active.pixels.copy()
    _painted([(0.5, 12.0), (2.0, 12.0)], wrap="both", size=9, doc=doc)
    assert doc.history.can_undo
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, blank)
    # ``head`` is a serial, not a depth, so "exactly one" is asserted as *no
    # second step to undo* on a document that had none before the stroke.
    assert not doc.history.can_undo


# --- fill -------------------------------------------------------------------


def _striped():
    """A canvas that is white but for one opaque black column, so a flood from
    column 0 reaches column width-1 only by going round the outside."""
    pixels = np.full((SIZE[1], SIZE[0], 4), 255, dtype=np.uint8)
    pixels[:, 12] = (0, 0, 0, 255)
    return inker.Document.from_pixels(pixels)


def test_a_wrapped_fill_joins_the_two_halves_across_the_seam():
    doc = _striped()
    doc.fill((0, 5), RED, thresh=0, wrap="x")
    painted = doc.stack.active.pixels
    assert tuple(painted[5, 0]) == RED
    assert tuple(painted[5, SIZE[0] - 1]) == RED  # round the outside
    assert tuple(painted[5, 12]) == (0, 0, 0, 255)  # the wall still stops it


def test_the_same_fill_untiled_stops_at_the_canvas_edge():
    doc = _striped()
    doc.fill((0, 5), RED, thresh=0, wrap="off")
    painted = doc.stack.active.pixels
    assert tuple(painted[5, 0]) == RED
    assert tuple(painted[5, SIZE[0] - 1]) == (255, 255, 255, 255)


def test_y_tiling_does_not_join_a_left_right_region():
    """The per-axis control again, on the wand's seam seeds this time."""
    doc = _striped()
    doc.fill((0, 5), RED, thresh=0, wrap="y")
    assert tuple(doc.stack.active.pixels[5, SIZE[0] - 1]) == (255, 255, 255, 255)


def test_a_non_contiguous_wand_answers_the_same_tiled_or_not():
    """Similarity is per pixel and has no notion of an edge, so wrapping can
    only ever change *contiguity*."""
    doc = _striped()
    plain = inker.magic_wand(doc.composite, (0, 5), tolerance=0, contiguous=False)
    tiled = inker.magic_wand(
        doc.composite, (0, 5), tolerance=0, contiguous=False, wrap="both"
    )
    assert np.array_equal(plain.mask, tiled.mask)


# --- shapes -----------------------------------------------------------------


def test_a_wide_line_at_the_edge_wraps_its_overhang():
    doc = inker.Document.blank(*SIZE)
    doc.shape("line", (0, 2), (0, 20), RED, 7, wrap="x")
    alpha = doc.stack.active.pixels[..., 3]
    assert int(alpha[10, -1]) > 0
    assert int(alpha[10, 0]) > 0


def test_the_same_line_untiled_loses_its_overhang():
    doc = inker.Document.blank(*SIZE)
    doc.shape("line", (0, 2), (0, 20), RED, 7, wrap="off")
    assert int(doc.stack.active.pixels[..., 3][10, -1]) == 0


def test_a_shape_drawn_a_whole_tile_away_lands_on_the_canvas():
    """The canvas subtracts the tile offset at press, but a *drag* can still
    end one tile out -- the release point is where the cursor is, not where the
    press was."""
    doc = inker.Document.blank(*SIZE)
    doc.shape("rect", (SIZE[0] + 4, 4), (SIZE[0] + 10, 10), RED, 1, wrap="x", filled=True)
    assert int(doc.stack.active.pixels[..., 3][7, 6]) > 0


def test_an_interior_shape_is_identical_however_the_flag_is_set():
    """The shape tool's off-path parity: the plane is the canvas at the origin
    and the fold is the identity, so this is byte equality rather than
    approximate agreement."""
    plain = inker.Document.blank(*SIZE)
    plain.shape("ellipse", (5, 5), (16, 16), RED, 3)
    tiled = inker.Document.blank(*SIZE)
    tiled.shape("ellipse", (5, 5), (16, 16), RED, 3, wrap="both")
    assert np.array_equal(plain.stack.active.pixels, tiled.stack.active.pixels)


# --- the selection stays canonical ------------------------------------------


def test_a_selection_still_scopes_a_wrapped_stroke():
    """Free, and worth pinning: the pieces are dest rects in canvas space, so
    the clip is the same multiply it always was and the selection needs no
    notion of tiles."""
    doc = inker.Document.blank(*SIZE)
    doc.select(inker.SelectionMask.from_rect(SIZE, (0, 0, SIZE[0], 8)))
    _painted([(0.5, 4.0), (1.0, 4.0)], wrap="both", size=9, doc=doc)
    alpha = doc.stack.active.pixels[..., 3]
    assert int(alpha[:8, -1].max()) > 0  # wrapped, inside the selection
    assert int(alpha[8:].max()) == 0  # and nothing outside it


def test_axes_of_is_what_the_document_threads():
    """One spelling of the mode string reaching the engine, so a typo in a pane
    is a ValueError rather than silently untiled painting."""
    doc = inker.Document.blank(*SIZE)
    with pytest.raises(ValueError):
        doc.begin_stroke((1.0, 1.0), RED, wrap="sideways")
    assert tiling.axes_of("both") == (True, True)
