"""A dithered gradient lands on its stops, and an undithered one is untouched.

The point of dithering a ramp is that it stops being a ramp of *colours*: every
pixel comes out as one of the stops, so the result survives an indexed
document's snap and a pixel-art palette exactly. Which means the assertions are
about the stop set, not about smoothness.

The off-path pin at the bottom is the load-bearing one. Dithering is applied
inside ``gradient.render`` between adjacent stops, which is a branch the
undithered call must never enter -- so ``dither=None`` has to produce the exact
bytes it produced before the option existed.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import dither as dith
from warlock.studio.inker import gradient as grad
from warlock.studio.inker.document import Document

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RED = (255, 0, 0, 255)
SIZE = (32, 16)


def _bytes(rgba: np.ndarray) -> set[tuple[int, int, int]]:
    scaled = np.rint(rgba[..., :3] * 255.0).astype(int)
    return {tuple(int(c) for c in px) for px in scaled.reshape(-1, 3)}


# --- the promise -------------------------------------------------------------


def test_a_dithered_ramp_uses_only_its_stop_colours():
    rgba, _weight = grad.render(
        SIZE, (0, 0), (31, 0), BLACK, WHITE, "linear", dither="bayer4"
    )
    assert _bytes(rgba) <= {BLACK[:3], WHITE[:3]}


def test_the_undithered_ramp_uses_the_colours_between_them():
    """The control: without this the assertion above would pass on a render
    that was quietly doing nothing."""
    rgba, _weight = grad.render(SIZE, (0, 0), (31, 0), BLACK, WHITE, "linear")
    assert len(_bytes(rgba)) > 2


def test_a_three_stop_ramp_only_ever_picks_two_adjacent_stops():
    stops = [(0.0, BLACK), (0.5, RED), (1.0, WHITE)]
    rgba, _weight = grad.render(
        SIZE, (0, 0), (31, 0), stops=stops, dither="bayer8"
    )
    assert _bytes(rgba) <= {BLACK[:3], RED[:3], WHITE[:3]}
    # And the far ends are pure: a pixel past the last stop has nothing to mix
    # with, and picking up a speck of the middle colour there is the artefact
    # centred thresholds exist to prevent.
    assert tuple(np.rint(rgba[0, 0, :3] * 255).astype(int)) == BLACK[:3]
    assert tuple(np.rint(rgba[0, -1, :3] * 255).astype(int)) == WHITE[:3]


def test_a_radial_ramp_dithers_the_same_way():
    """Kind-agnostic by construction: ``dithered`` takes the ramp parameter and
    knows nothing about how it was computed."""
    rgba, _weight = grad.render(
        SIZE, (16, 8), (31, 8), BLACK, WHITE, "radial", dither="bayer4"
    )
    assert _bytes(rgba) <= {BLACK[:3], WHITE[:3]}


def test_alpha_dithers_with_its_stop_rather_than_being_blended_under_it():
    """Otherwise the drag produces a soft gradient with hard colour banding on
    top of it -- two different gradients at once."""
    stops = [(0.0, (255, 0, 0, 0)), (1.0, (255, 0, 0, 255))]
    _rgba, weight = grad.render(SIZE, (0, 0), (31, 0), stops=stops, dither="bayer4")
    assert set(np.unique(weight).tolist()) <= {0.0, 1.0}


def test_bayer2_and_bayer8_are_different_pictures():
    a, _ = grad.render(SIZE, (0, 0), (31, 0), BLACK, WHITE, "linear", dither="bayer2")
    b, _ = grad.render(SIZE, (0, 0), (31, 0), BLACK, WHITE, "linear", dither="bayer8")
    assert not np.array_equal(a, b)


def test_the_matrix_is_the_shared_one_and_is_anchored_at_the_origin():
    """The same object ``dither.convert`` thresholds against, so a dithered
    gradient interlocks with a converted document rather than beating against
    it -- and the phase follows the canvas, so a ramp drawn in two halves lines
    up with itself."""
    flat = [(0.0, BLACK), (1.0, WHITE)]
    # A ramp whose parameter is 0.5 everywhere: the picture is then exactly the
    # matrix, thresholded at a half.
    t = np.full((4, 4), 0.5, dtype=np.float32)
    picked = grad.dithered(flat, t, "bayer4")[..., 0]
    expect = (dith.bayer_matrix(4) < 0.5).astype(np.float32)
    assert np.array_equal(picked, expect)


def test_a_single_stop_is_a_flat_colour_and_not_an_error():
    rgba, _weight = grad.render(SIZE, (0, 0), (31, 0), stops=[(0.3, RED)], dither="bayer2")
    assert _bytes(rgba) == {RED[:3]}


def test_an_unknown_matrix_is_refused():
    import pytest

    with pytest.raises(ValueError):
        grad.render(SIZE, (0, 0), (31, 0), BLACK, WHITE, "linear", dither="floyd")


# --- the off-path pin --------------------------------------------------------


def test_dither_none_is_byte_identical_to_the_call_without_it():
    """The pin. ``None`` is a separate path, not a value the new code handles:
    the undithered arithmetic is the arithmetic it always was."""
    for kind in grad.KINDS:
        plain = grad.render(SIZE, (2, 3), (29, 12), BLACK, WHITE, kind)
        explicit = grad.render(SIZE, (2, 3), (29, 12), BLACK, WHITE, kind, dither=None)
        assert np.array_equal(plain[0], explicit[0]), kind
        assert np.array_equal(plain[1], explicit[1]), kind


def test_the_document_gradient_defaults_to_no_dither():
    plain = Document.blank(*SIZE)
    plain.gradient((0, 0), (31, 0), BLACK, WHITE)
    explicit = Document.blank(*SIZE)
    explicit.gradient((0, 0), (31, 0), BLACK, WHITE, dither=None)
    assert np.array_equal(plain.stack.active.pixels, explicit.stack.active.pixels)


# --- through the document ----------------------------------------------------


def test_a_dithered_gradient_lands_on_the_stops_in_the_layer_too():
    doc = Document.blank(*SIZE)
    doc.gradient((0, 0), (31, 0), BLACK, WHITE, dither="bayer4")
    rgb = doc.stack.active.pixels[..., :3]
    assert set(np.unique(rgb).tolist()) <= {0, 255}


def test_the_selection_edge_stays_soft_while_the_colour_is_hard():
    """Coverage is not dithered: a feathered selection means one thing across
    every tool, and a soft edge chopped into a chequer is not that thing."""
    from warlock.studio.inker.selection import SelectionMask

    doc = Document.blank(*SIZE)
    doc.stack.active.pixels[:] = (0, 128, 0, 255)
    mask = np.zeros((16, 32), dtype=np.uint8)
    mask[:, 8:24] = 255
    doc.select(SelectionMask(mask), "replace")
    doc.feather_selection(3.0)

    doc.gradient((0, 0), (31, 0), BLACK, WHITE, dither="bayer4")

    rgb = doc.stack.active.pixels[..., :3]
    partial = set(np.unique(rgb).tolist()) - {0, 128, 255}
    assert partial, "the feathered edge still blends towards what was under it"


def test_an_indexed_document_takes_a_dithered_gradient_without_a_second_snap():
    """The stops are palette members, so ``_commit_patch``'s snap is a no-op --
    which is the property that makes this useful on pixel art at all."""
    doc = Document.blank(*SIZE)
    doc.set_palette([BLACK, WHITE, RED])
    doc.gradient((0, 0), (31, 0), BLACK, WHITE, dither="bayer8")

    from warlock.studio.inker import indexed

    snapped = indexed.snap(doc.stack.active.pixels, doc.palette)
    assert np.array_equal(snapped, doc.stack.active.pixels)


def test_a_dithered_gradient_is_one_undo_step_like_any_other():
    doc = Document.blank(*SIZE)
    before = doc.stack.active.pixels.copy()
    doc.gradient((0, 0), (31, 0), BLACK, WHITE, dither="bayer2")
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, before)
