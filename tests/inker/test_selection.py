"""Masks, the wand, the outline, and floating pixels.

The wand has an oracle here rather than being compared against itself: a
scanline flood in plain Python is slow and obviously correct, which is exactly
what a vectorised predicate plus a Pillow flood needs checking against.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import selection as sel
from warlock.studio.inker.selection import SelectionMask


def _rgba(plane: np.ndarray) -> np.ndarray:
    out = np.zeros((*plane.shape, 4), dtype=np.uint8)
    out[..., 0] = plane
    out[..., 3] = 255
    return out


# --- building a mask --------------------------------------------------------


def test_a_rectangle_selection_has_no_soft_edge_at_all():
    """A marquee is pixel-aligned by construction; half-covered edge pixels
    would be an antialiasing artefact the user did not ask for."""
    mask = sel.SelectionMask.from_rect((8, 8), (2, 2, 6, 6))
    assert set(np.unique(mask.mask).tolist()) == {0, 255}
    assert mask.bounds == (2, 2, 6, 6)


def test_a_rectangle_is_clipped_to_the_canvas():
    mask = sel.SelectionMask.from_rect((8, 8), (-4, -4, 4, 4))
    assert mask.bounds == (0, 0, 4, 4)


def test_an_empty_rectangle_selects_nothing():
    mask = sel.SelectionMask.from_rect((8, 8), (3, 3, 3, 3))
    assert mask.is_empty and mask.bounds is None


def test_an_ellipse_is_round_and_antialiased():
    mask = sel.SelectionMask.from_ellipse((32, 32), (0, 0, 32, 32))
    assert mask.mask[16, 16] == 255
    assert mask.mask[1, 1] == 0  # the corner is outside the circle
    assert np.any((mask.mask > 0) & (mask.mask < 255))


def test_a_polygon_selects_its_interior():
    mask = sel.SelectionMask.from_polygon((16, 16), [(1, 1), (14, 1), (14, 14)])
    assert mask.contains((12, 10))
    assert not mask.contains((2, 12))


def test_a_full_mask_selects_every_pixel():
    mask = sel.SelectionMask.full(4, 4)
    assert mask.bounds == (0, 0, 4, 4)
    assert int(mask.mask.min()) == 255


def test_a_mask_must_be_a_single_channel_plane():
    with pytest.raises(ValueError):
        sel.SelectionMask(np.zeros((4, 4, 4), dtype=np.uint8))


# --- algebra ----------------------------------------------------------------


def test_adding_two_selections_keeps_both():
    a = sel.SelectionMask.from_rect((8, 8), (0, 0, 4, 8))
    b = sel.SelectionMask.from_rect((8, 8), (4, 0, 8, 8))
    both = a.combined(b, "add")
    assert both.bounds == (0, 0, 8, 8)


def test_subtracting_removes_the_overlap_and_nothing_else():
    a = sel.SelectionMask.from_rect((8, 8), (0, 0, 8, 8))
    b = sel.SelectionMask.from_rect((8, 8), (0, 0, 4, 8))
    left = a.combined(b, "subtract")
    assert left.bounds == (4, 0, 8, 8)


def test_intersecting_keeps_only_the_overlap():
    a = sel.SelectionMask.from_rect((8, 8), (0, 0, 6, 8))
    b = sel.SelectionMask.from_rect((8, 8), (4, 0, 8, 8))
    both = a.combined(b, "intersect")
    assert both.bounds == (4, 0, 6, 8)


def test_replace_ignores_what_was_selected_before():
    a = sel.SelectionMask.from_rect((8, 8), (0, 0, 8, 8))
    b = sel.SelectionMask.from_rect((8, 8), (1, 1, 2, 2))
    assert a.combined(b, "replace").bounds == (1, 1, 2, 2)


def test_an_unknown_combine_op_is_a_programming_error():
    a = sel.SelectionMask.full(4, 4)
    with pytest.raises(ValueError):
        a.combined(a, "xor")


def test_inverting_swaps_inside_and_outside():
    mask = sel.SelectionMask.from_rect((8, 8), (0, 0, 4, 8)).inverted()
    assert mask.bounds == (4, 0, 8, 8)


def test_feathering_produces_the_intermediate_values_that_are_the_feather():
    hard = sel.SelectionMask.from_rect((32, 32), (8, 8, 24, 24))
    soft = hard.feathered(3.0)
    assert np.any((soft.mask > 0) & (soft.mask < 255))
    assert soft.mask[16, 16] > 240
    # A feather with no radius is not a blur of zero -- it is the same mask.
    assert np.array_equal(hard.feathered(0).mask, hard.mask)


def test_bounds_are_cached_but_still_right_after_a_copy():
    mask = sel.SelectionMask.from_rect((8, 8), (1, 1, 3, 3))
    assert mask.bounds == (1, 1, 3, 3)
    assert mask.copy().bounds == (1, 1, 3, 3)


def test_a_point_outside_the_canvas_is_not_inside_the_selection():
    mask = sel.SelectionMask.full(4, 4)
    assert not mask.contains((-1, 0))
    assert not mask.contains((0, 99))


# --- the wand ---------------------------------------------------------------


def _oracle(pixels, seed, tolerance):
    """A plain scanline flood: slow, obviously correct, and not the code."""
    height, width = pixels.shape[:2]
    target = pixels[seed[1], seed[0]].astype(int)
    out = np.zeros((height, width), dtype=bool)
    todo = [seed]
    while todo:
        x, y = todo.pop()
        if not (0 <= x < width and 0 <= y < height) or out[y, x]:
            continue
        if np.abs(pixels[y, x].astype(int) - target).max() > tolerance:
            continue
        out[y, x] = True
        todo.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return out


def test_the_wand_agrees_with_a_plain_scanline_flood():
    rng = np.random.default_rng(5)
    pixels = _rgba(rng.integers(0, 256, (24, 24), dtype=np.uint8))
    got = sel.magic_wand(pixels, (12, 12), tolerance=60).mask >= 128
    assert np.array_equal(got, _oracle(pixels, (12, 12), 60))


def test_a_contiguous_wand_stops_at_a_wall_a_global_one_does_not():
    plane = np.zeros((16, 16), dtype=np.uint8)
    plane[8, :] = 255
    pixels = _rgba(plane)
    near = sel.magic_wand(pixels, (0, 0), tolerance=10).mask
    assert near[0, 0] == 255 and near[15, 0] == 0
    far = sel.magic_wand(pixels, (0, 0), tolerance=10, contiguous=False).mask
    assert far[15, 0] == 255


def test_a_zero_tolerance_wand_takes_only_the_exact_colour():
    plane = np.zeros((8, 8), dtype=np.uint8)
    plane[4:, :] = 4
    got = sel.magic_wand(_rgba(plane), (0, 0), tolerance=0).mask
    assert got[0, 0] == 255 and got[7, 0] == 0


def test_a_wand_off_the_canvas_selects_nothing_rather_than_raising():
    assert sel.magic_wand(_rgba(np.zeros((4, 4), np.uint8)), (-1, 0)).is_empty


def test_the_distance_is_chebyshev_so_the_tolerance_means_per_channel():
    a = np.array([[[10, 10, 10, 255]]], dtype=np.uint8)
    assert int(sel.colour_distance(a, np.array([20, 10, 10, 255]))[0, 0]) == 10
    assert int(sel.colour_distance(a, np.array([20, 20, 20, 255]))[0, 0]) == 10


# --- the outline ------------------------------------------------------------


def test_a_rectangles_contour_is_its_four_corners_and_nothing_inside():
    mask = sel.SelectionMask.from_rect((16, 16), (4, 4, 12, 12))
    loops = mask.contours()
    assert len(loops) == 1
    points = set(loops[0])
    assert {(4, 4), (12, 4), (12, 12), (4, 12)} <= points
    assert all(4 <= x <= 12 and 4 <= y <= 12 for x, y in points)


def test_the_contour_sits_on_the_pixel_edge_not_the_pixel_centre():
    """A one-pixel selection's ants have to enclose that pixel; drawn through
    its centre they would look like they enclosed its neighbour."""
    mask = sel.SelectionMask.from_rect((8, 8), (3, 3, 4, 4))
    assert sorted(mask.contours()[0]) == [(3, 3), (3, 4), (4, 3), (4, 4)]


def test_two_separate_regions_give_two_loops():
    a = sel.SelectionMask.from_rect((16, 8), (1, 1, 4, 7))
    b = sel.SelectionMask.from_rect((16, 8), (10, 1, 14, 7))
    assert len(a.combined(b, "add").contours()) == 2


def test_an_empty_selection_has_no_outline():
    assert sel.SelectionMask(np.zeros((8, 8), dtype=np.uint8)).contours() == []


# --- floating pixels --------------------------------------------------------


def _floating():
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    return sel.FloatingBuffer(pixels=pixels, mask=mask, offset=(2, 2), layer_uid=1)


def test_moving_a_floating_buffer_changes_where_it_is_not_what_it_holds():
    """``rev`` drives a texture upload -- a move must not trigger one."""
    buf = _floating()
    rev = buf.rev
    buf.moved(3, -1)
    assert buf.offset == (5, 1)
    assert buf.rev == rev


def test_a_floating_buffer_knows_which_points_are_over_it():
    buf = _floating()
    assert buf.contains((2, 2)) and buf.contains((5, 5))
    assert not buf.contains((1, 2)) and not buf.contains((6, 6))


def test_a_hole_in_the_floating_mask_is_not_part_of_it():
    buf = _floating()
    buf.mask[1, 1] = 0
    assert not buf.contains((3, 3))


# --- the clipboard ----------------------------------------------------------


def test_the_clipboard_copies_in_and_copies_out():
    board = sel.Clipboard()
    assert board.empty
    pixels = np.full((2, 2, 4), 7, dtype=np.uint8)
    mask = np.full((2, 2), 255, dtype=np.uint8)
    board.put(pixels, mask)
    pixels[0, 0] = 0  # the clipboard took a copy, not a reference
    got, got_mask = board.take()
    assert int(got[0, 0, 0]) == 7
    assert np.array_equal(got_mask, mask)
    assert not board.empty


def test_taking_from_an_empty_clipboard_gives_nothing():
    assert sel.Clipboard().take() is None


# --- the bounded supersample --------------------------------------------------


def _draw_shape_reference(size, kind, geometry):
    """The whole-canvas supersample ``_draw_shape`` used to do.

    A 4x buffer the size of the document, drawn and box-resized in full however
    small the shape was -- 64 MB at 2048 square for a fifty-pixel lasso. The
    replacement bounds that buffer by the shape; this is here so "same output"
    is a measured claim rather than an argument about how BOX resizing works.
    """
    from PIL import Image, ImageDraw

    width, height = size
    big = Image.new("L", (width * sel._SS, height * sel._SS), 0)
    draw = ImageDraw.Draw(big)
    if kind == "rect":
        x0, y0, x1, y1 = (v * sel._SS for v in geometry)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    elif kind == "ellipse":
        x0, y0, x1, y1 = (v * sel._SS for v in geometry)
        draw.ellipse((x0, y0, x1 - 1, y1 - 1), fill=255)
    elif kind == "polygon":
        points = [(x * sel._SS, y * sel._SS) for x, y in geometry]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return np.asarray(big.resize((width, height), Image.BOX), dtype=np.uint8)


SHAPES = [
    ("ellipse", (10, 12, 40, 39)),
    ("ellipse", (0, 0, 64, 48)),
    ("ellipse", (-8, -6, 12, 9)),  # partly off the top-left
    ("ellipse", (55, 40, 90, 70)),  # partly off the bottom-right
    ("polygon", [(4.5, 5.25), (30.75, 8.0), (18.0, 33.5), (9.0, 20.0)]),
    ("polygon", [(1.0, 1.0), (2.0, 1.0), (1.5, 2.0)]),  # a few pixels
    ("polygon", [(-20.0, -20.0), (100.0, -5.0), (40.0, 90.0)]),  # overhangs
    ("polygon", [(1.0, 1.0), (5.0, 5.0)]),  # too few points to fill
    ("polygon", [(200.0, 200.0), (240.0, 210.0), (220.0, 250.0)]),  # off-canvas
]


@pytest.mark.parametrize("kind,geometry", SHAPES)
def test_the_bounded_supersample_is_bit_identical_to_the_whole_canvas_one(
    kind, geometry
):
    size = (64, 48)
    got = sel._draw_shape(size, kind, geometry)
    assert got.shape == (48, 64)
    assert np.array_equal(got, _draw_shape_reference(size, kind, geometry))


def test_a_small_shape_no_longer_supersamples_the_document(monkeypatch):
    """The point of the change, asserted as the thing it actually bought.

    The old path allocated ``(W * 4, H * 4)`` regardless; the new one asks
    Pillow for a buffer bounded by the shape. A 50-pixel lasso on a 2048 square
    document must not touch anything of that order.
    """
    from PIL import Image

    sizes: list[tuple[int, int]] = []
    real_new = Image.new

    def spy(mode, size, *args, **kwargs):
        sizes.append(size)
        return real_new(mode, size, *args, **kwargs)

    monkeypatch.setattr(Image, "new", spy)
    sel.SelectionMask.from_polygon(
        (2048, 2048), [(1000.0, 1000.0), (1050.0, 1010.0), (1020.0, 1048.0)]
    )
    assert sizes, "the rasteriser stopped going through Image.new"
    assert max(w * h for w, h in sizes) < (2048 * 4) ** 2 // 100


# --- morphology and layer alpha (Ink5) ---------------------------------------


def _square(size=16, box=(6, 6, 10, 10)) -> SelectionMask:
    return SelectionMask.from_rect((size, size), box)


def test_growing_moves_the_edge_out_by_the_radius_asked_for():
    grown = _square().grown(3)
    assert grown.mask[8, 3] == 255, "three pixels to the left of the old edge"
    assert grown.mask[8, 2] == 0, "and not four"


def test_shrinking_moves_it_in_by_the_radius_asked_for():
    shrunk = _square(box=(4, 4, 12, 12)).shrunk(2)
    assert shrunk.mask[8, 6] == 255
    assert shrunk.mask[8, 5] == 0


def test_growing_then_shrinking_by_the_same_amount_recovers_a_convex_shape():
    """Not a general property of morphology -- a closing fills concavities --
    but it holds for a convex region and is the sanity check that says the two
    passes use the same neighbourhood."""
    original = _square(box=(4, 4, 12, 12))
    back = original.grown(3).shrunk(3)
    assert np.array_equal(back.mask, original.mask)


def test_the_neighbourhood_is_an_octagon_rather_than_a_square():
    """A square kernel is one line shorter and turns a circle into a rectangle
    with visibly square corners, which is what this rules out."""
    grown = SelectionMask.from_rect((21, 21), (10, 10, 11, 11)).grown(4)
    assert grown.mask[10, 6] == 255, "four out along the axis"
    assert grown.mask[6, 6] == 0, "but not four out on both axes at once"


def test_growing_keeps_a_soft_edge_soft():
    """On the 8-bit mask, not on a threshold of it: hardening every
    antialiased selection the moment somebody nudged it by a pixel is the bug
    this rules out."""
    soft = _square().feathered(2.0)
    partial = int(((soft.mask > 0) & (soft.mask < 255)).sum())
    grown = soft.grown(1)
    assert int(((grown.mask > 0) & (grown.mask < 255)).sum()) >= partial * 0.5
    assert grown.mask.max() == soft.mask.max()


def test_a_border_is_a_band_either_side_of_the_edge():
    band = _square(box=(4, 4, 12, 12)).bordered(2)
    assert band.mask[8, 4] == 255, "on the old edge"
    assert band.mask[8, 8] == 0, "hollow in the middle"
    assert band.mask[8, 1] == 0, "and bounded outside"


def test_a_zero_width_border_selects_nothing():
    assert _square().bordered(0).is_empty


def test_a_zero_radius_grow_or_shrink_is_the_identity_and_a_copy():
    original = _square()
    for out in (original.grown(0), original.shrunk(0)):
        assert np.array_equal(out.mask, original.mask)
        assert out.mask is not original.mask


def test_growing_a_selection_that_touches_the_canvas_edge_stays_against_it():
    """Edge padding, not zero padding: the selection continues off-canvas,
    which is what the user drew -- zero padding would peel it off the border."""
    grown = SelectionMask.from_rect((8, 8), (0, 0, 4, 8)).grown(2)
    assert grown.mask[0, 0] == 255
    assert grown.mask[7, 0] == 255


def test_selecting_a_layers_alpha_copies_the_coverage_rather_than_thresholding():
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[2:6, 2:6] = (255, 0, 0, 128)
    doc.select_layer_alpha()
    assert doc.mask is not None
    assert int(doc.mask.mask[3, 3]) == 128, "half coverage stays half"
    assert int(doc.mask.mask[0, 0]) == 0


def test_selecting_a_layers_alpha_ignores_its_opacity_and_visibility():
    """The same split ``eyedrop(layer_only=True)`` makes: both are about how a
    layer is shown, not about what is painted in it."""
    doc = inker.Document.blank(4, 4)
    layer = doc.add_layer("Top")
    layer.pixels[:, :] = (10, 10, 10, 255)
    layer.opacity = 0.25
    layer.visible = False
    doc.invalidate_all()
    doc.select_layer_alpha()
    assert doc.mask is not None and int(doc.mask.mask[0, 0]) == 255
