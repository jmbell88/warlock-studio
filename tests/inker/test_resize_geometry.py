"""The arithmetic behind the Image size and Canvas size dialogs.

All pure, and all in ``transform`` rather than in the pane, because every one
of these is a decision a person can get wrong in a way a screenshot will not
show: an anchor grid that highlights one cell and resizes towards another, a
proportion lock that walks a 3:2 document off its ratio, a percentage that
disagrees with the pixels beside it.
"""

from __future__ import annotations

from warlock.studio.inker import transform as tf

# --- the anchor grid ----------------------------------------------------------


def test_the_centre_anchor_points_every_way_and_leaves_no_cell_empty():
    cells = [name for row in tf.ANCHOR_GRID for name in row]
    answers = {name: tf.anchor_cell("centre", name) for name in cells}
    assert answers["centre"] == (0, 0)
    assert None not in answers.values()
    # Eight directions, each used exactly once.
    arrows = [value for name, value in answers.items() if name != "centre"]
    assert len(arrows) == 8
    assert len(set(arrows)) == 8
    assert all(abs(dx) <= 1 and abs(dy) <= 1 for dx, dy in arrows)


def test_a_corner_anchor_leaves_five_cells_empty():
    """An arrow in one of those would promise room that anchor never makes."""
    cells = [name for row in tf.ANCHOR_GRID for name in row]
    answers = {name: tf.anchor_cell("top-left", name) for name in cells}
    assert answers["top-left"] == (0, 0)
    assert answers["top"] == (1, 0), "room opens to the right"
    assert answers["left"] == (0, 1), "room opens below"
    assert answers["centre"] == (1, 1)
    assert sum(1 for value in answers.values() if value is None) == 5
    assert answers["bottom-right"] is None


def test_every_anchor_the_grid_names_is_one_the_engine_knows():
    """The grid and ``ANCHORS`` are two lists of the same nine names."""
    assert {name for row in tf.ANCHOR_GRID for name in row} == set(tf.ANCHORS)
    assert len(tf.ANCHOR_GRID) == 3
    assert all(len(row) == 3 for row in tf.ANCHOR_GRID)


def test_an_unknown_anchor_behaves_as_top_left_here_too():
    """``anchor_offset``'s own rule. Two different answers to one typo is the
    highlight-one-cell-resize-towards-another bug."""
    for cell in ("top-left", "top", "bottom-right", "centre"):
        assert tf.anchor_cell("nonsense", cell) == tf.anchor_cell("top-left", cell)
    assert tf.anchor_cell("centre", "nonsense") is None


# --- percentages --------------------------------------------------------------


def test_percent_and_pixels_round_trip():
    assert tf.percent_size((100, 200), (50.0, 50.0)) == (50, 100)
    assert tf.percent_size((100, 200), (100.0, 100.0)) == (100, 200)
    assert tf.size_percent((100, 200), (50, 100)) == (50.0, 50.0)


def test_a_percentage_floors_at_one_pixel_rather_than_zero():
    """Half of three pixels is one, not none. The snap is honest and is what
    Photoshop does; a zero-width document is not a thing."""
    assert tf.percent_size((3, 3), (10.0, 10.0)) == (1, 1)
    assert tf.percent_size((3, 3), (0.0, 0.0)) == (1, 1)


def test_percent_size_does_not_clamp_growth():
    """The ceiling is ``inker_mode.clamp_resize``'s and lives one layer up.

    Two ceilings is two things to keep in step, and the one that drifted would
    be the one nothing routed through.
    """
    assert tf.percent_size((1000, 1000), (10000.0, 10000.0)) == (100000, 100000)


# --- the proportion chain -----------------------------------------------------


def test_the_chain_holds_the_ratio_of_the_axis_that_was_not_typed():
    assert tf.linked_size((100, 50), (200, 50), "w") == (200, 100)
    assert tf.linked_size((100, 50), (100, 100), "h") == (200, 100)


def test_the_chain_is_stable_under_repeated_application():
    """The drift guard, and the reason the ratio comes from ``old``.

    Taking it from the previous pending pair lets a chain of roundings walk a
    3:2 off its own ratio after a few keystrokes -- a proportion lock that does
    not lock.
    """
    old = (300, 200)
    size = (301, 200)
    for _ in range(20):
        size = tf.linked_size(old, size, "w")
    assert size == (301, 201)


def test_the_chain_needs_to_be_told_which_axis_moved():
    """A single "something changed" flag makes whichever field is read second
    win, so typing a width would silently rewrite it from the height."""
    old = (100, 50)
    typed_width = tf.linked_size(old, (200, 50), "w")
    typed_height = tf.linked_size(old, (200, 50), "h")
    assert typed_width == (200, 100)
    assert typed_height == (100, 50)
    assert typed_width != typed_height


# --- the preview --------------------------------------------------------------


def _inside(outer, inner) -> bool:
    return (
        outer[0] <= inner[0] + 1e-6
        and outer[1] <= inner[1] + 1e-6
        and outer[2] >= inner[2] - 1e-6
        and outer[3] >= inner[3] - 1e-6
    )


def test_a_growth_shows_the_picture_inside_the_frame():
    new_rect, old_rect = tf.preview_boxes((100, 100), (200, 200), "centre", 80.0)
    assert _inside(new_rect, old_rect)
    # Centred: equal margins all round.
    assert abs((old_rect[0] - new_rect[0]) - (new_rect[2] - old_rect[2])) < 1e-6
    assert abs((old_rect[1] - new_rect[1]) - (new_rect[3] - old_rect[3])) < 1e-6


def test_a_crop_shows_the_frame_inside_the_picture():
    new_rect, old_rect = tf.preview_boxes((200, 200), (100, 100), "centre", 80.0)
    assert _inside(old_rect, new_rect)


def test_the_anchor_decides_which_side_the_room_lands_on():
    """The one thing the numbers on the dialog cannot say."""
    top_left, old_tl = tf.preview_boxes((100, 100), (200, 200), "top-left", 80.0)
    bottom_right, old_br = tf.preview_boxes(
        (100, 100), (200, 200), "bottom-right", 80.0
    )
    # Anchored top-left the old image sits at the frame's top-left corner.
    assert abs(old_tl[0] - top_left[0]) < 1e-6
    assert abs(old_tl[1] - top_left[1]) < 1e-6
    # Anchored bottom-right it sits at the opposite one.
    assert abs(old_br[2] - bottom_right[2]) < 1e-6
    assert abs(old_br[3] - bottom_right[3]) < 1e-6


def test_an_unchanged_size_previews_two_identical_boxes():
    new_rect, old_rect = tf.preview_boxes((64, 64), (64, 64), "centre", 80.0)
    assert all(abs(a - b) < 1e-6 for a, b in zip(new_rect, old_rect, strict=True))


def test_the_preview_stays_inside_its_box():
    for old, new, anchor in (
        ((100, 20), (20, 100), "centre"),
        ((16, 16), (512, 512), "bottom-right"),
        ((512, 128), (64, 64), "top"),
    ):
        for rect in tf.preview_boxes(old, new, anchor, 80.0):
            assert -1e-6 <= rect[0] <= rect[2] <= 80.0 + 1e-6
            assert -1e-6 <= rect[1] <= rect[3] <= 80.0 + 1e-6
