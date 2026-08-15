"""What the panes offer against what the engine implements.

Every one of these is a table in a pane beside a table in the engine, and each
pair is a place the two have already drifted: the symmetry combo stopped at
``xy`` while ``brush.SYMMETRY`` carried ``radial``, so the mode the engine
implements, the "Ways" slider next to the combo and the manual chapter all
described a setting no user could select. Asserting membership in *both*
directions is the point -- a pane that offers something the engine does not have
fails on the first click, and one that offers less simply hides the feature.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import inker_state
from warlock.studio.inker import animation, brush, selection, transform
from warlock.studio.panes import inker_canvas, inker_timeline, inker_tools


def test_the_symmetry_combo_offers_every_mode_the_brush_implements():
    assert tuple(key for key, _label in inker_tools.SYMMETRY_LABELS) == brush.SYMMETRY


class _Lines:
    """A draw list that only takes notes. ``to_screen`` at identity view and
    zero origin makes a screen coordinate the image coordinate, so the numbers
    recorded here are the ones ``_mirror`` reflects about."""

    def __init__(self) -> None:
        self.lines: list[tuple] = []
        self.circles: list[tuple] = []

    def add_line(self, a, b, colour) -> None:
        self.lines.append((a, b))

    def add_circle(self, centre, radius, colour) -> None:
        self.circles.append((centre, radius))


@pytest.fixture
def guide(monkeypatch):
    """``_symmetry`` with its one imgui call stubbed. ``_u32`` reaches into a
    live context and takes the process down without one, which is the whole
    reason this pane's drawing was untested."""
    monkeypatch.setattr(inker_canvas, "_u32", lambda colour, alpha=1.0: 0)

    def draw_guide(symmetry, size, axis=None, ways=6):
        state = SimpleNamespace(symmetry=symmetry, symmetry_axis=axis, radial_count=ways)
        draw = _Lines()
        # Identity view at a zero origin, so a screen coordinate *is* the image
        # coordinate and the recorded numbers can be compared with ``_mirror``.
        view = inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0))
        inker_canvas._symmetry(state, draw, view, (0.0, 0.0), size)
        return draw

    return draw_guide


@pytest.mark.parametrize("size", [(16, 9), (33, 33), (64, 48)])
def test_the_symmetry_guide_is_drawn_where_the_engine_reflects(guide, size):
    """The guide used to draw at ``width / 2`` while ``brush._mirror``
    reflected about ``(width - 1) / 2`` -- half a pixel out on every canvas, and
    a whole one on an odd-sized canvas. Both now read
    ``brush.axis_or_default``, so this compares the line the user sees against
    the reflection the brush actually performs rather than against a formula.
    """
    ax, ay = brush.axis_or_default(size, None)
    vertical, horizontal = guide("xy", size).lines
    assert vertical[0][0] == vertical[1][0] == ax
    assert horizontal[0][1] == horizontal[1][1] == ay
    # And it is the *reflection* line, not merely a number that matches: a dab
    # at x = 0 comes back at the far column, and the guide sits between them.
    _origin, reflected = brush._mirror((0.0, 0.0), size, "x")
    assert (0.0 + reflected[0]) / 2.0 == ax


def test_a_moved_symmetry_axis_moves_its_guide(guide):
    """The bug this pair exists for: ``_mirror`` honoured ``symmetry_axis`` and
    the guide ignored it, so a moved axis left the line pointing at the middle
    of the page while the strokes came out somewhere else."""
    size = (64, 64)
    vertical, horizontal = guide("xy", size, axis=(10.0, 48.0)).lines
    assert vertical[0][0] == vertical[1][0] == 10.0
    assert horizontal[0][1] == horizontal[1][1] == 48.0
    assert brush._mirror((2.0, 5.0), size, "xy", axis=(10.0, 48.0))[1][0] == 18.0


def test_radial_symmetry_shows_its_pivot_rather_than_nothing(guide):
    """A rotation has no mirror line to draw, so the guide is the point it turns
    about. Before this, selecting radial drew no guide at all -- the one mode
    where the axis matters most and the only one that showed nothing."""
    draw = guide("radial", (64, 64), axis=(20.0, 30.0))
    assert draw.circles and draw.circles[0][0] == (20.0, 30.0)
    # A crosshair through the ring, so the pivot reads as a point and not as a
    # small selection.
    assert len(draw.lines) == 2


def test_the_nib_combo_offers_every_nib_the_brush_implements():
    assert tuple(key for key, _label in inker_tools.NIB_LABELS) == brush.NIBS


def test_the_tag_menu_names_every_direction_playback_understands():
    assert tuple(inker_timeline.DIRECTION_NOTES) == animation.DIRECTIONS


def test_the_default_direction_is_written_as_nothing_at_all():
    """A forward loop is what a tag has always been; labelling it would put a
    word on every tag in the band to distinguish the ordinary case from itself.
    """
    assert inker_timeline.DIRECTION_NOTES["forward"] == ""
    assert inker_timeline._tag_note(animation.Tag(name="x")) == ""
    assert inker_timeline._tag_note(animation.Tag(name="x", loop=False)) == " (once)"
    assert (
        inker_timeline._tag_note(animation.Tag(name="x", direction="pingpong"))
        == " (ping-pong)"
    )


@pytest.mark.parametrize(
    ("shift", "alt", "expected"),
    [(False, False, "replace"), (True, False, "add"), (False, True, "subtract"),
     (True, True, "intersect")],
)
def test_every_combine_op_is_reachable_from_the_keyboard(monkeypatch, shift, alt, expected):
    """The fourth was not: the Shift branch answered the pair, so ``intersect``
    existed in the engine and in no gesture."""
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=shift, key_alt=alt),
    )
    assert inker_canvas._combine_op() == expected


def test_the_modifiers_cover_the_engines_whole_vocabulary():
    reachable = set()
    for shift in (False, True):
        for alt in (False, True):
            reachable.add(
                "intersect" if shift and alt else "add" if shift else "subtract" if alt
                else "replace"
            )
    assert reachable == set(selection.COMBINE_OPS)


def test_the_resize_popup_offers_both_resamples():
    """Derived from ``transform.RESAMPLES`` at the call site rather than written
    out, so this asserts the tuple is the one thing there is to offer."""
    assert transform.RESAMPLES == ("smooth", "nearest")


# --- the timeline's cell hit test --------------------------------------------
#
# Extracted from the pane because the marquee cannot use hover: a pressed imgui
# button suppresses hover on every neighbour, so a drag has to be measured
# geometrically -- and the arithmetic that does it is then the one part of the
# gesture that can be asserted without a window.

GRID = {
    # Frame 0's cells start at x=100, each 20 wide with a 2px gutter, and the
    # rows are drawn top-first so track 1 sits *above* track 0.
    "x0": 100.0,
    "tops": {1: 50.0, 0: 72.0},
    "cell": 20.0,
    "gutter": 2.0,
    "frames": 4,
}


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((100.0, 50.0), (1, 0)),          # the top-left corner of cell (1, 0)
        ((119.0, 69.0), (1, 0)),          # its bottom-right
        ((122.0, 55.0), (1, 1)),          # the next column across
        ((100.0, 80.0), (0, 0)),          # the row below
        ((180.0, 55.0), (1, 3)),          # the last column
    ],
)
def test_a_point_maps_to_the_cell_it_is_inside(point, expected):
    assert inker_timeline.cell_index(point, **GRID) == expected


@pytest.mark.parametrize(
    "point",
    [
        (99.0, 55.0),    # left of the first column
        (120.5, 55.0),   # in the gutter between two columns
        (100.0, 71.0),   # in the gap between two rows
        (210.0, 55.0),   # past the last column
        (100.0, 200.0),  # below every row
    ],
)
def test_a_point_between_or_beyond_the_cells_is_nothing(point):
    """The nearest cell would be the wrong answer: a drag that snapped would
    select cells the cursor never crossed."""
    assert inker_timeline.cell_index(point, **GRID) is None


def test_an_empty_grid_has_no_cells_to_hit():
    assert inker_timeline.cell_index((100.0, 50.0), **{**GRID, "frames": 0}) is None
