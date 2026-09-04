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

import numpy as np
import pytest

from warlock.studio import inker, inker_state
from warlock.studio.inker import animation, brush, selection, transform
from warlock.studio.panes import inker_canvas, inker_timeline, inker_tools


def test_every_symmetry_the_engine_composes_has_a_control():
    """The combo this replaced pinned the same fact one model earlier.

    A symmetry is a *set* now (``brush.SYMMETRY_AXES``), so a seven-way combo
    cannot express one and the control is five: the four mirrors as a pill
    group on the context bar, and the radial as a checkbox in the popover
    behind the same ``Sym`` word, where the count it needs already lives. The
    two directions still both matter -- a toggle the engine does not implement
    fails on the first click, and an axis with no control hides the feature.

    Sourced from ``inker_context`` throughout since 2026-08-31, when the radial
    followed the mirrors out of the toolbox.
    """
    from warlock.studio.panes import inker_context

    mirrors = tuple(axis for axis, _label, _tip in inker_context.SYMMETRY_TOGGLES)
    assert mirrors + (inker_context.RADIAL_AXIS,) == brush.SYMMETRY_AXES
    # And each carries a tooltip: a bar of "H V \ /" with no words is four
    # buttons a user has to press to find out about.
    assert all(tip for _axis, _label, tip in inker_context.SYMMETRY_TOGGLES)
    # Every mirror also has a *word* for the popover, which is where the
    # characters stop being enough.
    assert set(inker_context._MIRROR_WORDS) == set(mirrors)


def test_the_legacy_symmetry_names_all_still_read():
    """``brush.SYMMETRY`` is a stored value in every settings file that has
    ever run Warlock, so every one of its members has to keep meaning what it
    meant -- which is what makes the composed model an addition rather than a
    migration."""
    assert brush.axes_of("none") == ()
    assert brush.axes_of("xy") == ("x", "y")
    for mode in brush.SYMMETRY:
        axes = brush.axes_of(mode)
        assert all(axis in brush.SYMMETRY_AXES for axis in axes), mode
        # And it round-trips: the canonical spelling of what a legacy name
        # means reads back as the same set.
        assert brush.axes_of(brush.compose(axes)) == axes


def test_a_symmetry_this_build_does_not_have_is_ignored_rather_than_fatal():
    """The value comes out of a settings file."""
    assert brush.axes_of("sideways") == ()
    assert brush.axes_of("x+sideways") == ("x",)
    assert brush.axes_of(None) == ()


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
def guide(monkeypatch, patch_canvas):
    """``_symmetry`` with its one imgui call stubbed. ``_u32`` reaches into a
    live context and takes the process down without one, which is the whole
    reason this pane's drawing was untested."""
    patch_canvas("_u32", lambda colour, alpha=1.0: 0)

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


def test_the_two_diagonal_mirrors_get_guides_of_their_own(guide):
    """They had none: the guide drew ``x`` and ``y`` and nothing else, so the
    two 45-degree mirrors reflected with no line on screen saying where. That
    was survivable while symmetry was one mode at a time and is not now -- a
    composed ``x+diag`` draws one line and reflects about two."""
    size = (64, 64)
    ax, ay = brush.axis_or_default(size, None)
    for mode in ("diag", "anti"):
        (line,) = guide(mode, size).lines
        (a, b) = line
        # Through the axis, at 45 degrees, and in the direction the toggle's
        # own label draws: ``diag`` descends to the right on screen.
        assert a[0] != b[0]
        slope = (b[1] - a[1]) / (b[0] - a[0])
        assert slope == (1.0 if mode == "diag" else -1.0), mode
        assert a[1] - slope * a[0] == ay - slope * ax, mode


def test_a_composed_symmetry_draws_every_guide_it_reflects_about(guide):
    """The whole point of the composed model, at the one place a mismatch is
    invisible until a stroke lands somewhere unexpected."""
    size = (64, 64)
    assert len(guide("x+y+diag+anti", size).lines) == 4
    assert len(guide("x+diag", size).lines) == 2
    # And the engine agrees about how many places a dab lands: D4 is eight.
    assert len(brush._mirror((3.0, 5.0), size, "x+y+diag+anti")) == 8


def test_the_nib_combo_offers_every_nib_the_brush_implements():
    """The nib picker is the context bar's now (W2.4) and it gained the line
    nib in 6.1's sibling wave; a nib the engine has and no control offers is a
    feature that exists only in the source."""
    from warlock.studio.panes import inker_context

    assert tuple(key for key, _label in inker_context.NIB_LABELS) == brush.NIBS


def test_the_tag_menu_names_every_direction_playback_understands():
    assert tuple(inker_timeline.DIRECTION_NOTES) == animation.DIRECTIONS


def test_the_default_direction_is_written_as_nothing_at_all():
    """A forward loop is what a tag has always been; labelling it would put a
    word on every tag in the band to distinguish the ordinary case from itself.
    """
    assert inker_timeline.DIRECTION_NOTES["forward"] == ""
    assert inker_timeline._tag_note(animation.Tag(name="x")) == ""
    assert inker_timeline._tag_note(animation.Tag(name="x", loop=False)) == " (once)"
    assert inker_timeline._tag_note(animation.Tag(name="x", direction="pingpong")) == " (ping-pong)"


@pytest.mark.parametrize(
    ("shift", "alt", "ctrl", "expected"),
    [
        (False, False, False, "replace"),
        (True, False, False, "add"),
        (False, True, False, "replace"),
        (True, True, False, "subtract"),
        (True, False, True, "intersect"),
    ],
)
def test_every_combine_op_is_reachable_from_the_keyboard(monkeypatch, shift, alt, ctrl, expected):
    """The fourth was not: the Shift branch answered the pair, so ``intersect``
    existed in the engine and in no gesture."""
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=shift, key_alt=alt, key_ctrl=ctrl),
    )
    assert inker_canvas._combine_op() == expected


def test_the_modifiers_cover_the_engines_whole_vocabulary():
    reachable = set()
    for shift in (False, True):
        for alt in (False, True):
            for ctrl in (False, True):
                if shift and ctrl:
                    reachable.add("intersect")
                elif shift and alt:
                    reachable.add("subtract")
                elif shift:
                    reachable.add("add")
                else:
                    reachable.add("replace")
    assert reachable == set(selection.COMBINE_OPS)


def test_the_palette_sort_combo_offers_every_key_the_engine_implements():
    """Both directions, which is the whole point: a key the pane offers and the
    engine does not fails on the first click, and one the engine has and the
    pane does not is a feature no user can reach."""
    from warlock.studio import inker
    from warlock.studio.panes import inker_colors

    keys = tuple(key for key, _label in inker_colors.SORT_LABELS)
    assert keys == inker.PALETTE_SORT_KEYS


def test_every_sort_key_is_labelled_and_no_label_is_repeated():
    from warlock.studio.panes import inker_colors

    labels = [label for _key, label in inker_colors.SORT_LABELS]
    assert all(labels)
    assert len(set(labels)) == len(labels)


def test_the_convert_popup_offers_every_method_the_engine_implements():
    """The dither combo is built from ``dither.METHODS`` at the call site rather
    than written out, so this asserts the tuple is the one thing there is to
    offer -- and that the default is one of them."""
    from warlock.studio import inker
    from warlock.studio.inker import dither

    assert inker.DITHER_METHODS == dither.METHODS
    assert dither.METHODS[0] == "nearest"
    assert inker_state.InkerState().convert_method in dither.METHODS


def test_the_gradient_dither_option_defaults_to_off_and_names_a_real_matrix():
    from warlock.studio.inker import dither

    assert inker_state.TOOL_OPTION_DEFAULTS["gradient_dither"] == "none"
    assert set(dither.ORDERED) <= set(dither.METHODS)


def test_the_resize_popup_offers_every_resample():
    """Derived from ``transform.RESAMPLES`` at the call site rather than written
    out, so this asserts the tuple is the one thing there is to offer.

    ``rotsprite`` is in it and is deliberately offered to the resize popup too,
    even though a *scale* asked for it falls through to nearest: the setting is
    a statement about the kind of art being made rather than about one
    operation, and hiding it from one control would make it look like two
    settings."""
    assert transform.RESAMPLES == ("smooth", "nearest", "rotsprite")


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
        ((100.0, 50.0), (1, 0)),  # the top-left corner of cell (1, 0)
        ((119.0, 69.0), (1, 0)),  # its bottom-right
        ((122.0, 55.0), (1, 1)),  # the next column across
        ((100.0, 80.0), (0, 0)),  # the row below
        ((180.0, 55.0), (1, 3)),  # the last column
    ],
)
def test_a_point_maps_to_the_cell_it_is_inside(point, expected):
    assert inker_timeline.cell_index(point, **GRID) == expected


@pytest.mark.parametrize(
    "point",
    [
        (99.0, 55.0),  # left of the first column
        (120.5, 55.0),  # in the gutter between two columns
        (100.0, 71.0),  # in the gap between two rows
        (210.0, 55.0),  # past the last column
        (100.0, 200.0),  # below every row
    ],
)
def test_a_point_between_or_beyond_the_cells_is_nothing(point):
    """The nearest cell would be the wrong answer: a drag that snapped would
    select cells the cursor never crossed."""
    assert inker_timeline.cell_index(point, **GRID) is None


def test_an_empty_grid_has_no_cells_to_hit():
    assert inker_timeline.cell_index((100.0, 50.0), **{**GRID, "frames": 0}) is None


# --- the timeline's group-indent depth (L3 v1) --------------------------------
#
# v1 is indent only: no header row, no fold. ``_track_row`` only needs the
# length of this to decide how far to shift a label, but the chain itself is
# what proves the depth comes from ``groups.ancestry`` rather than from a
# hand-rolled walk that could disagree with it.


def test_an_ungrouped_track_has_no_indent_depth():
    doc = inker.Document.blank(4, 4)
    assert inker_timeline.track_depth(doc, doc.member_uids()[0]) == []


def test_a_grouped_track_is_one_level_deep():
    doc = inker.Document.blank(4, 4)
    doc.add_layer()
    group = doc.group_layers([0, 1])
    member = doc.member_uids()[0]
    assert inker_timeline.track_depth(doc, member) == [group.uid]


def test_nested_groups_stack_the_depth_innermost_first():
    doc = inker.Document.blank(4, 4)
    doc.add_layer()
    doc.add_layer()
    outer = doc.group_layers([0, 1, 2], name="outer")
    inner = doc.group_layers([0, 1], name="inner")
    member = doc.member_uids()[0]
    assert inker_timeline.track_depth(doc, member) == [inner.uid, outer.uid]


def test_a_document_with_no_groups_at_all_never_walks_group_of():
    """The short-circuit on ``doc.groups`` matters for a still document too --
    it is why this is safe to call on a document that has never grown a group
    tree, not just one that once had a group and dissolved it."""
    doc = inker.Document.blank(4, 4)
    assert doc.groups == {}
    assert inker_timeline.track_depth(doc, doc.member_uids()[0]) == []


# --- the transform box's handles ---------------------------------------------


def test_every_transform_handle_has_axes_declared_for_it():
    """The table beside the thing it describes, in both directions. A handle
    ``_handles`` draws that ``HANDLE_AXES`` does not know about is one the drag
    code silently scales in both axes; an entry with no handle is a scale
    nothing can grab.

    Two exemptions, and both are grab points that are not scales at all: the
    rotate arm, and the pivot ring the transform turns about. Named here rather
    than allowed by a blanket rule, so a ninth *scale* handle still cannot slip
    in without an entry."""
    doc = inker.Document.blank(16, 16)
    doc.select(selection.SelectionMask.from_rect(doc.size, (2, 2, 10, 10)))
    assert doc.lift()
    tab = SimpleNamespace(doc=doc, view=inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0)))
    drawn = set(inker_canvas._handles(tab, (0.0, 0.0)))
    assert drawn - {"rotate", "pivot"} == set(inker_canvas.HANDLE_AXES)


def test_the_edge_handles_sit_on_the_edge_midpoints():
    doc = inker.Document.blank(16, 16)
    doc.select(selection.SelectionMask.from_rect(doc.size, (2, 2, 10, 10)))
    assert doc.lift()
    tab = SimpleNamespace(doc=doc, view=inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0)))
    handles = inker_canvas._handles(tab, (0.0, 0.0))
    assert handles["n"] == (6.0, 2.0)
    assert handles["s"] == (6.0, 10.0)
    assert handles["w"] == (2.0, 6.0)
    assert handles["e"] == (10.0, 6.0)


def test_the_resample_combo_offers_every_mode_the_engine_implements():
    """Same pin as the symmetry combo above, for the setting that decides what
    a rotate and a scale do to a pixel-art drawing."""
    from warlock.studio.panes import inker_bridge

    assert inker_bridge.transform.RESAMPLES == transform.RESAMPLES


# --- the timeline's range verbs (Wave 2) --------------------------------------
#
# Nine menu items over one table, so the pane's list and the engine's methods
# can be asserted against each other. Every verb is run against a real document
# with the range's four integers, which is the only thing the menu row does that
# can be wrong: an off-by-one in the rect, or a verb wired to the wrong engine
# method, both look identical until something moves.


def _range_clip(frames: int = 3, size=(4, 4)):
    doc = inker.Document.blank(*size)
    for index in range(frames - 1):
        doc.add_frame()
        doc.write_colour((0, 0, 2, 2), (10 * index + 5, 0, 0, 255), np.ones((2, 2), np.float32))
    doc.set_current_frame(0)
    doc.write_colour((0, 0, 2, 2), (255, 0, 0, 255), np.ones((2, 2), np.float32))
    doc.history.clear()
    return doc


@pytest.mark.parametrize("label", [v[0] for v in inker_timeline.RANGE_VERBS])
def test_every_range_verb_reaches_the_document_as_one_step(label):
    doc = _range_clip()
    run = next(r for name, r, _sq in inker_timeline.RANGE_VERBS if name == label)
    before = [doc.anim.cels[(doc.anim.tracks[0].uid, f.uid)].pixels.copy() for f in doc.anim.frames]

    run(doc, (0, 0, 0, 1))
    assert len(doc.history) == 1
    cels = [doc.anim.cels[(doc.anim.tracks[0].uid, f.uid)] for f in doc.anim.frames]
    # The two frames the rect covers moved; the one outside it did not.
    assert not np.array_equal(cels[0].pixels, before[0])
    assert not np.array_equal(cels[1].pixels, before[1])
    assert np.array_equal(cels[2].pixels, before[2])


def test_only_the_quarter_turns_are_gated_on_a_square_canvas():
    """The pane's copy of the engine's refusal, so the row greys before the
    click. Both exist deliberately: neither the menu nor the engine alone
    covers every door into the op."""
    gated = {name for name, _run, square in inker_timeline.RANGE_VERBS if square}
    assert gated == {"Rotate 90 clockwise", "Rotate 90 anticlockwise"}


def test_a_gated_verb_on_a_non_square_canvas_refuses_by_name():
    doc = _range_clip(size=(4, 6))
    run = next(r for name, r, sq in inker_timeline.RANGE_VERBS if sq)
    with pytest.raises(ValueError, match="square"):
        run(doc, (0, 0, 0, 1))


def test_a_refused_verb_is_toasted_as_a_sentence_not_a_bare_exception():
    doc = _range_clip(size=(4, 6))
    said: list[tuple] = []
    ctx = SimpleNamespace(toast=lambda text, kind="info": said.append((text, kind)))
    run = next(r for name, r, sq in inker_timeline.RANGE_VERBS if sq)

    inker_timeline._run_range_verb(ctx, doc, run, (0, 0, 0, 1))
    assert len(said) == 1
    assert said[0][0].startswith("That range was not changed:")
    assert "square" in said[0][0]


def test_the_shift_verbs_all_wrap():
    """No number beside them, so they must not be able to push a drawing off
    the edge one press at a time."""
    doc = _range_clip()
    before = doc.anim.cels[(doc.anim.tracks[0].uid, doc.anim.frames[0].uid)].pixels.copy()
    for label in ("Shift left", "Shift right", "Shift up", "Shift down"):
        run = next(r for name, r, _sq in inker_timeline.RANGE_VERBS if name == label)
        run(doc, (0, 0, 0, 0))
    after = doc.anim.cels[(doc.anim.tracks[0].uid, doc.anim.frames[0].uid)].pixels
    # Left then right then up then down is the identity -- which it can only be
    # if every one of them wrapped.
    assert np.array_equal(after, before)


# --- the layers panel and the timeline range ----------------------------------


def _range_tab(doc, rect=None):
    return SimpleNamespace(doc=doc, range_sel=rect)


def test_the_timeline_reads_the_range_track_span():
    from warlock.studio.panes import inker_timeline

    doc = inker.Document.blank(4, 4)
    doc.add_layer()
    doc.add_frame()
    assert inker_timeline.track_range(_range_tab(doc), doc) is None
    assert inker_timeline.track_range(_range_tab(doc, (1, 0, 0, 0)), doc) == (0, 1)


def test_the_track_span_is_clamped_at_use_like_every_other_reader():
    from warlock.studio.panes import inker_timeline

    doc = inker.Document.blank(4, 4)
    doc.add_frame()
    assert inker_timeline.track_range(_range_tab(doc, (0, 9, 0, 0)), doc) == (0, 0)
    assert inker_timeline.track_range(_range_tab(doc, (7, 9, 0, 0)), doc) is None


def test_a_still_document_has_no_track_span_to_read():
    from warlock.studio.panes import inker_timeline

    doc = inker.Document.blank(4, 4)
    assert inker_timeline.track_range(_range_tab(doc, (0, 0, 0, 0)), doc) is None


def test_shift_clicking_a_layer_row_extends_the_range_over_the_whole_clip(monkeypatch):
    from warlock.studio.panes import inker_timeline

    doc = inker.Document.blank(4, 4)
    doc.add_layer()
    doc.add_layer()
    doc.add_frame()
    doc.set_active_layer(0)
    tab = _range_tab(doc)
    monkeypatch.setattr(inker_timeline.imgui, "get_io", lambda: SimpleNamespace(key_shift=True))

    assert inker_timeline.extend_range(tab, doc, 2)
    # Anchored on the row that *was* active, and with no range yet the frame
    # span is the whole clip.
    assert tab.range_sel == (0, 2, 0, 1)


def test_shift_clicking_keeps_the_frame_span_a_range_already_has(monkeypatch):
    from warlock.studio.panes import inker_timeline

    doc = inker.Document.blank(4, 4)
    doc.add_layer()
    doc.add_frame()
    doc.add_frame()
    doc.set_active_layer(1)
    tab = _range_tab(doc, (1, 1, 1, 2))
    monkeypatch.setattr(inker_timeline.imgui, "get_io", lambda: SimpleNamespace(key_shift=True))

    assert inker_timeline.extend_range(tab, doc, 0)
    assert tab.range_sel == (0, 1, 1, 2)


def test_an_unmodified_click_does_not_extend_the_range(monkeypatch):
    from warlock.studio.panes import inker_timeline

    doc = inker.Document.blank(4, 4)
    doc.add_layer()
    doc.add_frame()
    tab = _range_tab(doc)
    monkeypatch.setattr(inker_timeline.imgui, "get_io", lambda: SimpleNamespace(key_shift=False))

    assert not inker_timeline.extend_range(tab, doc, 1)
    assert tab.range_sel is None


def test_the_timeline_builds_its_two_lists_once_per_draw():
    """``frame_uids`` was called once per *track* and ``member_uids`` once per
    *row*, inside the row loop -- so a twenty-track fifty-frame clip allocated
    seventy lists a frame to answer two questions whose answers cannot change
    while the grid is drawing. Both are hoisted into ``geom``, the per-draw
    scratch the rows are already handed."""
    import inspect

    from warlock.studio.panes import inker_timeline

    grid = inspect.getsource(inker_timeline._grid)
    row = inspect.getsource(inker_timeline._track_row)

    assert '"columns": columns' in grid
    assert '"order": doc.member_uids()' in grid
    assert 'geom["columns"]' in row
    assert "doc.member_uids()" not in row


def test_depth_still_answers_without_a_hoisted_order():
    """The callers outside the grid pass nothing and must still work."""
    from warlock.studio import inker
    from warlock.studio.panes import inker_timeline

    doc = inker.Document.blank(4, 4)
    assert inker_timeline._depth(doc, 0) == 0
    assert inker_timeline._depth(doc, 99) == 0, "off the end is not an error"
    assert inker_timeline._depth(doc, 0, doc.member_uids()) == 0


def test_the_two_u32_helpers_are_deliberately_different():
    """They share a name and pack a colour, and that is where the resemblance
    stops: ``get_color_u32`` multiplies by imgui's global ``style.Alpha`` and
    ``color_convert_float4_to_u32`` does not. The canvas wants its overlays to
    fade inside a disabled block; the timeline needs its range bands to stay
    legible while the panel waits on a save. Folding them into one would break
    one or the other, so this is the guard against a tidy-up."""
    import inspect

    from warlock.studio.panes import inker_canvas, inker_timeline

    canvas = inspect.getsource(inker_canvas._u32)
    timeline = inspect.getsource(inker_timeline._u32)
    assert "get_color_u32" in canvas
    assert "color_convert_float4_to_u32" in timeline
    assert "color_convert_float4_to_u32" not in canvas.split('"""')[-1]
    assert "get_color_u32" not in timeline.split('"""')[-1]


def test_the_two_rgba_helpers_are_deliberately_different():
    """One clamps and one does not, and the clamping one reads a colour back
    out of a drag where a value can leave 0..1 mid-gesture."""

    assert inker_tools._rgba((1.5, -0.2, 0.5, 1.0)) == (255, 0, 128, 255)
