"""Image brushes: a captured tip, its variants, and how overlap accumulates.

The two questions this feature had to answer before it could be written are
asserted here rather than described anywhere else.

*Overlap.* An image dab composites onto the **live** layer -- it has to, because
it lays down a different colour per pixel and the recompute-from-``before`` path
the paint modes use would need the whole stroke's colours kept to do it. A live
composite is exactly where a half-transparent tip dragged slowly over itself
piles up into an opaque one. It does not, because a dab writes the *increment*
from the coverage the stroke has already laid rather than its own alpha, so the
stroke's total contribution is the maximum over its dabs -- the same rule the
class docstring states for every other kind of dab. In ``aligned`` mode the
question does not arise at all: every position inside one lattice cell produces
the same rectangle of the same pixels, so a second visit adds nothing.

*Undo.* One ``PatchEdit`` over the union of every dab, which is what the stroke
lifecycle already gives: an image stamp is a dab like any other, so it inherits
symmetry, the selection clip, the alpha lock, tiled wrapping, the spray's
emission and the single-patch undo without a line of its own. Each of those is
asserted below rather than assumed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import inker, inker_mode, inker_state
from warlock.studio.inker import brush
from warlock.studio.inker.selection import SelectionMask
from warlock.studio.panes import inker_canvas, inker_tools

SIZE = (48, 48)
RED = (255, 0, 0, 255)


def _tip(width=8, height=8, alpha=255, colour=(255, 0, 0), mark=True) -> np.ndarray:
    """A flat RGBA block, with a marked corner so a turn is visible.

    ``mark=False`` for the overlap tests. Within one stroke the first dab to
    reach a pixel owns its *colour*, so a tip with a distinguishable corner
    makes two different walks over the same ground legitimately disagree about
    which dab painted the marker -- which is ordering, not the accumulation
    those tests are about.
    """
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., :3] = colour
    pixels[..., 3] = alpha
    if mark:
        pixels[0, 0, :3] = (0, 255, 0)
    return pixels


def _doc(size=SIZE) -> inker.Document:
    return inker.Document.blank(*size)


def _rect_mask(size, box) -> SelectionMask:
    x0, y0, x1, y1 = box
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return SelectionMask(mask)


def _stamped(stamp, *, doc=None, points=((20.5, 20.5),), **kw):
    doc = doc if doc is not None else _doc()
    doc.begin_stroke(points[0], RED, stamp=stamp, **kw)
    for point in points[1:]:
        doc.stroke_to(point)
    doc.end_stroke()
    return doc


def _sweeps(passes, x0=14.0, x1=26.0, y=20.0):
    """A run of samples that walks the same segment ``passes`` times over."""
    points = [(x0, y)]
    for index in range(passes):
        points.append((x1 if index % 2 == 0 else x0, y))
    return tuple(points)


@pytest.fixture
def dabs(monkeypatch):
    """Every image dab this test emits, recorded.

    A guard against the failure mode these tests are most exposed to: with the
    spacing at a tenth of an eight-pixel tip a dab lands every 0.8 px, so a walk
    of half a pixel emits **one** dab and an anti-compounding assertion over it
    is a single dab compared with itself. Counting is the only way a test can
    say it exercised what it claims to.
    """
    recorded: list[tuple[float, float]] = []
    original = brush.StrokeState._image_dab

    def counting(self, point, target):
        recorded.append(point)
        original(self, point, target)

    monkeypatch.setattr(brush.StrokeState, "_image_dab", counting)
    return recorded


# --- capture ----------------------------------------------------------------


def test_a_capture_is_the_selected_pixels_with_the_mask_in_their_alpha():
    """``copy``'s invariant, which is what makes the tip's own alpha the dab's
    coverage with no second mask to carry."""
    doc = _doc()
    doc.stack.active.pixels[10:18, 10:18] = _tip()
    doc.select(_rect_mask(SIZE, (10, 10, 18, 18)))
    stamp = doc.capture_stamp()
    assert stamp is not None
    assert np.array_equal(stamp.pixels, doc.stack.active.pixels[10:18, 10:18])


def test_a_feathered_selection_captures_a_feathered_tip():
    doc = _doc()
    doc.stack.active.pixels[10:18, 10:18] = _tip()
    mask = _rect_mask(SIZE, (10, 10, 18, 18))
    mask.mask[10:18, 10] = 64
    doc.select(mask)
    stamp = doc.capture_stamp()
    assert stamp is not None
    # 255 * 64 / 255 truncated, which is what ``_masked_alpha`` writes.
    assert int(stamp.pixels[0, 0, 3]) == 64
    assert int(stamp.pixels[0, 1, 3]) == 255


def test_a_capture_reads_rather_than_cuts():
    """It is ``copy``, not ``lift``: no hole, no undo step, no cel."""
    doc = _doc()
    doc.stack.active.pixels[10:18, 10:18] = _tip()
    before = doc.stack.active.pixels.copy()
    doc.select(_rect_mask(SIZE, (10, 10, 18, 18)))
    head = doc.history.head
    assert doc.capture_stamp() is not None
    assert np.array_equal(doc.stack.active.pixels, before)
    assert doc.history.head == head


def test_capturing_with_nothing_selected_is_refused():
    assert _doc().capture_stamp() is None


def test_a_selection_bigger_than_the_cap_is_refused_rather_than_cropped():
    """Refused rather than cropped or scaled: both would hand back a brush that
    is not the thing that was selected."""
    span = brush.MAX_STAMP + 8
    doc = _doc((span, span))
    doc.stack.active.pixels[...] = 255
    doc.select(_rect_mask((span, span), (0, 0, span, span)))
    assert doc.capture_stamp() is None
    doc.select(_rect_mask((span, span), (0, 0, brush.MAX_STAMP, brush.MAX_STAMP)))
    assert doc.capture_stamp() is not None


def test_a_capture_round_trips_onto_the_canvas():
    """Capture, stamp, and the pixels that come back are the ones that went in."""
    source = _doc()
    source.stack.active.pixels[10:18, 10:18] = _tip()
    source.select(_rect_mask(SIZE, (10, 10, 18, 18)))
    stamp = source.capture_stamp()
    doc = _stamped(stamp, points=((24.0, 24.0),))
    # ``free`` centres the tip on the pixel the cursor is in, growing left and
    # up by half; an eight-wide tip therefore starts four before it.
    assert np.array_equal(doc.stack.active.pixels[20:28, 20:28], stamp.pixels)


# --- variants ---------------------------------------------------------------


def test_a_quarter_turn_is_an_exact_reindexing():
    pixels = _tip(width=8, height=4)
    turned = brush.Stamp(pixels, rotation=90)
    assert np.array_equal(turned.image, np.rot90(pixels, -1))
    assert turned.size == (4, 8)


@pytest.mark.parametrize("rotation", (0, 90, 180, 270))
def test_every_turn_is_byte_exact_against_numpy(rotation):
    pixels = _tip(width=6, height=10)
    assert np.array_equal(
        brush.Stamp(pixels, rotation=rotation).image,
        np.rot90(pixels, -(rotation // 90)),
    )


def test_the_flips_are_byte_exact_against_numpy():
    pixels = _tip(width=6, height=10)
    assert np.array_equal(brush.Stamp(pixels, flip_x=True).image, pixels[:, ::-1])
    assert np.array_equal(brush.Stamp(pixels, flip_y=True).image, pixels[::-1, :])


def test_the_flips_come_before_the_rotation():
    """An order has to be picked because the two do not commute, and this is the
    one that keeps the horizontal flip button mirroring horizontally at every
    turn."""
    pixels = _tip(width=6, height=10)
    both = brush.Stamp(pixels, rotation=90, flip_x=True)
    assert np.array_equal(both.image, np.rot90(pixels[:, ::-1], -1))
    assert not np.array_equal(both.image, np.rot90(pixels, -1)[:, ::-1])


def test_four_rotations_come_back_to_where_they_started():
    """The variants cycle, because the control is one button pressed again."""
    stamp = brush.Stamp(_tip(width=6, height=10))
    turned = stamp.rotated().rotated().rotated().rotated()
    assert turned.rotation == 0
    assert np.array_equal(turned.image, stamp.image)


def test_flipping_twice_is_flipping_not_at_all():
    stamp = brush.Stamp(_tip(width=6, height=10))
    assert np.array_equal(stamp.flipped("x").flipped("x").image, stamp.image)
    assert stamp.flipped("x").flipped("x").flip_x is False


def test_a_variant_keeps_the_original_pixels_to_take_the_next_one_from():
    """Variants are taken from the capture, never from the last variant --
    otherwise a rotate then a flip would be a flip of a rotation."""
    pixels = _tip(width=6, height=10)
    stamp = brush.Stamp(pixels).rotated().flipped("x")
    assert np.array_equal(stamp.pixels, pixels)
    assert np.array_equal(stamp.image, brush.Stamp(pixels, rotation=90, flip_x=True).image)


def test_a_stamps_pixels_are_write_locked():
    """``make_stamp``'s rule: one capture is handed to every stroke drawn with
    it, so a caller drawing into one in place would change them all."""
    stamp = brush.Stamp(_tip())
    with pytest.raises(ValueError):
        stamp.pixels[0, 0, 0] = 1
    with pytest.raises(ValueError):
        stamp.image[0, 0, 0] = 1


def test_nonsense_is_refused_at_the_door():
    with pytest.raises(ValueError):
        brush.Stamp(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        brush.Stamp(np.zeros((4, 4, 3), dtype=np.uint8))
    over = brush.MAX_STAMP + 1
    with pytest.raises(ValueError):
        brush.Stamp(np.zeros((over, 4, 4), dtype=np.uint8))


# --- overlap: free ----------------------------------------------------------


def test_a_half_transparent_tip_does_not_build_up_where_a_stroke_crosses_itself(dabs):
    """The whole overlap decision: coverage accumulates with maximum, so a
    stroke that goes over its own ground six times is what one pass leaves.

    Six passes over a twelve-pixel segment rather than a wobble in one place --
    a dab lands every 0.8 px here, so a walk shorter than that emits one dab and
    an assertion over it would be a single dab compared with itself. The counts
    are asserted for exactly that reason.
    """
    stamp = brush.Stamp(_tip(alpha=128, mark=False))
    once = _stamped(stamp, points=_sweeps(1)).stack.active.pixels
    single = len(dabs)
    assert single > 4
    dabs.clear()
    many = _stamped(stamp, points=_sweeps(6)).stack.active.pixels
    assert len(dabs) > 4 * single

    # Nothing anywhere carries more than one dab's worth, at any coverage.
    painted = many[many[..., 3] > 0]
    assert {tuple(int(c) for c in v) for v in np.unique(painted, axis=0)} == {
        (255, 0, 0, 128)
    }
    # And byte-identical to the single pass everywhere the single pass reached.
    # Only the far edge may differ: the spacing carry leaves the last dab of a
    # longer walk at a different sub-position, which is the walk rather than the
    # accumulation.
    reached = once[..., 3] > 0
    assert np.array_equal(once[reached], many[reached])


def test_two_strokes_do_pile_up_where_one_does_not():
    """Coverage is per stroke, which is what makes lifting the button the way
    to put a second picture over a first."""
    stamp = brush.Stamp(_tip(alpha=128))
    doc = _stamped(stamp, points=((20.5, 20.5),))
    _stamped(stamp, doc=doc, points=((20.5, 20.5),))
    assert int(doc.stack.active.pixels[20, 20, 3]) > 128


def test_the_picture_does_not_depend_on_how_often_the_mouse_was_sampled():
    """The spacing carry's property, inherited: a stroke drawn at two frames a
    second and the same stroke at sixty are the same picture."""
    stamp = brush.Stamp(_tip(alpha=128))
    coarse = _stamped(stamp, points=((10.0, 20.0), (34.0, 20.0)))
    fine = _stamped(stamp, points=[(10.0 + i, 20.0) for i in range(25)])
    assert np.array_equal(coarse.stack.active.pixels, fine.stack.active.pixels)


# --- overlap: aligned -------------------------------------------------------


def test_aligned_dabs_snap_to_a_lattice_of_the_tips_own_size():
    stamp = brush.Stamp(_tip(width=8, height=8))
    doc = _stamped(stamp, points=((19.0, 19.0),), stamp_align="aligned")
    rows = np.flatnonzero(doc.stack.active.pixels[..., 3].any(axis=1))
    assert (int(rows[0]), int(rows[-1])) == (16, 23)


def test_every_point_in_one_cell_stamps_the_same_cell():
    """What makes aligned mode a *pattern*: neighbouring stamps line up rather
    than meeting wherever the mouse happened to be."""
    stamp = brush.Stamp(_tip(width=8, height=8))
    low = _stamped(stamp, points=((16.0, 16.0),), stamp_align="aligned")
    high = _stamped(stamp, points=((23.9, 23.9),), stamp_align="aligned")
    assert np.array_equal(low.stack.active.pixels, high.stack.active.pixels)


def test_aligned_overlap_is_idempotent_even_at_half_alpha():
    """Identical pixels restamped add no coverage, so a drag back and forth
    over one cell is exactly one dab of it -- which is the property free mode
    can only get by accumulating with maximum."""
    stamp = brush.Stamp(_tip(alpha=128))
    once = _stamped(stamp, points=((18.0, 18.0),), stamp_align="aligned")
    # Back and forth *inside* one cell: 16..23 is the lattice square an
    # eight-wide tip lands on here.
    dragged = _stamped(
        stamp,
        points=[(18.0 + 0.2 * i, 18.0) for i in range(20)]
        + [(22.0 - 0.2 * i, 18.0) for i in range(20)],
        stamp_align="aligned",
    )
    assert int(once.stack.active.pixels[18, 18, 3]) == 128
    assert np.array_equal(once.stack.active.pixels, dragged.stack.active.pixels)


def test_the_lattice_is_anchored_on_the_canvas_rather_than_on_the_press():
    """The only anchoring under which a pattern painted in two sittings still
    tiles."""
    stamp = brush.Stamp(_tip(width=8, height=8))
    first = _stamped(stamp, points=((3.0, 3.0),), stamp_align="aligned")
    second = _stamped(stamp, points=((5.0, 5.0),), stamp_align="aligned")
    rows = np.flatnonzero(first.stack.active.pixels[..., 3].any(axis=1))
    assert (int(rows[0]), int(rows[-1])) == (0, 7)
    assert np.array_equal(first.stack.active.pixels, second.stack.active.pixels)


def test_an_unknown_alignment_falls_back_to_free_rather_than_raising():
    stroke = brush.StrokeState(
        layer_uid=1, size=SIZE, before=np.zeros((48, 48, 4), np.uint8),
        colour=RED, stamp=brush.Stamp(_tip()), stamp_align="pattern",
    )
    assert stroke.stamp_align == "free"


# --- undo -------------------------------------------------------------------


def test_a_whole_image_stroke_is_exactly_one_undo_step():
    """One ``PatchEdit`` over the union of every dab -- never a step per dab."""
    stamp = brush.Stamp(_tip())
    doc = _doc()
    _stamped(stamp, doc=doc, points=[(6.0 + i, 20.0) for i in range(36)])
    # ``len`` rather than ``head``, which is a serial rather than a depth.
    assert len(doc.history) == 1


def test_undoing_an_image_stroke_restores_every_pixel_it_touched():
    stamp = brush.Stamp(_tip())
    doc = _doc()
    doc.stack.active.pixels[...] = (0, 0, 255, 255)
    before = doc.stack.active.pixels.copy()
    _stamped(stamp, doc=doc, points=[(6.0 + i, 20.0) for i in range(36)])
    assert not np.array_equal(doc.stack.active.pixels, before)
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, before)


def test_a_stroke_that_lands_nothing_pushes_nothing():
    stamp = brush.Stamp(_tip())
    doc = _doc()
    doc.begin_stroke((-400.0, -400.0), RED, stamp=stamp)
    doc.end_stroke()
    assert len(doc.history) == 0


# --- what a dab inherits ----------------------------------------------------


def test_the_alpha_lock_holds_against_an_image_stamp():
    stamp = brush.Stamp(_tip())
    doc = _doc()
    doc.stack.active.pixels[..., 3] = 0
    doc.stack.active.alpha_lock = True
    _stamped(stamp, doc=doc, points=((20.5, 20.5),))
    assert not doc.stack.active.pixels[..., 3].any()


def test_the_selection_clips_an_image_stamp():
    stamp = brush.Stamp(_tip())
    doc = _doc()
    doc.select(_rect_mask(SIZE, (20, 20, 24, 24)))
    _stamped(stamp, doc=doc, points=((20.5, 20.5),))
    painted = doc.stack.active.pixels[..., 3] > 0
    assert painted[20:24, 20:24].all()
    assert not painted[:20].any()
    assert not painted[24:].any()


def test_a_feathered_selection_softens_an_image_stamp_like_everything_else():
    stamp = brush.Stamp(_tip())
    doc = _doc()
    mask = _rect_mask(SIZE, (16, 16, 26, 26))
    mask.mask[16:26, 20] = 128
    doc.select(mask)
    _stamped(stamp, doc=doc, points=((20.5, 20.5),))
    assert int(doc.stack.active.pixels[20, 20, 3]) == 128


def test_an_image_stamp_wraps_on_a_tiled_document():
    stamp = brush.Stamp(_tip())
    doc = _doc()
    _stamped(stamp, doc=doc, points=((1.0, 20.5),), wrap="both")
    painted = doc.stack.active.pixels[..., 3] > 0
    assert painted[20, 0]
    # The half that hangs off the left edge arrives on the right.
    assert painted[20, -1]


def test_symmetry_mirrors_where_an_image_stamp_lands():
    """The *position* is mirrored, not the picture: symmetry is applied above
    the dab, which is what lets every mode inherit it."""
    stamp = brush.Stamp(_tip())
    doc = _stamped(stamp, points=((8.0, 20.5),), symmetry="x")
    painted = doc.stack.active.pixels[..., 3] > 0
    assert painted[20, 8]
    assert painted[20, SIZE[0] - 1 - 8]


def test_the_spray_emits_image_dabs_too():
    stamp = brush.Stamp(_tip(width=3, height=3))
    doc = _doc()
    doc.begin_stroke((24.0, 24.0), RED, stamp=stamp, scatter=10.0, seed=5)
    doc.spray_at((24.0, 24.0), 40)
    doc.end_stroke()
    assert doc.stack.active.pixels[..., 3].any()


# --- which modes take a tip -------------------------------------------------


@pytest.mark.parametrize("mode", ("blur", "smudge", "shade", "replace"))
def test_a_mode_that_cannot_use_a_picture_drops_the_tip(mode):
    """Dropped rather than refused: it is a tip that stays loaded while the
    user tries the smudge tool.

    ``replace`` is in the list for the reason ``STAMP_MODES`` gives at length: a
    captured tip's alpha is both its shape and its transparency, so a copy ink
    has nothing left to express about one. The panel hides the ink radio while a
    tip is loaded, so the two settings never contradict each other on screen.
    """
    stroke = brush.StrokeState(
        layer_uid=1, size=SIZE, before=np.zeros((48, 48, 4), np.uint8),
        colour=RED, mode=mode, stamp=brush.Stamp(_tip()),
    )
    assert stroke.stamp is None


def test_an_image_eraser_cuts_exactly_the_tips_alpha_however_many_passes(dabs):
    """The increment from the other side: alpha multiplies by ``1 - share`` per
    dab, and that product telescopes to ``1 - max``, so six passes cut exactly
    what one does."""
    stamp = brush.Stamp(_tip(alpha=128, mark=False))
    doc = _doc()
    doc.stack.active.pixels[...] = (0, 0, 255, 255)
    _stamped(stamp, doc=doc, mode="erase", points=_sweeps(1))
    once = doc.stack.active.pixels.copy()
    single = len(dabs)
    assert single > 4
    dabs.clear()
    doc.undo()
    _stamped(stamp, doc=doc, mode="erase", points=_sweeps(6))
    assert len(dabs) > 4 * single

    # 255 * (1 - 128/255) exactly, and nowhere any deeper however many passes.
    assert int(doc.stack.active.pixels[20, 20, 3]) == 127
    assert int(doc.stack.active.pixels[..., 3].min()) == 127
    cut = once[..., 3] < 255
    assert np.array_equal(once[cut], doc.stack.active.pixels[cut])


def test_a_replace_ink_brush_with_a_tip_loaded_paints_the_ordinary_way():
    """The tip is dropped rather than the stroke refused, so the stroke is the
    round replace stroke it was before a tip was captured."""
    doc = _doc()
    doc.stack.active.pixels[...] = (0, 0, 255, 255)
    _stamped(brush.Stamp(_tip()), doc=doc, mode="replace", points=((20.5, 20.5),))
    plain = _doc()
    plain.stack.active.pixels[...] = (0, 0, 255, 255)
    plain.begin_stroke((20.5, 20.5), RED, mode="replace")
    plain.end_stroke()
    assert np.array_equal(doc.stack.active.pixels, plain.stack.active.pixels)


def test_an_image_stroke_walks_the_spacing_whatever_the_nib_says():
    """The pixel walk emits a dab per whole pixel of travel, which is a full-tip
    composite per pixel for anything that is not a one-pixel pencil."""
    stroke = brush.StrokeState(
        layer_uid=1, size=SIZE, before=np.zeros((48, 48, 4), np.uint8),
        colour=RED, nib="pixel", stamp=brush.Stamp(_tip()),
    )
    assert stroke.pixel is False


def test_spacing_is_a_fraction_of_the_tip_rather_than_of_the_size_slider():
    stroke = brush.StrokeState(
        layer_uid=1, size=SIZE, before=np.zeros((48, 48, 4), np.uint8),
        colour=RED, diameter=4, spacing=0.5, stamp=brush.Stamp(_tip(width=20, height=8)),
    )
    assert stroke.step == pytest.approx(10.0)


# --- the app layer ----------------------------------------------------------


class _Settings:
    """The two methods ``persist`` and ``ensure`` use, over a plain dict."""

    def __init__(self, block=None) -> None:
        self.data = {"inker": block} if block is not None else {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


def _ctx(block=None) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(inker=None),
        settings=_Settings(block),
        toast=lambda text, kind="info": None,
    )


def test_the_alignment_combo_offers_every_mode_the_brush_implements():
    assert tuple(key for key, _label in inker_tools.STAMP_ALIGN_LABELS) == brush.STAMP_ALIGN


def test_the_stamping_tools_are_exactly_the_ones_whose_mode_takes_a_tip():
    """``STAMP_TOOLS`` is a *derivation*, written out because the panel and the
    press both read it -- so it is pinned against what it is derived from. The
    two lists drifting apart is the same seam the replace-ink bug came through:
    a tool the panel offers the controls for and the engine then drops the tip
    of is a feature that silently does nothing."""
    derived = {
        tool
        for tool, mode in inker_state.BRUSH_MODES.items()
        if mode in brush.STAMP_MODES
    }
    assert derived == inker_state.STAMP_TOOLS


def test_the_press_lays_down_whatever_tip_for_advertises():
    """``tip_for`` is the single source of truth, and the ink may not overrule
    it. The panel hides the ink radio and the cursor draws the tip's box while a
    tip is loaded, so a press that quietly took the ``replace`` path instead
    would stamp nothing, advertise the tip on screen, and leave no control to
    turn the ink back off.

    Both ways in are covered: setting Replace and *then* capturing (Ctrl+B does
    not touch the ink), and applying a preset -- ``paint_ink`` is one of the
    options a preset carries.
    """
    state = inker_state.InkerState()
    state.set_tool("brush")
    state.paint_ink = "replace"
    assert inker_canvas._press_mode(state, "brush", state.tip_for("brush")) == "replace"

    state.stamp = brush.Stamp(_tip())
    state.use_stamp = True
    assert state.tip_for("brush") is not None
    assert inker_canvas._press_mode(state, "brush", state.tip_for("brush")) == "paint"
    # The ink is ignored rather than cleared: forget the tip and the ink the
    # user chose is still theirs.
    assert state.paint_ink == "replace"
    state.use_stamp = False
    assert inker_canvas._press_mode(state, "brush", state.tip_for("brush")) == "replace"


def test_a_preset_carrying_a_replace_ink_cannot_disarm_a_loaded_tip():
    """A preset stores ``paint_ink`` and ``use_stamp`` side by side, so one can
    restore both at once -- the second of the two routes into the bug."""
    state = inker_state.InkerState()
    state.set_tool("brush")
    state.paint_ink = "replace"
    state.use_stamp = True
    state.save_preset("copy pen")

    fresh = inker_state.InkerState()
    fresh.presets = dict(state.presets)
    fresh.stamp = brush.Stamp(_tip())
    fresh.apply_preset("copy pen")
    assert fresh.paint_ink == "replace"
    assert fresh.tip_for("brush") is not None
    assert inker_canvas._press_mode(fresh, "brush", fresh.tip_for("brush")) == "paint"


@pytest.mark.parametrize("tool", sorted(inker_state.PAINT_TOOLS))
def test_no_other_tool_reads_the_brushs_ink(tool):
    state = inker_state.InkerState()
    state.set_tool(tool)
    state.paint_ink = "replace"
    expected = "replace" if tool == "brush" else inker_state.BRUSH_MODES[tool]
    assert inker_canvas._press_mode(state, tool, None) == expected


def test_a_tool_that_cannot_stamp_never_reports_a_tip():
    state = inker_state.InkerState()
    state.stamp = brush.Stamp(_tip())
    for tool in inker_state.STAMP_TOOLS:
        state.options_for(tool)["use_stamp"] = True
    assert state.tip_for("brush") is state.stamp
    assert state.tip_for("smudge") is None
    assert state.tip_for("line") is None


def test_the_tip_is_per_tool_so_an_eraser_can_keep_a_round_one():
    state = inker_state.InkerState()
    state.stamp = brush.Stamp(_tip())
    state.options_for("brush")["use_stamp"] = True
    assert state.tip_for("brush") is state.stamp
    assert state.tip_for("eraser") is None


def test_capturing_picks_the_brush_up_and_turns_the_tip_on():
    """Capturing into a tool the user then has to find is a feature that looks
    like it did nothing -- and the tool in hand is a selection tool by
    definition."""
    ctx = _ctx()
    state = inker_mode.ensure(ctx)
    doc = _doc()
    doc.stack.active.pixels[10:18, 10:18] = _tip()
    state.add(inker_state.InkerDoc(doc=doc, title="t"))
    state.set_tool("lasso")
    doc.select(_rect_mask(SIZE, (10, 10, 18, 18)))
    assert inker_mode.capture_brush(ctx) is True
    assert state.tool == "brush"
    assert state.use_stamp is True
    assert state.tip_for("brush") is state.stamp


def test_ctrl_b_captures_the_way_aseprite_does():
    ctx = _ctx()
    state = inker_mode.ensure(ctx)
    doc = _doc()
    doc.stack.active.pixels[10:18, 10:18] = _tip()
    state.add(inker_state.InkerDoc(doc=doc, title="t"))
    doc.select(_rect_mask(SIZE, (10, 10, 18, 18)))
    # Through the registry, which is where Ctrl+B is bound now (W2.7): the
    # chord is ``Op.key`` and the op is the only implementation of it.
    from warlock.studio import inker_ops

    inker_ops.run(ctx, inker_ops.get("capture_brush"))
    assert state.stamp is not None
    # Not a mutating chord: it reads pixels and changes app state, which is what
    # Ctrl+C does and is safe while a save is in flight for the same reason.
    assert "b" not in inker_mode._MUTATING_CTRL


def test_capturing_nothing_is_refused_out_loud():
    said: list[tuple[str, str]] = []
    ctx = _ctx()
    ctx.toast = lambda text, kind="info": said.append((text, kind))
    state = inker_mode.ensure(ctx)
    state.add(inker_state.InkerDoc(doc=_doc(), title="t"))
    assert inker_mode.capture_brush(ctx) is False
    assert said and said[0][1] == "warn"


def test_forgetting_the_tip_unticks_every_tool_not_just_the_one_in_hand():
    ctx = _ctx()
    state = inker_mode.ensure(ctx)
    state.stamp = brush.Stamp(_tip())
    for tool in inker_state.STAMP_TOOLS:
        state.options_for(tool)["use_stamp"] = True
    inker_mode.clear_brush(ctx)
    assert state.stamp is None
    assert all(state.tip_for(tool) is None for tool in inker_state.STAMP_TOOLS)


# --- presets ----------------------------------------------------------------


def test_a_preset_round_trips_through_save_and_apply():
    state = inker_state.InkerState()
    state.set_tool("brush")
    state.brush_size = 40
    state.hardness = 0.2
    state.save_preset("inking pen")
    state.brush_size = 3
    state.set_tool("eraser")
    assert state.apply_preset("inking pen") is True
    assert state.tool == "brush"
    assert state.brush_size == 40
    assert state.hardness == pytest.approx(0.2)


def test_applying_a_preset_replaces_the_tool_rather_than_updating_it():
    """Over a base of the defaults, so a preset saved before an option existed
    leaves that option at its default -- otherwise applying the same preset
    twice in one session gives two different brushes."""
    state = inker_state.InkerState()
    state.set_tool("brush")
    state.save_preset("plain")
    state.opacity = 0.1
    state.apply_preset("plain")
    assert state.opacity == inker_state.TOOL_OPTION_DEFAULTS["opacity"]


def test_a_preset_is_deleted_by_name():
    state = inker_state.InkerState()
    state.save_preset("gone")
    assert state.delete_preset("gone") is True
    assert state.delete_preset("gone") is False
    assert state.apply_preset("gone") is False


def test_saving_over_a_name_replaces_it():
    state = inker_state.InkerState()
    state.brush_size = 11
    state.save_preset("one")
    state.brush_size = 22
    state.save_preset("one")
    assert len(state.presets) == 1
    state.brush_size = 3
    state.apply_preset("one")
    assert state.brush_size == 22


def test_an_empty_name_saves_nothing():
    state = inker_state.InkerState()
    assert state.save_preset("   ") == ""
    assert not state.presets


def test_the_preset_shelf_has_a_ceiling_and_drops_the_oldest():
    state = inker_state.InkerState()
    for index in range(inker_state.MAX_PRESETS + 5):
        state.save_preset(f"p{index}")
    assert len(state.presets) == inker_state.MAX_PRESETS
    assert "p0" not in state.presets
    assert f"p{inker_state.MAX_PRESETS + 4}" in state.presets


def test_applying_a_preset_goes_through_set_tool():
    """A tool cannot be changed without a half-drawn multi-click gesture going
    with it, whichever door does the changing."""
    state = inker_state.InkerState()
    state.set_tool("brush")
    state.save_preset("pen")
    state.set_tool("lasso_poly")
    state.gesture_pts = [(1.0, 1.0), (2.0, 2.0)]
    state.apply_preset("pen")
    assert state.gesture_pts == []


def test_presets_survive_a_persist_and_a_fresh_session():
    ctx = _ctx()
    state = inker_mode.ensure(ctx)
    state.set_tool("brush")
    state.brush_size = 33
    state.save_preset("thick")
    inker_mode.persist(ctx)

    reopened = _ctx(ctx.settings.get("inker"))
    restored = inker_mode.ensure(reopened)
    assert restored.apply_preset("thick") is True
    assert restored.brush_size == 33


def test_deleting_a_preset_persists_too():
    ctx = _ctx()
    state = inker_mode.ensure(ctx)
    state.save_preset("temporary")
    inker_mode.persist(ctx)
    state.delete_preset("temporary")
    inker_mode.persist(ctx)
    assert inker_mode.ensure(_ctx(ctx.settings.get("inker"))).presets == {}


def test_persisting_presets_leaves_the_swatches_and_the_legacy_key_alone():
    ctx = _ctx({"recent": ["a.ora"], "swatches": [[1, 2, 3, 4]]})
    inker_mode.ensure(ctx)
    inker_mode.persist(ctx)
    assert ctx.settings.get("inker")["recent"] == ["a.ora"]
    assert ctx.settings.get("inker")["swatches"] == [[1, 2, 3, 4]]


@pytest.mark.parametrize(
    "block",
    (
        "not a dict",
        {"pen": "not a dict"},
        {"pen": {"tool": "no-such-tool", "options": {}}},
        {"pen": {"tool": "brush"}},
    ),
)
def test_a_stored_block_that_is_not_a_preset_is_dropped_rather_than_crashing(block):
    """A JSON file a user may edit and a build may have changed the option
    table under."""
    assert inker_mode.ensure(_ctx({"presets": block})).presets == {}


def test_a_stored_option_the_table_no_longer_has_is_dropped():
    state = inker_mode.ensure(
        _ctx({"presets": {"pen": {"tool": "brush", "options": {"brush_size": 9, "gone": 1}}}})
    )
    assert state.presets["pen"]["options"] == {"brush_size": 9}
    state.apply_preset("pen")
    assert state.brush_size == 9


def test_the_captured_tip_is_not_part_of_a_preset():
    """It is pixels rather than a setting: the settings block is small JSON
    rewritten on every swatch click."""
    ctx = _ctx()
    state = inker_mode.ensure(ctx)
    state.stamp = brush.Stamp(_tip())
    state.use_stamp = True
    state.save_preset("stamped")
    inker_mode.persist(ctx)
    restored = inker_mode.ensure(_ctx(ctx.settings.get("inker")))
    restored.apply_preset("stamped")
    # The flag comes back, the pixels do not -- and ``tip_for`` is what keeps
    # that from meaning a stroke stamps something that is not there.
    assert restored.use_stamp is True
    assert restored.stamp is None
    assert restored.tip_for("brush") is None
