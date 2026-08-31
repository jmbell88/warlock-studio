"""The movable transform pivot: engine, ranged replay, and the real handle.

Three things are being defended here and they are not the same thing.

**Nothing changes for anyone who never touches it.** ``pivot`` is None until a
user drags the ring, and with it None the render is the code it has always
been. That is pinned below against hashes captured from the tree *before* this
feature existed, not against a re-derivation of what the maths ought to
produce -- a re-derivation would agree with a rewritten kernel by construction,
which is exactly the agreement worth nothing.

**A ranged commit carries the pivot.** ``_replay_transform_on``'s whole
correctness argument is that every cel renders to the same shape, so one
``dest`` places them all. The pivot decides the padding and the crop, so a
replay that dropped it would render every cel about the source box's centre --
the same shape, landing at the same place, and a different picture from the one
the user watched on the active cel. Nothing would raise; it would simply be
wrong, and only in motion.

**The handle is not merely drawn.** It goes through ``_input`` with a fake
mouse, one frame per call, which is the same harness the canvas's other
gestures are driven by.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import inker, inker_state
from warlock.studio.inker import _doc_selection
from warlock.studio.inker.selection import SelectionMask, render_transform_about
from warlock.studio.panes import inker_canvas

SIZE = (32, 24)


# --- the standing negative control ------------------------------------------
#
# Captured by running the gestures below against 3cf38f0c -- the commit before
# ``pivot`` existed -- and pasted here. They are what "bit-identical" means:
# every byte of the rendered plane, of its mask, and of the two commits those
# feed. If a future change to the padding or the crop leaks into the pivotless
# path, these are what says so.

PIVOTLESS = {
    "rot37_smooth": ("5b6d7e0731181707", "7108980126848521", (2, -1), (22, 22)),
    "rot37_nearest": ("8124034186d8cca0", "7108980126848521", (2, -1), (22, 22)),
    "scale_smooth": ("8c0e0e1479e37134", "a818847102ee5cad", (0, 6), (27, 8)),
    "shear_smooth": ("f4b3470abb536b7e", "55f1ff8cf607c01d", (4, 2), (19, 16)),
    "all_smooth": ("c0fd03ba80408899", "d3ee67575ba60239", (0, -1), (27, 22)),
    "flip_rot": ("72fc6e6ba4db0e3f", "3223250cef009406", (2, -1), (22, 22)),
}
PIVOTLESS_COMMIT = "431681d1a3399b7c"
PIVOTLESS_RANGE = ("8cdaf69f7e15ad53", "5d61d664081758e6", "16d078a3404aa237")

GESTURES = {
    "rot37_smooth": dict(angle=37.0, resample="smooth"),
    "rot37_nearest": dict(angle=37.0, resample="nearest"),
    "scale_smooth": dict(scale=(1.7, 0.6), resample="smooth"),
    "shear_smooth": dict(shear=(12.0, -7.0), resample="smooth"),
    "all_smooth": dict(
        angle=23.0, scale=(1.4, 0.8), shear=(5.0, 3.0), resample="smooth"
    ),
}

LIFT_BOX = (5, 3, 21, 17)


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def _patterned() -> inker.Document:
    """A document whose every pixel differs, so a hash pins the whole plane."""
    doc = inker.Document.blank(*SIZE)
    pixels = doc.stack.active.pixels
    ys, xs = np.mgrid[0 : SIZE[1], 0 : SIZE[0]]
    pixels[..., 0] = (xs * 7 % 251).astype(np.uint8)
    pixels[..., 1] = (ys * 13 % 241).astype(np.uint8)
    pixels[..., 2] = ((xs * ys) % 233).astype(np.uint8)
    pixels[..., 3] = 255
    doc.invalidate_all()
    return doc


def _floated() -> inker.Document:
    doc = _patterned()
    doc.select(SelectionMask.from_rect(SIZE, LIFT_BOX))
    assert doc.lift()
    return doc


@pytest.mark.parametrize("name", sorted(GESTURES))
def test_a_pivotless_transform_renders_exactly_what_it_always_did(name):
    doc = _floated()
    assert doc.transform_floating(**GESTURES[name])
    buf = doc.floating
    assert buf.pivot is None
    pixels, mask, offset, size = PIVOTLESS[name]
    assert (_sha(buf.pixels), _sha(buf.mask)) == (pixels, mask)
    assert (buf.offset, buf.size) == (offset, size)


def test_a_pivotless_flip_then_rotate_renders_exactly_what_it_always_did():
    doc = _floated()
    assert doc.flip_floating("horizontal")
    assert doc.transform_floating(angle=30.0, resample="smooth")
    buf = doc.floating
    pixels, mask, offset, size = PIVOTLESS["flip_rot"]
    assert (_sha(buf.pixels), _sha(buf.mask)) == (pixels, mask)
    assert (buf.offset, buf.size) == (offset, size)


def test_a_pivotless_commit_writes_exactly_the_bytes_it_always_did():
    doc = _floated()
    assert doc.transform_floating(angle=37.0, resample="smooth")
    doc.move_floating(3, -2)
    assert doc.commit_floating()
    assert _sha(doc.stack.active.pixels) == PIVOTLESS_COMMIT


def _cel(doc, track, frame):
    anim = doc.anim
    return anim.cels.get((anim.tracks[track].uid, anim.frames[frame].uid))


def _clip() -> inker.Document:
    """Three frames of the patterned document, each with its own block."""
    doc = _patterned()
    for index in range(1, 3):
        doc.add_frame()
        box = (6, 4 + index, 20, 18 + index)
        weight = np.ones((box[3] - box[1], box[2] - box[0]), np.float32)
        assert doc.write_colour(box, (10 * (index + 1), 60, 200, 255), weight)
    doc.set_current_frame(0)
    doc.invalidate_all()
    return doc


def test_a_pivotless_ranged_commit_writes_exactly_the_bytes_it_always_did():
    doc = _clip()
    doc.select(SelectionMask.from_rect(SIZE, LIFT_BOX))
    assert doc.lift()
    assert doc.transform_floating(angle=37.0, resample="smooth")
    doc.move_floating(2, 1)
    assert doc.commit_floating_range(0, 0, 0, 2)
    assert tuple(_sha(_cel(doc, 0, i).pixels) for i in range(3)) == PIVOTLESS_RANGE


# --- the buffer's own state --------------------------------------------------


def test_a_fresh_buffer_has_no_pivot():
    """Which is the whole of the negative control above, stated once."""
    doc = _floated()
    assert doc.floating.pivot is None
    assert doc.floating.pivot_local is None


def test_a_pivot_with_no_source_yet_has_no_local_form():
    """There is no frame to express it in until a render has captured one."""
    doc = _floated()
    assert doc.set_floating_pivot((4.0, 4.0))
    assert doc.floating.source is None
    assert doc.floating.pivot_local is None


def test_the_pivot_and_the_source_corner_travel_with_a_move():
    """A moved selection must not strand its pivot -- and the two move by the
    same delta, so what a render actually reads is untouched."""
    doc = _floated()
    assert doc.set_floating_pivot((7.0, 5.0))
    assert doc.transform_floating(angle=20.0, resample="nearest")
    buf = doc.floating
    local = buf.pivot_local
    offset, source_offset = buf.offset, buf.source_offset

    doc.move_floating(6, -3)

    assert buf.offset == (offset[0] + 6, offset[1] - 3)
    assert buf.pivot == (13.0, 2.0)
    assert buf.source_offset == (source_offset[0] + 6, source_offset[1] - 3)
    assert buf.pivot_local == local


def test_a_move_carries_a_pivot_through_the_document_door():
    doc = _floated()
    assert doc.set_floating_pivot((7.0, 5.0))
    doc.move_floating(2, 2)
    assert doc.floating.pivot == (9.0, 7.0)


def test_the_pivot_is_clamped_to_the_canvas():
    """It decides how far the source is padded, so an unclamped one dragged off
    the page allocates a plane wider than the drawing."""
    doc = _floated()
    assert doc.set_floating_pivot((-400.0, 900.0))
    assert doc.floating.pivot == (0.0, 24.0)


def test_setting_the_pivot_to_none_goes_back_to_the_centred_render():
    """And *in place*: dropping the pivot re-centres on where the subject
    already is rather than teleporting it back to where a never-pivoted gesture
    would have left it. Every further turn is about its own centre again, which
    is the only thing 'no pivot' can mean."""
    doc = _floated()
    assert doc.set_floating_pivot((2.0, 2.0))
    assert doc.transform_floating(angle=90.0, resample="nearest")
    centre = doc.floating.centre

    assert doc.set_floating_pivot(None)

    assert doc.floating.pivot is None
    assert doc.floating.pivot_local is None
    assert doc.floating.centre == pytest.approx(centre, abs=1.0)
    centred = _floated()
    assert centred.transform_floating(angle=90.0, resample="nearest")
    # The same picture, only placed where the pivoted gesture had left it.
    assert np.array_equal(doc.floating.pixels, centred.floating.pixels)


def test_setting_the_pivot_on_an_untransformed_buffer_renders_nothing():
    """No render means no source captured and no rev bump: the identity
    transform of a padded plane is the plane, and paying for a pad and a crop
    to learn that would churn the pane's texture cache for nothing."""
    doc = _floated()
    rev = doc.floating.rev
    assert doc.set_floating_pivot((1.0, 1.0))
    assert doc.floating.rev == rev
    assert doc.floating.source is None


def test_moving_the_pivot_re_renders_a_buffer_that_is_already_turned():
    """A preview that only caught up on the next drag of some other handle
    would be showing a picture the commit would not write."""
    doc = _floated()
    assert doc.set_floating_pivot((6.0, 5.0))
    assert doc.transform_floating(angle=45.0, resample="nearest")
    before = doc.floating.offset

    assert doc.set_floating_pivot((18.0, 15.0))

    assert doc.floating.offset != before
    assert doc.floating.angle == 45.0


# --- what a pivot actually does ---------------------------------------------


MARKED = (255, 0, 0, 255)
OTHER = (0, 255, 0, 255)


def _marked_doc() -> inker.Document:
    doc = inker.Document.blank(8, 8)
    pixels = doc.stack.active.pixels
    pixels[..., :3] = 30
    pixels[..., 3] = 255
    pixels[2, 2] = MARKED
    pixels[2, 5] = OTHER
    doc.invalidate_all()
    doc.select(SelectionMask.from_rect((8, 8), (0, 0, 8, 8)))
    assert doc.lift()
    return doc


def _at(doc, colour) -> list[tuple[int, int]]:
    """Where a marker pixel sits, in canvas coordinates."""
    ox, oy = doc.floating.offset
    plane = doc.floating.pixels
    ys, xs = np.nonzero(
        (plane[..., 0] == colour[0])
        & (plane[..., 1] == colour[1])
        & (plane[..., 2] == colour[2])
    )
    return [(int(x) + ox, int(y) + oy) for x, y in zip(xs, ys, strict=True)]


def test_the_pivot_is_the_fixed_point_of_a_rotation():
    """A half turn about the centre of pixel (2, 2) leaves that pixel where it
    was and reflects everything else through it."""
    doc = _marked_doc()
    assert doc.set_floating_pivot((2.5, 2.5))
    assert doc.transform_floating(angle=180.0, resample="nearest")
    assert _at(doc, MARKED) == [(2, 2)]
    # 2 * 2.5 - 5.5 = -0.5, which is the pixel covering [-1, 0).
    assert _at(doc, OTHER) == [(-1, 2)]


def test_without_a_pivot_the_same_half_turn_moves_that_pixel():
    """The control. Without it the test above would pass on a buffer that had
    simply not moved at all."""
    doc = _marked_doc()
    assert doc.transform_floating(angle=180.0, resample="nearest")
    assert _at(doc, MARKED) == [(5, 5)]


def test_a_pivot_dropped_on_the_centre_is_the_centred_render():
    """The continuity that makes the handle safe to grab: putting the ring back
    where it started changes nothing, so there is no jump at the moment a pivot
    first exists."""
    doc = _marked_doc()
    centre = doc.floating.centre
    assert doc.set_floating_pivot(centre)
    assert doc.transform_floating(angle=180.0, resample="nearest")

    plain = _marked_doc()
    assert plain.transform_floating(angle=180.0, resample="nearest")
    assert doc.floating.offset == plain.floating.offset
    assert np.array_equal(doc.floating.pixels, plain.floating.pixels)


def test_a_gesture_adjusted_back_and_forth_does_not_creep():
    """Every adjustment renders from the source again, so the pivot has to be
    read from a frame the render does not move -- ``source_offset``. Derived
    from the buffer's own offset instead, it would creep a little further with
    every drag of the same gesture: drag a scale handle out and back and the
    subject would not be where it started."""
    doc = _marked_doc()
    assert doc.set_floating_pivot((2.5, 2.5))
    assert doc.transform_floating(scale=(1.5, 1.5), resample="nearest")
    local = doc.floating.pivot_local
    first = (doc.floating.offset, doc.floating.size)
    marked = _at(doc, MARKED)
    for factor in (2.0, 0.7, 1.5):
        assert doc.transform_floating(scale=(factor, factor), resample="nearest")

    assert (doc.floating.offset, doc.floating.size) == first
    assert _at(doc, MARKED) == marked
    assert doc.floating.pivot_local == local
    # A scale about that pivot leaves the pixel it sits in covering it.
    assert (2, 2) in marked


def test_a_pivot_near_an_edge_does_not_leave_the_box_off_the_pixels():
    """The crop. Padding to centre a corner pivot doubles the plane; leaving it
    padded would put the transform box a subject's width away from anything
    drawn in it."""
    doc = _marked_doc()
    assert doc.set_floating_pivot((0.0, 0.0))
    assert doc.transform_floating(scale=(1.0, 1.0), resample="nearest")
    assert doc.floating.size == (8, 8)
    assert doc.floating.offset == (0, 0)


def test_the_crop_reads_the_mask_and_never_the_pixels():
    """Which is what lets a ranged replay crop every cel identically: the mask
    is shared across the range and the content is not."""
    source = np.zeros((6, 6, 4), np.uint8)
    source[..., 3] = 255
    mask = np.zeros((6, 6), np.uint8)
    mask[1:5, 1:5] = 255
    a, a_mask, a_origin = render_transform_about(
        source, mask, 0.0, (1.0, 1.0), (0.0, 0.0), "nearest", (1.5, 1.5)
    )
    other = source.copy()
    other[..., 0] = 200
    b, b_mask, b_origin = render_transform_about(
        other, mask, 0.0, (1.0, 1.0), (0.0, 0.0), "nearest", (1.5, 1.5)
    )
    assert a.shape == b.shape and a_origin == b_origin
    assert np.array_equal(a_mask, b_mask)


# --- the ranged replay -------------------------------------------------------


def _pivoted_clip():
    doc = _clip()
    doc.select(SelectionMask.from_rect(SIZE, LIFT_BOX))
    assert doc.lift()
    assert doc.set_floating_pivot((7.0, 5.0))
    assert doc.transform_floating(angle=90.0, resample="nearest")
    return doc


def test_every_cel_in_the_range_turns_about_the_same_pivot(monkeypatch):
    """The sharp edge, pinned on the argument itself. A ranged 'rotate this
    pose across eight frames' that re-pivoted per cel would look plausible on
    every single frame and only read as wrong in motion, so the assertion is on
    what each replay was handed rather than on how the result looks."""
    doc = _pivoted_clip()
    expected = doc.floating.pivot_local
    assert expected is not None

    seen: list[tuple[float, float] | None] = []
    real = _doc_selection.render_transform_about

    def spy(*args):
        seen.append(args[-1])
        return real(*args)

    monkeypatch.setattr(_doc_selection, "render_transform_about", spy)
    assert doc.commit_floating_range(0, 0, 0, 2)

    assert len(seen) == 3, "one replay per cel in the range"
    assert seen == [expected] * 3


def test_a_pivoted_ranged_commit_writes_what_the_plain_commit_writes():
    """The replay *is* the commit on the cel the user was looking at. If the
    pivot did not survive the hand-off, this is the frame that would disagree
    -- and it is the only frame anybody could have seen."""
    plain, ranged = _pivoted_clip(), _pivoted_clip()
    for doc in (plain, ranged):
        doc.move_floating(2, 1)
    assert plain.commit_floating()
    assert ranged.commit_floating_range(0, 0, 0, 2)
    assert np.array_equal(_cel(plain, 0, 0).pixels, _cel(ranged, 0, 0).pixels)


def test_a_pivoted_range_lands_somewhere_a_centred_one_would_not():
    """The control for the two above: with the pivot dropped on the way in,
    every cel would still render to the same shape and still land at the same
    ``dest``. It would simply be the wrong picture."""
    pivoted, centred = _pivoted_clip(), _clip()
    centred.select(SelectionMask.from_rect(SIZE, LIFT_BOX))
    assert centred.lift()
    assert centred.transform_floating(angle=90.0, resample="nearest")
    assert pivoted.commit_floating_range(0, 0, 0, 2)
    assert centred.commit_floating_range(0, 0, 0, 2)
    for index in range(3):
        assert not np.array_equal(
            _cel(pivoted, 0, index).pixels, _cel(centred, 0, index).pixels
        ), index


def test_a_pivoted_range_shows_each_cel_its_own_content():
    """Unchanged by the pivot and worth restating: every cel is its own drawing
    turned, not three copies of the active one."""
    doc = _pivoted_clip()
    assert doc.commit_floating_range(0, 0, 0, 2)
    planes = [_cel(doc, 0, i).pixels for i in range(3)]
    assert not np.array_equal(planes[0], planes[1])
    assert not np.array_equal(planes[1], planes[2])


def test_the_replay_takes_the_pivot_by_name_and_defaults_to_none():
    """Structural. A positional-only extra on the end is how the ranged commit
    and the replay drift apart the next time either grows an argument."""
    import inspect

    signature = inspect.signature(_doc_selection.SelectionOps._replay_transform_on)
    parameter = signature.parameters["pivot"]
    assert parameter.default is None


# --- the handle, through the real input path ---------------------------------


class _Mouse:
    """imgui's mouse, as much of it as ``_input`` reads."""

    def __init__(self) -> None:
        self.at = (0.0, 0.0)
        self.down = {0: False, 1: False, 2: False}
        self.clicked = {0: False, 1: False, 2: False}
        self.dragging = {0: False, 1: False, 2: False}
        self.wheel = 0.0
        self.cursors: list[str] = []
        self.shift = False

    def module(self) -> SimpleNamespace:
        return SimpleNamespace(
            get_io=lambda: SimpleNamespace(
                mouse_wheel=self.wheel,
                key_shift=self.shift,
                key_alt=False,
                key_ctrl=False,
                delta_time=1.0 / 60.0,
            ),
            get_mouse_pos=lambda: SimpleNamespace(x=self.at[0], y=self.at[1]),
            is_mouse_clicked=lambda button: self.clicked[button],
            is_mouse_down=lambda button: self.down[button],
            is_mouse_dragging=lambda button: self.dragging[button],
            get_mouse_drag_delta=lambda button: SimpleNamespace(x=0.0, y=0.0),
            reset_mouse_drag_delta=lambda button: None,
            set_mouse_cursor=self.cursors.append,
            # ``_transform_box`` packs the accent colour through imgui's global
            # alpha; the number itself is not what any of this asserts.
            get_color_u32=lambda value: 0xFFFFFFFF,
            ImVec4=lambda *parts: parts,
            MouseCursor_=SimpleNamespace(
                **{
                    name: SimpleNamespace(value=name)
                    for name in ("hand", "not_allowed", "resize_all", "arrow")
                }
            ),
        )


#: The pane fixture's document, and the rectangle it floats. Deliberately far
#: bigger than the engine tests' 32x24: at zoom 1 a handle is grabbed from
#: 12.5 screen pixels away, so on a small buffer *every* interior point is
#: within reach of some handle and the plain move drag is unreachable. That is
#: true of the eight scale handles too and has been since they landed -- it is
#: the marquee being smaller than the grab radius, not a claim the ring makes.
BIG = (96, 96)
BIG_BOX = (16, 16, 80, 80)


def _big() -> inker.Document:
    doc = inker.Document.blank(*BIG)
    pixels = doc.stack.active.pixels
    ys, xs = np.mgrid[0 : BIG[1], 0 : BIG[0]]
    pixels[..., 0] = (xs * 5 % 251).astype(np.uint8)
    pixels[..., 1] = (ys * 11 % 241).astype(np.uint8)
    pixels[..., 3] = 255
    doc.invalidate_all()
    doc.select(SelectionMask.from_rect(BIG, BIG_BOX))
    assert doc.lift()
    return doc


@pytest.fixture
def driven(monkeypatch):
    """``_input`` with a fake mouse over a lifted, transforming buffer.

    At zoom 1 with no pan and the origin at (0, 0), a screen coordinate *is* an
    image coordinate, so the numbers a test types are the pixels it means.
    """
    mouse = _Mouse()
    monkeypatch.setattr(inker_canvas, "imgui", mouse.module())
    state = inker_state.InkerState()
    doc = _big()
    state.transforming = True
    tab = SimpleNamespace(
        doc=doc,
        tiled="off",
        busy=False,
        view=inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True),
    )

    def frame(at, *, click=None, down=(), shift=False):
        mouse.at = (float(at[0]), float(at[1]))
        mouse.clicked = {0: False, 1: False, 2: False}
        if click is not None:
            mouse.clicked[click] = True
        mouse.down = {button: button in down for button in (0, 1, 2)}
        mouse.shift = shift
        inker_canvas._input(None, state, tab, (0.0, 0.0), active=True, hovered=True)

    frame.mouse = mouse
    return state, tab, frame


def test_the_pivot_ring_sits_on_the_centre_until_it_is_dragged(driven):
    _state, tab, _frame = driven
    handles = inker_canvas._handles(tab, (0.0, 0.0))
    assert handles["pivot"] == pytest.approx(tab.doc.floating.centre)


def test_dragging_the_ring_moves_the_pivot(driven):
    """The control that is drawn and does nothing is this codebase's commonest
    historical defect, so the ring is grabbed and dragged rather than merely
    asked where it is."""
    state, tab, frame = driven
    centre = tab.doc.floating.centre

    frame(centre, click=0, down=(0,))
    assert state.drag_kind == "pivot"
    assert state.transform_grab == "pivot"

    frame((30.0, 22.0), down=(0,))
    assert tab.doc.floating.pivot == (30.0, 22.0)

    frame((30.0, 22.0), down=())
    assert state.drag_kind == ""
    assert tab.doc.floating.pivot == (30.0, 22.0)


def test_dragging_the_ring_does_not_move_the_pixels(driven):
    """It is a reference point, not a nudge."""
    state, tab, frame = driven
    offset = tab.doc.floating.offset
    frame(tab.doc.floating.centre, click=0, down=(0,))
    frame((30.0, 22.0), down=(0,))
    assert tab.doc.floating.offset == offset
    assert state.drag_kind == "pivot"


def test_a_press_inside_the_buffer_and_away_from_the_ring_still_moves_it(driven):
    """The regression the ring could have caused. Handles have always won a
    press over the move -- ``near`` is tested before ``contains`` -- and the
    ring is one more handle, not a new claim on the middle of the subject."""
    state, tab, frame = driven
    frame((66.0, 66.0), click=0, down=(0,))
    assert state.drag_kind == "move"
    frame((69.0, 68.0), down=(0,))
    assert tab.doc.floating.offset == (19, 18)
    assert tab.doc.floating.pivot is None


def test_a_move_drag_carries_a_pivot_the_user_placed(driven):
    state, tab, frame = driven
    frame(tab.doc.floating.centre, click=0, down=(0,))
    frame((30.0, 22.0), down=(0,))
    frame((30.0, 22.0), down=())

    frame((66.0, 66.0), click=0, down=(0,))
    assert state.drag_kind == "move"
    frame((69.0, 68.0), down=(0,))
    assert tab.doc.floating.pivot == (33.0, 24.0)


def test_a_rotate_drag_turns_about_the_ring_the_user_moved(driven):
    """End to end: the ring is dragged, then the rotate arm is dragged, and
    what lands is what a headless transform about that same point produces --
    and is not what a centred one produces."""
    state, tab, frame = driven
    frame(tab.doc.floating.centre, click=0, down=(0,))
    frame((30.0, 22.0), down=(0,))
    frame((30.0, 22.0), down=())

    arm = inker_canvas._handles(tab, (0.0, 0.0))["rotate"]
    frame(arm, click=0, down=(0,))
    assert state.drag_kind == "rotate"
    frame((arm[0] + 40.0, arm[1] + 40.0), down=(0,))
    buf = tab.doc.floating
    assert abs(buf.angle) > 1.0, "the drag actually turned something"

    twin = _big()
    assert twin.set_floating_pivot((30.0, 22.0))
    assert twin.transform_floating(angle=buf.angle, resample=state.resample)
    assert twin.floating.offset == buf.offset
    assert np.array_equal(twin.floating.pixels, buf.pixels)

    centred = _big()
    assert centred.transform_floating(angle=buf.angle, resample=state.resample)
    assert centred.floating.offset != buf.offset


def test_a_scale_drag_measures_from_the_ring_the_user_moved(driven):
    """The other reference the pivot owns. Measured against the same drag on a
    buffer whose ring was left alone, which must not agree."""
    state, tab, frame = driven
    frame(tab.doc.floating.centre, click=0, down=(0,))
    frame((30.0, 22.0), down=(0,))
    frame((30.0, 22.0), down=())

    corner = inker_canvas._handles(tab, (0.0, 0.0))["se"]
    frame(corner, click=0, down=(0,))
    assert state.drag_kind == "scale"
    frame((corner[0] + 12.0, corner[1] + 12.0), down=(0,))
    buf = tab.doc.floating
    assert buf.scale != (1.0, 1.0)
    assert buf.pivot == (30.0, 22.0)

    twin = _big()
    assert twin.set_floating_pivot((30.0, 22.0))
    assert twin.transform_floating(scale=buf.scale, resample=state.resample)
    assert twin.floating.offset == buf.offset

    centred = _big()
    assert centred.transform_floating(scale=buf.scale, resample=state.resample)
    assert centred.floating.offset != buf.offset


class _Recorder:
    """A draw list that remembers the rings and the boxes it was asked for."""

    def __init__(self) -> None:
        self.circles: list[tuple] = []
        self.rects: list[tuple] = []
        self.lines: list[tuple] = []

    def add_rect(self, a, b, colour):
        self.rects.append((a, b))

    def add_rect_filled(self, a, b, colour):
        self.rects.append((a, b))

    def add_circle(self, point, radius, colour):
        self.circles.append((point, radius))

    def add_circle_filled(self, point, radius, colour):
        self.circles.append((point, radius))

    def add_line(self, a, b, colour):
        self.lines.append((a, b))


def test_the_pivot_is_drawn_as_the_symmetry_guide_s_ring(driven):
    """One visual language for 'the point this turns about'. The radial
    symmetry guide draws a ring and a crosshair; so does this."""
    state, tab, frame = driven
    frame(tab.doc.floating.centre, click=0, down=(0,))
    frame((30.0, 22.0), down=(0,))
    frame((30.0, 22.0), down=())

    draw_list = _Recorder()
    inker_canvas._transform_box(state, tab, draw_list, (0.0, 0.0))

    radius = inker_canvas.sp(inker_canvas.SYMMETRY_PIVOT_RADIUS)
    assert ((30.0, 22.0), radius) in draw_list.circles
    assert ((30.0 - radius, 22.0), (30.0 + radius, 22.0)) in draw_list.lines
    assert ((30.0, 22.0 - radius), (30.0, 22.0 + radius)) in draw_list.lines
