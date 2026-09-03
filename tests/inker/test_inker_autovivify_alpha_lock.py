"""Autovivifying a cel must carry *every* track property, alpha lock included.

``_ensure_cel_for`` builds the real ``Layer`` that replaces an empty slot's
shared placeholder, and it copied four of the five properties a track owns. The
missing one was ``alpha_lock`` -- "preserve transparency" -- and the write that
autovivifies a cel is routinely the same statement that reads it back:
``begin_stroke`` samples ``layer.alpha_lock`` one line after
``_ensure_active_cel``. So the omission was not a mislabelled cel, it was the
lock silently off for the first gesture on every fresh frame and on again for
the second.

``animation.placeholder`` and ``animation.layers_for`` both copy all five, which
makes ``_ensure_cel_for`` the third copy of one list; the tests below assert the
three agree rather than asserting the one that happened to be wrong.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
TRACK_PROPS = ("name", "opacity", "visible", "blend", "alpha_lock")


def _animated(width: int = 8, height: int = 8) -> Document:
    """A two-frame document with the playhead on an empty slot.

    ``add_frame`` with neither ``link`` nor ``copy`` adds a frame holding no
    cels at all, which is exactly the state autovivification exists for.
    """
    doc = Document.blank(width, height)
    doc.add_frame()
    assert doc.anim is not None and doc.anim.is_placeholder(doc.stack.active)
    return doc


def test_the_autovivified_cel_carries_all_five_track_properties():
    doc = _animated()
    doc.set_layer_props(alpha_lock=True, opacity=0.5, visible=False, blend="multiply")
    placeholder = doc.stack.active
    before = {key: getattr(placeholder, key) for key in TRACK_PROPS}

    doc._ensure_cel_for(placeholder.uid)

    cel = doc.stack.active
    assert doc.anim is not None
    assert not doc.anim.is_placeholder(cel)
    assert cel.uid == placeholder.uid, "the cel keeps the slot's uid"
    assert {key: getattr(cel, key) for key in TRACK_PROPS} == before
    # And ``layers_for``, the third copy of the same list, agrees with both.
    rebuilt = doc.anim.layers_for(doc.anim.frame, doc.size)[0]
    assert {key: getattr(rebuilt, key) for key in TRACK_PROPS} == before


def test_a_stroke_begun_on_a_fresh_cel_sees_the_lock():
    """The reported failure path, one line long: ``begin_stroke`` autovivifies
    and then reads the lock off the layer it just made."""
    doc = _animated()
    doc.set_layer_props(alpha_lock=True)
    doc.begin_stroke((2.0, 2.0), RED, size=4)
    assert doc._stroke is not None
    assert doc._stroke.alpha_lock is True
    doc.end_stroke()


def test_preserve_transparency_holds_on_the_first_stroke_of_a_fresh_frame():
    """The behaviour, not the flag. The new cel is transparent everywhere, so a
    locked stroke must leave it exactly as it found it -- and an unlocked one on
    the same document must not, or this would pass on a brush that never
    painted."""
    locked = _animated()
    locked.set_layer_props(alpha_lock=True)
    locked.begin_stroke((2.0, 2.0), RED, size=4)
    locked.stroke_to((5.0, 5.0))
    locked.end_stroke()
    assert int(locked.stack.active.pixels[..., 3].max()) == 0

    free = _animated()
    free.begin_stroke((2.0, 2.0), RED, size=4)
    free.stroke_to((5.0, 5.0))
    free.end_stroke()
    assert int(free.stack.active.pixels[..., 3].max()) > 0


def test_the_lock_is_about_alpha_and_the_cel_still_becomes_real():
    """Stated so the test above is not read as "the stroke did nothing".

    "Preserve transparency" is exactly *the alpha does not change*, which is the
    definition rather than an approximation of one: the brush still writes
    colour, and colour under zero alpha is invisible. So the stroke is a real
    change to the pixels, it pushes a step, and the cel it autovivified stays --
    which is what makes this different from the no-op case that takes its cel
    back out.
    """
    doc = _animated()
    doc.set_layer_props(alpha_lock=True)
    before = doc.history.head
    doc.begin_stroke((2.0, 2.0), RED, size=4)
    doc.stroke_to((5.0, 5.0))
    doc.end_stroke()
    assert doc.history.head != before
    assert doc.anim is not None
    assert not doc.anim.is_placeholder(doc.stack.active)
    assert int(doc.stack.active.pixels[..., 3].max()) == 0


def test_write_colour_on_a_fresh_unlocked_cel_still_paints():
    doc = _animated()
    assert doc.write_colour((0, 0, 4, 4), RED, np.ones((4, 4), dtype=np.float32))
    assert doc.stack.active.alpha_lock is False
    assert tuple(int(v) for v in doc.stack.active.pixels[0, 0]) == RED


# --- continuous tracks (Wave 2) -----------------------------------------------
#
# The sixth thing autovivification can be asked to do. A *continuous* track is
# one where a new cel starts as a copy of the drawing before it rather than
# blank -- Aseprite's "continuous layer", and the way a background or a held
# pose is drawn once and carried forward.
#
# A copy, not a link. Aseprite links; we copy. Linking would make the first
# stroke on the new frame edit the old one too, which is the opposite of what
# "carry it forward and change it" means, and unlinking afterwards is a verb
# the timeline already has.


def _continuous(frames: int = 3) -> Document:
    doc = Document.blank(8, 8)
    doc.write_colour((0, 0, 4, 4), RED, np.ones((4, 4), dtype=np.float32))
    for _ in range(frames - 1):
        doc.add_frame()
    doc.anim.tracks[0].continuous = True
    doc.history.clear()
    return doc


def test_a_continuous_track_autovivifies_a_copy_of_the_nearest_earlier_cel():
    doc = _continuous()
    source = doc.anim.cels[(doc.anim.tracks[0].uid, doc.anim.frames[0].uid)]
    doc.set_current_frame(2)
    assert doc.anim.is_placeholder(doc.stack.active)

    assert doc.write_colour((6, 6, 7, 7), (0, 0, 255, 255), np.ones((1, 1), np.float32))
    fresh = doc.stack.active
    assert not doc.anim.is_placeholder(fresh)
    # A copy: the earlier drawing is there, the new pixel is on top, and the
    # object is not the one frame 0 holds.
    assert fresh is not source
    assert tuple(int(v) for v in fresh.pixels[0, 0]) == RED
    assert tuple(int(v) for v in fresh.pixels[6, 6]) == (0, 0, 255, 255)
    assert tuple(int(v) for v in source.pixels[6, 6]) == (0, 0, 0, 0)


def test_a_continuous_track_with_no_earlier_cel_autovivifies_blank():
    doc = Document.blank(8, 8)
    doc.add_frame()
    doc.anim.tracks[0].continuous = True
    doc.set_current_frame(1)

    assert doc.write_colour((0, 0, 2, 2), RED, np.ones((2, 2), dtype=np.float32))
    assert int(doc.stack.active.pixels[4, 4, 3]) == 0


def test_a_continuous_track_walks_back_past_empty_frames():
    """The *nearest earlier* cel, not the first one: a gap in the middle of a
    track must not send the copy all the way back to frame zero."""
    doc = _continuous(4)
    doc.set_current_frame(1)
    doc.write_colour((4, 4, 6, 6), (0, 255, 0, 255), np.ones((2, 2), np.float32))
    doc.set_current_frame(3)

    assert doc.write_colour((7, 7, 8, 8), (0, 0, 255, 255), np.ones((1, 1), np.float32))
    assert tuple(int(v) for v in doc.stack.active.pixels[4, 4]) == (0, 255, 0, 255)


def test_a_discontinuous_track_still_autovivifies_blank():
    """The negative control: nothing about a plain track changes."""
    doc = _continuous()
    doc.anim.tracks[0].continuous = False
    doc.set_current_frame(2)
    assert doc.write_colour((6, 6, 7, 7), RED, np.ones((1, 1), dtype=np.float32))
    assert int(doc.stack.active.pixels[0, 0, 3]) == 0


def test_a_continuous_indexed_track_copies_the_index_plane_too():
    doc = _continuous()
    assert doc.convert_to_indexed([(0, 0, 0, 255), RED])
    doc.check_materialized()
    doc.set_current_frame(2)

    assert doc.write_colour((6, 6, 7, 7), RED, np.ones((1, 1), dtype=np.float32))
    doc.check_materialized()
    assert doc.stack.active.indices is not None
    assert int(doc.stack.active.indices[0, 0]) == 1


def test_undoing_a_first_draw_on_a_continuous_track_removes_the_cel():
    """The copy rides the same compound the blank cel always did, so one
    Ctrl+Z takes the stroke and the cel it conjured together."""
    doc = _continuous()
    doc.set_current_frame(2)
    assert doc.write_colour((6, 6, 7, 7), (0, 0, 255, 255), np.ones((1, 1), np.float32))
    assert not doc.anim.is_placeholder(doc.stack.active)

    assert doc.history.undo(doc)
    assert doc.anim.is_placeholder(doc.stack.active)


# --- the one write path that refused instead of autovivifying ----------------


def test_a_regeneration_lands_on_an_empty_animated_cel():
    """``apply_pixels`` is the landing for a picture the image model returned
    while the user carried on editing, and it is addressed by uid because the
    active layer may have moved. On an empty animated cel the uid captured at
    submit is the *placeholder's*, and this refused it -- so "Regenerate
    selection" on an empty frame always came back "The layer it was for is
    gone", which was both a refusal of the ordinary case and untrue.
    """
    doc = _animated()
    uid = doc.stack.active.uid
    assert doc.anim is not None and doc.anim.is_placeholder(doc.stack.active)
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 0] = 255
    pixels[..., 3] = 255

    assert doc.apply_pixels(uid, (0, 0, 4, 4), pixels) is True

    cel = doc.stack.active
    assert not doc.anim.is_placeholder(cel)
    assert cel.uid == uid, "the cel keeps the slot's uid the caller held"
    assert tuple(cel.pixels[0, 0]) == RED
    # And the blank plane every other empty slot shares is still blank.
    assert doc.anim.blank_plane(*doc.size)[0, 0].tolist() == [0, 0, 0, 0]


def test_a_regeneration_that_lands_on_an_empty_cel_is_one_undo_step():
    """Autovivification plus the blend, not two steps: the placeholder is not
    a state the history should be able to stop at."""
    doc = _animated()
    uid = doc.stack.active.uid
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)

    assert doc.apply_pixels(uid, (0, 0, 4, 4), pixels) is True
    doc.undo()

    assert doc.anim is not None
    assert tuple(doc.stack.active.pixels[0, 0]) == (0, 0, 0, 0)


def test_a_regeneration_is_still_refused_when_the_layer_has_gone():
    """The refusal survives for what it was really about."""
    doc = _animated()
    pixels = np.full((4, 4, 4), 255, dtype=np.uint8)

    assert doc.apply_pixels(9_999_999, (0, 0, 4, 4), pixels) is False
