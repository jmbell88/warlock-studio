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
