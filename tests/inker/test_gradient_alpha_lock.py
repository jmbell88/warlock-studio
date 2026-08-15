""""Preserve transparency" held for every tool except the gradient.

``write_colour`` -- which is fill, shapes and every flat-colour write -- restores
the alpha channel after its composite, and the docstring beside it argues that
restoring the channel *is* the definition of the lock rather than an
approximation of it. ``gradient`` performs the same composite with the colour
varying per pixel and did not restore anything, so a ramp dragged across a
locked layer filled its transparent part in.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(*, locked: bool) -> Document:
    """A layer that is opaque on the left half and empty on the right."""
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:, :4] = (30, 30, 30, 255)
    doc.set_layer_props(alpha_lock=locked)
    return doc


def test_a_ramp_over_a_locked_layer_writes_no_new_alpha():
    doc = _doc(locked=True)

    assert doc.gradient((0.0, 0.0), (7.0, 0.0), RED, BLUE) is True

    alpha = doc.stack.active.pixels[..., 3]
    assert np.array_equal(alpha[:, :4], np.full((8, 4), 255, dtype=np.uint8))
    assert not alpha[:, 4:].any(), "the empty half stayed empty"


def test_the_lock_is_about_alpha_and_the_colours_still_move():
    doc = _doc(locked=True)
    before = doc.stack.active.pixels[:, :4, :3].copy()

    doc.gradient((0.0, 0.0), (7.0, 0.0), RED, BLUE)

    assert not np.array_equal(doc.stack.active.pixels[:, :4, :3], before)


def test_an_unlocked_layer_is_unchanged_by_the_fix():
    doc = _doc(locked=False)

    doc.gradient((0.0, 0.0), (7.0, 0.0), RED, BLUE)

    assert doc.stack.active.pixels[:, 4:, 3].any(), "the ramp filled the empty half"


def test_the_lock_survives_the_selection_clip():
    """The two multiply into the same write, so a feathered selection must not
    reintroduce the alpha the lock is holding back."""
    doc = _doc(locked=True)
    doc.select_all()

    doc.gradient((0.0, 0.0), (7.0, 0.0), RED, BLUE)

    assert not doc.stack.active.pixels[:, 4:, 3].any()
